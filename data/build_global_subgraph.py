"""
Module 3: Random-Walk-with-Restart (RWR) global subgraph sampling.

Replaces the 2-hop neighbour sampling done by ``data/get_his_subg.py``. For
every training timestep ``t_q`` and every query at that step we

  1. Restrict edges to the historical window ``[t_q - m, t_q - 1]``.
  2. Build a sparse transition matrix whose non-zero weights combine
     time decay and deep relation-semantic similarity:
         P_{u,v} ~ sum_{r', tau} exp(-lambda * (t_q - tau)) * exp(sim(h_{r'}^0, h_r^0))
     where the relation embeddings ``h_*^0`` are taken from
     ``relation_text_emb.pt`` (Module 1's offline LLM encoding) so that
     this script remains query-aware without depending on training state.
  3. Run the iteration  pi^k = (1 - alpha) P^T pi^(k-1) + alpha e_s  for K
     steps starting from each query subject ``s``.
  4. Take the union of the Top-N highest-scoring nodes across all queries
     at this timestep and emit the corresponding edges in the standard
     ``[src, rel, dst, freq]`` layout used by ``rgcn.utils.build_graph``.

Outputs are saved under
    data/<DATASET>/his_graph_for/train_s_r_{t}.npy   # forward subgraph
    data/<DATASET>/his_graph_inv/train_o_r_{t}.npy   # inverse subgraph
    data/<DATASET>/his_dict/train_s_r.npy            # sr -> dst dict
which is the exact format consumed by ``src/main.py``.

Example::
    python data/build_global_subgraph.py -d ICEWS14 \
        --window 7 --alpha 0.15 --num-iters 4 --top-n 50 --time-decay 0.5
"""
import argparse
import os
import sys
from collections import defaultdict

import numpy as np
import torch
from tqdm import tqdm

try:
    import scipy.sparse as sp
except ImportError as e:
    print("ERROR: scipy is required. pip install scipy", file=sys.stderr)
    raise e


def load_quadruples(path):
    """Read a (head, rel, tail, time) file. Times must be integer-coded."""
    quads = []
    with open(path, "r", encoding="utf-8") as fr:
        for line in fr:
            parts = line.split()
            head = int(parts[0])
            rel = int(parts[1])
            tail = int(parts[2])
            t = int(parts[3])
            quads.append((head, rel, tail, t))
    return np.asarray(quads, dtype=np.int64)


def get_total_number(stat_path):
    with open(stat_path, "r", encoding="utf-8") as fr:
        for line in fr:
            parts = line.split()
            return int(parts[0]), int(parts[1])


def split_by_time(data):
    """Group rows of (h, r, t, time) by their time column, sorted ascending."""
    snapshots = []
    if len(data) == 0:
        return snapshots, []
    order = np.argsort(data[:, 3], kind="stable")
    data = data[order]
    times_unique, idx = np.unique(data[:, 3], return_index=True)
    idx = np.append(idx, len(data))
    for k in range(len(times_unique)):
        snap = data[idx[k]:idx[k + 1], :3]
        snapshots.append(snap.astype(np.int64))
    return snapshots, times_unique.tolist()


def update_sr_dict(snap, num_rels, sr_to_sro):
    """Accumulate (src, rel) -> {dst} for both directions (with inverse rel)."""
    inv = snap[:, [2, 1, 0]].copy()
    inv[:, 1] += num_rels
    for src, rel, dst in np.concatenate([snap, inv], axis=0):
        sr_to_sro[(int(src), int(rel))].add(int(dst))


def build_window_edges(snapshots_in_window, num_rels):
    """Materialise (src, rel, dst, time_index) for a list of snapshots.

    ``time_index`` is the position of the snapshot within the window: 0 is the
    oldest snapshot and ``len(snapshots)-1`` is the most recent one.
    """
    rows = []
    for i, snap in enumerate(snapshots_in_window):
        if snap.size == 0:
            continue
        # forward edges
        forward = np.column_stack([snap[:, 0], snap[:, 1], snap[:, 2],
                                    np.full(len(snap), i, dtype=np.int64)])
        # inverse edges (so that random walks can follow either direction).
        inverse = np.column_stack([snap[:, 2], snap[:, 1] + num_rels, snap[:, 0],
                                    np.full(len(snap), i, dtype=np.int64)])
        rows.append(forward)
        rows.append(inverse)
    if not rows:
        return np.empty((0, 4), dtype=np.int64)
    return np.concatenate(rows, axis=0)


def compute_relation_similarity(text_emb):
    """Pre-compute exp(sim(h_{r'}, h_r)) for every (r', r) pair.

    ``text_emb`` is a numpy array of shape [num_rels, D]. We L2 normalise
    each row and use cosine similarity, returning a [num_rels, num_rels]
    matrix on cpu in float32 (small enough for ICEWS scales).
    """
    norm = np.linalg.norm(text_emb, axis=1, keepdims=True) + 1e-8
    unit = text_emb / norm
    sim = unit @ unit.T                 # cosine in [-1, 1]
    return np.exp(sim).astype(np.float32)


def rwr_topn(P_T, seed, num_iters, alpha, top_n):
    """Run RWR for ``num_iters`` steps starting at ``seed`` and return the
    indices of the Top-N nodes in the final probability vector ``pi``.

    P_T: scipy sparse CSR matrix, shape [N, N], representing P^T.
    seed: int, query subject id.
    """
    n = P_T.shape[0]
    e_s = np.zeros(n, dtype=np.float32)
    e_s[seed] = 1.0
    pi = e_s.copy()
    for _ in range(num_iters):
        pi = (1.0 - alpha) * (P_T @ pi) + alpha * e_s
    if top_n >= n:
        return np.arange(n)
    # Always keep the seed itself.
    pi[seed] = max(pi[seed], 1.0)
    idx = np.argpartition(-pi, top_n - 1)[:top_n]
    return idx


def build_query_subgraph(window_edges, query_subjects_with_rel, num_nodes,
                         num_rels, rel_sim_exp, time_decay, num_iters, alpha,
                         top_n, time_index_of_now):
    """Run RWR per (subject, query_relation) and emit a single union subgraph.

    Returns a numpy array of shape [E, 4]: ``(src, rel, dst, freq)``.
    """
    if window_edges.size == 0 or len(query_subjects_with_rel) == 0:
        return np.empty((0, 4), dtype=np.int64)

    src = window_edges[:, 0]
    rel = window_edges[:, 1]
    dst = window_edges[:, 2]
    t_idx = window_edges[:, 3]
    # exp(-lambda * (t_q - tau)); the most recent snapshot in the window
    # has t_idx == time_index_of_now, the oldest is 0.
    delta_t = (time_index_of_now - t_idx).astype(np.float32)
    time_w = np.exp(-time_decay * delta_t)

    # Per-query loop: for each query relation we build a different P (cheap
    # because we reuse the structural sparsity).
    # Pre-aggregate same-(src, dst, r') multiplicity to amortise sparse build.
    selected_nodes = set()
    for (s, r) in query_subjects_with_rel:
        if s < 0 or s >= num_nodes:
            continue
        # Edge weights = time decay * exp(sim(h_{r'}, h_r))
        edge_w = time_w * rel_sim_exp[rel, r]
        if edge_w.sum() == 0.0:
            selected_nodes.add(int(s))
            continue
        A = sp.csr_matrix((edge_w, (src, dst)), shape=(num_nodes, num_nodes))
        # Row-normalise to obtain the row-stochastic P (zero-rows stay zero).
        row_sum = np.asarray(A.sum(axis=1)).ravel()
        nz = row_sum > 0
        inv = np.zeros_like(row_sum)
        inv[nz] = 1.0 / row_sum[nz]
        D_inv = sp.diags(inv)
        P = D_inv @ A
        P_T = P.T.tocsr()

        top = rwr_topn(P_T, int(s), num_iters, alpha, top_n)
        selected_nodes.update(int(x) for x in top)
        selected_nodes.add(int(s))

    if not selected_nodes:
        return np.empty((0, 4), dtype=np.int64)

    selected = np.fromiter(selected_nodes, dtype=np.int64)
    sel_set = set(selected.tolist())
    keep = np.fromiter((int(u) in sel_set and int(v) in sel_set
                        for u, v in zip(src, dst)),
                       dtype=bool, count=len(src))
    if not keep.any():
        return np.empty((0, 4), dtype=np.int64)
    sub_src = src[keep]
    sub_rel = rel[keep]
    sub_dst = dst[keep]
    # Aggregate frequencies across the window for each (src, rel, dst).
    triples = np.stack([sub_src, sub_rel, sub_dst], axis=1)
    triples_view = np.ascontiguousarray(triples).view(
        [("", triples.dtype)] * triples.shape[1])
    uniq, counts = np.unique(triples_view, return_counts=True)
    uniq = uniq.view(triples.dtype).reshape(-1, triples.shape[1])
    out = np.column_stack([uniq, counts.astype(np.int64)])
    return out.astype(np.int64)


def split_forward_inverse(query_subjects_with_rel, num_rels):
    """For Module 3 the forward subgraph is seeded at original subjects /
    relations, while the inverse subgraph mirrors the LogCL convention by
    using object-as-subject and ``r + num_rels``.
    """
    forward = []
    inverse = []
    for (s, r, o) in query_subjects_with_rel:
        forward.append((int(s), int(r)))
        inverse.append((int(o), int(r) + num_rels))
    return forward, inverse


def main():
    parser = argparse.ArgumentParser(description="Module 3 - RWR global subgraph sampling")
    parser.add_argument("-d", "--dataset", type=str, required=True,
                        choices=["ICEWS14", "ICEWS18", "ICEWS05-15", "GDELT"])
    parser.add_argument("--window", type=int, default=7,
                        help="number of historical snapshots used per t_q")
    parser.add_argument("--alpha", type=float, default=0.15,
                        help="RWR restart probability")
    parser.add_argument("--num-iters", type=int, default=4,
                        help="number of RWR iterations K")
    parser.add_argument("--top-n", type=int, default=50,
                        help="keep the Top-N nodes per query")
    parser.add_argument("--time-decay", type=float, default=0.5,
                        help="lambda in the temporal decay term exp(-lambda*Delta t)")
    parser.add_argument("--data-root", type=str, default="./data")
    parser.add_argument("--llm-emb-dir", type=str, default=None,
                        help="directory holding relation_text_emb.pt; defaults to <data-root>/<dataset>")
    args = parser.parse_args()

    data_dir = os.path.join(args.data_root, args.dataset)
    llm_emb_dir = args.llm_emb_dir or data_dir

    train_data = load_quadruples(os.path.join(data_dir, "train.txt"))
    num_nodes, num_rels = get_total_number(os.path.join(data_dir, "stat.txt"))

    rel_path = os.path.join(llm_emb_dir, "relation_text_emb.pt")
    if not os.path.isfile(rel_path):
        raise FileNotFoundError(
            f"{rel_path} not found. Run data/encode_llm_prior.py first.")
    rel_text = torch.load(rel_path, map_location="cpu").float().numpy()
    if rel_text.shape[0] != num_rels:
        raise RuntimeError(
            f"relation_text_emb.pt has {rel_text.shape[0]} rows but stat.txt "
            f"says num_rels={num_rels}.")
    # The transition matrix doubles the relation index for inverse edges, so
    # we mirror the relation similarity tensor too: r' ∈ [0, 2*num_rels).
    rel_text_full = np.concatenate([rel_text, rel_text], axis=0)
    rel_sim_exp = compute_relation_similarity(rel_text_full)

    save_dir_subg = os.path.join(data_dir, "his_graph_for")
    save_dir_obj = os.path.join(data_dir, "his_graph_inv")
    save_dir_sub = os.path.join(data_dir, "his_dict")
    os.makedirs(save_dir_subg, exist_ok=True)
    os.makedirs(save_dir_obj, exist_ok=True)
    os.makedirs(save_dir_sub, exist_ok=True)

    snapshots, time_keys = split_by_time(train_data)
    print(f"[{args.dataset}] training snapshots = {len(snapshots)}, "
          f"window = {args.window}, alpha = {args.alpha}, K = {args.num_iters}, "
          f"Top-N = {args.top_n}, lambda = {args.time_decay}")

    sr_to_sro = defaultdict(set)

    for t in tqdm(range(len(snapshots)), desc="RWR subgraph sampling"):
        if t == 0:
            update_sr_dict(snapshots[0], num_rels, sr_to_sro)
            continue
        # Update the running (s, r) -> {dst} dictionary using the previous
        # snapshot, mirroring the LogCL ``get_his_subg.py`` semantics.
        update_sr_dict(snapshots[t - 1], num_rels, sr_to_sro)

        window_start = max(0, t - args.window)
        window_snaps = snapshots[window_start:t]
        if len(window_snaps) == 0:
            empty = np.empty((0, 4), dtype=np.int64)
            np.save(os.path.join(save_dir_subg, f"train_s_r_{t}.npy"), empty)
            np.save(os.path.join(save_dir_obj, f"train_o_r_{t}.npy"), empty)
            continue

        window_edges = build_window_edges(window_snaps, num_rels)
        time_index_of_now = len(window_snaps) - 1

        triples_now = snapshots[t]
        forward_queries = [(int(s), int(r)) for s, r, _ in triples_now]
        inverse_queries = [(int(o), int(r) + num_rels) for _, r, o in triples_now]

        sub_forward = build_query_subgraph(
            window_edges, forward_queries, num_nodes, num_rels,
            rel_sim_exp, args.time_decay, args.num_iters, args.alpha,
            args.top_n, time_index_of_now)
        sub_inverse = build_query_subgraph(
            window_edges, inverse_queries, num_nodes, num_rels,
            rel_sim_exp, args.time_decay, args.num_iters, args.alpha,
            args.top_n, time_index_of_now)

        np.save(os.path.join(save_dir_subg, f"train_s_r_{t}.npy"), sub_forward)
        np.save(os.path.join(save_dir_obj, f"train_o_r_{t}.npy"), sub_inverse)

    # The (s, r) -> {dst} dictionary is consumed by the online subgraph
    # sampler in ``src/main.py`` for valid / test, so we always emit it.
    np.save(os.path.join(save_dir_sub, "train_s_r.npy"), dict(sr_to_sro))
    print("Saved -> {}/<his_graph_for, his_graph_inv, his_dict>".format(data_dir))


if __name__ == "__main__":
    main()
