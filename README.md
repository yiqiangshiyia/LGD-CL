# LogCL
The code of LogCL

### Process data
First, unpack the data files. Then generate the offline history subgraphs and the
`(s, r) -> {dst}` dictionary used by valid / test sampling.

**Option A (original 2-hop sampler):**
```
python data/get_his_subg.py
```
Run from the `data/` directory, or adjust paths accordingly.

**Option B (Module 3 RWR sampler, window `[t_q - m, t_q - 1]`):**
```
python data/build_global_subgraph.py -d ICEWS14 \
    --window 7 --alpha 0.15 --num-iters 4 --top-n 50 --time-decay 0.5
```
Requires `relation_text_emb.pt` from Module 1 (`encode_llm_prior.py`).

Both scripts write `data/<DATASET>/his_graph_for/`, `data/<DATASET>/his_graph_inv/`
and `data/<DATASET>/his_dict/` in the format expected by `src/main.py`.

### Module 1: pre-compute the LLM semantic prior
Module 1 (LLM-driven semantic prior injection) replaces the original `e-w-graph`
static graph. The frozen LLM (Qwen2.5-1.5B by default) is invoked once **offline**
to encode every entity and relation into a text-view embedding. Run once per
dataset:
```
python data/encode_llm_prior.py -d ICEWS14 --llm-path ./Qwen2.5-1.5B --gpu 0
```
This writes `entity_text_emb.pt` and `relation_text_emb.pt` next to
`entity2id.txt`, which the main model loads at construction time and combines
with random structural embeddings via
`h^{(0)} = h^{struct} + LayerNorm(W_proj * h^{text})`.

### Module 2: query-guided local entity encoder
Module 2 replaces the GRU-based local sequential aggregator with a
**query-aware time attention**. For each historical snapshot `tau` in
`[t_q - m, t_q - 1]`, an L-layer R-GCN starts from the shared `h^(0)` features
(no cross-snapshot recurrent state) to produce per-entity features
`h_{i,tau}^(L)` and a snapshot summary `H_tau = AvgPool(h_{i,tau}^(L))`.
A query joint feature `q_emb = MLP([h_s^(0) || h_r^(0)])` is then used to score
each snapshot:
`e_tau = v^T tanh(W1 h_{i,tau} + W2 q_emb + W3 Delta t_tau)`,
followed by `alpha = softmax(e_tau)` and
`Z_local = sum_tau alpha_tau H_tau`.

### Module 3: relation-guided global entity encoder
Module 3 uses the offline history subgraphs produced by `data/get_his_subg.py`
(forward: `his_graph_for/`, inverse: `his_graph_inv/`) together with the online
RWR sampler in `RecurrentRGCN.rwr_sampler` and a standard R-GCN over the
extracted Top-N nodes.

### Module 4: local-global contrastive learning and prediction
Module 4 fuses the Module 2 local view `Z_local` and the Module 3 global view
`Z_global`, scores candidates against the LLM-fused `h^(0)` and trains under a
hard-negative-aware InfoNCE objective.

1. **Feature fusion + KG decoder**.
   `Z_fuse(s) = W_out * ReLU(W_in * [Z_local(s) || Z_global(s)])`.
   The ConvTransE decoder now scores every candidate `o` directly against the
   LLM-fused initial features:
   `Score(s, r, o, t_q) = Decoder(Z_fuse(s), h_r, h_o^(0))`.
   The previous ``--pre-weight`` weighted-sum mixing inside the decoder is
   removed; the flag is kept only for argparse compatibility.
2. **Hard-negative-aware InfoNCE**. For every batch query `q = (s, r, o, t_q)`
   and every entity `e_k` that appears in the RWR-sampled global sub-graph
   (excluding the true tail) we compute
   `beta_k = exp(sim(Z_local(q), Z_global(q_k^-)) / tau_hard)` and
   `L_CL = -log( exp(sim(Z_local, Z_global_pos)/tau)
                / (exp(...) + sum_k beta_k * exp(sim(...)/tau)) )`.
   Sub-graph membership realises the *structural* hard-negative set (high
   `pi_v` in the RWR ranking); the implementation in
   `RecurrentRGCN.get_loss_cl` is fully vectorised and can be extended with
   *semantic* hard negatives derived from the LLM relation embeddings.
3. **Joint loss**. `L_total = L_CE + lambda * L_CL`, controlled by
   ``--cl-weight`` and ``--tau-hard``.

### Train models
Then the following commands can be used to train the proposed models. By default, dev set evaluation results will be printed when training terminates.

1. Train models
```
python src/main.py -d ICEWS14 --train-history-len 7 --test-history-len 7 --dilate-len 1 --lr 0.001 --n-layers 2 --evaluate-every 1 --gpu=0 --n-hidden 200 --self-loop --decoder convtranse --encoder uvrgcn --layer-norm --weight 0.5  --entity-prediction --angle 10 --discount 1 --pre-type all --use-llm-prior --use-cl --temperature 0.03 --tau-hard 0.1 --cl-weight 0.5
```
Pass `--no-use-llm-prior` (or set `--llm-text-dim` / `--llm-emb-dir`) to ablate
or relocate the semantic prior. Drop ``--use-cl`` to fall back to a pure
cross-entropy objective.
### Cite
Please cite our paper if you find this code useful for your research.
~~~
@article{chen2023local,
  title={Local-Global History-aware Contrastive Learning for Temporal Knowledge Graph Reasoning},
  author={Chen, Wei and Wan, Huaiyu and Wu, Yuting and Zhao, Shuyuan and Cheng, Jiayaqi and Li, Yuxin and Lin, Youfang},
  journal={arXiv preprint arXiv:2312.01601},
  year={2023}
}
~~~


