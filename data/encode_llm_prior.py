"""
Module 1: LLM-driven Semantic Prior Encoding (offline pre-computation).

For every entity e in E and relation r in R of a TKG dataset, build a natural
language prompt from its name (and parsed description attributes), feed it to a
frozen pretrained LLM (Qwen2.5-1.5B by default), and mean-pool the last layer
hidden states to obtain text-view semantic embeddings:

    h_e^text  in R^{D_text}
    h_r^text  in R^{D_text}

The resulting vectors are saved next to the dataset files as

    data/<DATASET>/entity_text_emb.pt   shape: (num_ents, D_text)
    data/<DATASET>/relation_text_emb.pt shape: (num_rels, D_text)

The main model loads these tensors once at construction time, projects them
through a learnable linear matrix W_proj followed by LayerNorm, and adds the
result residually to the random structural embeddings to form h_e^(0) and
h_r^(0) (see src/rrgcn.py).

Example:
    python data/encode_llm_prior.py -d ICEWS14 \
        --llm-path ./Qwen2.5-1.5B --gpu 0 --batch-size 16
"""
import argparse
import os
import re
import sys

import torch
from tqdm import tqdm


def load_index(input_path):
    items = []
    with open(input_path, encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) != 2:
                continue
            name, idx = parts
            items.append((int(idx), name))
    items.sort(key=lambda x: x[0])
    if items and items[0][0] != 0:
        raise RuntimeError(f"{input_path} ids do not start at 0")
    for expected, (idx, _) in enumerate(items):
        if idx != expected:
            raise RuntimeError(f"{input_path} ids are not contiguous at {idx}")
    return [name for _, name in items]


def normalize_entity_name(raw):
    name = raw.replace("_", " ").strip()
    m = re.match(r"^(.*?)\s*\((.*)\)\s*$", name)
    if m:
        head, tail = m.group(1).strip(), m.group(2).strip()
        if head and tail:
            return f"{head} (affiliated with {tail})"
    return name


def normalize_relation_name(raw):
    return raw.replace("_", " ").strip()


def build_entity_prompt(raw):
    canon = normalize_entity_name(raw)
    return (
        f"In a temporal knowledge graph of geopolitical events, the entity is "
        f"\"{canon}\". This entity participates in events involving political, "
        f"diplomatic, military, social, or economic actions."
    )


def build_relation_prompt(raw):
    canon = normalize_relation_name(raw)
    return (
        f"In a temporal knowledge graph of geopolitical events, the relation "
        f"\"{canon}\" describes an action, interaction, or stance between two "
        f"actors at a specific point in time."
    )


@torch.no_grad()
def encode_texts(texts, tokenizer, model, device, batch_size, max_length):
    feats = []
    model.eval()
    for start in tqdm(range(0, len(texts), batch_size), desc="LLM encode"):
        batch = texts[start:start + batch_size]
        enc = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        ).to(device)
        out = model(**enc, use_cache=False)
        last_hidden = out.last_hidden_state                  # [B, L, H]
        mask = enc["attention_mask"].unsqueeze(-1).float()   # [B, L, 1]
        pooled = (last_hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
        feats.append(pooled.float().cpu())
    return torch.cat(feats, dim=0)


def main():
    parser = argparse.ArgumentParser(description="Module 1 - LLM semantic prior encoding")
    parser.add_argument("-d", "--dataset", type=str, required=True,
                        choices=["ICEWS14", "ICEWS18", "ICEWS05-15", "GDELT"])
    parser.add_argument("--llm-path", type=str, default="./Qwen2.5-1.5B",
                        help="path to local Qwen2.5-1.5B checkpoint")
    parser.add_argument("--gpu", type=int, default=0,
                        help="gpu id, set -1 to run on cpu")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=64)
    parser.add_argument("--dtype", type=str, default="bf16",
                        choices=["fp16", "bf16", "fp32"])
    parser.add_argument("--data-root", type=str, default="./data")
    parser.add_argument("--out-dir", type=str, default=None,
                        help="defaults to <data-root>/<dataset>")
    args = parser.parse_args()

    try:
        from transformers import AutoModel, AutoTokenizer
    except ImportError as e:
        print("ERROR: transformers is required. pip install 'transformers>=4.43'",
              file=sys.stderr)
        raise e

    data_dir = os.path.join(args.data_root, args.dataset)
    out_dir = args.out_dir or data_dir
    os.makedirs(out_dir, exist_ok=True)

    entity_names = load_index(os.path.join(data_dir, "entity2id.txt"))
    relation_names = load_index(os.path.join(data_dir, "relation2id.txt"))
    print(f"[{args.dataset}] #entities={len(entity_names)}  "
          f"#relations={len(relation_names)}")

    use_cuda = args.gpu >= 0 and torch.cuda.is_available()
    device = torch.device(f"cuda:{args.gpu}") if use_cuda else torch.device("cpu")
    dtype_map = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}
    torch_dtype = dtype_map[args.dtype] if use_cuda else torch.float32

    print(f"Loading LLM from {args.llm_path} (dtype={torch_dtype}) -> {device} ...")
    tokenizer = AutoTokenizer.from_pretrained(args.llm_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModel.from_pretrained(
        args.llm_path,
        trust_remote_code=True,
        torch_dtype=torch_dtype,
    )
    model.to(device)
    for p in model.parameters():
        p.requires_grad_(False)

    ent_prompts = [build_entity_prompt(n) for n in entity_names]
    rel_prompts = [build_relation_prompt(n) for n in relation_names]
    print("Sample entity prompt :", ent_prompts[0])
    print("Sample relation prompt:", rel_prompts[0])

    print("Encoding entities ...")
    ent_feat = encode_texts(ent_prompts, tokenizer, model, device,
                            args.batch_size, args.max_length)
    print("Encoding relations ...")
    rel_feat = encode_texts(rel_prompts, tokenizer, model, device,
                            args.batch_size, args.max_length)

    print(f"entity_text_emb shape   = {tuple(ent_feat.shape)}")
    print(f"relation_text_emb shape = {tuple(rel_feat.shape)}")

    ent_path = os.path.join(out_dir, "entity_text_emb.pt")
    rel_path = os.path.join(out_dir, "relation_text_emb.pt")
    torch.save(ent_feat, ent_path)
    torch.save(rel_feat, rel_path)
    print(f"Saved -> {ent_path}")
    print(f"Saved -> {rel_path}")


if __name__ == "__main__":
    main()
