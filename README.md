# Local-Global Disentangled Contrastive Learning for Temporal Knowledge Graph Reasoning
<img width="3605" height="1811" alt="图片1" src="https://github.com/user-attachments/assets/b66079fb-ef37-48e0-90a2-f19ef7787d3a" />

## Sensitivity Analysis
To evaluate the impact of the global subgraph sampling size on model performance, we test the hyperparameter N over {10, 30, 50, 100, 200, 300}. The results show that model performance first improves and then declines as N increases, which verifies the effectiveness of the RWR sampling strategy. When N is too small, such as N=10, the sampled subgraph misses key multi-hop topological information, leading to a substantial performance drop. In this case, the MRR on ICEWS14 is only 0.535.

The model achieves the best performance when N=50. At this scale, the transition matrix that incorporates temporal decay and relation similarity can effectively focus on high-value historical entities. This provides the best trade-off between structural information coverage and noise filtering. However, when N further increases to 200 or 300, a large amount of irrelevant global noise is introduced. This weakens the targeted focusing capability of RWR and causes performance to decline again.
<img width="3587" height="2391" alt="topn_mrr" src="https://github.com/user-attachments/assets/05f972c1-4285-40a0-9e6a-386c489bcc86" />

## Running Instructions
### Environment Requirements
```
pip install -r requirement.txt
pip install scipy transformers>=4.43
```

### LLM Semantic Prior Encoding
```
python data/encode_llm_prior.py -d ICEWS14 --llm-path ./Qwen2.5-1.5B --gpu 0 --dtype bf16
```

### RWR Global Subgraph Sampling
```
python data/build_global_subgraph.py \
    -d ICEWS14 \
    --window 7 \
    --alpha 0.15 \
    --num-iters 4 \
    --top-n 50 \
    --time-decay 0.5
```

### Train
```
python src/main.py \
    -d ICEWS14 \
    --train-history-len 7 \
    --test-history-len 7 \
    --dilate-len 1 \
    --lr 1e-3 \
    --n-layers 2 \
    --n-hidden 200 \
    --self-loop --layer-norm \
    --decoder convtranse --encoder uvrgcn \
    --pre-type all \
    --weight 0.5 --discount 1 --angle 10 \
    --entity-prediction \
    --use-llm-prior --llm-text-dim 1536 \
    --use-cl --temperature 0.07 --tau-hard 0.1 --cl-weight 0.5 \
    --evaluate-every 1 \
    --n-epochs 100 \
    --patience 30 \
    --dropout 0.3 --input-dropout 0.3 --hidden-dropout 0.3 --feat-dropout 0.3 \
    --gpu 0
```

### Test
```
python src/main.py \
    -d ICEWS14 \
    --train-history-len 7 \
    --test-history-len 7 \
    --dilate-len 1 \
    --lr 1e-3 \
    --n-layers 2 \
    --n-hidden 200 \
    --self-loop --layer-norm \
    --decoder convtranse --encoder uvrgcn \
    --pre-type all \
    --weight 0.5 --discount 1 --angle 10 \
    --entity-prediction \
    --use-llm-prior --llm-text-dim 1536 \
    --use-cl --temperature 0.07 --tau-hard 0.1 --cl-weight 0.5 \
    --gpu 0 \
    --test
```
