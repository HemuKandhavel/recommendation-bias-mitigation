# recommendation-bias-mitigation
A GRU-based sequential recommendation model that mitigates popularity bias using temporal user behavior modeling. Reduces popular item bias from ~100% to ~70% while increasing long-tail exposure to ~30%. Evaluated using Recall@K and NDCG metrics. Deployed with Gradio.
# Mitigating Popularity Bias in Recommendation Systems

A sequential recommendation system using GRU-based temporal modeling to reduce 
popularity bias and increase exposure of long-tail items.

## Overview
Traditional recommendation systems tend to recommend only popular items, ignoring 
niche content that users might actually enjoy. This project addresses that bias using 
temporal sequential modeling on real user interaction data.

## Tech Stack
- Python, TensorFlow, Keras
- GRU (Gated Recurrent Unit)
- SVD (baseline comparison)
- Gradio (deployment)
- Amazon dataset

## Results
| Metric | Value |
|--------|-------|
| Popular item bias (baseline SVD) | ~100% |
| Popular item bias (GRU model) | ~70% |
| Long-tail item exposure | ~30% |
| Evaluation metrics | Recall@K, NDCG |



## Author
**Hema Priya K**   
📧 hemukandhavel01@gmail.com
