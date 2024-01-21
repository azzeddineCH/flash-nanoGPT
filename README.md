# flash-nanoGPT (Under development)

a jax (flax) re-write of Andrej Karpathy NanoGPT, this repository will hold a collection of Jax/Flax new features like :
Pallas kernel language for flashAttention on TPU, Data and tensor sharding with Jax on TPU

## Todos

- [x] GPT2 alike model in flax
- [x] Mixed precision training with jmp
- [x] Gradient accumulation with optax
- [x] Data sharding across GPUs/TPUs using the new Jax shmap
- [x] Loading and Saving checkpoints
- [x] Reproduce the results on shakespear-char dataset
- [x] TF Record reader/writer with support for data sharding across hosts
- [x] Multi-host training
- [x] Reproducing results on OpenWebText dataset
- [x] Loading huggingface GPTs pre-trained models
- [x] Fine tuning GPT-2 weights on Shakespear dataset
- [x] Sampling
- [x] Estimating MFU (Model flops utilization)
- [x] Profiling training iteration
- [ ] Flash attention with Pallas

## data generation
in order to run training using TPU VM, copy the generated data files into a GCP bucket

## Acknowledgement
Big thanks to [TPU Research Cloud](https://sites.research.google/trc/about/) for providing v2-8/v3-8/v3-32 TPU instances on Google Cloud.

## References
- Original nanoGPT repositories [[1]](https://github.com/karpathy/nanoGPT)
- jax based nanoGPT repositories [[1]](https://github.com/jenkspt/gpt-jax?tab=readme-ov-file) [[2]](https://github.com/cgarciae/nanoGPT-jax)
- Nvidia mixed precision training [[1]](https://docs.nvidia.com/deeplearning/performance/mixed-precision-training/index.html)
- Google Cloud documentation [[1]](https://cloud.google.com/tpu/docs/)

