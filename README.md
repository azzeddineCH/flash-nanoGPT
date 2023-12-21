# flash-nanoGPT (Under developement)

a jax (flax) re-write of Andrej Karpathy NanoGPT, this repository will hold a collection of Jax/Flax new features like :
Pallas kernel language for flashAttention on TPU, Data and tensor sharding with Jax on TPU

## Todos

- [x] GPT model in flax
- [x] Core data pipeline
- [x] Core training loop
- [x] Mixed precision training
- [x] Gradient accumulation
- [x] Data sharding across GPUs/TPUs
- [x] Loading and Saving checkpoints
- [x] Reproduce the results on shakespear-char datasets
- [ ] Multi-node support
- [ ] Reproducing results on OpenWebText dataset if compute is available
- [ ] Profiling/Benchmarking and speeding up iteration
- [ ] Batch pre-fetching
- [ ] Loading GPT-2 weights
- [ ] Fine tuning GPT-2 weights
- [ ] Flash attention with Pallas