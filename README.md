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
- [ ] Reproducing results on OpenWebText dataset
- [x] Loading GPTs pre-trained models
- [ ] Fine tuning GPT-2 weights on Shakespear dataset
- [ ] Profiling training iteration, estimating MFU (Model flops utilization)
- [ ] Optimizing Inference
- [ ] Flash attention with Pallas

## Future work

- Experimenting with Jax tensor sharding
- Experimenting with advanced fine-tuning techniques
- ...

## Acknowledgement
...

## References

