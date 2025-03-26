## Multi-Task DNN Kernel Testbench

For detailed description of each kernel, check the [preprint](https://arxiv.org/abs/2502.08807) paper in Table III.

> [!NOTE]
> Speedup and DSP efficiency are normalized based on the sequential kernel. For verification, you can divide the value by whatever value for sequential kernel to get the speedup or DSP efficiency.

| Kernel Name | Cycle Count (InTAR/dataflow/sequential) | Latency (ms) (InTAR/dataflow/sequential) | DSP count (InTAR/dataflow/sequential) |
| ---- | ---- | ---- | ---- |
| [Self Attention](self-attn/README.md) (`self-attn`) | 40520051/73143507/348574419 | 139/252/1267 | 43/51/52|
| [FFN Layers](mlp/README.md) (`mlp`) | 140957/181996/1374935 | 0.582/0.742/4.77 | 32/36/36 |
| [Multi-layer CNN](cnn-4L/README.md) (`cnn-4L`) | 118729/713144/2215282 | 0.546/2.53/7.63 | 439/583/580 |
| [Variational Autoencoder](vae/README.md) (`vae`) | 10295/28383/306935 | 0.145/0.232/1.6 | 239/183/190 |
| [Gating Network](gating-net/README.md) (`gating-net`) | 48668637/48668637/117702892 | 211/211/446 | 109/109/104 |

Please `cd` to the corresponding folder first, then following the instructions in the link.
