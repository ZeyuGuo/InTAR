### InTAR

Variational Autoencoder kernel in InTAR design

```sh
make vae-intrra
./vae-intrra --bitstream bitstreams/VAE_xilinx_u280_xdma_201920_3.xclbin
```

### Spatial

Dataflow accelerator for Variational Autoencoder kernel

```sh
cd spatial
make vae-spatial
./vae-spatial --bitstream bitstream/VAE_xilinx_u280_xdma_201920_3.xclbin
```

### Sequential

Sequential accelerator for Variational Autoencoder kernel

```sh
cd sequential
make build/vae-sequential
./build/vae-sequential --bitstream bitstreams/VAE_xilinx_u280_xdma_201920_3_hw.xclbin
```
