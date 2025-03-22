### InTAR

MLP kernel in InTAR design

```sh
make mlp-intrra
./mlp-intrra --bitstream bitstreams/MLP_xilinx_u280_xdma_201920_3.xclbin
```

### Spatial

Dataflow accelerator for MLP kernel

```sh
cd spatial
make build/mlp-spatial
./build/mlp-spatial --bitstream bitstreams/MLP_xilinx_u280_xdma_201920_3_hw.xclbin
```

### Sequential

Sequential accelerator for MLP kernel

```sh
cd sequential
make build/mlp-sequential
./build/mlp-sequential --bitstream bitstreams/MLP_xilinx_u280_xdma_201920_3_hw.xclbin
```
