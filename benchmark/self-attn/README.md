### InTAR

Self Attention kernel in InTAR design

```sh
make self-attn-intrra
./self-attn-intrra --bitstream bitstreams/selfAttention_xilinx_u280_xdma_201920_3.xclbin
```

### Spatial

Dataflow accelerator for Self Attention kernel

```sh
cd spatial
make build/self-attn-spatial
./build/self-attn-spatial --bitstream bitstreams/selfAttention_xilinx_u280_xdma_201920_3.xclbin
```

### Sequential

Sequential accelerator for Self Attention kernel

```sh
cd sequential
make build/self-attn-sequential
./build/self-attn-sequential --bitstream bitstreams/self_attn_sequential_xilinx_u280_xdma_201920_3_hw.xclbin
```
