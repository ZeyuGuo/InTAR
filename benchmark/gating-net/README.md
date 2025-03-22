### InTAR / Spatial

Gating Network kernel in InTAR design. The schedule is the same as the dataflow accelerator.

```sh
cd spatial
make build/gating-net-spatial
./build/gating-net-spatial --bitstream bitstreams/gating_net_spatial_xilinx_u280_xdma_201920_3_hw.xclbin
```

### Sequential

Sequential accelerator for Gating Network kernel

```sh
cd sequential
make build/gating-net-sequential
./build/gating-net-sequential --bitstream bitstreams/gating_net_sequential_xilinx_u280_xdma_201920_3_hw.xclbin
```
