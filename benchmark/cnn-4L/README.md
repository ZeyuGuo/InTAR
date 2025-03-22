### InTAR

CNN kernel in InTAR design

```sh
make cnn-4L-intrra
./cnn-4L-intrra --bitstream bitstreams/CNN4L_xilinx_u280_xdma_201920_3.xclbin
```

### Spatial

Dataflow accelerator for CNN kernel

```sh
cd spatial
make build/cnn-4L-spatial
./build/cnn-4L-spatial --bitstream bitstreams/CNN4L_xilinx_u280_xdma_201920_3_hw.xclbin
```

### Sequential

Sequential accelerator for CNN kernel

```sh
cd sequential
make build/cnn-4L-sequential
./build/cnn-4L-sequential --bitstream bitstreams/CNN4L_xilinx_u280_xdma_201920_3_hw.xclbin
```
