tapac \
    -o w3a6.hw.xo \
    --platform xilinx_u280_xdma_201920_3 \
    --top w3a6o48linear \
    --work-dir w3a6.hw.xo.tapa \
    --enable-synth-util \
    --max-parallel-synth-jobs 16 \
    w3a6o48linear_kernel.cpp 