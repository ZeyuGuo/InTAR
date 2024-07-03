tapac \
    -o fluid.hw.xo \
    --platform xilinx_u280_xdma_201920_3 \
    --top fluid_spatial_kernel \
    --work-dir fluid.hw.xo.tapa \
    --enable-synth-util \
    --max-parallel-synth-jobs 16 \
    fluid_sa_kernel.cpp 