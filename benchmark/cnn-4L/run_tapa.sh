#!/usr/bin/env bash
ml load xilinx/vivado/2021.2
tapac \
    -o cnn-4L.xo \
    --platform xilinx_u280_xdma_201920_3 \
    --top CNN4L \
    --work-dir cnn-4L.tapa \
    --connectivity hbm_config.ini \
    --enable-hbm-binding-adjustment \
    --enable-synth-util \
    --run-floorplan-dse \
    --min-area-limit 0.55 \
    --min-slr-width-limit 5000 \
    --max-slr-width-limit 19000 \
    --max-parallel-synth-jobs 16 \
    --floorplan-output cnn-4L.tcl \
    cnn-4L-intrra-kernel.cpp 