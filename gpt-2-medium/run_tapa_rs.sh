#!/usr/bin/env bash
ml load xilinx/vivado/2024.1
tapac \
    --work-dir opt-stage4-dot-prod.tapa \
    --top opt_kernel \
    --part-num xcvp1802-lsvc4072-2MP-e-S \
    --clock-period 3.33 \
    -o "opt-stage4-dot-prod.tapa/opt.hw.xo" \
    --connectivity link_config_versal.ini \
    --run-tapacc        \
    --run-hls           \
    --generate-task-rtl \
    --run-floorplanning \
    --generate-top-rtl \
    kernel-versal.cpp 

ml load xilinx/vivado/2024.1
tapac \
    --work-dir opt-stage4-dot-prod.tapa \
    --top opt_kernel \
    --part-num xcvp1802-lsvc4072-2MP-e-S \
    --clock-period 3.33 \
    -o "opt-stage4-dot-prod.tapa/opt.hw.xo" \
    --connectivity link_config_versal.ini \
    --pack-xo \
    kernel-versal.cpp 


