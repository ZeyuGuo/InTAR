#!/bin/bash
# TARGET=hw
TARGET=hw_emu
DEBUG=-g

TOP=opt_kernel
XO='/path/to/opt_kernel.xo'
CONSTRAINT='/path/to/constraints.tcl'
>&2 echo "Using the default clock target of the platform."
PLATFORM="/path/to/vpk180_pfm_vitis.xpfm"
VERSAL="/path/to/xilinx-versal-common-v2023.2"
TARGET_FREQUENCY=300000000
if [ -z $PLATFORM ]; then echo Please edit this file and set a valid PLATFORM= on line "${LINENO}"; exit; fi

OUTPUT_DIR="$(pwd)/vitis_run_${TARGET}_ln"

MAX_SYNTH_JOBS=16
STRATEGY="Default"
PLACEMENT_STRATEGY="Default"

emconfigutil --platform ${PLATFORM} --od "${OUTPUT_DIR}/"

v++ ${DEBUG}\
  --platform ${PLATFORM} \
  --target ${TARGET} \
  --package \
  "${OUTPUT_DIR}/${TOP}_vpk180.xsa" \
  --temp_dir "${OUTPUT_DIR}/${TOP}_vpk180.temp/package.build" \
  --save-temps \
  --package.out_dir "${OUTPUT_DIR}/package" \
  --package.boot_mode sd \
  --package.rootfs "${VERSAL}/rootfs.ext4" \
  --package.kernel_image "${VERSAL}/Image" \
  --package.sd_file "${OUTPUT_DIR}/emconfig.json" \
  --package.sd_file "./host-opencl" \
  --package.sd_file "./run_app.sh" \
  --package.sd_file "./xrt.ini" \
  -o "${OUTPUT_DIR}/${TOP}_vpk180.xclbin" 