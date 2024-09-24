#!/bin/bash
TARGET=hw
# TARGET=hw_emu
# DEBUG=-g

TOP=opt_kernel
XO='/path/to/opt_kernel.xo'
CONSTRAINT='/path/to/floorplanning/constraint.tcl'
>&2 echo "Using the default clock target of the platform."
PLATFORM="/path/to/vitis/vpk180.xpfm"
TARGET_FREQUENCY=240000000
if [ -z $PLATFORM ]; then echo Please edit this file and set a valid PLATFORM= on line "${LINENO}"; exit; fi

OUTPUT_DIR="$(pwd)/vitis_run_${TARGET}_ln"

MAX_SYNTH_JOBS=16
STRATEGY="Explore"
PLACEMENT_STRATEGY="Explore"

v++ ${DEBUG} \
  --link \
  --output "${OUTPUT_DIR}/${TOP}_vpk180.xsa" \
  --kernel ${TOP} \
  --platform ${PLATFORM} \
  --target ${TARGET} \
  --report_level 2 \
  --temp_dir "${OUTPUT_DIR}/${TOP}_vpk180.temp" \
  --optimize 3 \
  --connectivity.nk ${TOP}:1:${TOP} \
  --save-temps \
  "${XO}" \
  --vivado.synth.jobs ${MAX_SYNTH_JOBS} \
  --vivado.prop=run.impl_1.STEPS.PHYS_OPT_DESIGN.IS_ENABLED=1 \
  --vivado.prop=run.impl_1.STEPS.OPT_DESIGN.ARGS.DIRECTIVE=$STRATEGY \
  --vivado.prop=run.impl_1.{STEPS.OPT_DESIGN.ARGS.MORE\ OPTIONS}={-debug_log} \
  --vivado.prop=run.impl_1.STEPS.PLACE_DESIGN.ARGS.DIRECTIVE=$PLACEMENT_STRATEGY \
  --vivado.prop=run.impl_1.STEPS.PHYS_OPT_DESIGN.ARGS.DIRECTIVE=$STRATEGY \
  --vivado.prop=run.impl_1.STEPS.ROUTE_DESIGN.ARGS.DIRECTIVE=$STRATEGY \
  --clock.default_freqhz ${TARGET_FREQUENCY} \
  --vivado.prop=run.impl_1.STEPS.OPT_DESIGN.TCL.PRE=$CONSTRAINT \