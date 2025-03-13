# Place & Route Instructions

## Generate Vitis Platform

Follow this [tutorial](https://docs.amd.com/r/2023.2-English/Vitis-Tutorials-Vitis-Platform-Creation/Versal-Platform-Creation-Quick-Start) to generate the Vitis Platform for VPK180. There are a couple of changes:

1. Step 1-3: Select VPK180 as the device. Generate 3 clocks: 100MHz, 200MHz, 300MHz.
2. Step 2-2: git-branch should be `xlnx_rel_v2023.2`. `system-user.dtsi` is on [Vitis Tutorial Github Repo](https://github.com/Xilinx/Vitis-Tutorials/blob/2023.2/Vitis_Platform_Creation/Design_Tutorials/03_Edge_VCK190/ref_files/step2_pfm/system-user.dtsi). Change the name to Xilinx custom-vpk180. Board name is `versal-vpk180-reva`.

## Launch V++ Script for P&R

After exporting the xo container, replace the platform path, xo path, and constraint path in `generate_bitstream_sample.sh` and launch the script to start P&R.

## Hardware Emulation Using QEMU

After exporting the xo container, replace the platform path, xo path, and constraint path in `generate_bitstream_sample.sh`. Change target to `hw_emu` and turn on debug mode `-g`. After generating the xsa file for hardware emulation, run `package_sample.sh` with the same modifications as `generate_bitstream_sample.sh`, with the files you want to include in the SD card image (including the host binary, launch scripts, and configuration file `xrt.ini`). You will find a script `/package/launch_hw_emu.sh` to start QEMU directly.
