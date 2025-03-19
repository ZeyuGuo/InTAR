# InTAR: Inter-Task Auto-Reconfigurable Accelerator Design

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.15016321.svg)](https://doi.org/10.5281/zenodo.15016321)

![intar-template](/figures/intrra-arch-template.png)

We propose a novel accelerator design paradigm on FPGAs: inter-task auto-reconfigurable accelerator (InTAR). InTAR can switch execution patterns automatically based on on-chip memory and computation resources. When a task produces large intermediate data, InTAR pipelines multiple tasks to avoid accessing off-chip memory for these data. Otherwise, InTAR will process tasks sequentially to maximize compute efficiency by eliminating pipeline stalls. Compared with other reconfigurable accelerators, InTAR allows model-specific circuit optimization that keeps only necessary control logic and interconnects. Hence, InTAR requires fewer reconfiguration resources, achieves a high clock frequency, and has a low reconfiguration overhead (10 to 20 ns). Since computations are reconfigured at the task level, InTAR is one of the first works regarding FPGA-based reconfigurable accelerators that support high-level hardware generation tools such as High-Level Synthesis (HLS) for fast accelerator design.

## Project Structure

- `/benchmark`: contains multi-task DNN kernels that are HDV
- `/gpt-2-medium`: HLS code and bitstreams for hardware emulation and on-board execution of GPT-2 Medium model with InTAR

## Dependencies

### Software

- Vitis/Vivado 2024.1+ for VPK180
- Vitis/Vivado 2021.2+ for U280
- TAPA: https://github.com/rapidstream-org/rapidstream-tapa
    - For the older version (TAPA 2024), Autobridge/RapidStream is integrated. Please install from the source or use `apptainer` on the VAST Lab cluster.
    - For the main version, TAPA is only used for HLS code generation, and you need to use RapidStream for floorplanning. It can still be used to compile the host code.
- RapidStream: https://docs.rapidstream-da.com/

> [!TIP]
> On the VASTlab cluster, you can execute the following to enable the TAPA old version in a container
> ```sh
> ml load tapa apptainer
> apptainer instance start $(which tapa.sif) tapa-2024 # now the container has the name tapa-2024
> apptainer shell instance://tapa-2024
> ```

### Hardware Platform

- Alveo U280: `xilinx_u280_xdma_201920_3` ([dev](https://drive.google.com/file/d/1GvZ1_x8_W5q_h4U76dH9iQXDN9xPeLvv/view?usp=drive_link), [deploy](https://drive.google.com/file/d/1wQywrYvW9r0oBccn-PqoS4KPZJfEW3_J/view?usp=drive_link))
- Versal VPK180: Custom Platform (Please follow [this documentation](/gpt-2-medium/README.md))

> [!NOTE]
> You can still use the most recent U280 platform, but you have to use the most updated TAPA with Rapidstream to regenerate the bitstream for your own FPGA board. It may be easier to flash the older platform with the given link.

## Multi-Task Kernel Testbench

Run `make <kernel-name>-intrra` and `./<kernel-name>-intrra --bitstreams <bitstream-path>` to perform on-board execution on Alveo U280. Similar for sequential and dataflow kernels in the corresponding folders.

## GPT-2 Medium

<p align="center">
    <img src="figures/intra_fpga_design_v2.png" alt="intar-gpt2" width="450">
</p>

- U280: `make opt350` and `./opt350 <sequence-length> --bitstream bitstreams/opt_kernel_latest.xclbin` to execute an older version of kernel (less optimized, lower frequency). `make opt350-ultrascale` and `./opt350 <sequence-length> --bitstream bitstreams/opt_kernel_xilinx_u280_full.xclbin` for an optimized version of kernel.

- VPK180: Follow [this documentation](/gpt-2-medium/README.md) to generate a custom platform, run hardware emulation using QEMU, and perform bitstream generation.
