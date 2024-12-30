[tapa.sif:~/repo/LLM-InTRRA/benchmark/self-attn/sequential]$ make run_sequential_hw
./build/self-attn-sequential --bitstream bitstreams/self_attn_sequential_xilinx_u280_xdma_201920_3_hw.xclbin
I1229 18:13:29.451143 409059 self-attn-sequential-host.cpp:54] Initializing input matrix...
I1229 18:13:29.451256 409059 self-attn-sequential-host.cpp:61] Input matrix initialized with all 1s
I1229 18:13:29.451704 409059 self-attn-sequential-host.cpp:73] Weight matrices initialized with all 1s
I1229 18:13:29.452287 409059 self-attn-sequential-host.cpp:86] Weight matrices transposed
I1229 18:13:29.452421 409059 self-attn-sequential-host.cpp:95] Offchip memory initialized
I1229 18:13:29.452427 409059 frt.cpp:18] Loading bitstreams/self_attn_sequential_xilinx_u280_xdma_201920_3_hw.xclbin
I1229 18:13:29.696909 409059 xilinx_opencl_device.cpp:299] Running on-board execution with Xilinx OpenCL
I1229 18:13:29.702123 409059 opencl_device.cpp:160] Found platform: Xilinx
XRT build version: 2.14.384
Build hash: 090bb050d570d2b668477c3bd0f979dc3a34b9db
Build date: 2022-12-09 00:55:08
Git branch: 2022.2
PID: 409059
UID: 10047
[Mon Dec 30 02:13:29 2024 GMT]
HOST: c02
EXE: /home/yingqi/repo/LLM-InTRRA/benchmark/self-attn/sequential/build/self-attn-sequential
[XRT] WARNING: dev_init failed: -1
[XRT] ERROR: Operation not permitted Device index 0
[XRT] ERROR: Could not open device with index '0'
[XRT] WARNING: dev_init failed: -1
[XRT] ERROR: Operation not permitted Device index 0
[XRT] ERROR: Could not open device with index '0'
I1229 18:13:30.160164 409059 xilinx_opencl_device.cpp:105] Found device: xilinx_u280_xdma_201920_3 (bdf=0000:c2:00.1)
I1229 18:13:30.160185 409059 opencl_device.cpp:167] Using xilinx_u280_xdma_201920_3 (bdf=0000:c2:00.1)
I1229 18:13:36.300172 409059 self-attn-sequential-host.cpp:109] Kernel invoked
Cycle count: 1650459
Kernel time (ns): 5684712
Kernel time (us): 5684.71
Kernel time (ms): 5.68471