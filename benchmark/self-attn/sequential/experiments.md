Problem:
Bitstream run on hardware hangs. 

# Small Size Cosim
1. Cosim finished (32, 128) with warning about DSP. 
    - Might Change datatype to float and recompile the bitstream. 

2. Cosim finished (64, 256) with warning about DSP. 

# Hardware
No hangs. 
```
[tapa.sif:~/repo/LLM-InTRRA/benchmark/self-attn/sequential]$ make run_sequential_hw
./build/self-attn-sequential --bitstream bitstreams/self_attn_sequential_xilinx_u280_xdma_201920_3_hw.xclbin
I1230 02:31:39.548687    50 self-attn-sequential-host.cpp:54] Initializing input matrix...
I1230 02:31:39.549538    50 self-attn-sequential-host.cpp:61] Input matrix initialized with all 1s
I1230 02:31:39.556267    50 self-attn-sequential-host.cpp:73] Weight matrices initialized with all 1s
I1230 02:31:39.588603    50 self-attn-sequential-host.cpp:86] Weight matrices transposed
I1230 02:31:39.590281    50 self-attn-sequential-host.cpp:95] Offchip memory initialized
I1230 02:31:39.590286    50 frt.cpp:18] Loading bitstreams/self_attn_sequential_xilinx_u280_xdma_201920_3_hw.xclbin
I1230 02:31:39.782213    50 xilinx_opencl_device.cpp:299] Running on-board execution with Xilinx OpenCL
I1230 02:31:39.786180    50 opencl_device.cpp:160] Found platform: Xilinx
XRT build version: 2.14.384
Build hash: 090bb050d570d2b668477c3bd0f979dc3a34b9db
Build date: 2022-12-09 00:55:08
Git branch: 2022.2
PID: 50
UID: 10047
[Mon Dec 30 10:31:39 2024 GMT]
HOST: c02
EXE: /home/yingqi/repo/LLM-InTRRA/benchmark/self-attn/sequential/build/self-attn-sequential
[XRT] WARNING: dev_init failed: -1
[XRT] ERROR: Operation not permitted Device index 0
[XRT] ERROR: Could not open device with index '0'
[XRT] WARNING: dev_init failed: -1
[XRT] ERROR: Operation not permitted Device index 0
[XRT] ERROR: Could not open device with index '0'
I1230 02:31:40.060158    50 xilinx_opencl_device.cpp:105] Found device: xilinx_u280_xdma_201920_3 (bdf=0000:c2:00.1)
I1230 02:31:40.060176    50 opencl_device.cpp:167] Using xilinx_u280_xdma_201920_3 (bdf=0000:c2:00.1)
I1230 02:31:41.456338    50 self-attn-sequential-host.cpp:109] Kernel invoked
Cycle count: 348574419
Kernel time (ns): 1267933015
Kernel time (us): 1.26793e+06
Kernel time (ms): 1267.93
```