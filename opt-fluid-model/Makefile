GCC=g++
ARMGCC=$(XILINX_VITIS)/gnu/aarch64/lin/aarch64-linux/bin/aarch64-linux-gnu-g++
SYSROOT=/home/oswaldhe/vpk180_custom_platform/vpk180_custom_platform.vitis/xilinx-versal-common-v2023.2/sysroots/cortexa72-cortexa53-xilinx-linux
# TAPA_ROOT=$(shell spack location -i tapa@2024-05-18)
# FRT_ROOT=$(shell spack location -i fpga-runtime)
# GLOG_ROOT=$(shell spack location -i glog/pqucikz)
# GFLAGS_ROOT=$(shell spack location -i gflags/y2uaz43)
INCLUDE_FLAGS=-I$(TAPA_ROOT)/include -I$(FRT_ROOT)/include -I$(GLOG_ROOT)/include -I$(GFLAGS_ROOT)/include -I$(XILINX_HLS)/include
LDFLAGS=-L$(TAPA_ROOT)/lib -L$(FRT_ROOT)/lib -L$(GLOG_ROOT)/lib -L$(GFLAGS_ROOT)/lib -ltapa -lfrt -lglog -lgflags -lm
# RPATH_FLAGS=-Wl,-rpath,$(TAPA_ROOT)/lib -Wl,-rpath,$(FRT_ROOT)/lib -Wl,-rpath,$(GLOG_ROOT)/lib -Wl,-rpath,$(GFLAGS_ROOT)/lib 
#OPT=-I$(shell spack location -i tapa@2023-01-08)/include -I$(shell spack location -i fpga-runtime)/include -I$(shell spack location -i glog/pqucikz)/include -I${shell spack location -i gflags/y2uaz43}/include -ltapa -lfrt -lglog -lgflags -lOpenCL -lm -I${XILINX_HLS}/include
#RPATH_FLAGS=-Wl,-rpath,$(shell spack location -i tapa@2023-01-08)/lib -Wl,-rpath,$(shell spack location -i fpga-runtime)/lib -Wl,-rpath,$(shell spack location -i glog/pqucikz)/lib -Wl,-rpath,$(shell spack location -i gflags/y2uaz43)/lib 

opt350-new: kernel.cpp host.cpp
	$(GCC) -o $@ -O2 $^ -L/lib/x86_64-linux-gnu -L/usr/local/lib -ltapa -lfrt -lgflags -lglog -lm -lOpenCL -I$(XILINX_HLS)/include  

opt350-ultrascale: kernel-ultrascale.cpp host-u280.cpp
	$(GCC) -o $@ -O2 $^ -L/lib/x86_64-linux-gnu -L/usr/local/lib -ltapa -lfrt -lgflags -lglog -lm -lOpenCL -I$(XILINX_HLS)/include  

host-opencl: host_opencl.o
	$(ARMGCC) -o $@ $^ -L$(SYSROOT)/usr/lib/ -lxrt_coreutil -lpthread -lrt -lstdc++ -lgmp -lOpenCL --sysroot=$(SYSROOT)

host_opencl.o: host_opencl.cpp
	$(ARMGCC) -c -D__USE_XOPEN2K8 -I$(SYSROOT)/usr/include/xrt -I$(XILINX_VIVADO)/include -I$(SYSROOT)/usr/include -I$(XILINX_HLS)/include -fmessage-length=0 -std=c++17 --sysroot=$(SYSROOT) -o $@ $^

opt350-versal: kernel-versal.cpp host-versal.cpp
	$(GCC) -o $@ -O2 $^ -L/lib/x86_64-linux-gnu -L/usr/local/lib -ltapa -lfrt -lglog -lgflags -lm -lOpenCL -I$(XILINX_HLS)/include 

opt350-scale: kernel-versal-scale.cpp host-versal.cpp
	$(GCC) -o $@ -O2 $^ -L/lib/x86_64-linux-gnu -L/usr/local/lib -ltapa -lfrt -lglog -lgflags -lm -lOpenCL -I$(XILINX_HLS)/include 

clean:
	rm opt350-new opt-versal
