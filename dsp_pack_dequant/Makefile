GCC=g++
OPT=-I${TAPA_ROOT}/include -I$(shell spack location -i fpga-runtime)/include -I$(shell spack location -i glog)/include -I${shell spack location -i gflags}/include -L${TAPA_ROOT}/lib -L$(shell spack location -i fpga-runtime)/lib -L${shell spack location -i glog}/lib -L$(shell spack location -i gflags)/lib -ltapa -lfrt -lglog -lgflags -lOpenCL -lm -I/opt/xilinx/Vitis_HLS/2021.2/include

w3a6: w3a6o48linear_kernel.cpp w3a6o48linear_host.cpp
	$(GCC) -o $@ -O2 $^ $(OPT)

clean:
	rm w3a6