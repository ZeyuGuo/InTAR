GCC=g++
OPT=-I$(shell spack location -i tapa@2023-01-08)/include -I$(shell spack location -i fpga-runtime)/include -I$(shell spack location -i glog)/include -I${shell spack location -i gflags}/include -L$(shell spack location -i tapa@2023-01-08)/lib -L$(shell spack location -i fpga-runtime)/lib -L${shell spack location -i glog}/lib -L$(shell spack location -i gflags)/lib -ltapa -lfrt -lglog -lgflags -lOpenCL -lm -I/opt/xilinx/Vitis_HLS/2021.2/include

opt350: kernel.cpp host.cpp
	$(GCC) -o $@ -O2 $^ $(OPT)

clean:
	rm opt350