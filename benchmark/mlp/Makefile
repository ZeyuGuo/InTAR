GCC=g++

mlp: mlp.cpp
	$(GCC) -o $@ -O2 $^ -lm

mlp-intrra: mlp-intrra-kernel.cpp mlp-intrra-host.cpp
	$(GCC) -o $@ -O2 $^ -L/lib/x86_64-linux-gnu -L/usr/local/lib -ltapa -lfrt -lgflags -lglog -lm -lOpenCL -I$(XILINX_HLS)/include


clean:
	rm mlp