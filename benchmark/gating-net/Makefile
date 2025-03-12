GCC=g++

gating-net: gating-net.cpp
	$(GCC) -o $@ -O2 $^ -lm

gn-intrra: gating-net-intrra-kernel.cpp gating-net-intrra-host.cpp
	$(GCC) -o $@ -O2 $^ -L/lib/x86_64-linux-gnu -L/usr/local/lib -ltapa -lfrt -lgflags -lglog -lm -lOpenCL -I$(XILINX_HLS)/include

gn-intrra-hls: gating-net-intrra-kernel.cpp
	bash run_tapa.sh

clean:
	rm -f gating-net gn-intrra