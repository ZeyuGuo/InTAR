GCC=g++

self-attn: self-attn.cpp
	$(GCC) -o $@ -O2 $^ -lm

self-attn-intrra: self-attn-intrra-kernel.cpp self-attn-intrra-host.cpp
	$(GCC) -o $@ -O2 $^ -L/lib/x86_64-linux-gnu -L/usr/local/lib -ltapa -lfrt -lgflags -lglog -lm -lOpenCL -I$(XILINX_HLS)/include

self-attn-hls: self-attn-intrra-kernel.cpp
	bash run_tapa.sh

clean:
	rm -f self-attn self-attn-intrra