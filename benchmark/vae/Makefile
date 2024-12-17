GCC=g++

vae: vae.cpp
	$(GCC) -o $@ -O2 $^ -lm

vae-intrra: vae-intrra-kernel.cpp vae-intrra-host.cpp
	$(GCC) -o $@ -O2 $^ -L/lib/x86_64-linux-gnu -L/usr/local/lib -ltapa -lfrt -lgflags -lglog -lm -lOpenCL -I$(XILINX_HLS)/include

vae-intrra-hls: vae-intrra-kernel.cpp
	bash run_tapa.sh

clean:
	rm vae