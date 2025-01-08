GCC=g++

vae: vae.cpp
	$(GCC) -o $@ -O2 $^ -lm

vae-spatial: vae-spatial-kernel.cpp vae-spatial-host.cpp
	$(GCC) -o $@ -O2 $^ -L/lib/x86_64-linux-gnu -L/usr/local/lib -ltapa -lfrt -lgflags -lglog -lm -lOpenCL -I$(XILINX_HLS)/include

vae-spatial-hls: vae-spatial-kernel.cpp
	bash run_tapa.sh

clean:
	rm -f vae vae-spatial