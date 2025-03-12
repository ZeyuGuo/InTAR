GCC=g++

cnn-4L: cnn-4L.cpp
	$(GCC) -o $@ -O2 $^ -lm

cnn-4L-intrra: cnn-4L-intrra-kernel.cpp cnn-4L-intrra-host.cpp
	$(GCC) -o $@ -O2 $^ -L/lib/x86_64-linux-gnu -L/usr/local/lib -ltapa -lfrt -lgflags -lglog -lm -lOpenCL -I$(XILINX_HLS)/include

cnn-4L-intrra-hls: cnn-4L-intrra-kernel.cpp
	bash run_tapa.sh

clean:
	rm -f cnn-4L cnn-4L-intrra
