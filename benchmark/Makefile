GCC=g++

self-attn: self-attn.cpp
	$(GCC) -o $@ -O2 $^ -lm

mlp: mlp.cpp
	$(GCC) -o $@ -O2 $^ -lm

cnn-4L: cnn-4L.cpp
	$(GCC) -o $@ -O2 $^ -lm

vae: vae.cpp
	$(GCC) -o $@ -O2 $^ -lm

yolov3: yolov3.cpp
	$(GCC) -o $@ -O2 $^ -lm

faster-r-cnn: faster-r-cnn.cpp
	$(GCC) -o $@ -O2 $^ -lm

gating-net: gating-net.cpp
	$(GCC) -o $@ -O2 $^ -lm

clean:
	rm self-attn mlp cnn-4L vae yolov3 faster-r-cnn gating-net