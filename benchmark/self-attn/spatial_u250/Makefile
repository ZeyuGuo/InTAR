GCC=g++
PLATFORM := xilinx_u250_gen3x16_xdma_4_1_202210_1
TOP_NAME := selfAttention
CONNECTIVITY_FILE := hbm_config_spatial.ini

CPU_EXECUTABLE := build/self-attn
SPATIAL_EXECUTABLE := build/self-attn-spatial

CPU_SRCS := self-attn.cpp
SPATIAL_HOST_SRCS := self-attn-spatial-host.cpp
SPATIAL_KERNEL_SRCS := self-attn-spatial-kernel.cpp

DUMMY := $(shell mkdir -p build)

$(CPU_EXECUTABLE): $(CPU_SRCS)
	$(GCC) -o $(CPU_EXECUTABLE) -O2 $^ -lm

$(SPATIAL_EXECUTABLE): $(BASELINE_SRCS)
	$(GCC) -o $(SPATIAL_EXECUTABLE) -O2 $^ -L/lib/x86_64-linux-gnu -L/usr/local/lib -ltapa -lfrt -lgflags -lglog -lm -lOpenCL -I$(XILINX_HLS)/include


.PHONY: build_spatial
build_spatial: $(SPATIAL_KERNEL_SRCS)
	rm -rf work.out/ build/
	mkdir -p build
	tapac \
		-o build/solver.$(PLATFORM).hw.xo \
		--platform $(PLATFORM) \
		--top $(TOP_NAME) \
		--work-dir build/solver.$(PLATFORM).hw.xo.tapa \
		--connectivity $(CONNECTIVITY_FILE) \
		--enable-hbm-binding-adjustment \
		--enable-synth-util \
		--run-floorplan-dse \
		--max-parallel-synth-jobs 8 \
		--floorplan-output build/solver.tcl \
		$^

.PHONY: build_spatial_hw_emu
build_spatial_hw_emu:
	cd work.out/run-1/;\
	sed -i '2s/.*/# TARGET=hw/' generate-bitstream.sh; \
	sed -i '3s/.*/TARGET=hw_emu/' generate-bitstream.sh; \
	bash generate-bitstream.sh
	mkdir -p bitstreams/; \
	cp work.out/run-1/vitis_run_hw_emu/selfAttention_$(PLATFORM).xclbin bitstreams/self_attn_spatial_$(PLATFORM)_hw_emu.xclbin

.PHONY: build_spatial_hw
build_spatial_hw:
	cd work.out/run-1/;\
	sed -i '2s/.*/TARGET=hw/' generate-bitstream.sh; \
	sed -i '3s/.*/# TARGET=hw_emu/' generate-bitstream.sh; \
	bash generate-bitstream.sh
	mkdir -p bitstreams/; \
	cp work.out/run-1/vitis_run_hw/selfAttention_$(PLATFORM).xclbin bitstreams/self_attn_spatial_$(PLATFORM)_hw.xclbin


self-attn: self-attn.cpp
	$(GCC) -o $@ -O2 $^ -lm

self-attn-spatial: self-attn-spatial-kernel.cpp self-attn-spatial-host.cpp
	$(GCC) -o $@ -O2 $^ -L/lib/x86_64-linux-gnu -L/usr/local/lib -ltapa -lfrt -lgflags -lglog -lm -lOpenCL -I$(XILINX_HLS)/include


# Run the designs
run_cpu: $(CPU_EXECUTABLE)
	./$(CPU_EXECUTABLE)

run_spatial: $(SPATIAL_EXECUTABLE)
	./$(SPATIAL_EXECUTABLE)

run_spatial_hw_emu:
	./$(SPATIAL_EXECUTABLE) --bitstream bitstreams/self_attn_spatial_$(PLATFORM)_hw_emu.xclbin

run_spatial_hw:
	./$(SPATIAL_EXECUTABLE) --bitstream bitstreams/self_attn_spatial_$(PLATFORM)_hw.xclbin

clean:
	rm -rf self-attn self-attn-spatial work.out/ build/
