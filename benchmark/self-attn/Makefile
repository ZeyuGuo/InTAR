GCC=g++
PLATFORM := xilinx_u280_xdma_201920_3
TOP_NAME := selfAttention
CONNECTIVITY_FILE := hbm_config.ini

CPU_EXECUTABLE := build/self-attn
HLS_EXECUTABLE := build/self-attn-intrra

CPU_SRCS := self-attn.cpp
HLS_HOST_SRCS := self-attn-intrra-host.cpp
HLS_KERNEL_SRCS := self-attn-intrra-kernel.cpp
BASELINE_HOST_SRCS := self-attn-spatial-host.cpp
BASELINE_KERNEL_SRCS := self-attn-spatial-kernel.cpp
HLS_SRCS := $(HLS_KERNEL_SRCS) $(HLS_HOST_SRCS)
BASELINE_SRCS := $(BASELINE_KERNEL_SRCS) $(BASELINE_HOST_SRCS)

DUMMY := $(shell mkdir -p build)

$(CPU_EXECUTABLE): $(CPU_SRCS)
	$(GCC) -o $(CPU_EXECUTABLE) -O2 $^ -lm

$(HLS_EXECUTABLE): $(HLS_SRCS)
	$(GCC) -o $(HLS_EXECUTABLE) -O2 $^ -L/lib/x86_64-linux-gnu -L/usr/local/lib -ltapa -lfrt -lgflags -lglog -lm -lOpenCL -I$(XILINX_HLS)/include

$(BASELINE_EXECUTABLE): $(BASELINE_SRCS)
	$(GCC) -o $(BASELINE_EXECUTABLE) -O2 $^ -L/lib/x86_64-linux-gnu -L/usr/local/lib -ltapa -lfrt -lgflags -lglog -lm -lOpenCL -I$(XILINX_HLS)/include

.PHONY: build_tapa
build_tapa: $(HLS_KERNEL_SRCS)
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
		--max-parallel-synth-jobs 16 \
		--floorplan-output build/solver.tcl \
		$^
# Don't know if the following are needed
# --min-area-limit 0.55 \
# --min-slr-width-limit 5000 \
# --max-slr-width-limit 19000 \

.PHONY: build_hw_emu
build_hw_emu:
	cd work.out/run-1/;\
	sed -i '2s/.*/# TARGET=hw/' generate-bitstream.sh; \
	sed -i '3s/.*/TARGET=hw_emu/' generate-bitstream.sh; \
	bash generate-bitstream.sh
	mkdir -p bitstreams/; \
	cp work.out/run-1/vitis_run_hw_emu/self_attn_$(PLATFORM).xclbin bitstreams/self_attn_$(PLATFORM)_hw_emu.xclbin

.PHONY: build_hw
build_hw:
	cd work.out/run-1/;\
	sed -i '2s/.*/TARGET=hw/' generate-bitstream.sh; \
	sed -i '3s/.*/# TARGET=hw_emu/' generate-bitstream.sh; \
	bash generate-bitstream.sh
	mkdir -p bitstreams/; \
	cp work.out/run-1/vitis_run_hw/self_attn_$(PLATFORM).xclbin bitstreams/self_attn_$(PLATFORM)_hw.xclbin


.PHONY: build_baseline
build_baseline: $(BASELINE_KERNEL_SRCS)
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
		--max-parallel-synth-jobs 16 \
		--floorplan-output build/solver.tcl \
		$^
# Don't know if the following are needed
# --min-area-limit 0.55 \
# --min-slr-width-limit 5000 \
# --max-slr-width-limit 19000 \

.PHONY: build_baseline_hw_emu
build_baseline_hw_emu:
	cd work.out/run-1/;\
	sed -i '2s/.*/# TARGET=hw/' generate-bitstream.sh; \
	sed -i '3s/.*/TARGET=hw_emu/' generate-bitstream.sh; \
	bash generate-bitstream.sh
	mkdir -p bitstreams/; \
	cp work.out/run-1/vitis_run_hw_emu/self_attn_baseline_$(PLATFORM).xclbin bitstreams/self_attn_baseline_$(PLATFORM)_hw_emu.xclbin

.PHONY: build_baseline_hw
build_baseline_hw:
	cd work.out/run-1/;\
	sed -i '2s/.*/TARGET=hw/' generate-bitstream.sh; \
	sed -i '3s/.*/# TARGET=hw_emu/' generate-bitstream.sh; \
	bash generate-bitstream.sh
	mkdir -p bitstreams/; \
	cp work.out/run-1/vitis_run_hw/self_attn_baseline_$(PLATFORM).xclbin bitstreams/self_attn_baseline_$(PLATFORM)_hw.xclbin


self-attn: self-attn.cpp
	$(GCC) -o $@ -O2 $^ -lm

self-attn-baseline: self-attn-spatial-kernel.cpp self-attn-spatial-host.cpp
	$(GCC) -o $@ -O2 $^ -L/lib/x86_64-linux-gnu -L/usr/local/lib -ltapa -lfrt -lgflags -lglog -lm -lOpenCL -I$(XILINX_HLS)/include

self-attn-intrra: self-attn-intrra-kernel.cpp self-attn-intrra-host.cpp
	$(GCC) -o $@ -O2 $^ -L/lib/x86_64-linux-gnu -L/usr/local/lib -ltapa -lfrt -lgflags -lglog -lm -lOpenCL -I$(XILINX_HLS)/include

# self-attn-hls is moved to to the target "build_tapa"
# self-attn-hls: self-attn-intrra-kernel.cpp
# 	bash run_tapa.sh

# Run the designs
run_cpu: $(CPU_EXECUTABLE)
	./$(CPU_EXECUTABLE)

run_baseline: $(BASELINE_EXECUTABLE)
	./$(BASELINE_EXECUTABLE)

run_hls: $(HLS_EXECUTABLE)
	./$(HLS_EXECUTABLE)

run_hw_emu:
	./$(HLS_EXECUTABLE) --bitstream bitstreams/self_attn_$(PLATFORM)_hw_emu.xclbin

run_hw:
	./$(HLS_EXECUTABLE) --bitstream bitstreams/self_attn_$(PLATFORM)_hw.xclbin

run_baseline_hw_emu:
	./$(BASELINE_EXECUTABLE) --bitstream bitstreams/self_attn_baseline_$(PLATFORM)_hw_emu.xclbin

run_baseline_hw:
	./$(BASELINE_EXECUTABLE) --bitstream bitstreams/self_attn_baseline_$(PLATFORM)_hw.xclbin

clean:
	rm -rf self-attn self-attn-intrra work.out/ build/