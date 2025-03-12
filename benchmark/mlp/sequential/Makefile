GCC := g++
PLATFORM := xilinx_u280_xdma_201920_3
TOP_NAME := MLP
CONNECTIVITY_FILE := hbm_config.ini

CPU_EXECUTABLE := build/mlp
HLS_EXECUTABLE := build/mlp-sequential

CPU_SRCS := mlp.cpp

HLS_HOST_SRCS := mlp-sequential-host.cpp
HLS_KERNEL_SRCS := mlp-sequential-kernel.cpp
HLS_SRCS := $(HLS_KERNEL_SRCS) $(HLS_HOST_SRCS)

DUMMY := $(shell mkdir -p build)

$(CPU_EXECUTABLE): $(CPU_SRCS)
	$(GCC) -o $(CPU_EXECUTABLE) -O2 $^ -lm

$(HLS_EXECUTABLE): $(HLS_SRCS)
	$(GCC) -o $(HLS_EXECUTABLE) -O2 $^ -L/lib/x86_64-linux-gnu -L/usr/local/lib -ltapa -lfrt -lgflags -lglog -lm -lOpenCL -I$(XILINX_HLS)/include

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

.PHONY: build_hw_emu
build_hw_emu:
	cd work.out/run-1/;\
	sed -i '2s/.*/# TARGET=hw/' generate-bitstream.sh; \
	sed -i '3s/.*/TARGET=hw_emu/' generate-bitstream.sh; \
	bash generate-bitstream.sh
	mkdir -p bitstreams/; \
	cp work.out/run-1/vitis_run_hw_emu/$(TOP_NAME)_$(PLATFORM).xclbin bitstreams/$(TOP_NAME)_$(PLATFORM)_hw_emu.xclbin

.PHONY: build_hw
build_hw:
	cd work.out/run-1/;\
	sed -i '2s/.*/TARGET=hw/' generate-bitstream.sh; \
	sed -i '3s/.*/# TARGET=hw_emu/' generate-bitstream.sh; \
	bash generate-bitstream.sh
	mkdir -p bitstreams/; \
	cp work.out/run-1/vitis_run_hw/$(TOP_NAME)_$(PLATFORM).xclbin bitstreams/$(TOP_NAME)_$(PLATFORM)_hw.xclbin

run_cpu: $(CPU_EXECUTABLE)
	./$(CPU_EXECUTABLE)

run_hls: $(HLS_EXECUTABLE)
	./$(HLS_EXECUTABLE)

run_hw_emu:
	./$(HLS_EXECUTABLE) --bitstream bitstreams/$(TOP_NAME)_$(PLATFORM)_hw_emu.xclbin

run_hw:
	./$(HLS_EXECUTABLE) --bitstream bitstreams/$(TOP_NAME)_$(PLATFORM)_hw.xclbin

.PHONY: build_hw
clean:
	rm -rf work.out/ build/
