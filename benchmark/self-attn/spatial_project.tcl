open_project self_attential_spatial
set_top selfAttention

# v++ -g, -D, -I, --advanced.prop kernel.seq_align_multiple_static.kernel_flags
add_files "/home/yingqi/repo/LLM-InTRRA/benchmark/self-attn/self-attn-spatial-kernel.cpp" -cflags "-I ./include"
add_files -tb "/home/yingqi/repo/LLM-InTRRA/benchmark/self-attn/self-attn-spatial-testbench.cpp" -cflags "-I ./include"

# The followings should be put in another script after reopening the project created. 
# The problem is that we must close and reopen the project to successfully create a solution. There is a bug to create solution with tcl script. 
# open_solution -flow_target vitis solution1
# set_part xcvu9p-flgb2104-2-i
# # v++ --hls.clock or --kernel_frequency
# create_clock -period {{ clock_frequency }}Hz -name default  # This must be something like -period 5
# # v++ --advanced.param compiler.hlsDataflowStrictMode
# config_dataflow -strict_mode warning
# # v++ --advanced.param compiler.deadlockDetection
# config_export -deadlock_detection none
# # v++ --advanced.param compiler.axiDeadLockFree
# config_interface -m_axi_conservative_mode=1
# config_interface -m_axi_addr64
# # v++ --hls.max_memory_ports
# config_interface -m_axi_auto_max_ports=0
# config_export -format xo -ipname seq_align_multiple_static
# csynth_design
# export_design
close_project
puts "HLS Project Successfully Created"
exit