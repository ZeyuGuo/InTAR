open_project self_attential_spatial
set_top selfAttention

open_solution "solution1" -flow_target vitis
set_part xcvu9p-flgb2104-1-e
create_clock -period 5ns -name default
config_dataflow -strict_mode warning
config_interface -m_axi_alignment_byte_size 64 -m_axi_latency 64 -m_axi_max_widen_bitwidth 512
config_rtl -register_reset_num 3
config_export -deadlock_detection none
config_interface -m_axi_conservative_mode=1
config_interface -m_axi_addr64
config_interface -m_axi_auto_max_ports=0
config_export -format xo -ipname selfAttention
csim_design
csynth_design -dump_cfg -dump_post_cfg 
cosim_design -enable_dataflow_profiling -trace_level all
exit
