from rapidstream import RapidStreamTAPA, DeviceFactory, get_u250_vitis_device_factory
from pathlib import Path
import os

CURR_DIR = os.path.dirname(os.path.abspath(__file__))
BUILD_DIR = "rs_build"
VITIS_PLATFORM = "~/vpk180_linux_platform/vpk180_pfm_vitis/export/vpk180_pfm_vitis/vpk180_pfm_vitis.xpfm"


rs = RapidStreamTAPA(BUILD_DIR)

# factory = get_u250_vitis_device_factory(VITIS_PLATFORM)
factory = DeviceFactory(
    row=4,
    col=2,
    part_num="xcvp1802-lsvc4072-2MP-e-S",
    board_name="xilinx.com:vpk180:part0:1.1",
)

# Set the pblocks of the device so that each slot contains half of an SLR:
factory.set_slot_pblock(0, 0, ["-add CLOCKREGION_X0Y1:CLOCKREGION_X4Y4"])
factory.set_slot_pblock(1, 0, ["-add CLOCKREGION_X5Y1:CLOCKREGION_X9Y4"])
factory.set_slot_pblock(0, 1, ["-add CLOCKREGION_X0Y5:CLOCKREGION_X4Y7"])
factory.set_slot_pblock(1, 1, ["-add CLOCKREGION_X5Y5:CLOCKREGION_X9Y7"])

factory.set_slot_pblock(0, 2, ["-add CLOCKREGION_X0Y8:CLOCKREGION_X4Y10"])
factory.set_slot_pblock(1, 2, ["-add CLOCKREGION_X5Y8:CLOCKREGION_X9Y10"])
factory.set_slot_pblock(0, 3, ["-add CLOCKREGION_X0Y11:CLOCKREGION_X4Y13"])
factory.set_slot_pblock(1, 3, ["-add CLOCKREGION_X5Y11:CLOCKREGION_X9Y13"])

# There are 18870 total SLL nodes for VP1552:
factory.set_slot_capacity(0, 0, north=9435)
factory.set_slot_capacity(1, 0, north=9435)
factory.set_slot_capacity(0, 1, north=9435)
factory.set_slot_capacity(1, 1, north=9435)
factory.set_slot_capacity(0, 2, north=9435)
factory.set_slot_capacity(1, 2, north=9435)

# Call factory to extract the slot resources automatically from Vivado:
factory.extract_slot_resources()

rs.set_virtual_device(factory.generate_virtual_device())

rs.add_xo_file("./gpt2-sa.tapa/gpt2.xo")
rs.set_top_module_name("opt_kernel")
rs.add_clock("ap_clk", period_ns=3.33)
rs.set_vitis_connectivity_config("link_config_versal.ini")

work_dir_to_ir = {Path(f'{CURR_DIR}/{BUILD_DIR}/dse/candidate_5'): Path(f'{CURR_DIR}/{BUILD_DIR}/dse/candidate_5/add_pipeline.json')}
rs.remote_ip_cache = Path(f"{CURR_DIR}/{BUILD_DIR}")
rs.set_vitis_platform(VITIS_PLATFORM)
rs.parallel_export_candidates(work_dir_to_ir)