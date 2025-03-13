from rapidstream import RapidStreamTAPA, DeviceFactory

rs = RapidStreamTAPA("rs_build/")
rs.reset()
factory = DeviceFactory(
    row=4,
    col=2,
    part_num="xcvp1802-lsvc4072-2MP-e-S"
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
factory.set_slot_crossing_capacity(0, 0, north=9435)
factory.set_slot_crossing_capacity(1, 0, north=9435)
factory.set_slot_crossing_capacity(0, 1, north=9435)
factory.set_slot_crossing_capacity(1, 1, north=9435)
factory.set_slot_crossing_capacity(0, 2, north=9435)
factory.set_slot_crossing_capacity(1, 2, north=9435)

# Call factory to extract the slot resources automatically from Vivado:
factory.extract_slot_resources()

# The device can be supplied as the virtual device for the RapidStream APIs:
device = factory.generate_virtual_device()
rs.set_virtual_device(device)

rs.add_xo_file("./opt-stage4-dot-prod.tapa/opt.hw.xo")
rs.set_top_module_name("opt_kernel")
rs.add_clock("ap_clk", period_ns=3.33)

rs.set_vitis_connectivity_config("link_config_versal.ini")
rs.assign_port_to_region(".*", "SLOT_X0Y0:SLOT_X1Y0")
rs.run_dse(max_workers=1, max_dse_limit=0.9, min_dse_limit=0.6)
