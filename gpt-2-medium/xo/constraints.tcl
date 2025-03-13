

# Initialize an empty list to store undefined cells
set undefined_cells {}


# begin defining a slot for logic resources
create_pblock SLOT_X0Y0_TO_SLOT_X0Y0
resize_pblock SLOT_X0Y0_TO_SLOT_X0Y0 -add CLOCKREGION_X0Y1:CLOCKREGION_X4Y4


# begin defining a slot for logic resources
create_pblock SLOT_X0Y2_TO_SLOT_X0Y2
resize_pblock SLOT_X0Y2_TO_SLOT_X0Y2 -add CLOCKREGION_X0Y8:CLOCKREGION_X4Y10


# begin defining a slot for logic resources
create_pblock SLOT_X1Y2_TO_SLOT_X1Y2
resize_pblock SLOT_X1Y2_TO_SLOT_X1Y2 -add CLOCKREGION_X5Y8:CLOCKREGION_X9Y10


# begin defining a slot for logic resources
create_pblock SLOT_X0Y3_TO_SLOT_X0Y3
resize_pblock SLOT_X0Y3_TO_SLOT_X0Y3 -add CLOCKREGION_X0Y11:CLOCKREGION_X4Y13


# begin defining a slot for logic resources
create_pblock SLOT_X1Y3_TO_SLOT_X1Y3
resize_pblock SLOT_X1Y3_TO_SLOT_X1Y3 -add CLOCKREGION_X5Y11:CLOCKREGION_X9Y13


# begin defining a slot for logic resources
create_pblock SLOT_X1Y0_TO_SLOT_X1Y0
resize_pblock SLOT_X1Y0_TO_SLOT_X1Y0 -add CLOCKREGION_X5Y1:CLOCKREGION_X9Y4


# begin defining a slot for logic resources
create_pblock SLOT_X1Y1_TO_SLOT_X1Y1
resize_pblock SLOT_X1Y1_TO_SLOT_X1Y1 -add CLOCKREGION_X5Y5:CLOCKREGION_X9Y7


# begin defining a slot for logic resources
create_pblock SLOT_X0Y1_TO_SLOT_X0Y1
resize_pblock SLOT_X0Y1_TO_SLOT_X0Y1 -add CLOCKREGION_X0Y5:CLOCKREGION_X4Y7

set SLOT_X0Y0_TO_SLOT_X0Y0_cells {
    ext_platform_i/VitisRegion/opt_kernel/inst/SLOT_X0Y0_TO_SLOT_X0Y0.*
}
add_cells_to_pblock [get_pblocks SLOT_X0Y0_TO_SLOT_X0Y0] [get_cells -regex $SLOT_X0Y0_TO_SLOT_X0Y0_cells]

# Iterate through each cell in the list
foreach cell $SLOT_X0Y0_TO_SLOT_X0Y0_cells {
    set defined [llength [get_cells $cell]]
    if { $defined == 0 } {
        lappend undefined_cells $cell
    }
}

set SLOT_X0Y2_TO_SLOT_X0Y2_cells {
    ext_platform_i/VitisRegion/opt_kernel/inst/SLOT_X0Y2_TO_SLOT_X0Y2.*
}
add_cells_to_pblock [get_pblocks SLOT_X0Y2_TO_SLOT_X0Y2] [get_cells -regex $SLOT_X0Y2_TO_SLOT_X0Y2_cells]

# Iterate through each cell in the list
foreach cell $SLOT_X0Y2_TO_SLOT_X0Y2_cells {
    set defined [llength [get_cells $cell]]
    if { $defined == 0 } {
        lappend undefined_cells $cell
    }
}

set SLOT_X1Y2_TO_SLOT_X1Y2_cells {
    ext_platform_i/VitisRegion/opt_kernel/inst/SLOT_X1Y2_TO_SLOT_X1Y2.*
}
add_cells_to_pblock [get_pblocks SLOT_X1Y2_TO_SLOT_X1Y2] [get_cells -regex $SLOT_X1Y2_TO_SLOT_X1Y2_cells]

# Iterate through each cell in the list
foreach cell $SLOT_X1Y2_TO_SLOT_X1Y2_cells {
    set defined [llength [get_cells $cell]]
    if { $defined == 0 } {
        lappend undefined_cells $cell
    }
}

set SLOT_X0Y3_TO_SLOT_X0Y3_cells {
    ext_platform_i/VitisRegion/opt_kernel/inst/SLOT_X0Y3_TO_SLOT_X0Y3.*
}
add_cells_to_pblock [get_pblocks SLOT_X0Y3_TO_SLOT_X0Y3] [get_cells -regex $SLOT_X0Y3_TO_SLOT_X0Y3_cells]

# Iterate through each cell in the list
foreach cell $SLOT_X0Y3_TO_SLOT_X0Y3_cells {
    set defined [llength [get_cells $cell]]
    if { $defined == 0 } {
        lappend undefined_cells $cell
    }
}

set SLOT_X1Y3_TO_SLOT_X1Y3_cells {
    ext_platform_i/VitisRegion/opt_kernel/inst/SLOT_X1Y3_TO_SLOT_X1Y3.*
}
add_cells_to_pblock [get_pblocks SLOT_X1Y3_TO_SLOT_X1Y3] [get_cells -regex $SLOT_X1Y3_TO_SLOT_X1Y3_cells]

# Iterate through each cell in the list
foreach cell $SLOT_X1Y3_TO_SLOT_X1Y3_cells {
    set defined [llength [get_cells $cell]]
    if { $defined == 0 } {
        lappend undefined_cells $cell
    }
}


set SLOT_X1Y0_TO_SLOT_X1Y0_cells {
    ext_platform_i/VitisRegion/opt_kernel/inst/SLOT_X1Y0_TO_SLOT_X1Y0.*
}
add_cells_to_pblock [get_pblocks SLOT_X1Y0_TO_SLOT_X1Y0] [get_cells -regex $SLOT_X1Y0_TO_SLOT_X1Y0_cells]

# Iterate through each cell in the list
foreach cell $SLOT_X1Y0_TO_SLOT_X1Y0_cells {
    set defined [llength [get_cells $cell]]
    if { $defined == 0 } {
        lappend undefined_cells $cell
    }
}

set SLOT_X1Y1_TO_SLOT_X1Y1_cells {
    ext_platform_i/VitisRegion/opt_kernel/inst/SLOT_X1Y1_TO_SLOT_X1Y1.*
}
add_cells_to_pblock [get_pblocks SLOT_X1Y1_TO_SLOT_X1Y1] [get_cells -regex $SLOT_X1Y1_TO_SLOT_X1Y1_cells]

# Iterate through each cell in the list
foreach cell $SLOT_X1Y1_TO_SLOT_X1Y1_cells {
    set defined [llength [get_cells $cell]]
    if { $defined == 0 } {
        lappend undefined_cells $cell
    }
}

set SLOT_X0Y1_TO_SLOT_X0Y1_cells {
    ext_platform_i/VitisRegion/opt_kernel/inst/SLOT_X0Y1_TO_SLOT_X0Y1.*
}
add_cells_to_pblock [get_pblocks SLOT_X0Y1_TO_SLOT_X0Y1] [get_cells -regex $SLOT_X0Y1_TO_SLOT_X0Y1_cells]

# Iterate through each cell in the list
foreach cell $SLOT_X0Y1_TO_SLOT_X0Y1_cells {
    set defined [llength [get_cells $cell]]
    if { $defined == 0 } {
        lappend undefined_cells $cell
    }
}


if {[llength $undefined_cells] > 0} {
    puts "Undefined cells:"
    foreach cell $undefined_cells {
        puts $cell
    }
}
