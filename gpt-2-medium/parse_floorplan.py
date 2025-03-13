import json
from enum import Enum, auto
from typing import Any
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("-f", "--file", dest="filename", type=str,
                    help="input floorplan json file", metavar="FILE")

class IREnum(Enum):
    """Enums to parse Rapidstream NOC IR."""

    PIPELINE = "__rs_hs_pipeline"
    REGION = "REGION"
    BODY = "BODY"
    HEAD_REGION = "__HEAD_REGION"
    TAIL_REGION = "__TAIL_REGION"
    DATA_WIDTH = "DATA_WIDTH"
    DEPTH = "DEPTH"
    BODY_LEVEL = "BODY_LEVEL"
    IF_DOUT = "if_dout"
    IF_EMPTY_N = "if_empty_n"
    IF_READ = "if_read"
    IF_DIN = "if_din"
    IF_FULL_N = "if_full_n"
    IF_WRITE = "if_write"
    NMU = "nmu_"
    NSU = "nsu_"
    CC_MASTER = "_cc_master"
    CC_RET = "_cc_ret"
    RS_ROUTE = "RS_ROUTE"
    FLOORPLAN_REGION = "floorplan_region"
    PRAGMAS = "pragmas"
    LIT = "lit"

PIPELINE_MAPPING = {
    "__rs_ap_ctrl_start_ready_pipeline": "AP",
    "__rs_ff_pipeline": "FF",
    "__rs_hs_pipeline": "HS",
}

def parse_top_mod(ir: dict[str, Any]) -> Any:
    """Parses the top_mod dict in the Rapidstream IR.

    Return a dictionary.

    Example:
    >>> design = {
    ...     "modules": {
    ...         "top_name": "FINDME",
    ...         "module_definitions": [{"name": "FINDME"}],
    ...     }
    ... }
    >>> parse_top_mod(design)
    {'name': 'FINDME'}
    """
    top_mod = ir["modules"]["top_name"]
    for mod in ir["modules"]["module_definitions"]:
        if mod["name"] == top_mod:
            return mod
    raise AssertionError()

def parse_mod(ir: dict[str, Any], name: str) -> Any:
    """Parses a given module's IR in the Rapidstream IR.

    Return a dictionary.
    """
    for mod in ir["modules"]["module_definitions"]:
        if mod["name"] == name:
            return mod
    return {}

def find_repr(source: list[dict[str, Any]], key: str) -> str:
    """Finds the first type repr value of a key in the Rapidstream list IR.

    Returns a string.
    """
    for e in find_expr(source, key):
        return str(e["repr"])
    print(f"WARNING: repr for key {key} not found!")
    return ""

def find_expr(
    source: list[dict[str, Any | list[dict[str, str]]]], key: str
) -> list[dict[str, str]]:
    """Finds the expr value of a key in the Rapidstream list IR.

    Returns a string.
    """
    for c in source:
        if c["name"] == key:
            return c["expr"]
    print(f"WARNING: expr for key {key} not found!")
    return []

def parse_floorplan(ir: dict[str, Any], grouped_mod_name: str) -> dict[str, list[str]]:
    """Parses the top module and grouped module's floorplan regions.

    Return a dictionary where keys are slots and values are submodules.
    """
    combined_mods = {
        # top
        "": parse_top_mod(ir)["submodules"],
    }
    if grouped_mod_ir := parse_mod(ir, grouped_mod_name):
        # grouped module
        combined_mods[f"{grouped_mod_name}_0/"] = grouped_mod_ir["submodules"]

    insts = {}
    for parent, mods in combined_mods.items():
        for sub_mod in mods:
            sub_mod_name = parent + sub_mod["name"]
            if sub_mod["floorplan_region"] is not None:
                # regular module
                insts[sub_mod_name] = sub_mod["floorplan_region"]
            elif sub_mod["module"] in PIPELINE_MAPPING:
                # pipeline module, needs to extract slot of each reg
                mapped_name = PIPELINE_MAPPING[sub_mod["module"]]
                body_level = find_repr(sub_mod["parameters"], IREnum.BODY_LEVEL.value)
                insts[f"{sub_mod_name}/RS_{mapped_name}_PP_HEAD"] = find_repr(
                    sub_mod["parameters"], IREnum.HEAD_REGION.value
                ).strip('"')
                insts[f"{sub_mod_name}/RS_{mapped_name}_PP_TAIL"] = find_repr(
                    sub_mod["parameters"], IREnum.TAIL_REGION.value
                ).strip('"')
                for i in range(int(body_level)):
                    insts[f"{sub_mod_name}/RS_{mapped_name}_PP_BODY_{i}"] = find_repr(
                        sub_mod["parameters"], f"__BODY_{i}_REGION"
                    ).strip('"')

    # convert {instance: slot} to {slot: [instances]}
    floorplan: dict[str, list[str]] = {}
    for sub_mod_name, slot in insts.items():
        assert slot is not None, f"{sub_mod_name} cannot have null slot!"
        if slot not in floorplan:
            floorplan[slot] = []
        floorplan[slot].append(sub_mod_name)
    return floorplan


def extract_slot_coord(slot_name: str) -> tuple[int, int]:
    """Extracts the x and y coordinates from the slot name.

    Returns a coordinate tuple as (x, y) in int.

    Example:
    >>> extract_slot_coord("SLOT_X0Y1")
    (0, 1)
    """
    return int(slot_name.split("X")[1].split("Y")[0]), int(slot_name.split("Y")[1])

def export_constraint(floorplan: dict[str, list[str]], kernel_name: str) -> list[str]:
    """Generates tcl constraints given the floorplan dictionary.

    Returns a list of tcl commands.
    """
    tcl = [
        """

# Initialize an empty list to store undefined cells
set undefined_cells {}
"""
    ]

    cr_map = [
        ["CLOCKREGION_X0Y1:CLOCKREGION_X4Y4", "CLOCKREGION_X0Y5:CLOCKREGION_X4Y7", "CLOCKREGION_X0Y8:CLOCKREGION_X4Y10", "CLOCKREGION_X0Y11:CLOCKREGION_X4Y13"],
        ["CLOCKREGION_X5Y1:CLOCKREGION_X9Y4", "CLOCKREGION_X5Y5:CLOCKREGION_X9Y7", "CLOCKREGION_X5Y8:CLOCKREGION_X9Y10", "CLOCKREGION_X5Y11:CLOCKREGION_X9Y13"]
    ]

    for slot in floorplan.keys():
        slot1, slot2 = slot.split("_TO_")
        assert slot1 == slot2
        x, y = extract_slot_coord(slot1)
        cr = cr_map[x][y]
        tcl += [
            f"""
# begin defining a slot for logic resources
create_pblock {slot}
resize_pblock {slot} -add {cr}
"""
        ]

    for slot, _ in floorplan.items():
        tcl += [f"set {slot}_cells {{"]
        tcl += [f"    ext_platform_i/VitisRegion/{kernel_name}/inst/{slot}_0/.*"]
        tcl += [
            f"""}}
add_cells_to_pblock [get_pblocks {slot}] [get_cells -regex ${slot}_cells]

# Iterate through each cell in the list
foreach cell ${slot}_cells {{
    set defined [llength [get_cells $cell]]
    if {{ $defined == 0 }} {{
        lappend undefined_cells $cell
    }}
}}
"""
        ]

    tcl += [
        """
if {[llength $undefined_cells] > 0} {
    puts "Undefined cells:"
    foreach cell $undefined_cells {
        puts $cell
    }
}
"""
    ]

    return tcl

if __name__ == "__main__":
    args = parser.parse_args()

    with open(args.filename, "r", encoding="utf-8") as file:
        ir = json.load(file)
    
    pipeline_dict = parse_floorplan(ir, "")
    tcl = export_constraint(pipeline_dict, "opt_kernel")

    with open("constraints.tcl", "w", encoding="utf-8") as file:
        file.write("\n".join(tcl))
