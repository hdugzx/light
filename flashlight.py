#!/usr/bin/env python3
"""Control the flashlight of an OPPO/OnePlus (ColorOS) phone over ADB.
No root required. Works by opening the Quick Settings panel, locating the
flashlight tile in the UI hierarchy (so it is resilient to layout changes),
reading its on/off state and tapping it only when needed.
Usage:
    python3 flashlight.py on        # turn flashlight on
    python3 flashlight.py off       # turn flashlight off
    python3 flashlight.py toggle    # flip current state (default)
    python3 flashlight.py status    # just print current state
"""
import re
import sys
import time
import subprocess
import xml.etree.ElementTree as ET
TILE_LABEL = "手电筒"          # flashlight tile label on ColorOS (Chinese)
TILE_LABEL_EN = "Flashlight"   # fallback for English locale
REMOTE_DUMP = "/sdcard/_flashlight_ui.xml"
def adb(*args):
    return subprocess.run(["adb", *args], capture_output=True, text=True)
def ensure_device():
    out = adb("get-state").stdout.strip()
    if out != "device":
        sys.exit("No authorized ADB device. Plug in the phone and enable USB debugging.")
def wake():
    adb("shell", "input", "keyevent", "KEYCODE_WAKEUP")
    time.sleep(0.3)
def expand_quick_settings():
    adb("shell", "cmd", "statusbar", "expand-settings")
    time.sleep(0.9)
def collapse():
    adb("shell", "cmd", "statusbar", "collapse")
def dump_ui():
    """Return the UI hierarchy XML of the current screen."""
    for _ in range(3):
        adb("shell", "uiautomator", "dump", REMOTE_DUMP)
        xml = adb("exec-out", "cat", REMOTE_DUMP).stdout
        if "<hierarchy" in xml:
            return xml
        time.sleep(0.6)
    sys.exit("Could not read the UI hierarchy from the device.")
def parse_bounds(bounds):
    m = re.match(r"\[(\d+),(\d+)\]\[(\d+),(\d+)\]", bounds or "")
    if not m:
        return None
    x1, y1, x2, y2 = map(int, m.groups())
    return (x1 + x2) // 2, (y1 + y2) // 2
def find_tile(xml):
    """Locate the flashlight tile node. Returns (center_xy, is_on)."""
    root = ET.fromstring(xml)
    parents = {c: p for p in root.iter() for c in p}
    target = None
    for node in root.iter("node"):
        text = node.get("text", "")
        desc = node.get("content-desc", "")
        if TILE_LABEL in text or TILE_LABEL in desc \
                or TILE_LABEL_EN.lower() in text.lower() \
                or TILE_LABEL_EN.lower() in desc.lower():
            target = node
            break
    if target is None:
        return None, None
    # Walk up to find a clickable ancestor (the actual tile button) and its state.
    node = target
    center = parse_bounds(target.get("bounds"))
    is_on = None
    for _ in range(6):
        if node.get("checkable") == "true" or node.get("clickable") == "true":
            c = parse_bounds(node.get("bounds"))
            if c:
                center = c
            if node.get("checked") in ("true", "false"):
                is_on = node.get("checked") == "true"
            desc = node.get("content-desc", "")
            if "开启" in desc or "已打开" in desc:
                is_on = True
            elif "关闭" in desc or "已关闭" in desc:
                is_on = False
            break
        node = parents.get(node)
        if node is None:
            break
    return center, is_on
def get_state():
    """Return (center_xy, is_on) with Quick Settings expanded."""
    wake()
    expand_quick_settings()
    xml = dump_ui()
    return find_tile(xml)
def main():
    action = (sys.argv[1] if len(sys.argv) > 1 else "toggle").lower()
    ensure_device()
    center, is_on = get_state()
    if center is None:
        collapse()
        sys.exit(f"Flashlight tile ('{TILE_LABEL}') not found in Quick Settings.")
    state_str = {True: "ON", False: "OFF", None: "UNKNOWN"}[is_on]
    if action == "status":
        print(f"Flashlight is currently {state_str}")
        collapse()
        return
    should_tap = (
        action == "toggle"
        or (action == "on" and is_on is not True)
        or (action == "off" and is_on is not False)
    )
    if should_tap:
        x, y = center
        adb("shell", "input", "tap", str(x), str(y))
        time.sleep(0.4)
        print(f"Tapped flashlight tile at ({x}, {y}). Previous state: {state_str}, action: {action}.")
    else:
        print(f"Flashlight already {state_str}; nothing to do for action '{action}'.")
    collapse()
if __name__ == "__main__":
    main()
