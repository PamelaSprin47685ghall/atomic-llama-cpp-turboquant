#!/usr/bin/env python3
"""Exercise recovery control flow against an isolated fake PCI filesystem."""

import os
import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def exercise(source: str, *, native: bool, fail_clear: bool = False, missing_dpm: bool = False,
             busy_before: str = "0", clocks_only: bool = False):
    with tempfile.TemporaryDirectory(prefix="eagle-mock-") as tmp:
        root = Path(tmp)
        replacements = {
            "/sys/bus/pci/devices": str(root / "sys/bus/pci/devices"),
            "/sys/bus/pci/rescan": str(root / "sys/bus/pci/rescan"),
            "/sys/class/sound": str(root / "sys/class/sound"),
            "/dev/dri": str(root / "dev/dri"),
            "/dev/snd": str(root / "dev/snd"),
            "/run/lock/gpu-hard-unlock.lock": str(root / "recovery.lock"),
            "$(id -u)": "0",
        }
        for old, new in replacements.items():
            assert old in source, f"fixture must isolate {old}"
            source = source.replace(old, new)
        script = root / "gpu-hard-unlock.sh"
        script.write_text(source)

        devices = root / "sys/bus/pci/devices"
        devices.mkdir(parents=True)
        bridge = root / "topology/0000:0e:00.0"
        gpu = bridge / "0000:0f:00.0"
        (gpu / "drm/card1").mkdir(parents=True)
        (devices / "0000:0e:00.0").symlink_to(bridge, target_is_directory=True)
        (devices / "0000:0f:00.0").symlink_to(gpu, target_is_directory=True)
        (root / "sys/bus/pci/rescan").write_text("")
        busy_file = gpu / "gpu_busy_percent"
        busy_file.write_text(busy_before + "\n")
        perf = gpu / "power_dpm_force_performance_level"
        perf.write_text("high\n")
        sclk = gpu / "pp_dpm_sclk"
        if not missing_dpm:
            sclk.write_text("S: 0Mhz *\n1: 500Mhz\n2: 1200Mhz\n")
        (gpu / "remove").write_text("")
        if native:
            (gpu / "reset").write_text("")

        mocks = root / "bin"
        mocks.mkdir()
        state = root / "bridge-control"
        state.write_text("0000")
        setpci = mocks / "setpci"
        setpci.write_text("""#!/usr/bin/env python3
import os, sys
from pathlib import Path
state = Path(os.environ["GPU_MOCK_PCI_STATE"])
arg = sys.argv[-1]
if "=" not in arg:
    print(state.read_text())
else:
    value = arg.split("=", 1)[1].lower()
    failed = Path(os.environ["GPU_MOCK_CLEAR_FAILED"])
    if os.getenv("GPU_MOCK_FAIL_CLEAR") == "1" and value == "0000" and state.read_text().strip() == "0040" and not failed.exists():
        failed.write_text("1")
        sys.exit(1)
    if value == "0000" and state.read_text().strip() == "0040":
        Path(os.environ["GPU_MOCK_BUSY_FILE"]).write_text("0\\n")
    state.write_text(value)
""")
        setpci.chmod(0o700)
        for name, content in (("fuser", "#!/bin/sh\nexit 1\n"), ("sleep", "#!/bin/sh\nexit 0\n")):
            path = mocks / name
            path.write_text(content)
            path.chmod(0o700)
        env = {
            **os.environ,
            "PATH": str(mocks) + ":" + os.environ["PATH"],
            "GPU_MOCK_PCI_STATE": str(state),
            "GPU_MOCK_CLEAR_FAILED": str(root / "clear-failed"),
            "GPU_MOCK_BUSY_FILE": str(busy_file),
            "GPU_MOCK_FAIL_CLEAR": "1" if fail_clear else "0",
        }
        args = ["bash", str(script)] + (["--clocks-only"] if clocks_only else []) + ["card1"]
        result = subprocess.run(args, env=env, capture_output=True, text=True, timeout=15)
        return {
            "status": result.returncode,
            "perf": perf.read_text().strip(),
            "sclk": sclk.read_text().strip() if sclk.exists() else None,
            "native_reset": (gpu / "reset").read_text().strip() if native else None,
            "removed": (gpu / "remove").read_text().strip(),
            "rescanned": (root / "sys/bus/pci/rescan").read_text().strip(),
            "bridge": state.read_text().strip(),
            "trap": "Emergency EXIT trap" in result.stderr,
            "stderr": result.stderr,
        }


def check(result, *, status, perf, bridge="0000", sclk=None):
    assert result["status"] == status, result["stderr"]
    assert result["perf"] == perf, result["stderr"]
    assert result["bridge"] == bridge, result["stderr"]
    if sclk is not None:
        assert result["sclk"] == sclk, result["stderr"]


def main():
    source = (ROOT / "scripts/gpu-hard-unlock.sh").read_text()
    kernel_reset = exercise(source, native=True, busy_before="99", clocks_only=True)
    check(kernel_reset, status=0, perf="manual", sclk="2")
    assert kernel_reset["native_reset"] == "" and kernel_reset["removed"] == "" and kernel_reset["rescanned"] == ""

    clock_failure = exercise(source, native=True, missing_dpm=True, clocks_only=True)
    check(clock_failure, status=7, perf="high")
    assert clock_failure["native_reset"] == "" and clock_failure["removed"] == ""

    direct = exercise(source, native=True)
    check(direct, status=0, perf="manual", sclk="2")
    assert direct["native_reset"] == "1" and direct["removed"] == "" and direct["rescanned"] == ""

    bus = exercise(source, native=False)
    check(bus, status=0, perf="manual", sclk="2")
    assert bus["removed"] == "1" and bus["rescanned"] == "1"

    still_busy = exercise(source, native=True, busy_before="99")
    check(still_busy, status=0, perf="manual", sclk="2")
    assert still_busy["removed"] == "1" and still_busy["rescanned"] == "1", still_busy["stderr"]

    deassert_failure = exercise(source, native=False, fail_clear=True)
    check(deassert_failure, status=6, perf="high")
    assert deassert_failure["trap"] and deassert_failure["rescanned"] == "", deassert_failure["stderr"]

    dpm_failure = exercise(source, native=True, missing_dpm=True)
    check(dpm_failure, status=7, perf="high")
    assert dpm_failure["native_reset"] == "1", dpm_failure["stderr"]
    print("Isolated native/SBR recovery, emergency release, and DPM failure checks passed")


if __name__ == "__main__":
    main()
