#!/usr/bin/env python3
"""TP5 cost arithmetic, not a benchmark. See PERFORMANCE.md section 6.

All costs must use the same population and non-overlapping time intervals.
No model, GPU, network, subprocess, or optional package is required.
"""

import argparse
import json
import math
from typing import Any, Dict, Optional


def number(name: str, value: float, *, positive: bool = False) -> float:
    """Reject non-finite/negative costs, including booleans passed by callers."""
    if isinstance(value, bool):
        raise ValueError(f"{name}: expected a number, not bool")
    value = float(value)
    if not math.isfinite(value) or value < 0 or (positive and value == 0):
        raise ValueError(f"{name}: expected a finite {'positive' if positive else 'nonnegative'} number")
    return value


def checked(result: Dict[str, Any]) -> Dict[str, Any]:
    for key, value in result.items():
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError(f"{key}: arithmetic overflow or non-finite result")
    return {"status": "CALCULATION_NOT_BENCHMARK", **result}


def cycle(single_ms: float, draft_ms: float, verify_ms: float,
          catchup_ms: float, other_ms: float, committed: float,
          horizon: Optional[int] = None) -> Dict[str, Any]:
    single_ms = number("single_ms", single_ms, positive=True)
    committed = number("committed", committed, positive=True)
    costs = [number(name, value) for name, value in (
        ("draft_ms", draft_ms), ("verify_ms", verify_ms),
        ("catchup_ms", catchup_ms), ("other_ms", other_ms))]
    total = number("cycle_ms", sum(costs), positive=True)
    if horizon is not None:
        if isinstance(horizon, bool) or not isinstance(horizon, int) or horizon < 0:
            raise ValueError("horizon: expected a nonnegative integer")
        if committed > horizon + 1:
            raise ValueError("committed exceeds horizon + 1 for a standard speculative cycle")
    per_token = number("ms_per_token", total / committed, positive=True)
    result = {
        "cycle_ms": total,
        "committed_per_cycle": committed,
        "ms_per_token": per_token,
        "tokens_per_second": 1000.0 / per_token,
        "speedup_vs_single": single_ms / per_token,
        "break_even_committed_per_cycle": total / single_ms,
        "net_faster": per_token < single_ms,
        "assumption": "Same cycle population; component intervals do not overlap; committed means public tokens.",
    }
    if horizon == 1:
        # Unclamped on purpose: a threshold >= 1 means no strict win under
        # the ideal h=1 model. EOS/budget-truncated cycles need measured K.
        result["ideal_h1_acceptance_threshold"] = total / single_ms - 1.0
    return checked(result)


def budget(target_tps: float, committed: float,
           baseline_tps: Optional[float] = None) -> Dict[str, Any]:
    target_tps = number("target_tps", target_tps, positive=True)
    committed = number("committed", committed, positive=True)
    per_token = 1000.0 / target_tps
    result = {"target_tps": target_tps, "ms_per_token_budget": per_token,
              "cycle_ms_budget": per_token * committed,
              "assumption": "Assumed committed tokens per cycle stay unchanged."}
    if baseline_tps is not None:
        baseline_tps = number("baseline_tps", baseline_tps, positive=True)
        result["required_speedup"] = target_tps / baseline_tps
        result["required_time_reduction_fraction"] = 1.0 - baseline_tps / target_tps
    return checked(result)


def amdahl(fraction: float, factor: Optional[float] = None,
           eliminate: bool = False) -> Dict[str, Any]:
    fraction = number("fraction", fraction)
    if fraction > 1:
        raise ValueError("fraction must be in [0, 1]")
    if eliminate == (factor is not None):
        raise ValueError("choose exactly one of factor or eliminate")
    if eliminate:
        if fraction == 1:
            raise ValueError("eliminating all time gives no finite speed bound")
        relative = 1.0 - fraction
    else:
        factor = number("factor", factor, positive=True)
        relative = 1.0 - fraction + fraction / factor
    relative = number("relative_time", relative, positive=True)
    return checked({"relative_time": relative, "speedup": 1.0 / relative,
                    "time_reduction_fraction": 1.0 - relative,
                    "assumption": "Only this non-overlapping time fraction changes; workload is fixed."})


def resource(byte_count: float, bandwidth_gbs: float,
             flops: float = 0.0, compute_tflops: Optional[float] = None) -> Dict[str, Any]:
    byte_count = number("bytes", byte_count)
    bandwidth_gbs = number("bandwidth_gbs", bandwidth_gbs, positive=True)
    flops = number("flops", flops)
    if flops and compute_tflops is None:
        raise ValueError("compute_tflops is required when flops > 0")
    memory_ms = byte_count / bandwidth_gbs / 1e6  # decimal GB/s
    compute_ms = 0.0
    if compute_tflops is not None:
        compute_tflops = number("compute_tflops", compute_tflops, positive=True)
        compute_ms = flops / compute_tflops / 1e9
    return checked({"memory_ms": memory_ms, "compute_ms": compute_ms,
                    "resource_max_ms": max(memory_ms, compute_ms),
                    "assumption": "Conditional resource model only, not measured time or an end-to-end hardware limit."
                                  " Bytes must refer to the specified cache boundary; arithmetic types must match."})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    cyc = sub.add_parser("cycle", help="MTP cycle cost and break-even")
    cyc.add_argument("--single-ms", type=float, required=True)
    cyc.add_argument("--draft-ms", type=float, default=0.0)
    cyc.add_argument("--verify-ms", type=float, required=True)
    cyc.add_argument("--catchup-ms", type=float, default=0.0)
    cyc.add_argument("--other-ms", type=float, default=0.0)
    cyc.add_argument("--committed", type=float, required=True)
    cyc.add_argument("--horizon", type=int)
    bud = sub.add_parser("budget", help="Convert a target token rate to a cycle budget")
    bud.add_argument("--target-tps", type=float, required=True)
    bud.add_argument("--committed", type=float, required=True)
    bud.add_argument("--baseline-tps", type=float)
    amd = sub.add_parser("amdahl", help="Bound the gain from one time fraction")
    amd.add_argument("--fraction", type=float, required=True)
    group = amd.add_mutually_exclusive_group(required=True)
    group.add_argument("--factor", type=float)
    group.add_argument("--eliminate", action="store_true")
    res = sub.add_parser("resource", help="Conditional bandwidth/compute arithmetic")
    res.add_argument("--bytes", dest="byte_count", type=float, required=True)
    res.add_argument("--bandwidth-gbs", type=float, required=True)
    res.add_argument("--flops", type=float, default=0.0)
    res.add_argument("--compute-tflops", type=float)
    args = vars(parser.parse_args())
    command = args.pop("command")
    try:
        result = {"cycle": cycle, "budget": budget, "amdahl": amdahl,
                  "resource": resource}[command](**args)
        print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    except (ValueError, OverflowError, ZeroDivisionError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
