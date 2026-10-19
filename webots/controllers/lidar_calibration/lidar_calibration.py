"""Standalone LIDAR ray-index <-> body-relative-angle calibration tool.

Diagnostic tool, not part of the training/ROS2 pipeline -- created to answer a concrete question:
does ray index N of ``lidar.getRangeImage()`` really correspond to the angle a naive formula
predicts? youbot_mecanum.py's own ``_lidar_min_range`` deliberately avoids trusting that mapping
(see its comment) because it was never verified live. This script verifies it live, the same way
``_lidar_calibrate_self_mask`` already verifies self-hit rays live instead of assuming them.

Fully autonomous, no manual robot dragging needed -- the two-step problem a human hit trying to do
this by hand (position a robot AND a landmark object precisely, then catch a message that only
prints once) is solved by doing it all in code:
  1. Reads this robot's own live pose (x, y, theta) the exact same way youbot_mecanum.py does
     (``self_node.getOrientation()``, ``theta = atan2(orientation[3], orientation[0])``) -- so
     "front" here means the same thing it means everywhere else in this codebase.
  2. Re-runs youbot_mecanum.py's own self-mask calibration (reused, not reimplemented) so the
     sweep below ignores whichever rays are hitting this robot's own chassis/arm -- confirmed by
     that function's own docstring to read ~0.067m, a value a live test independently reproduced.
  3. Spawns ONE small probe obstacle via the Supervisor API (``importMFNodeFromString`` on the
     world root's children field) -- no world file editing required, and no risk of leaving stray
     objects behind (it's removed at the end).
  4. Moves the probe through a full circle around the robot, in the robot's OWN body frame
     (angle 0 = straight ahead, +90 = left, 180 = behind, 270/-90 = right -- standard
     forward/left/up right-handed convention, matching ``theta``'s own meaning), a fixed radius
     away. At each stop: settles a few sim steps, reads the range image, finds which ray reports
     the shortest non-self-masked, non-inf distance, and records (angle, ray_index, range).
  5. Prints a full table plus the empirical index-per-degree step (unwrapped across the 0/179
     boundary), and writes everything to ``outputs/lidar_calibration_<robot_name>.txt`` so the
     result survives past terminal scrollback -- this is what actually fixes the "it only prints
     once and I can't grab it in time" problem.

Usage: temporarily point one robot's ``controller`` field at "lidar_calibration" instead of
"youbot_mecanum" (e.g. robot_0 in keyboard_test.wbt, which already has open floor around its
spawn cell), run the simulation, wait for "CALIBRATION COMPLETE" in the console or the output
file, then switch the controller field back.

Pick a robot spawn location with open space in every direction for a clean read -- if a real wall
or shelf is within SWEEP_RADIUS_M of some sweep angles, those specific angles will report the
wall instead of the probe, which is still informative (it tells you a real obstacle is there) but
adds noise to the ray-index fit for this specific purpose.
"""
from __future__ import annotations

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "youbot_mecanum"))
from youbot_mecanum import (  # noqa: E402
    LIDAR_MAX_RANGE_M,
    LIDAR_SAFETY_STOP_M,
    LIDAR_SELF_CALIBRATION_MARGIN_M,
    TIME_STEP_MS,
    _lidar_calibrate_self_mask,
)

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
_OUTPUT_DIR = os.path.join(_REPO_ROOT, "outputs")

SETTLE_STEPS = 8
SWEEP_RADIUS_M = 0.6  # inside [minRange, maxRange]=[0.05, 1.5], clear of the chassis itself
SWEEP_STEP_DEG = 15
PROBE_SIZE_XY_M = 0.15
PROBE_HEIGHT_M = 0.5  # matches this project's own wall Solids (translation z=0.25, size z=0.5)
PROBE_CENTER_Z_M = 0.25


def _log(lines: list[str], text: str) -> None:
    print(text, flush=True)
    lines.append(text)


def _spawn_probe(robot):
    root = robot.getRoot()
    children_field = root.getField("children")
    vrml = (
        "DEF LIDAR_CAL_PROBE Solid {\n"
        "  translation 0 0 -10\n"
        "  children [\n"
        "    Shape {\n"
        "      appearance PBRAppearance { baseColor 1 0.1 0.1 }\n"
        f"      geometry Box {{ size {PROBE_SIZE_XY_M} {PROBE_SIZE_XY_M} {PROBE_HEIGHT_M} }}\n"
        "    }\n"
        "  ]\n"
        '  name "lidar_cal_probe"\n'
        f"  boundingObject Box {{ size {PROBE_SIZE_XY_M} {PROBE_SIZE_XY_M} {PROBE_HEIGHT_M} }}\n"
        "}\n"
    )
    children_field.importMFNodeFromString(-1, vrml)
    return robot.getFromDef("LIDAR_CAL_PROBE")


def _robot_pose(self_node) -> tuple[float, float, float]:
    x, y, _z = self_node.getField("translation").getSFVec3f()
    orientation = self_node.getOrientation()
    theta = math.atan2(orientation[3], orientation[0])
    return x, y, theta


def _unwrap_index_delta(delta: int, resolution: int) -> int:
    """Smallest-magnitude equivalent of ``delta`` modulo ``resolution`` -- handles the ray index
    wrapping past 0/resolution-1 between two consecutive sweep angles."""
    half = resolution / 2.0
    while delta > half:
        delta -= resolution
    while delta < -half:
        delta += resolution
    return delta


def main() -> None:
    from controller import Supervisor

    robot = Supervisor()
    robot_name = robot.getName()

    self_node = robot.getSelf()
    if self_node is None:
        print(
            f"[lidar_calibration] FATAL: getSelf() returned None for {robot_name!r} -- this "
            f"controller needs `supervisor TRUE` on the robot node (already set on every Youbot "
            f"in this project's worlds -- check the world file wasn't hand-edited).",
            flush=True,
        )
        return

    lidar = robot.getDevice("lidar")
    if lidar is None:
        print(f"[lidar_calibration] FATAL: no 'lidar' device found on {robot_name!r}.", flush=True)
        return
    lidar.enable(TIME_STEP_MS)

    lines: list[str] = []
    _log(lines, f"[lidar_calibration] robot={robot_name!r} starting up, settling {SETTLE_STEPS} steps...")
    for _ in range(SETTLE_STEPS):
        if robot.step(TIME_STEP_MS) == -1:
            return

    horizontal_resolution = lidar.getHorizontalResolution()
    field_of_view = lidar.getFov()
    _log(
        lines,
        f"[lidar_calibration] lidar config: horizontalResolution={horizontal_resolution} "
        f"fieldOfView={field_of_view:.6f}rad ({math.degrees(field_of_view):.1f}deg) "
        f"minRange={lidar.getMinRange():.3f}m maxRange={lidar.getMaxRange():.3f}m",
    )

    self_mask = _lidar_calibrate_self_mask(lidar.getRangeImage())
    excluded = sum(self_mask)
    threshold = LIDAR_SAFETY_STOP_M + LIDAR_SELF_CALIBRATION_MARGIN_M
    self_hit_indices = [i for i, is_self in enumerate(self_mask) if is_self]
    _log(
        lines,
        f"[lidar_calibration] self-mask: {excluded}/{len(self_mask)} rays excluded as permanent "
        f"self-hits (closer than {threshold:.3f}m at calibration time). indices={self_hit_indices}",
    )

    rx, ry, rtheta = _robot_pose(self_node)
    _log(
        lines,
        f"[lidar_calibration] robot pose at calibration start: x={rx:.4f} y={ry:.4f} "
        f"theta={rtheta:.4f}rad ({math.degrees(rtheta):.1f}deg)",
    )

    probe_node = _spawn_probe(robot)
    probe_translation = probe_node.getField("translation")

    _log(
        lines,
        f"[lidar_calibration] sweeping probe at radius={SWEEP_RADIUS_M}m, step={SWEEP_STEP_DEG}deg "
        f"(0=front, +90=left, 180=behind, -90/270=right, relative to the robot's own heading)...",
    )

    results: list[tuple[int, int, float]] = []  # (angle_deg, ray_index, range_m)
    for angle_deg in range(0, 360, SWEEP_STEP_DEG):
        phi = math.radians(angle_deg)
        world_angle = rtheta + phi
        probe_x = rx + SWEEP_RADIUS_M * math.cos(world_angle)
        probe_y = ry + SWEEP_RADIUS_M * math.sin(world_angle)
        probe_translation.setSFVec3f([probe_x, probe_y, PROBE_CENTER_Z_M])

        for _ in range(SETTLE_STEPS):
            if robot.step(TIME_STEP_MS) == -1:
                return

        ranges = lidar.getRangeImage()
        candidates = [
            i for i, r in enumerate(ranges)
            if i < len(self_mask) and not self_mask[i] and r == r and r != float("inf")
        ]
        if not candidates:
            _log(
                lines,
                f"  angle={angle_deg:>4}deg  NO RETURN (nothing within {LIDAR_MAX_RANGE_M}m on any "
                f"non-self-masked ray -- probe may be outside the scan plane's height or blocked)",
            )
            continue

        min_idx = min(candidates, key=lambda i: ranges[i])
        min_range = ranges[min_idx]
        results.append((angle_deg, min_idx, min_range))
        _log(
            lines,
            f"  angle={angle_deg:>4}deg  ray_index={min_idx:>3}  range={min_range:.4f}m  "
            f"(expected ~{SWEEP_RADIUS_M - PROBE_SIZE_XY_M / 2:.3f}-{SWEEP_RADIUS_M:.3f}m)",
        )

    probe_node.remove()

    _log(lines, "[lidar_calibration] --- summary ---")
    if len(results) >= 2:
        deltas = []
        for (a0, i0, _), (a1, i1, _) in zip(results, results[1:]):
            step_deg = a1 - a0
            raw_delta = i1 - i0
            unwrapped = _unwrap_index_delta(raw_delta, horizontal_resolution)
            deltas.append(unwrapped / step_deg)
        avg_index_per_deg = sum(deltas) / len(deltas)
        theoretical = horizontal_resolution / math.degrees(field_of_view)
        _log(
            lines,
            f"[lidar_calibration] empirical index-per-degree (unwrapped, averaged over "
            f"{len(deltas)} consecutive pairs): {avg_index_per_deg:.4f}  "
            f"(theoretical horizontalResolution/fieldOfView = {theoretical:.4f})",
        )
        _log(
            lines,
            f"[lidar_calibration] sign: {'increasing index = counterclockwise (angle increasing)' if avg_index_per_deg > 0 else 'increasing index = clockwise (angle increasing)'}",
        )
        per_step_str = ", ".join(f"{d:.3f}" for d in deltas)
        _log(lines, f"[lidar_calibration] per-step index/deg values (watch for outliers): {per_step_str}")
    else:
        _log(lines, "[lidar_calibration] too few valid sweep points to compute a fit.")

    _log(lines, "[lidar_calibration] CALIBRATION COMPLETE.")

    os.makedirs(_OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(_OUTPUT_DIR, f"lidar_calibration_{robot_name}.txt")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"[lidar_calibration] full report written to {out_path}", flush=True)

    while robot.step(TIME_STEP_MS) != -1:
        pass


if __name__ == "__main__":
    main()
