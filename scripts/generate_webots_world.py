#!/usr/bin/env python3
"""Generate a Webots ``.wbt`` world from the same procedural map + robot start positions
``WarehouseEnv.reset()`` would use, so the physical world and the Python model start in sync.

Usage:
    python scripts/generate_webots_world.py --env-config configs/env_small.yaml \
        --output webots/worlds/hiverl_warehouse.wbt
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import yaml

from hiverl.bridge.webots_world_generator import WEBOTS_VERSION, write_world_file
from hiverl.env.warehouse_grid import generate_warehouse_map

ROOT = os.path.join(os.path.dirname(__file__), "..")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-config", default=os.path.join(ROOT, "configs", "env_small.yaml"))
    parser.add_argument(
        "--output", default=os.path.join(ROOT, "webots", "worlds", "hiverl_warehouse.wbt")
    )
    parser.add_argument(
        "--robot-model", choices=["youbot", "agv"], default="youbot",
        help="'youbot' = real KUKA youBot PROTO (mecanum base, default); "
        "'agv' = self-authored differential-drive body, lower risk/complexity",
    )
    parser.add_argument(
        "--controller", default=None,
        help="defaults to the matching controller for --robot-model "
        "(youbot_mecanum or diff_drive_agv)",
    )
    parser.add_argument(
        "--webots-version", default=WEBOTS_VERSION,
        help="must match your installed Webots version (Help > About) -- controls the .wbt "
        "header and the EXTERNPROTO URLs for TexturedBackground/TexturedBackgroundLight/Floor"
        "/Youbot",
    )
    parser.add_argument(
        "--robots", type=int, default=None,
        help="override num_robots from the env config (e.g. --robots 5)",
    )
    parser.add_argument(
        "--render-shelves", action="store_true",
        help="draw shelf Solid boxes in the Webots visual world (off by default, per user "
        "request -- see docs/webots_ros2_integration.md section 8f). Purely visual: the "
        "underlying grid/A*/Safety Shield/policy always route around shelf cells either way, "
        "since none of that logic reads the Webots world file.",
    )
    args = parser.parse_args()

    with open(args.env_config) as f:
        cfg = yaml.safe_load(f)
    if args.robots is not None:
        cfg["num_robots"] = args.robots

    warehouse_map = generate_warehouse_map(
        width=cfg["width"],
        height=cfg["height"],
        num_shelf_blocks_x=cfg["num_shelf_blocks_x"],
        num_shelf_blocks_y=cfg["num_shelf_blocks_y"],
        shelf_block_w=cfg["shelf_block_w"],
        shelf_block_h=cfg["shelf_block_h"],
        num_pickup_stations=cfg["num_pickup_stations"],
        num_dropoff_stations=cfg["num_dropoff_stations"],
    )

    rng = np.random.default_rng(cfg["seed"])
    free_cells = [c for c in warehouse_map.free_cells() if warehouse_map.grid[c] == 0]
    rng.shuffle(free_cells)
    start_cells = free_cells[: cfg["num_robots"]]

    resolved_controller = args.controller or ("youbot_mecanum" if args.robot_model == "youbot" else "diff_drive_agv")

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    write_world_file(
        warehouse_map, start_cells, args.output,
        controller=args.controller, webots_version=args.webots_version,
        robot_model=args.robot_model, render_shelves=args.render_shelves,
    )

    sidecar_path = args.output + ".robots.json"
    with open(sidecar_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "robot_model": args.robot_model,
                "controller": resolved_controller,
                "start_cells": [[int(r), int(c)] for r, c in start_cells],
            },
            f,
            indent=2,
        )

    print(
        f"Wrote {args.output} ({cfg['num_robots']} robots, {cfg['width']}x{cfg['height']} grid, "
        f"robot_model={args.robot_model}, render_shelves={args.render_shelves}, "
        f"webots_version={args.webots_version})"
    )
    print(f"Wrote {sidecar_path} (per-robot start cells, for scripts/run_extern_controllers.sh)")


if __name__ == "__main__":
    main()
