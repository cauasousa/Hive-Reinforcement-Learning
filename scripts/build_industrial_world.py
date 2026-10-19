#!/usr/bin/env python3
"""Build webots/worlds/hiverl_warehouse_industrial.wbt from the training world with an industrial look.

Walls, pads, packages and robots are copied unchanged; only visual nodes are added (no physics),
either as floor decals or above the 0.5 m wall, so the robots' LiDAR sees the same world. Run from
the repo root: python3 scripts/build_industrial_world.py
"""
from __future__ import annotations

import json
import math
import os
import re
import shutil

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
WORLDS = os.path.join(ROOT, "webots", "worlds")
BASE = os.path.join(WORLDS, "hiverl_warehouse_extern.wbt")
OUT = os.path.join(WORLDS, "hiverl_warehouse_industrial.wbt")
TEX_DIR = os.path.join(WORLDS, "textures")
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"

# Map geometry in meters (cell (row, col) center is x = col + 0.5, y = -(row + 0.5)).
MAP_W, MAP_H = 17.0, 13.0
WALL_T = 0.2
WALL_TOP_Z = 0.5
CLAD_TOP_Z = 2.4
SOUTH_CLAD_TOP_Z = 0.9
# Shelf blocks of the env map (logical obstacles, no physical shelf).
SHELF_ROWS = [(3, 5), (7, 9)]
SHELF_COLS = [(4, 5), (7, 8), (10, 11)]

DECAL_Z = 0.0012
DECAL_H = 0.0016


def _font(size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(FONT, size)


def _centered_text(draw: ImageDraw.ImageDraw, box, text: str, font, fill) -> None:
    x0, y0, x1, y1 = box
    l, t, r, b = draw.textbbox((0, 0), text, font=font)
    draw.text(((x0 + x1 - (r - l)) / 2 - l, (y0 + y1 - (b - t)) / 2 - t), text, font=font, fill=fill)


def _smooth_noise(rng: np.random.Generator, size: int, cells: int) -> np.ndarray:
    """Tileable value noise in [-1, 1]."""
    grid = rng.uniform(-1.0, 1.0, (cells, cells))
    tiled = np.tile(grid, (3, 3))
    img = Image.fromarray(((tiled + 1.0) * 127.5).astype(np.uint8))
    big = img.resize((size * 3, size * 3), Image.BICUBIC)
    arr = np.asarray(big, dtype=np.float32)[size:2 * size, size:2 * size]
    return arr / 127.5 - 1.0


def make_concrete(path: str) -> None:
    """Concrete floor texture with tone variation and joints on the tile edges."""
    size = 1024
    rng = np.random.default_rng(7)
    base = 0.33
    tone = (
        0.050 * _smooth_noise(rng, size, 4)
        + 0.030 * _smooth_noise(rng, size, 12)
        + 0.015 * _smooth_noise(rng, size, 48)
    )
    grain = rng.normal(0.0, 0.018, (size, size))
    lum = base + tone + grain
    for _ in range(2500):
        x, y = rng.integers(0, size, 2)
        lum[y, x] += rng.choice([-0.08, 0.07])
    lum = np.clip(lum, 0.0, 1.0)
    rgb = np.stack([lum * 0.98, lum * 0.99, lum * 1.02], axis=-1)
    img = Image.fromarray((np.clip(rgb, 0, 1) * 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(0.6))
    draw = ImageDraw.Draw(img)
    joint = (48, 49, 51)
    for e in (0, size - 1):
        draw.line([(e, 0), (e, size)], fill=joint, width=3)
        draw.line([(0, e), (size, e)], fill=joint, width=3)
    img.save(path)


def make_hazard(path: str) -> None:
    w, h = 512, 64
    img = Image.new("RGB", (w, h), (24, 24, 24))
    draw = ImageDraw.Draw(img)
    stripe = 32
    for x0 in range(-h, w + h, stripe * 2):
        draw.polygon([(x0, h), (x0 + stripe, h), (x0 + stripe + h, 0), (x0 + h, 0)], fill=(242, 190, 22))
    img.save(path)


def make_hatch(path: str) -> None:
    size = 256
    img = Image.new("RGB", (size, size), (58, 59, 61))
    draw = ImageDraw.Draw(img)
    for x0 in range(-size, size * 2, 64):
        draw.line([(x0, size), (x0 + size, 0)], fill=(196, 160, 30), width=10)
    img.save(path)


def make_floor_label(path: str, text: str, color) -> None:
    w, h = 1024, 256
    img = Image.new("RGB", (w, h), (66, 67, 69))
    draw = ImageDraw.Draw(img)
    draw.rectangle([8, 8, w - 9, h - 9], outline=color, width=10)
    _centered_text(draw, (0, 0, w, h), text, _font(140), color)
    img.save(path)


def make_estop_label(path: str) -> None:
    w, h = 1024, 256
    img = Image.new("RGB", (w, h), (244, 196, 24))
    draw = ImageDraw.Draw(img)
    draw.rectangle([8, 8, w - 9, h - 9], outline=(20, 20, 20), width=10)
    _centered_text(draw, (0, 0, w, h * 0.62), "EMERGENCY", _font(128), (20, 20, 20))
    _centered_text(draw, (0, h * 0.50, w, h), "STOP", _font(106), (190, 20, 20))
    img.save(path)


def make_voltage_label(path: str) -> None:
    w, h = 512, 256
    img = Image.new("RGB", (w, h), (244, 196, 24))
    draw = ImageDraw.Draw(img)
    draw.rectangle([5, 5, w - 6, h - 6], outline=(20, 20, 20), width=8)
    cx, top, bot = w / 2, 22, 140
    draw.polygon([(cx, top), (cx - 70, bot), (cx + 70, bot)], outline=(20, 20, 20), width=10)
    draw.polygon([(cx + 8, 50), (cx - 18, 98), (cx + 2, 98), (cx - 10, 128), (cx + 22, 84), (cx + 2, 84)],
                 fill=(20, 20, 20))
    _centered_text(draw, (0, 150, w, 200), "DANGER", _font(46), (20, 20, 20))
    _centered_text(draw, (0, 198, w, 246), "HIGH VOLTAGE", _font(36), (20, 20, 20))
    img.save(path)


def make_display(path: str) -> None:
    w, h = 256, 128
    img = Image.new("RGB", (w, h), (8, 30, 14))
    draw = ImageDraw.Draw(img)
    _centered_text(draw, (0, 0, w, h * 0.55), "SYSTEM OK", _font(34), (80, 255, 120))
    _centered_text(draw, (0, h * 0.50, w, h), "FLEET: 5", _font(28), (80, 255, 120))
    img.save(path)


def make_sign(path: str, text: str) -> None:
    w, h = 1024, 256
    img = Image.new("RGB", (w, h), (28, 64, 120))
    draw = ImageDraw.Draw(img)
    draw.rectangle([8, 8, w - 9, h - 9], outline=(235, 235, 235), width=10)
    _centered_text(draw, (0, 0, w, h), text, _font(138), (240, 240, 240))
    img.save(path)


def f(*v: float) -> str:
    return " ".join(f"{x:.4f}".rstrip("0").rstrip(".") if x != int(x) else str(int(x)) for x in v)


def appearance(color=(0.5, 0.5, 0.5), rough=0.6, metal=0.0, emissive=None, emissive_intensity=1.0,
               texture=None, tex_scale=None, tex_rotation=0.0) -> str:
    """PBRAppearance node text."""
    parts = [f"baseColor {f(*color)}", f"roughness {f(rough)}", f"metalness {f(metal)}"]
    if emissive is not None:
        parts.append(f"emissiveColor {f(*emissive)}")
        parts.append(f"emissiveIntensity {f(emissive_intensity)}")
    if texture is not None:
        parts.append(f'baseColorMap ImageTexture {{ url [ "textures/{texture}" ] filtering 4 }}')
        if tex_scale is not None or tex_rotation:
            sx, sy = tex_scale or (1, 1)
            parts.append(f"textureTransform TextureTransform {{ scale {f(sx, sy)} rotation {f(tex_rotation)} }}")
    return "PBRAppearance { " + " ".join(parts) + " }"


def shape(geometry: str, app: str) -> str:
    return f"Shape {{ appearance {app} geometry {geometry} castShadows FALSE }}"


def box(sx, sy, sz) -> str:
    return f"Box {{ size {f(sx, sy, sz)} }}"


def cylinder(height, radius, sub=16) -> str:
    return f"Cylinder {{ height {f(height)} radius {f(radius)} subdivision {sub} }}"


def sphere(radius, sub=2) -> str:
    return f"Sphere {{ radius {f(radius)} subdivision {sub} }}"


def pose(children: list[str], t=(0, 0, 0), r=(0, 0, 1, 0)) -> str:
    return f"Pose {{ translation {f(*t)} rotation {f(*r)} children [ " + " ".join(children) + " ] }"


def group(name: str, children: list[str]) -> str:
    return f"# {name}\nGroup {{ children [\n  " + "\n  ".join(children) + "\n] }"


# Rotations that lay a cylinder (axis +z) along x or y.
ALONG_X = (0, 1, 0, math.pi / 2)
ALONG_Y = (1, 0, 0, math.pi / 2)

STEEL = appearance((0.62, 0.64, 0.66), rough=0.35, metal=0.9)
DARK_STEEL = appearance((0.22, 0.23, 0.25), rough=0.45, metal=0.8)
PANEL = appearance((0.56, 0.59, 0.62), rough=0.45, metal=0.6)
SEAM = appearance((0.18, 0.19, 0.21), rough=0.6, metal=0.4)
BLACK = appearance((0.04, 0.04, 0.04), rough=0.4)
WHITE_PAINT = appearance((0.92, 0.92, 0.9), rough=0.8)
YELLOW_PAINT = appearance((0.95, 0.75, 0.08), rough=0.75)
RED = appearance((0.75, 0.05, 0.04), rough=0.4)


def wall_mount(side: str, along: float, z: float, depth: float) -> tuple[tuple[float, float, float], tuple]:
    """Center and rotation of an item mounted on a wall (local +y points into the room)."""
    half = WALL_T / 2
    if side == "north":
        return (along, -half - depth / 2, z), (0, 0, 1, math.pi)
    if side == "south":
        return (along, -MAP_H + half + depth / 2, z), (0, 0, 1, 0)
    if side == "west":
        return (half + depth / 2, along, z), (0, 0, 1, -math.pi / 2)
    if side == "east":
        return (MAP_W - half - depth / 2, along, z), (0, 0, 1, math.pi / 2)
    raise ValueError(side)


def on_wall(side: str, along: float, z: float, depth: float, children: list[str]) -> str:
    """Pose node of an item mounted on a wall."""
    t, r = wall_mount(side, along, z, depth)
    return pose(children, t, r)


def decal_strip(x0, y0, x1, y1, width, app) -> str:
    length = math.hypot(x1 - x0, y1 - y0)
    ang = math.atan2(y1 - y0, x1 - x0)
    return pose([shape(box(length, width, DECAL_H), app)], ((x0 + x1) / 2, (y0 + y1) / 2, DECAL_Z), (0, 0, 1, ang))


def dashed(x0, y0, x1, y1, width, app, dash=0.6, gap=0.4) -> list[str]:
    length = math.hypot(x1 - x0, y1 - y0)
    ux, uy = (x1 - x0) / length, (y1 - y0) / length
    out, s = [], 0.0
    while s < length:
        e = min(s + dash, length)
        out.append(decal_strip(x0 + ux * s, y0 + uy * s, x0 + ux * e, y0 + uy * e, width, app))
        s += dash + gap
    return out


def hazard_square(cx, cy, outer, band) -> list[str]:
    """Hazard-striped square frame on the floor."""
    h = outer / 2 - band / 2
    out = []
    for (x, y, ang) in ((cx, cy + h, 0), (cx, cy - h, 0), (cx - h, cy, math.pi / 2), (cx + h, cy, math.pi / 2)):
        app = appearance((1, 1, 1), rough=0.75, texture="hazard_stripes.png",
                         tex_scale=(outer / (band * 8), 1))
        out.append(pose([shape(box(outer, band, DECAL_H), app)], (x, y, DECAL_Z + 0.0002), (0, 0, 1, ang)))
    return out


def floor_markings(pads: list[tuple[str, int, int]]) -> str:
    """Lanes, pad safety squares and labels, keep-out zones."""
    items = []
    yellow = appearance((0.95, 0.76, 0.1), rough=0.7)
    white = appearance((0.88, 0.88, 0.86), rough=0.7)

    x_start, x_end = 2.45, MAP_W - 2.45
    for y_edges, y_center in (((-0.35, -2.85), -1.5), ((-6.07, -6.93), -6.5), ((-10.15, -12.65), -11.5)):
        for ye in y_edges:
            items.append(decal_strip(x_start, ye, x_end, ye, 0.07, yellow))
        items += dashed(x_start + 0.3, y_center, x_end - 0.3, y_center, 0.06, white)
    for x in (6.5, 9.5):
        items += dashed(x, -3.15, x, -9.85, 0.06, white)
    for x in (3.0, 14.0):
        items += dashed(x, -3.15, x, -9.85, 0.06, white)

    for kind, row, col in pads:
        cx, cy = col + 0.5, -(row + 0.5)
        items += hazard_square(cx, cy, 1.34, 0.1)
        label = "label_pickup.png" if kind == "pickup" else "label_dropoff.png"
        lx = cx + (1.25 if col == 0 else -1.25)
        app = appearance((1, 1, 1), rough=0.8, texture=label)
        items.append(pose([shape(box(0.95, 0.28, DECAL_H), app)], (lx, cy, DECAL_Z)))

    for r0, r1 in SHELF_ROWS:
        for c0, c1 in SHELF_COLS:
            w, h = (c1 - c0 + 1), (r1 - r0 + 1)
            cx, cy = c0 + w / 2, -(r0 + h / 2)
            app = appearance((1, 1, 1), rough=0.8, texture="keepout_hatch.png", tex_scale=(w * 1.2, h * 1.2))
            items.append(pose([shape(box(w - 0.24, h - 0.24, DECAL_H), app)], (cx, cy, DECAL_Z)))
            for (x0, y0, x1, y1) in (
                (c0 + 0.08, -(r0 + 0.08), c1 + 0.92, -(r0 + 0.08)),
                (c0 + 0.08, -(r1 + 0.92), c1 + 0.92, -(r1 + 0.92)),
                (c0 + 0.08, -(r0 + 0.08), c0 + 0.08, -(r1 + 0.92)),
                (c1 + 0.92, -(r0 + 0.08), c1 + 0.92, -(r1 + 0.92)),
            ):
                items.append(decal_strip(x0, y0, x1, y1, 0.08, yellow))
    return group("Floor markings: lanes, pad safety squares, keep-out zones (visual only)", items)


def cladding() -> str:
    """Metal wall panels above the real wall."""
    items = []
    h = CLAD_TOP_Z - WALL_TOP_Z
    zc = WALL_TOP_Z + h / 2
    for (x, y, sx, sy) in ((MAP_W / 2, 0, MAP_W + WALL_T, WALL_T),
                           (0, -MAP_H / 2, WALL_T, MAP_H + WALL_T),
                           (MAP_W, -MAP_H / 2, WALL_T, MAP_H + WALL_T)):
        items.append(pose([shape(box(sx, sy, h), PANEL)], (x, y, zc)))
    sh = SOUTH_CLAD_TOP_Z - WALL_TOP_Z
    items.append(pose([shape(box(MAP_W + WALL_T, WALL_T, sh), PANEL)],
                      (MAP_W / 2, -MAP_H, WALL_TOP_Z + sh / 2)))
    band_app = appearance((1, 1, 1), rough=0.7, texture="hazard_stripes.png", tex_scale=(10, 1))
    for side, length, mid in (("north", MAP_W - 0.2, MAP_W / 2), ("south", MAP_W - 0.2, MAP_W / 2),
                              ("west", MAP_H - 0.2, -MAP_H / 2), ("east", MAP_H - 0.2, -MAP_H / 2)):
        items.append(on_wall(side, mid, WALL_TOP_Z + 0.06, 0.012, [shape(box(length, 0.012, 0.1), band_app)]))
    for side, length, start, top in (("north", MAP_W, 0.0, CLAD_TOP_Z), ("west", MAP_H, -MAP_H, CLAD_TOP_Z),
                                     ("east", MAP_H, -MAP_H, CLAD_TOP_Z)):
        n = int(length // 2)
        for k in range(1, n + 1):
            a = start + 2.0 * k
            if (side == "north" and a >= MAP_W) or (side != "north" and a >= 0):
                continue
            items.append(on_wall(side, a, (WALL_TOP_Z + 0.12 + top) / 2, 0.014,
                                 [shape(box(0.035, 0.014, top - WALL_TOP_Z - 0.12), SEAM)]))
        mid = start + length / 2
        items.append(on_wall(side, mid, 1.55, 0.02, [shape(box(length - 0.25, 0.02, 0.05), SEAM)]))
    return group("Wall cladding above the real 0.5 m wall: modular metal panels (visual only)", items)


def pipes() -> str:
    """Pipes, conduits and brackets on the walls."""
    items = []
    bracket = DARK_STEEL
    yellow_pipe = appearance((0.93, 0.72, 0.1), rough=0.4, metal=0.3)
    gray_pipe = appearance((0.6, 0.62, 0.64), rough=0.3, metal=0.85)
    conduit = appearance((0.4, 0.42, 0.44), rough=0.5, metal=0.5)
    for z, radius, app, off in ((2.18, 0.05, gray_pipe, 0.09), (2.04, 0.035, yellow_pipe, 0.08),
                                (1.88, 0.018, conduit, 0.04)):
        t, r = wall_mount("north", MAP_W / 2, z, off * 2)
        items.append(pose([pose([shape(cylinder(MAP_W - 0.3, radius), app)], r=ALONG_X)], t, r))
    for k in range(1, 9):
        items.append(on_wall("north", 2.0 * k, 2.11, 0.18, [shape(box(0.05, 0.18, 0.3), bracket)]))
    for side in ("west", "east"):
        t, r = wall_mount(side, -MAP_H / 2, 2.15, 0.12)
        items.append(pose([pose([shape(cylinder(MAP_H - 0.3, 0.045), gray_pipe)], r=ALONG_X)], t, r))
        for k in range(1, 7):
            items.append(on_wall(side, -2.0 * k, 2.15, 0.15, [shape(box(0.05, 0.15, 0.16), bracket)]))
    for side, along, z_top, z_bot, radius, app, off in (
        ("north", 4.3, 2.18, 1.75, 0.022, conduit, 0.05),
        ("north", 4.7, 2.18, 1.75, 0.022, conduit, 0.05),
        ("north", 12.4, 2.04, 1.75, 0.022, conduit, 0.05),
        ("north", 7.2, 2.18, 0.62, 0.04, gray_pipe, 0.09),
        ("north", 15.3, 2.04, 0.62, 0.03, yellow_pipe, 0.08),
        ("west", -4.0, 2.15, 1.45, 0.02, conduit, 0.05),
        ("east", -9.0, 2.15, 0.62, 0.04, gray_pipe, 0.1),
    ):
        h = z_top - z_bot
        t, r = wall_mount(side, along, z_bot + h / 2, off * 2)
        items.append(pose([shape(cylinder(h, radius), app)], t, r))
    for side, along, off, radius, app in (("north", 7.2, 0.09, 0.055, gray_pipe), ("east", -9.0, 0.1, 0.055, gray_pipe)):
        t, r = wall_mount(side, along, 0.66, off * 2)
        items.append(pose([shape(cylinder(0.08, radius), DARK_STEEL)], t, r))
    return group("Pipes, conduits and brackets (visual only, all above the real wall)", items)


def electrical_cabinet(side: str, along: float, status: str) -> str:
    """Wall cabinet with label and one status light."""
    w, d, h, z = 0.8, 0.22, 0.95, 1.22
    body = appearance((0.72, 0.73, 0.69), rough=0.5, metal=0.4)
    light_color = (0.1, 1.0, 0.25) if status == "online" else (1.0, 0.08, 0.05)
    children = [
        shape(box(w, d, h), body),
        pose([shape(box(0.006, 0.004, h - 0.06), SEAM)], (0, d / 2 + 0.002, 0)),
        pose([shape(box(0.03, 0.03, 0.16), DARK_STEEL)], (0.09, d / 2 + 0.015, 0.0)),
        pose([shape(box(0.22, 0.004, 0.15), appearance((1, 1, 1), rough=0.6, texture="label_voltage.png"))],
             (-0.2, d / 2 + 0.003, 0.25)),
        pose([shape(box(w + 0.04, d + 0.02, 0.04), DARK_STEEL)], (0, 0, h / 2 + 0.02)),
        pose([shape(cylinder(0.02, 0.035), BLACK)], (0.28, d / 2 + 0.01, 0.36), ALONG_Y),
        pose([shape(sphere(0.025), appearance(light_color, rough=0.2, emissive=light_color, emissive_intensity=3))],
             (0.28, d / 2 + 0.022, 0.36)),
    ]
    return on_wall(side, along, z, d, children)


def emergency_stop(side: str, along: float) -> str:
    """EMERGENCY STOP panel with button, display and label."""
    w, d, h, z = 0.42, 0.13, 0.55, 1.18
    face = d / 2
    children = [
        shape(box(w, d, h), RED),
        pose([shape(box(0.26, 0.006, 0.1), appearance((1, 1, 1), rough=0.3, texture="display_ok.png",
                                                       emissive=(0.25, 0.9, 0.35), emissive_intensity=0.6))],
             (0, face + 0.003, 0.17)),
        pose([shape(cylinder(0.02, 0.11, 24), YELLOW_PAINT)], (0, face + 0.01, -0.02), ALONG_Y),
        pose([shape(cylinder(0.05, 0.035, 20), DARK_STEEL)], (0, face + 0.04, -0.02), ALONG_Y),
        pose([shape(cylinder(0.035, 0.085, 28), appearance((0.85, 0.02, 0.02), rough=0.25))],
             (0, face + 0.075, -0.02), ALONG_Y),
        pose([shape(box(0.4, 0.006, 0.1), appearance((1, 1, 1), rough=0.6, texture="label_estop.png"))],
             (0, face + 0.003, -0.205)),
    ]
    return on_wall(side, along, z, d, children)


def security_camera(side: str, along: float, z: float, yaw_deg: float) -> str:
    """Small wall camera pitched down, with a tiny red light."""
    box_d = 0.12
    cam = [
        shape(box(0.2, 0.08, 0.08), appearance((0.85, 0.86, 0.86), rough=0.4, metal=0.3)),
        pose([shape(box(0.24, 0.1, 0.012), DARK_STEEL)], (0.01, 0, 0.048)),
        pose([shape(cylinder(0.02, 0.026, 20), BLACK)], (0.105, 0, 0), ALONG_X),
        pose([shape(sphere(0.008), appearance((1, 0.05, 0.03), rough=0.2, emissive=(1, 0.04, 0.02),
                                               emissive_intensity=6))], (0.1, 0.03, 0.03)),
    ]
    head = pose([pose(cam, r=(0, 1, 0, math.radians(30)))], (0, box_d / 2 + 0.16, -0.05),
                (0, 0, 1, math.radians(90 + yaw_deg)))
    children = [
        shape(box(0.16, box_d, 0.18), DARK_STEEL),
        pose([shape(cylinder(0.14, 0.015), STEEL)], (0, box_d / 2 + 0.07, 0), ALONG_Y),
        pose([shape(sphere(0.025), STEEL)], (0, box_d / 2 + 0.14, -0.01)),
        head,
    ]
    return on_wall(side, along, z, box_d, children)


def wall_sign(side: str, along: float, z: float, text_png: str) -> str:
    return on_wall(side, along, z, 0.01, [shape(box(1.2, 0.01, 0.3), appearance((1, 1, 1), rough=0.6, texture=text_png))])


def lighting() -> tuple[str, list[str]]:
    """Hanging luminaires and their point lights."""
    items, lights = [], []
    housing = appearance((0.3, 0.31, 0.33), rough=0.5, metal=0.6)
    diffuser = appearance((0.95, 0.96, 1.0), rough=0.3, emissive=(1.0, 0.97, 0.9), emissive_intensity=2.5)
    cable_h = 1.6
    for y in (-4.5, -8.5):
        for x in (5.0, 8.0, 11.0):
            z = 2.9
            items.append(pose([
                shape(box(1.8, 0.2, 0.07), housing),
                pose([shape(box(1.72, 0.14, 0.012), diffuser)], (0, 0, -0.04)),
                pose([shape(cylinder(cable_h, 0.005), STEEL)], (-0.75, 0, 0.035 + cable_h / 2)),
                pose([shape(cylinder(cable_h, 0.005), STEEL)], (0.75, 0, 0.035 + cable_h / 2)),
            ], (x, y, z)))
            lights.append(
                f"PointLight {{ location {f(x, y, 2.75)} color 1 0.97 0.9 intensity 0.45 "
                f"attenuation 0 0 0.08 radius 9 castShadows FALSE }}"
            )
    return group("Industrial luminaires on thin cables (visual only)", items), lights


def main() -> None:
    os.makedirs(TEX_DIR, exist_ok=True)
    make_concrete(os.path.join(TEX_DIR, "concrete_floor.png"))
    make_hazard(os.path.join(TEX_DIR, "hazard_stripes.png"))
    make_hatch(os.path.join(TEX_DIR, "keepout_hatch.png"))
    make_floor_label(os.path.join(TEX_DIR, "label_pickup.png"), "PICK UP", (246, 166, 97))
    make_floor_label(os.path.join(TEX_DIR, "label_dropoff.png"), "DROP OFF", (60, 190, 175))
    make_estop_label(os.path.join(TEX_DIR, "label_estop.png"))
    make_voltage_label(os.path.join(TEX_DIR, "label_voltage.png"))
    make_display(os.path.join(TEX_DIR, "display_ok.png"))
    make_sign(os.path.join(TEX_DIR, "sign_pickup.png"), "PICK-UP ZONE")
    make_sign(os.path.join(TEX_DIR, "sign_dropoff.png"), "DROP-OFF ZONE")

    with open(BASE, encoding="utf-8") as fh:
        src = fh.read()

    pads = [(m.group(1), int(m.group(2)), int(m.group(3)))
            for m in re.finditer(r'name "(pickup|dropoff)_(\d+)_(\d+)"', src)]

    floor_old = re.search(r"Floor \{.*?\n\}\n", src, re.S).group(0)
    floor_new = (
        "Floor {\n  translation 8.5 -6.5 0\n  size 17 13\n  tileSize 4 4\n"
        "  appearance PBRAppearance {\n    baseColorMap ImageTexture {\n"
        '      url [\n        "textures/concrete_floor.png"\n      ]\n      filtering 4\n    }\n'
        "    roughness 0.85\n    metalness 0\n  }\n}\n"
    )
    src = src.replace(floor_old, floor_new, 1)

    vp_old = re.search(r"Viewpoint \{.*?\n\}\n", src, re.S).group(0)
    src = src.replace(vp_old, _viewpoint((8.5, -27.0, 21.0), (8.5, -5.0, 0.0)), 1)

    src = src.replace("TexturedBackgroundLight {\n}", "TexturedBackgroundLight {\n  luminosity 0.75\n}", 1)

    light_group, point_lights = lighting()
    decor = [
        floor_markings(pads),
        cladding(),
        pipes(),
        group("Electrical cabinets with status lights (green = system online, red = maintenance)", [
            electrical_cabinet("north", 4.5, "online"),
            electrical_cabinet("north", 12.4, "maintenance"),
        ]),
        group("EMERGENCY STOP panel (not wired to anything)", [emergency_stop("west", -4.0)]),
        group("Surveillance cameras on wall boxes, tiny red light", [
            security_camera("north", 0.6, 2.2, 35),
            security_camera("east", -12.4, 2.2, -35),
        ]),
        group("Zone signs", [
            wall_sign("west", -9.0, 1.9, "sign_pickup.png"),
            wall_sign("east", -4.0, 1.9, "sign_dropoff.png"),
        ]),
        light_group,
        *point_lights,
    ]
    marker = "Youbot {"
    idx = src.index(marker)
    src = (src[:idx]
           + "# ---- Industrial decoration (scripts/build_industrial_world.py) -- visual only, no physics,\n"
           + "# ---- nothing inside the robots' LiDAR plane. Everything functional is unchanged.\n"
           + "\n".join(decor) + "\n" + src[idx:])

    with open(OUT, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(src)
    shutil.copyfile(BASE + ".robots.json", OUT + ".robots.json")
    print(f"wrote {OUT} ({len(src) // 1024} KiB), robots json and {len(os.listdir(TEX_DIR))} textures")


def _viewpoint(position, target) -> str:
    """Viewpoint node looking from position at target."""
    fwd = np.subtract(target, position).astype(float)
    fwd /= np.linalg.norm(fwd)
    left = np.cross([0, 0, 1], fwd)
    left /= np.linalg.norm(left)
    up = np.cross(fwd, left)
    m = np.column_stack([fwd, left, up])
    tr = np.trace(m)
    if tr > 0:
        s4 = math.sqrt(tr + 1.0) * 2
        w, x, y, z = 0.25 * s4, (m[2, 1] - m[1, 2]) / s4, (m[0, 2] - m[2, 0]) / s4, (m[1, 0] - m[0, 1]) / s4
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s4 = math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2
        w, x, y, z = (m[2, 1] - m[1, 2]) / s4, 0.25 * s4, (m[0, 1] + m[1, 0]) / s4, (m[0, 2] + m[2, 0]) / s4
    elif m[1, 1] > m[2, 2]:
        s4 = math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2
        w, x, y, z = (m[0, 2] - m[2, 0]) / s4, (m[0, 1] + m[1, 0]) / s4, 0.25 * s4, (m[1, 2] + m[2, 1]) / s4
    else:
        s4 = math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2
        w, x, y, z = (m[1, 0] - m[0, 1]) / s4, (m[0, 2] + m[2, 0]) / s4, (m[1, 2] + m[2, 1]) / s4, 0.25 * s4
    angle = 2 * math.acos(max(-1.0, min(1.0, w)))
    n = math.sqrt(x * x + y * y + z * z) or 1.0
    return (f"Viewpoint {{\n  orientation {f(x / n, y / n, z / n, angle)}\n  position {f(*position)}\n"
            f"  near 0.1\n}}\n")


if __name__ == "__main__":
    main()
