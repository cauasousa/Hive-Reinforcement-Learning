# LIDAR ray-index ↔ body-relative-angle calibration

`youbot_mecanum.py`'s LIDAR safety brake (`_lidar_min_range`) deliberately treats the whole
`lidar.getRangeImage()` array as one undirected pool — the comment on `_lidar_min_range` explains
why: nobody had verified live which ray index actually points in which direction, so trusting a
guessed mapping risked silently picking the wrong ray for a directional decision (e.g. "is my
right clear?").

This was verified live on 2026-08-31 using a purpose-built calibration controller
(`webots/controllers/lidar_calibration/lidar_calibration.py`). This doc records the method and the
result so it doesn't have to be re-derived from scratch later, and so it can be re-run and diffed
against if the LIDAR config, Webots version, or Youbot PROTO ever changes.

## Method

The calibration controller (assigned as a robot's `controller` field, replacing its normal one for
the duration of the test — it needs `supervisor TRUE` on the robot node):

1. Reads the robot's own live pose the same way `youbot_mecanum.py` does (`theta =
   atan2(orientation[3], orientation[0])` from `self_node.getOrientation()`).
2. Re-runs `_lidar_calibrate_self_mask` (imported, not reimplemented) to know which ray indices are
   permanent self-hits.
3. Spawns a small probe `Solid` via the Supervisor API (`importMFNodeFromString` on the world
   root's `children` field) — no manual world editing needed.
4. Sweeps the probe through a full circle around the robot (15° steps, 0.6m radius, in the robot's
   own body frame: 0°=front, +90°=left, 180°=behind, 270°/-90°=right — the same convention `theta`
   already uses everywhere else in this codebase), settling a few sim steps at each stop and
   recording which non-self-masked ray reports the shortest range.
5. Removes the probe and writes a full report to `outputs/lidar_calibration_<robot_name>.txt`.

To re-run: point one robot's `controller` field at `"lidar_calibration"` in a world where it has
open floor around it (`keyboard_test.wbt`'s `robot_0` already does, and already has `supervisor
TRUE` and a `Lidar` device added for this purpose), open the world directly in Webots (**not**
via `scripts/run_extern_controllers.sh` — that launcher is only for `<extern>`-controller robots
talking to the training pipeline over TCP; this is a normal embedded controller), and press play.

## Result (this project's LIDAR config: `horizontalResolution=180`, `fieldOfView=6.283185` (360°),
`numberOfLayers=1`)

Measured index-per-degree: `-0.4879`, matching the theoretical magnitude
(`horizontalResolution / fieldOfView_deg = 180/360 = 0.5`) with the opposite sign — i.e.
**increasing ray index sweeps clockwise**, not counterclockwise.

Fit against the 23 sweep points (residuals ±1-3 rays, consistent with the 15° step not being a
whole number of ray-widths and the probe's own footprint — not model error):

```
ray_index(angle_deg)  ≈ round((180 - angle_deg) / step_deg) mod horizontalResolution
angle_deg(ray_index)  ≈ (180 - step_deg * ray_index) mod 360
```
where `step_deg = fieldOfView_deg / horizontalResolution` (2° for this config), and `angle_deg` is
0=front / +90=left / 180=behind / 270 (or -90)=right, the same convention `theta` uses.

Implemented as `_ray_index_for_local_angle_deg` / `_local_angle_deg_for_ray_index` in
`youbot_mecanum.py`.

**Only confirmed for a full 360° FOV lidar** (every `Lidar` node in this project's worlds uses
this). Re-run this calibration if `fieldOfView` is ever changed to less than 360° — a partial scan
isn't guaranteed to share the same ray-ordering convention.

### The important finding: the robot's front is the arm side, not the "opposite of the arm" side

The self-mask's permanent self-hit block (indices 82-99 in this run) lands almost exactly on ray
index ~90 — which the formula above maps to **0° (front)**. In other words: looking straight ahead
(the same `+x`/`theta=0` direction the rest of this codebase already calls "front" — e.g.
`PICKUP_FACE_THETA_RAD`), the LIDAR mostly just sees the robot's own arm.

This matches the real youBot's physical layout (arm mounted at the front of the chassis, reaching
forward to grab whatever the robot just drove up to) and means: right after a pickup, when the
robot is still facing the shelf it just approached (`theta≈π` at that point, by construction — see
`youbot_mecanum.py`'s pickup routine), it has **no usable LIDAR coverage in the direction it needs
to leave** — not because nothing is scanning that way, but because that's exactly where its own arm
already blocks the view. This is the concrete mechanism behind the "robot finishes a pickup facing
a wall and can't see what's ahead of it" behavior discussed separately.

### Validated direction → ray-index sectors (this LIDAR config)

| direction | angle | ray index |
|---|---|---|
| behind | 180° | ~0 (or 180, same ray) |
| left | 90° | ~44-45 |
| right | 270° / -90° | ~134-135 |
| front | 0° | ~90 — **inside the self-hit block, no reliable coverage** |

## Raw data (2026-08-31 run, `outputs/lidar_calibration_robot_0.txt`)

```
angle=   0deg  NO RETURN (self-masked)
angle=  15deg  ray_index= 81  range=0.5269m
angle=  30deg  ray_index= 76  range=0.5069m
angle=  45deg  ray_index= 67  range=0.4935m
angle=  60deg  ray_index= 58  range=0.5023m
angle=  75deg  ray_index= 50  range=0.5140m
angle=  90deg  ray_index= 44  range=0.5251m
angle= 105deg  ray_index= 39  range=0.5140m
angle= 120deg  ray_index= 30  range=0.5083m
angle= 135deg  ray_index= 23  range=0.4965m
angle= 150deg  ray_index= 13  range=0.5036m
angle= 165deg  ray_index=  5  range=0.5147m
angle= 180deg  ray_index=  0  range=0.5258m
angle= 195deg  ray_index=174  range=0.5147m
angle= 210deg  ray_index=166  range=0.5066m
angle= 225deg  ray_index=158  range=0.4955m
angle= 240deg  ray_index=148  range=0.5045m
angle= 255deg  ray_index=140  range=0.5140m
angle= 270deg  ray_index=134  range=0.5251m
angle= 285deg  ray_index=129  range=0.5140m
angle= 300deg  ray_index=121  range=0.5047m
angle= 315deg  ray_index=113  range=0.4946m
angle= 330deg  ray_index=103  range=0.5032m
angle= 345deg  ray_index=100  range=0.5397m
```

self-mask (permanent self-hit ray indices at calibration time): `[82..99]` (18/180 rays).

## Also confirmed separately: the ~0.067m self-hit signature

While debugging this, a raw `lidar.getRangeImage()` print showed a persistent `~0.0673m` reading
at a fixed index across every frame, unrelated to the robot's surroundings. This matches
`_lidar_calibrate_self_mask`'s own docstring, which documents this exact value as the known
uncalibrated self-hit signature ("an uncalibrated LiDAR mounted on the chassis self-detected at
~0.067m on every robot") — confirming the self-mask calibration is working as designed for that
ray, not a new/unknown issue.
