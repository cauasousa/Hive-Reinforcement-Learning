"""Generate a Webots ``.wbt`` world from a HiveRL ``WarehouseMap``.

Reuses the exact same procedural generator (``warehouse_grid.generate_warehouse_map``) that
produces the training environment's grid, so the physical world and the grid the planner/shield/
policy reason about are always in sync by construction -- there is no second, hand-maintained
map definition to drift out of step with the Python one.

Scale (see docs/webots_ros2_integration.md section 1): 1 grid cell = 1 meter. Two robot bodies are
available (``robot_model``): ``"youbot"`` (default) instantiates the real KUKA youBot PROTO --
a genuine omnidirectional 4-mecanum-wheel research platform -- and ``"agv"`` falls back to a
self-authored ~0.4m x 0.34m warehouse-AGV chassis with two differential-drive wheels (see the
geometry constants below). Neither is the earlier e-puck (a real 7cm educational robot, too small
to read clearly against a 1m grid cell; see docs/webots_ros2_integration.md section 8 for that
full story, including why "youbot" was chosen next despite carrying materially more risk than
"agv" did -- an external PROTO with unconfirmed device names again, plus far more complex 4-wheel
mecanum kinematics instead of simple differential drive).

Webots version note: starting with recent Webots releases (confirmed against a real install --
see docs/webots_ros2_integration.md section 6/troubleshooting), ``TexturedBackground``,
``TexturedBackgroundLight``, and ``Floor`` are no longer implicit/base nodes -- they are PROTOs
that must be declared with an explicit ``EXTERNPROTO`` pointing at a URL matching the installed
Webots version, or the world fails to load with "Skipped unknown node" errors. This generator
always emits explicit EXTERNPROTO declarations (versioned GitHub raw URLs, which Webots
fetches/caches itself) for those three rather than relying on any node being built in, so it works
across Webots versions as long as ``WEBOTS_VERSION`` below matches (or is close to) what's
installed -- pass a different ``webots_version`` to ``generate_world``/``write_world_file`` if
your Webots reports a different version (shown in Webots' "Help > About" or in any version-
mismatch error message it prints). The robot body itself is authored directly as plain ``Robot``/
``HingeJoint``/``Solid`` nodes rather than an external PROTO, so it has no EXTERNPROTO dependency
and no risk of an unverified third-party device name -- every device name here is also the one
``webots/controllers/diff_drive_agv/diff_drive_agv.py`` expects, by construction.
"""
from __future__ import annotations

from hiverl.bridge.coords import CELL_SIZE_M, cell_to_world
from hiverl.env.warehouse_grid import DROPOFF, PICKUP, SHELF, WarehouseMap

WEBOTS_VERSION = "R2025a"

_GITHUB_PROTO_BASE = "https://raw.githubusercontent.com/cyberbotics/webots/{version}/projects"
_YOUBOT_PROTO_PATH = "robots/kuka/youbot/protos/Youbot.proto"

YOUBOT_WHEEL_RADIUS_M = 0.05
YOUBOT_HALF_LENGTH_M = 0.228
YOUBOT_HALF_WIDTH_M = 0.158
YOUBOT_ROOT_Z_M = 0.12

WHEEL_RADIUS_M = 0.09
WHEEL_THICKNESS_M = 0.03
HALF_AXLE_M = 0.21
CASTER_RADIUS_M = 0.03
CASTER_OFFSET_X_M = 0.15
CHASSIS_SIZE_M = (0.40, 0.34, 0.16)
PLATFORM_SIZE_M = (0.28, 0.28, 0.03)
ROBOT_ROOT_Z_M = WHEEL_RADIUS_M


def _externproto_block(version: str, robot_model: str) -> str:
    base = _GITHUB_PROTO_BASE.format(version=version)
    protos = [
        f"{base}/objects/backgrounds/protos/TexturedBackground.proto",
        f"{base}/objects/backgrounds/protos/TexturedBackgroundLight.proto",
        f"{base}/objects/floors/protos/Floor.proto",
    ]
    if robot_model == "youbot":
        protos.append(f"{base}/{_YOUBOT_PROTO_PATH}")
    return "".join(f'EXTERNPROTO "{url}"\n' for url in protos)


def _contact_properties_block() -> str:
    """Mecanum-wheel friction tuning for the youBot base -- without this, the wheels' rollers
    (which are only supposed to slip *sideways*, not along the roll direction) behave like plain
    frictionless casters against Webots' default contact model and the robot slides uncontrollably
    instead of driving. Values as commonly used in Webots' own youBot example worlds."""
    return (
        "  contactProperties [\n"
        "    ContactProperties {\n"
        '      material1 "InteriorWheelMat"\n'
        "      coulombFriction [ 0, 2, 0 ]\n"
        "      frictionRotation -0.785398 0\n"
        "      bounce 0\n"
        "      forceDependentSlip [ 10, 0 ]\n"
        "    }\n"
        "    ContactProperties {\n"
        '      material1 "ExteriorWheelMat"\n'
        "      coulombFriction [ 0, 2, 0 ]\n"
        "      frictionRotation 0.785398 0\n"
        "      bounce 0\n"
        "      forceDependentSlip [ 10, 0 ]\n"
        "    }\n"
        "  ]\n"
    )


def _world_info_block(robot_model: str) -> str:
    contact_properties = _contact_properties_block() if robot_model == "youbot" else ""
    return (
        "WorldInfo {\n"
        '  basicTimeStep 32\n'
        '  coordinateSystem "ENU"\n'
        f"{contact_properties}"
        "}\n"
        "Viewpoint {\n"
        "  orientation -0.5773502691896258 0.5773502691896258 0.5773502691896258 2.0943951023931953\n"
        "  position %.2f %.2f %.2f\n"
        "}\n"
        "TexturedBackground {\n}\n"
        "TexturedBackgroundLight {\n}\n"
    )


def _floor_block(width_cells: int, height_cells: int) -> str:
    w = width_cells * CELL_SIZE_M
    h = height_cells * CELL_SIZE_M
    cx, cy = w / 2, -h / 2
    return (
        "Floor {\n"
        f"  translation {cx:.2f} {cy:.2f} 0\n"
        f"  size {w:.2f} {h:.2f}\n"
        '  appearance PBRAppearance {\n'
        '    baseColor 0.85 0.85 0.85\n'
        '    roughness 0.9\n'
        "  }\n"
        "}\n"
    )


WALL_HEIGHT_M = 0.5
WALL_THICKNESS_M = 0.2


def _boundary_wall_block(name: str, cx: float, cy: float, size_x: float, size_y: float) -> str:
    cz = WALL_HEIGHT_M / 2
    return (
        "Solid {\n"
        f"  translation {cx:.2f} {cy:.2f} {cz:.2f}\n"
        "  children [\n"
        "    Shape {\n"
        "      appearance PBRAppearance { baseColor 0.4 0.4 0.4 }\n"
        f"      geometry Box {{ size {size_x:.2f} {size_y:.2f} {WALL_HEIGHT_M:.2f} }}\n"
        "    }\n"
        "  ]\n"
        f"  boundingObject Box {{ size {size_x:.2f} {size_y:.2f} {WALL_HEIGHT_M:.2f} }}\n"
        f'  name "{name}"\n'
        "}\n"
    )


def _boundary_walls_block(width_cells: int, height_cells: int) -> str:
    """4 Solid boxes fully enclosing the map, centered directly ON the map's boundary lines (each
    wall straddles the edge -- half inside the outermost cell row/col, half outside) so there is
    no open gap between the walkable area and any wall, and the four walls meet exactly (no gap,
    minimal overlap) at each corner.

    User-reported bug fixed here (2026-08-29): the previous version centered each wall
    WALL_OFFSET_M=0.5m *outward* from the map edge with only WALL_THICKNESS_M=0.2m of its own
    thickness -- leaving a uniform 0.5 - 0.2/2 = 0.4m gap of completely open space between the
    map's real boundary and the near face of every boundary wall, all the way around the
    perimeter. A robot could drive straight through that gap and end up outside the intended
    play area entirely. The old WALL_OFFSET_M=0.5m constant that produced that gap is removed
    entirely; the left/right walls now extend by WALL_THICKNESS_M/2 past each end -- just enough
    to meet the top/bottom walls' outer faces exactly, since all four walls now share the same
    centerline-on-the-edge convention.
    """
    w = width_cells * CELL_SIZE_M
    h = height_cells * CELL_SIZE_M
    cx_mid, cy_mid = w / 2, -h / 2
    return "".join(
        [
            _boundary_wall_block("wall_top", cx_mid, -h, w, WALL_THICKNESS_M),
            _boundary_wall_block("wall_bottom", cx_mid, 0.0, w, WALL_THICKNESS_M),
            _boundary_wall_block(
                "wall_left", 0.0, cy_mid, WALL_THICKNESS_M, h + WALL_THICKNESS_M
            ),
            _boundary_wall_block(
                "wall_right", w, cy_mid, WALL_THICKNESS_M, h + WALL_THICKNESS_M
            ),
        ]
    )


def _shelf_block(cell: tuple[int, int], name: str) -> str:
    x, y = cell_to_world(cell)
    size = CELL_SIZE_M * 0.9
    height = 0.5
    return (
        "Solid {\n"
        f"  translation {x:.2f} {y:.2f} {height / 2:.2f}\n"
        f'  name "{name}"\n'
        "  children [\n"
        "    Shape {\n"
        "      appearance PBRAppearance {\n"
        "        baseColor 0.36 0.25 0.16\n"
        "        roughness 0.8\n"
        "      }\n"
        f"      geometry Box {{ size {size:.2f} {size:.2f} {height:.2f} }}\n"
        "    }\n"
        "  ]\n"
        f"  boundingObject Box {{ size {size:.2f} {size:.2f} {height:.2f} }}\n"
        "}\n"
    )


def _station_pad(cell: tuple[int, int], name: str, color: tuple[float, float, float]) -> str:
    x, y = cell_to_world(cell)
    r, g, b = color
    size = CELL_SIZE_M * 0.95
    return (
        "Solid {\n"
        f"  translation {x:.2f} {y:.2f} 0.002\n"
        f'  name "{name}"\n'
        "  children [\n"
        "    Shape {\n"
        f"      appearance PBRAppearance {{ baseColor {r:.2f} {g:.2f} {b:.2f} roughness 1 metalness 0 }}\n"
        f"      geometry Box {{ size {size:.2f} {size:.2f} 0.002 }}\n"
        "    }\n"
        "  ]\n"
        "}\n"
    )


PACKAGE_BOX_SIZE_M = 0.06
PACKAGE_BOX_REST_Z_M = PACKAGE_BOX_SIZE_M / 2 + 0.002
PACKAGE_BOX_MASS_KG = 0.05


def _package_def_name(cell: tuple[int, int]) -> str:
    row, col = cell
    return f"PACKAGE_{row}_{col}"


def _package_box(cell: tuple[int, int]) -> str:
    """A real, physically-present package box sitting on a pickup pad -- DEF-named so the
    controller (as a ``Supervisor``) can look it up with ``getFromDef``. Has a real
    ``boundingObject``/``physics`` body (a real Webots dynamic Solid, not a purely visual one) so
    it actually rests on a surface under gravity/friction -- the controller (youbot_mecanum.py)
    only ever teleports it once, onto the dropoff pad on delivery (with a small drop gap so it
    visibly settles there rather than snapping to an exact position); the pickup side is left
    entirely to the gripper closing on it for real, no scripted repositioning."""
    x, y = cell_to_world(cell)
    s = PACKAGE_BOX_SIZE_M
    return (
        f"DEF {_package_def_name(cell)} Solid {{\n"
        f"  translation {x:.2f} {y:.2f} {PACKAGE_BOX_REST_Z_M:.3f}\n"
        f'  name "{_package_def_name(cell).lower()}"\n'
        "  recognitionColors [ 0.82 0.55 0.28 ]\n"
        "  children [\n"
        "    Shape {\n"
        "      appearance PBRAppearance { baseColor 0.82 0.55 0.28 roughness 0.85 }\n"
        f"      geometry Box {{ size {s:.2f} {s:.2f} {s:.2f} }}\n"
        "    }\n"
        "  ]\n"
        f"  boundingObject Box {{ size {s:.2f} {s:.2f} {s:.2f} }}\n"
        f"  physics Physics {{ density -1 mass {PACKAGE_BOX_MASS_KG} }}\n"
        "}\n"
    )


STATUS_MAST_HEIGHT_M = 0.35
STATUS_BEACON_RADIUS_M = 0.07

LIDAR_HORIZONTAL_RESOLUTION = 180
LIDAR_FIELD_OF_VIEW_RAD = 6.283185307179586
LIDAR_MAX_RANGE_M = 1.5
LIDAR_MOUNT_Z_M = 0.15

CAMERA_WIDTH_PX = 128
CAMERA_HEIGHT_PX = 96
CAMERA_FIELD_OF_VIEW_RAD = 2.0
CAMERA_MAX_RANGE_M = 2.0
CAMERA_NEAR_M = 0.01


def _camera_def_name(robot_id: int) -> str:
    return f"PICKUP_CAMERA_{robot_id}"


def _safety_lidar_block() -> str:
    return (
        "Lidar {\n"
        f"  translation 0 0 {LIDAR_MOUNT_Z_M:.2f}\n"
        '  name "lidar"\n'
        f"  horizontalResolution {LIDAR_HORIZONTAL_RESOLUTION}\n"
        f"  fieldOfView {LIDAR_FIELD_OF_VIEW_RAD:.6f}\n"
        "  numberOfLayers 1\n"
        "  minRange 0.05\n"
        f"  maxRange {LIDAR_MAX_RANGE_M:.2f}\n"
        "}\n"
    )


def _pickup_camera_block(robot_id: int) -> str:
    return (
        f"DEF {_camera_def_name(robot_id)} Camera {{\n"
        "  translation 0.156 0 0.3\n"
        '  name "pickup_camera"\n'
        f"  fieldOfView {CAMERA_FIELD_OF_VIEW_RAD:.2f}\n"
        f"  width {CAMERA_WIDTH_PX}\n"
        f"  height {CAMERA_HEIGHT_PX}\n"
        f"  near {CAMERA_NEAR_M:.2f}\n"
        f"  recognition Recognition {{\n"
        f"    maxRange {CAMERA_MAX_RANGE_M:.2f}\n"
        "  }\n"
        "}\n"
    )


def _status_beacon_block(x: float, y: float, base_z: float) -> str:
    """A thin pole topped with an enlarged, brightly-lit sphere for the ``led_to_item`` "en route
    to pickup" status -- much more visible from the default wide isometric camera than a small
    flat indicator sitting low on the chassis, which is easy to miss entirely in a screenshot at
    warehouse scale. The pole is a plain static ``Solid`` (no separate LED); the beacon itself is
    the ``LED`` device the controller already looks up by this same name, just relocated and
    enlarged -- no controller-side change needed."""
    pole_center_z = base_z + STATUS_MAST_HEIGHT_M / 2
    beacon_z = base_z + STATUS_MAST_HEIGHT_M
    return (
        "Solid {\n"
        f"  translation {x:.2f} {y:.2f} {pole_center_z:.3f}\n"
        '  name "status_mast"\n'
        "  children [\n"
        "    Shape {\n"
        "      appearance PBRAppearance { baseColor 0.1 0.1 0.1 roughness 0.6 }\n"
        f"      geometry Cylinder {{ radius 0.015 height {STATUS_MAST_HEIGHT_M:.2f} subdivision 12 }}\n"
        "    }\n"
        "  ]\n"
        "}\n"
        "LED {\n"
        f"  translation {x:.2f} {y:.2f} {beacon_z:.3f}\n"
        '  name "led_to_item"\n'
        "  color [\n"
        "    0.15 0.55 0.95\n"
        "  ]\n"
        "  children [\n"
        "    Shape {\n"
        "      appearance PBRAppearance { baseColor 0.12 0.12 0.12 roughness 0.5 }\n"
        f"      geometry Sphere {{ radius {STATUS_BEACON_RADIUS_M:.2f} subdivision 2 }}\n"
        "    }\n"
        "  ]\n"
        "}\n"
    )


def _wheel_hinge(name: str, motor_name: str, sensor_name: str, y_offset: float) -> str:
    r = WHEEL_RADIUS_M
    return (
        "HingeJoint {\n"
        "  jointParameters HingeJointParameters {\n"
        "    axis 0 1 0\n"
        f"    anchor 0 {y_offset:.2f} 0\n"
        "  }\n"
        "  device [\n"
        f'    RotationalMotor {{ name "{motor_name}" maxVelocity 10 }}\n'
        f'    PositionSensor {{ name "{sensor_name}" }}\n'
        "  ]\n"
        "  endPoint Solid {\n"
        f"    translation 0 {y_offset:.2f} 0\n"
        "    rotation 0 1 0 0\n"
        "    children [\n"
        "      Shape {\n"
        "        appearance PBRAppearance { baseColor 0.08 0.08 0.08 roughness 0.7 }\n"
        f"        geometry Cylinder {{ radius {r:.2f} height {WHEEL_THICKNESS_M:.2f} subdivision 24 }}\n"
        "      }\n"
        "    ]\n"
        f'    name "{name}"\n'
        f"    boundingObject Cylinder {{ radius {r:.2f} height {WHEEL_THICKNESS_M:.2f} }}\n"
        "    physics Physics { mass 0.15 }\n"
        "  }\n"
        "}\n"
    )


def _youbot_robot_node(robot_id: int, cell: tuple[int, int], controller: str) -> str:
    """A real KUKA youBot PROTO instance -- omnidirectional 4-mecanum-wheel base (plus its stock
    arm(s), left as-is/at rest; this project only drives the base). See
    docs/webots_ros2_integration.md section 8: wheel motor/sensor device names and the inverse/
    forward kinematics wheel-index-to-position mapping were confirmed against the real fetched
    PROTO source (not guessed) after a live Webots test showed the first, generically-assumed
    mapping drove robots off the map.

    Uses the PROTO's own ``bodySlot`` extension field (confirmed real, from the same PROTO fetch)
    to add a status beacon (``led_to_item``, tall mast + enlarged sphere so "heading to pickup" is
    visible from the default wide top-down camera -- see ``_status_beacon_block``). ``supervisor
    TRUE`` lets the controller (``youbot_mecanum.py``) look up a pickup station's real ``DEF
    PACKAGE_<row>_<col>`` box (see ``_package_box``) with ``getFromDef``. On arrival it runs a
    staged animation -- arm reaches down (gripper open) -> gripper closes -> arm lifts -- purely
    for the arm's own motion; the box itself (a real dynamic body, see ``_package_box``) is left
    untouched by this and governed entirely by its own physics, until it's teleported onto the
    dropoff pad on delivery. An earlier version added a ``carry_platform`` marker the box was
    teleported onto and tracked against; removed per direct user feedback that they don't use it
    and suspected it was affecting the box in unwanted ways. A separate, even earlier version had
    a large flat tray bolted to the chassis, also removed, per feedback that it looked like an
    ugly permanent attachment the real youBot doesn't have.
    """
    x, y = cell_to_world(cell)
    row, col = cell
    return (
        "Youbot {\n"
        f"  translation {x:.2f} {y:.2f} {YOUBOT_ROOT_Z_M:.2f}\n"
        f'  name "robot_{robot_id}"\n'
        f'  controller "{controller}"\n'
        f'  controllerArgs ["--robot-id" "{robot_id}" "--start-row" "{row}" "--start-col" "{col}"]\n'
        "  supervisor TRUE\n"
        "  bodySlot [\n"
        f"    {_status_beacon_block(-0.15, -0.14, 0.18).strip()}\n"
        f"    {_safety_lidar_block().strip()}\n"
        f"    {_pickup_camera_block(robot_id).strip()}\n"
        "  ]\n"
        "}\n"
    )


def _agv_robot_node(robot_id: int, cell: tuple[int, int], controller: str) -> str:
    """The earlier self-authored differential-drive AGV body -- no external PROTO, lower risk,
    kept available via ``robot_model="agv"`` (see module docstring).

    ``led_carrying`` is sized/positioned like an actual package box sitting on the tray (dark/
    shadow-colored idle, cardboard-brown when lit) rather than a small indicator light, and
    ``led_to_item`` is a tall mast + enlarged beacon rather than a small flat light -- same visual
    pickup/delivery representation and visibility reasoning as the youBot body, see
    ``_youbot_robot_node``'s docstring."""
    x, y = cell_to_world(cell)
    row, col = cell
    cl, cw, ch = CHASSIS_SIZE_M
    pl, pw, pt = PLATFORM_SIZE_M
    platform_rel_z = ch / 2 + pt / 2 + 0.005
    package_box_rel_z = platform_rel_z + pt / 2 + 0.04
    led_rel_z = platform_rel_z + pt / 2 + 0.015
    caster_rel_z = CASTER_RADIUS_M - ROBOT_ROOT_Z_M

    left_wheel = _wheel_hinge("left wheel", "left wheel motor", "left wheel sensor", HALF_AXLE_M)
    right_wheel = _wheel_hinge("right wheel", "right wheel motor", "right wheel sensor", -HALF_AXLE_M)

    return (
        "Robot {\n"
        f"  translation {x:.2f} {y:.2f} {ROBOT_ROOT_Z_M:.2f}\n"
        f'  name "robot_{robot_id}"\n'
        f'  controller "{controller}"\n'
        f'  controllerArgs ["--robot-id" "{robot_id}" "--start-row" "{row}" "--start-col" "{col}"]\n'
        "  children [\n"
        "    Shape {\n"
        "      appearance PBRAppearance { baseColor 0.75 0.75 0.78 roughness 0.5 metalness 0.3 }\n"
        f"      geometry Box {{ size {cl:.2f} {cw:.2f} {ch:.2f} }}\n"
        "    }\n"
        "    Solid {\n"
        f"      translation 0 0 {platform_rel_z:.3f}\n"
        '      name "package_platform"\n'
        "      children [\n"
        "        Shape {\n"
        "          appearance PBRAppearance { baseColor 0.55 0.42 0.28 roughness 0.9 }\n"
        f"          geometry Box {{ size {pl:.2f} {pw:.2f} {pt:.2f} }}\n"
        "        }\n"
        "      ]\n"
        "    }\n"
        f"    {_status_beacon_block(0, -0.14, led_rel_z).strip()}\n"
        "    LED {\n"
        f"      translation 0 0 {package_box_rel_z:.3f}\n"
        '      name "led_carrying"\n'
        "      color [\n"
        "        0.82 0.55 0.28\n"
        "      ]\n"
        "      children [\n"
        "        Shape {\n"
        "          appearance PBRAppearance { baseColor 0.12 0.1 0.08 roughness 0.85 }\n"
        "          geometry Box { size 0.14 0.14 0.08 }\n"
        "        }\n"
        "      ]\n"
        "    }\n"
        f"    {left_wheel.strip()}\n"
        f"    {right_wheel.strip()}\n"
        "    Solid {\n"
        f"      translation {CASTER_OFFSET_X_M:.2f} 0 {caster_rel_z:.3f}\n"
        '      name "caster"\n'
        "      children [\n"
        "        Shape {\n"
        "          appearance PBRAppearance { baseColor 0.1 0.1 0.1 roughness 0.6 }\n"
        f"          geometry Sphere {{ radius {CASTER_RADIUS_M:.2f} subdivision 2 }}\n"
        "        }\n"
        "      ]\n"
        f"      boundingObject Sphere {{ radius {CASTER_RADIUS_M:.2f} }}\n"
        "    }\n"
        "  ]\n"
        f"  boundingObject Box {{ size {cl:.2f} {cw:.2f} {ch:.2f} }}\n"
        "  physics Physics { mass 1.2 }\n"
        "}\n"
    )


def generate_world(
    warehouse_map: WarehouseMap,
    robot_start_cells: list[tuple[int, int]],
    controller: str | None = None,
    webots_version: str = WEBOTS_VERSION,
    robot_model: str = "youbot",
    render_shelves: bool = False,
) -> str:
    """Build the full ``.wbt`` file content as a string.

    ``robot_start_cells`` must have one entry per robot, in ``robot_id`` order (0..N-1) -- these
    should be the same start cells ``WarehouseEnv.reset()`` picked, so the Webots world and the
    Python model agree on the initial state. ``webots_version`` controls both the file header
    and the EXTERNPROTO URLs -- pass your installed Webots version if it differs from the
    default (see module docstring). ``robot_model`` is ``"youbot"`` (real KUKA youBot PROTO,
    default) or ``"agv"`` (self-authored differential-drive body, lower risk/complexity, see
    module docstring). ``controller`` defaults to the matching controller for ``robot_model``
    (``"youbot_mecanum"`` or ``"diff_drive_agv"``) if left as ``None``.

    ``render_shelves`` (default ``False``, per an explicit user decision -- see
    docs/webots_ros2_integration.md section 8f) controls only whether shelf ``Solid`` boxes are
    drawn in the Webots *visual* world. The underlying grid (``WarehouseMap.grid`` still marks
    those cells ``SHELF``) is completely unaffected either way -- A*, the Safety Shield, the
    trained policy, and every training/evaluation result all keep routing around them exactly as
    before, since none of that logic ever reads the Webots world file. This flag only changes what
    gets physically drawn for a human watching the simulation; it cannot let a robot be routed
    through a shelf cell, because the coordinator never asks it to.
    """
    if robot_model not in ("youbot", "agv"):
        raise ValueError(f"robot_model must be 'youbot' or 'agv', got {robot_model!r}")
    if controller is None:
        controller = "youbot_mecanum" if robot_model == "youbot" else "diff_drive_agv"

    header = f"#VRML_SIM {webots_version} utf8\n"
    parts = [header, _externproto_block(webots_version, robot_model)]
    cx = warehouse_map.width * CELL_SIZE_M / 2
    cy = -warehouse_map.height * CELL_SIZE_M / 2
    extent = max(warehouse_map.width, warehouse_map.height) * CELL_SIZE_M
    cam_x = cx + extent * 0.9
    cam_y = cy - extent * 0.9
    cam_z = extent * 1.1
    parts.append(_world_info_block(robot_model) % (cam_x, cam_y, cam_z))
    parts.append(_floor_block(warehouse_map.width, warehouse_map.height))
    parts.append(_boundary_walls_block(warehouse_map.width, warehouse_map.height))

    for row in range(warehouse_map.height):
        for col in range(warehouse_map.width):
            cell_value = warehouse_map.grid[row, col]
            if cell_value == SHELF:
                if render_shelves:
                    parts.append(_shelf_block((row, col), f"shelf_{row}_{col}"))
            elif cell_value == PICKUP:
                parts.append(_station_pad((row, col), f"pickup_{row}_{col}", (0.96, 0.65, 0.38)))
                parts.append(_package_box((row, col)))
            elif cell_value == DROPOFF:
                parts.append(_station_pad((row, col), f"dropoff_{row}_{col}", (0.16, 0.61, 0.56)))

    robot_node_fn = _youbot_robot_node if robot_model == "youbot" else _agv_robot_node
    for robot_id, cell in enumerate(robot_start_cells):
        parts.append(robot_node_fn(robot_id, cell, controller))

    return "\n".join(parts)


def write_world_file(
    warehouse_map: WarehouseMap,
    robot_start_cells: list[tuple[int, int]],
    output_path: str,
    controller: str | None = None,
    webots_version: str = WEBOTS_VERSION,
    robot_model: str = "youbot",
    render_shelves: bool = False,
) -> None:
    content = generate_world(
        warehouse_map, robot_start_cells,
        controller=controller, webots_version=webots_version, robot_model=robot_model,
        render_shelves=render_shelves,
    )
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(content)
