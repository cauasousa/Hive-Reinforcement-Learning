from hiverl.bridge.webots_world_generator import _boundary_walls_block, generate_world
from hiverl.env.warehouse_grid import generate_warehouse_map


def _small_map():
    return generate_warehouse_map(
        width=9,
        height=7,
        num_shelf_blocks_x=1,
        num_shelf_blocks_y=1,
        shelf_block_w=2,
        shelf_block_h=2,
        num_pickup_stations=1,
        num_dropoff_stations=1,
    )


def test_generated_world_has_valid_header_and_externproto():
    world = generate_world(_small_map(), robot_start_cells=[(1, 1)])
    assert world.startswith("#VRML_SIM")
    assert "EXTERNPROTO" in world
    assert "Floor.proto" in world
    assert "E-puck" not in world


def test_shelves_not_rendered_by_default():
    """render_shelves defaults to False -- a deliberate user decision (see
    webots_ros2_integration.md section 8f): the Webots *visual* world skips drawing shelf
    Solids, but WarehouseMap.grid still marks those cells SHELF, so A*/the Safety Shield/the
    trained policy still route around them exactly as before -- this flag only changes what's
    physically drawn, not the underlying grid any training/evaluation logic reads."""
    warehouse_map = _small_map()
    world = generate_world(warehouse_map, robot_start_cells=[(1, 1)])
    assert world.count('name "shelf_') == 0
    assert int((warehouse_map.grid == 1).sum()) > 0


def test_render_shelves_true_draws_one_solid_per_shelf_cell():
    warehouse_map = _small_map()
    world = generate_world(warehouse_map, robot_start_cells=[(1, 1)], render_shelves=True)
    num_shelf_cells = int((warehouse_map.grid == 1).sum())
    assert world.count('name "shelf_') == num_shelf_cells


def test_generated_world_always_includes_the_four_boundary_walls():
    world = generate_world(_small_map(), robot_start_cells=[(1, 1)])
    for name in ("wall_top", "wall_bottom", "wall_left", "wall_right"):
        assert f'name "{name}"' in world


def test_boundary_walls_match_the_exact_reference_layout_for_the_default_17x13_grid():
    """Fixed numbers for this project's default 17x13 grid, locking in that the generalized
    version (any width/height, not just 17x13) reproduces them. Updated 2026-08-29 (user report:
    a robot could drive through to "the other side" of a boundary wall) -- the previous reference
    layout centered each wall WALL_OFFSET_M=0.5m outward from the map edge with only
    WALL_THICKNESS_M=0.2m of its own thickness, leaving a genuine 0.4m open gap between the
    walkable area and every wall, all the way around the perimeter. Walls are now centered
    directly on the map's boundary lines (straddling the edge) so there is no gap and the
    corners meet exactly."""
    block = _boundary_walls_block(width_cells=17, height_cells=13)
    assert "translation 8.50 -13.00 0.25" in block
    assert "translation 8.50 0.00 0.25" in block
    assert "translation 0.00 -6.50 0.25" in block
    assert "translation 17.00 -6.50 0.25" in block
    assert block.count("size 17.00 0.20 0.50") == 4
    assert block.count("size 0.20 13.20 0.50") == 4


def test_generated_world_has_one_robot_node_per_start_cell_agv():
    starts = [(1, 1), (2, 2), (3, 3)]
    world = generate_world(_small_map(), robot_start_cells=starts, robot_model="agv")
    for robot_id in range(len(starts)):
        assert f'name "robot_{robot_id}"' in world
    assert world.count("Robot {") == len(starts)


def test_robot_controller_args_encode_start_cell():
    world = generate_world(
        _small_map(), robot_start_cells=[(2, 3)], robot_model="agv", controller="diff_drive_agv"
    )
    assert '"--start-row" "2"' in world
    assert '"--start-col" "3"' in world


def test_agv_robot_has_both_wheel_motors_and_status_leds():
    world = generate_world(_small_map(), robot_start_cells=[(1, 1), (2, 2)], robot_model="agv")
    for count, needle in [
        (2, '"left wheel motor"'),
        (2, '"right wheel motor"'),
        (2, '"left wheel sensor"'),
        (2, '"right wheel sensor"'),
        (2, 'name "led_to_item"'),
        (2, 'name "led_carrying"'),
        (2, 'name "caster"'),
    ]:
        assert world.count(needle) == count, needle


def test_agv_controller_default_is_diff_drive_agv():
    world = generate_world(_small_map(), robot_start_cells=[(1, 1)], robot_model="agv")
    assert 'controller "diff_drive_agv"' in world


def test_default_robot_model_is_youbot():
    world = generate_world(_small_map(), robot_start_cells=[(1, 1), (2, 2)])
    assert world.count("Youbot {") == 2
    assert 'controller "youbot_mecanum"' in world
    assert "Youbot.proto" in world
    assert "Robot {" not in world


def test_youbot_world_has_mecanum_contact_properties():
    world = generate_world(_small_map(), robot_start_cells=[(1, 1)])
    assert "contactProperties" in world
    assert "InteriorWheelMat" in world
    assert "ExteriorWheelMat" in world


def test_agv_world_has_no_mecanum_contact_properties():
    world = generate_world(_small_map(), robot_start_cells=[(1, 1)], robot_model="agv")
    assert "contactProperties" not in world


def test_youbot_controller_args_encode_start_cell():
    world = generate_world(_small_map(), robot_start_cells=[(2, 3)])
    assert '"--start-row" "2"' in world
    assert '"--start-col" "3"' in world


def test_youbot_bodyslot_has_only_status_beacon_and_is_a_supervisor():
    """The payload tray ("package_platform") and the fake-LED "led_carrying" parcel were removed
    -- per direct user feedback that the tray looked like a large, ugly attachment the real youBot
    doesn't have. Carrying is now shown with a real DEF-named package box (see
    test_pickup_stations_have_a_real_package_box) teleported by the controller via Supervisor
    APIs, which requires ``supervisor TRUE`` on the robot node."""
    world = generate_world(_small_map(), robot_start_cells=[(1, 1), (2, 2)])
    assert world.count("supervisor TRUE") == 2
    assert world.count('name "led_to_item"') == 2
    assert 'name "package_platform"' not in world
    assert 'name "led_carrying"' not in world


def test_pickup_stations_have_a_real_package_box():
    """Each pickup pad gets its own DEF-named package box (PACKAGE_<row>_<col>) so the controller
    can look it up with getFromDef and teleport it to the gripper on pickup / to the dropoff pad on
    delivery -- see youbot_mecanum.py."""
    warehouse_map = _small_map()
    world = generate_world(warehouse_map, robot_start_cells=[(1, 1)])
    from hiverl.env.warehouse_grid import PICKUP

    pickup_cells = [
        (row, col)
        for row in range(warehouse_map.height)
        for col in range(warehouse_map.width)
        if warehouse_map.grid[row, col] == PICKUP
    ]
    assert pickup_cells
    for row, col in pickup_cells:
        assert f"DEF PACKAGE_{row}_{col} Solid" in world


def test_agv_led_carrying_is_package_sized_not_a_small_indicator():
    """The AGV body has no arm/gripper at all, so the gripper-size constraint above doesn't apply
    -- kept at its original larger, more visible size."""
    world = generate_world(_small_map(), robot_start_cells=[(1, 1)], robot_model="agv")
    assert "geometry Box { size 0.14 0.14 0.08 }" in world
    assert "0.82 0.55 0.28" in world


def test_invalid_robot_model_rejected():
    import pytest

    with pytest.raises(ValueError):
        generate_world(_small_map(), robot_start_cells=[(1, 1)], robot_model="bogus")
