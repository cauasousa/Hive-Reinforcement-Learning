#!/usr/bin/env bash
set -uo pipefail

echo "=== 1. Is ros2 on PATH at all? ==="
if ! command -v ros2 >/dev/null 2>&1; then
  echo "ros2 CLI not found in this shell."
  echo "Trying to source a system ROS2 install..."
  found=""
  for f in /opt/ros/*/setup.bash; do
    if [ -f "$f" ]; then
      echo "Sourcing $f"
      source "$f"
      found="$f"
      break
    fi
  done
  if [ -z "$found" ]; then
    echo "ERROR: no /opt/ros/*/setup.bash found. ROS2 isn't installed in THIS shell's environment"
    echo "(is this WSL? the right distro? see docs/run_with_webots_ros2.md Part C.1/C.3)." >&2
    exit 1
  fi
  if ! command -v ros2 >/dev/null 2>&1; then
    echo "ERROR: sourced $found but 'ros2' still isn't on PATH -- installation looks incomplete." >&2
    exit 1
  fi
else
  echo "OK: ros2 CLI found at $(command -v ros2)"
fi

echo
echo "=== 2. ROS2 environment this shell is using ==="
echo "ROS_DISTRO=${ROS_DISTRO:-<unset>}"
echo "ROS_DOMAIN_ID=${ROS_DOMAIN_ID:-<unset, defaults to 0>}"
echo "RMW_IMPLEMENTATION=${RMW_IMPLEMENTATION:-<unset, using default rmw>}"
echo "-> Compare these three lines against the shell that ran run_extern_controllers.sh."
echo "   They must match EXACTLY (including 'unset' on both sides) or the two shells are on"
echo "   different DDS domains and will never see each other's topics/nodes, no matter what."

echo
echo "=== 3. Restarting the ros2 daemon (clears any stale discovery cache) ==="
ros2 daemon stop >/dev/null 2>&1
ros2 daemon start
sleep 1

echo
echo "=== 4. Current ROS2 graph ==="
echo "--- nodes ---"
ros2 node list
echo "--- topics (with types) ---"
ros2 topic list -t

echo
echo "If the node/topic lists above are still empty while run_extern_controllers.sh is clearly"
echo "printing '[youbot_mecanum robot_N] ...' lines, the controllers ARE creating rclpy nodes --"
echo "the two shells are just on different domains/networks. Fastest fix: run this exact script"
echo "in the SAME terminal (or a new tab of the SAME WSL distro/session) that ran"
echo "run_extern_controllers.sh, right after launching it."
