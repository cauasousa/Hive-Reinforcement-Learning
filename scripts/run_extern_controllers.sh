#!/usr/bin/env bash
# Launch one extern controller per robot of a Webots world.
# Usage: run_extern_controllers.sh <world>.wbt.robots.json [host] [webots_port] [ros_domain_id]
# Each Webots world needs its own port and ROS domain (same numbers as the trainer's --world-domain-ids).
set -euo pipefail

SIDECAR="${1:-webots/worlds/hiverl_warehouse.wbt.robots.json}"
HOST="${2:-localhost}"
PORT="${3:-1234}"
DOMAIN_ID="${4:-${ROS_DOMAIN_ID:-0}}"
export ROS_DOMAIN_ID="$DOMAIN_ID"

if [ -z "${WEBOTS_HOME:-}" ]; then
  echo "ERROR: WEBOTS_HOME is not set. Point it at a Linux Webots install's root directory" >&2
  echo "(e.g. export WEBOTS_HOME=/opt/webots) -- see docs/run_with_webots_ros2.md Part C." >&2
  echo "This must be a LINUX Webots install -- the Windows one's controller library is a" >&2
  echo ".dll/.pyd and cannot be imported from WSL's Linux Python." >&2
  exit 1
fi

if [ ! -f "$SIDECAR" ]; then
  echo "ERROR: sidecar file not found: $SIDECAR" >&2
  echo "Generate it (from the Windows venv) with scripts/generate_webots_world.py first." >&2
  exit 1
fi

mkdir -p outputs
LOG_FILE="outputs/run_extern_controllers_$(date +%Y%m%d_%H%M%S)_port${PORT}_domain${DOMAIN_ID}.log"
exec > >(tee "$LOG_FILE") 2>&1
echo "Logging this run's full output to $LOG_FILE"
echo "Webots port=$PORT ROS_DOMAIN_ID=$ROS_DOMAIN_ID"

CONTROLLER_PY_DIR="$(python3 - <<PYEOF
import glob, os
base = os.path.join(os.environ["WEBOTS_HOME"], "lib", "controller", "python")
candidates = [base] + sorted(glob.glob(os.path.join(base, "python3*")))
for c in candidates[::-1]:
    if os.path.isdir(c):
        print(c)
        break
PYEOF
)"
if [ -z "$CONTROLLER_PY_DIR" ]; then
  echo "ERROR: could not find lib/controller/python under WEBOTS_HOME=$WEBOTS_HOME" >&2
  echo "Check that this is a real Webots install directory (not just an AppImage mount point)." >&2
  exit 1
fi
export PYTHONPATH="$CONTROLLER_PY_DIR:${PYTHONPATH:-}"
echo "Using Webots controller library: $CONTROLLER_PY_DIR"

ROBOT_MODEL="$(python3 -c "import json; print(json.load(open('$SIDECAR'))['robot_model'])")"
NUM_ROBOTS="$(python3 -c "import json; print(len(json.load(open('$SIDECAR'))['start_cells']))")"

if [ "$ROBOT_MODEL" = "youbot" ]; then
  CONTROLLER_SCRIPT="webots/controllers/youbot_mecanum/youbot_mecanum.py"
else
  CONTROLLER_SCRIPT="webots/controllers/diff_drive_agv/diff_drive_agv.py"
fi
echo "robot_model=$ROBOT_MODEL -> $CONTROLLER_SCRIPT ($NUM_ROBOTS robots)"

PIDS=()
cleanup() {
  echo "Stopping ${#PIDS[@]} controller process(es)..."
  kill "${PIDS[@]}" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

for ((i = 0; i < NUM_ROBOTS; i++)); do
  ROW="$(python3 -c "import json; print(json.load(open('$SIDECAR'))['start_cells'][$i][0])")"
  COL="$(python3 -c "import json; print(json.load(open('$SIDECAR'))['start_cells'][$i][1])")"
  export WEBOTS_CONTROLLER_URL="tcp://${HOST}:${PORT}/robot_${i}"
  python3 "$CONTROLLER_SCRIPT" --robot-id "$i" --start-row "$ROW" --start-col "$COL" &
  PIDS+=("$!")
  echo "Launched robot_$i (pid $!) -> $WEBOTS_CONTROLLER_URL, start cell ($ROW, $COL)"
done

echo "All ${#PIDS[@]} controllers launched. Ctrl+C to stop them."
wait
