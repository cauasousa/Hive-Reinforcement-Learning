# HiveRL

A fleet of warehouse robots (KUKA youBot) that learns to pick up and deliver boxes without
colliding, using **multi-agent reinforcement learning (MAPPO)** trained directly in the **Webots**
simulator.

Each robot receives pickup → dropoff orders. A classical planner decides **where** to go (order
auction + A\*), the RL policy decides **how** to drive (linear and angular velocity), and a safety
layer on the robot prevents crashes.

---

## Requirements

| Technology | Purpose |
|---|---|
| Windows 11 + **WSL2 (Ubuntu 24.04)** | Webots runs on Windows; training and robot controllers run in WSL |
| **Webots R2025a** | Simulator. Installed on Windows (GUI) and in WSL at `/usr/local/webots` (controller library) |
| **ROS 2 Jazzy** (in WSL) | Communication between the trainer and the robots |
| **Python 3.12** + venv | `torch`, `numpy`, `gymnasium`, `pettingzoo`, `pyyaml`, `matplotlib`, `pytest` (see `requirements.txt`) |

---

## Setup (once)

In WSL:

```bash
python3 -m venv ~/hiverl-ros2-venv
source ~/hiverl-ros2-venv/bin/activate
pip install -r requirements.txt
echo 'export WEBOTS_HOME=/usr/local/webots' >> ~/.bashrc
```

In **every** WSL terminal used for the project:

```bash
cd /mnt/c/Users/CauaS/programacao/new_project/HiveRL
source ~/hiverl-ros2-venv/bin/activate
source /opt/ros/jazzy/setup.bash
```

---

## Training

Training can use one or more Webots worlds in parallel. Each world needs its own port and its own
ROS domain.

**1. Open the training world in Webots:** `webots/worlds/hiverl_warehouse_extern.wbt`.
For 2 worlds, start a second Webots instance with `--port=1235` on the same file.

**2. Start the robot controllers** (one terminal per world):

```bash
./scripts/run_extern_controllers.sh webots/worlds/hiverl_warehouse_extern.wbt.robots.json localhost 1234 0
```

```bash
./scripts/run_extern_controllers.sh webots/worlds/hiverl_warehouse_extern.wbt.robots.json localhost 1235 1
```

**3. Run the trainer** (another terminal). Example: continue from the current best model with 2
worlds:

```bash
python scripts/train_mappo_webots.py --continuous-actions --env-config configs/env_webots.yaml --robots 5 --checkpoint-in outputs/mappo_checkpoint_continuous_webots_finetuned_69_best.pt --checkpoint-out outputs/mappo_checkpoint_continuous_webots_finetuned_70.pt --rounds 300 --ticks 700 --arrival-timeout 500 --velocity-tick-sleep-s 0.3 --min-update-ticks 700 --update-epochs 15 --minibatches 4 --disable-legacy-retreats --worlds 2 --world-domain-ids 0,1 2>&1 | tee outputs/mappo_checkpoint_continuous_webots_finetuned_70_train.log
```

With a single world, drop `--worlds 2 --world-domain-ids 0,1`.

| Flag | Meaning |
|---|---|
| `--checkpoint-in` / `--checkpoint-out` | Starting model / where to save |
| `--rounds` | Number of rounds (one PPO update per round) |
| `--ticks` | Ticks per round in each world |
| `--velocity-tick-sleep-s` | Real duration of one tick (0.3 s) |
| `--worlds`, `--world-domain-ids` | Number of worlds and the ROS domain of each |
| `--disable-legacy-retreats` | Turns off the old automatic stop/reverse (always use it) |

**Output files** in `outputs/`:

- `..._70.pt`: latest model.
- `..._70_best.pt`: best 20-round mean reward.
- `..._70_best_deliveries.pt`: best delivery rate.
- `..._70_history.csv`: per-round metrics.
- `..._70_kl_diagnostics.csv`: PPO diagnostics.
- `run_extern_controllers_*.log`: robot controller logs.

**Monitor training** (writes `outputs/webots_finetune_progress.png`):

```bash
python scripts/plot_webots_finetune_history.py
```

---

## Testing a model

Inference only : the model is never modified.

**1.** Open `webots/worlds/hiverl_warehouse_industrial.wbt` (industrial look) or the training world
in Webots.

**2.** Start the controllers:

```bash
./scripts/run_extern_controllers.sh webots/worlds/hiverl_warehouse_industrial.wbt.robots.json localhost 1234 0
```

**3.** Run the model:

```bash
python scripts/watch_webots.py --checkpoint outputs/mappo_checkpoint_continuous_webots_finetuned_69_best.pt --robots 5 --ticks 700 --velocity-tick-sleep-s 0.3 --disable-legacy-retreats
```

Useful options:

- `--rounds N`: stop after N rounds (default: until Ctrl+C).
- `--execute-shield`: the robots execute the shield-corrected velocities. Useful for comparison only; the model was not trained this way. In this mode the shield checks only physical obstacles (perimeter walls and robots; the map's shelf blocks exist only logically and are still used by A\* for routes) and resolves conflicts actively: a stopped robot can still turn, the lower-priority robot backs away from the one it blocks, a stuck robot reverses along its trail, and a robot stopped by the shield gets the controller's yield nudge (right → back → left). The robot-robot distance at which it acts is `--shield-radius` (default 0.65 m, center to center) and `--shield-pickup-radius` around a robot picking up (default 0.9 m).
- `HIVERL_LIDAR_YIELD=0` in front of the controllers command: disables the LiDAR yield nudge.

If Webots reports that port 1234 is in use, it falls back to 1235. In that case, pass `1235` to the
controllers.

---

## Automated tests

```bash
python3 -m pytest -q tests
```

---

## How it works

```
 Orders ──► Auction (nearest idle robot) ──► A* (route on the grid)
                                                  │
                                                  ▼
 Each robot's observation ──► MAPPO policy (shared actor) ──► (v, ω)
   LiDAR, route direction, nearest robot,                       │
   own state (status, last command, measured motion)            ▼
                                            Lockstep coordinator (ROS 2)
                                                  │
                                                  ▼
                     youBot controller in Webots: directional LiDAR brake,
                     yield nudge (right → back → left), arm/gripper
                                                  │
                                                  ▼
                   Real poses ──► Environment computes rewards ──► PPO
```

- **Environment** (`warehouse_env_continuous.py`):
  - 17×13 cell map with shelves, 3 pickup stations and 3 dropoff stations;
  - robots have a continuous pose; robots going to a pickup or carrying a box follow the policy, while idle robots use a simple autopilot;
  - in Webots, rewards come from the real poses measured after each move.
- **Rewards:**
  - positive: pickup (+30), delivery (+50) and progress along the route;
  - negative: robot contact (−1 per tick), deep collision (−30), proximity, walls, excessive turning and standing still;
  - part of the task rewards is shared with the whole team (`--team-reward-share`).
- **Network** (`networks.py`):
  - the actor is shared by all robots and outputs a tanh-squashed Gaussian over (v, ω);
  - the critic is centralized and sees every robot;
  - observations are normalized online.
- **Training** (`train_mappo_webots.py`):
  - PPO with GAE and per-robot episode ends; inactive robots are masked out of the loss;
  - Huber value loss with clipping and a KL-adaptive learning rate;
  - parallel collection from several worlds.
- **Safety:**
  - the software shield (`safety_shield_continuous.py`) predicts conflicts and penalizes the policy, but does not brake the robots;
  - braking is done by the robot controller through the LiDAR: it stops for obstacles and performs the yield nudge;
  - on a real collision, the coordinator repositions only the robots involved and the round continues.

---

## Project structure

| Folder | Contents |
|---|---|
| `src/hiverl/env/` | Environments (grid and continuous), robot, orders, map and safety shields |
| `src/hiverl/agents/` | Networks, PPO, checkpoints and A\* baseline |
| `src/hiverl/bridge/` | Lockstep coordinator, ROS 2 driver, coordinate conversion, world generator |
| `src/hiverl/planners/` | A\* and order auction |
| `src/hiverl/utils/` | CSV history, metrics and plots |
| `scripts/` | Training, testing, plotting, world generation and tools |
| `webots/worlds/` | Worlds: `hiverl_warehouse_extern.wbt` (training), `hiverl_warehouse_industrial.wbt` (visual), test worlds |
| `webots/controllers/` | `youbot_mecanum` (main controller) and test/calibration controllers |
| `configs/` | Maps (`env_*.yaml`) and hyperparameters (`train_config*.yaml`) |
| `tests/` | Automated tests (pytest) |
| `docs/` | LiDAR calibration and the MAPPO paper |
| `outputs/` | Models, histories, logs and plots |

### Other scripts

| Script | Purpose |
|---|---|
| `train_mappo.py`, `train_single_agent.py`, `evaluate.py`, `visualize_episode.py` | Early phase: grid training and evaluation, without Webots |
| `run_lockstep_demo.py` | Coordinator demo in grid / A\* mode |
| `run_scripted_baseline.py` | A\* baseline (no RL) on Webots |
| `bc_warmstart_actor.py` | Actor pre-training by imitation |
| `generate_webots_world.py` | Generates a `.wbt` world from the map |
| `build_industrial_world.py` | Generates the industrial world from the training world |
| `ros2_env_check.sh` | Checks the ROS 2 / Webots installation |
