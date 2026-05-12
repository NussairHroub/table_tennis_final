# Robotic Table Tennis &mdash; KUKA LBR iiwa 7

A model-based controller for a 7-DOF KUKA manipulator playing table tennis in PyBullet.
Wins the single-player serve-return benchmark (5/5 games, ~74 % return rate) and
plays competitive two-robot matches.

<p align="center">
  <img src="assets/single_game.gif" alt="Single-player game (won 11–6)" width="720">
</p>

> **Single-player benchmark, full game (won 11–6).** Orange dashed = planner's
> projectile prediction; green &times; / red &#9733; = predicted bounce and strike;
> green / red &#10010; = actually observed bounce and paddle contact.

This repository is my submission for the **ECE 275 (Spring 2026) term project** at KAUST.
A complete write-up of the algorithm, the parameter-tuning methodology, and the results
is in the accompanying [`report.pdf`](../ECE275_Nussair_Hroub_TermProject/report.pdf).

---

## Table of contents

- [Highlights](#highlights)
- [Demo](#demo)
- [Quick start](#quick-start)
- [Algorithm overview](#algorithm-overview)
- [Results](#results)
- [Repository layout](#repository-layout)
- [Implementing a custom player](#implementing-a-custom-player)
- [Visualisation tools](#visualisation-tools)
- [Acknowledgements](#acknowledgements)

---

## Highlights

- **Plan-once, execute open-loop swing.** Locking the prediction at the moment the ball
  first heads toward the receiver removes the IK-branch chatter that re-planning every
  step would introduce.
- **Closed-form projectile + restitution model** for predicting the table bounce, then a
  damped-horizontal projection to the strike plane.
- **Mirror-IK trick.** The right-hand robot's base is yawed by &pi;, which causes
  PyBullet's IK to drift into a contorted joint branch with a poor Cartesian swing path.
  We always solve IK on the *left-hand* robot with a mirrored world target and apply the
  joint solution to the right-hand robot. This lifts the right-hand robot's
  paddle-contact rate from **11.8 % to 86.3 %**.
- **Exhaustive 8-dim parameter sweep** (5,184 configurations &times; 5 seeds, 16
  workers, ~16 min) with neighbourhood-robustness scoring to avoid lucky outliers.
- **Visualisation harness** that overlays the planner's locked prediction on the live
  PyBullet camera frames and exports the rally as a GIF.

---

## Demo

### Single-player serve-return (won 11–6)

<p align="center">
  <img src="assets/single_game.gif" alt="Single-player game" width="720">
</p>

### Two-robot match (Player 1 wins 11–7)

<p align="center">
  <img src="assets/double_game.gif" alt="Two-robot match" width="720">
</p>

Both clips show the same overlay convention:

| Marker | Meaning |
| --- | --- |
| <kbd>&minus; &minus; &minus;</kbd> orange dashed | Planner's locked projectile prediction |
| &#9711; orange | Prediction origin (ball state when the plan was locked) |
| &times; green | Predicted bounce |
| &#10010; green | Actual observed bounce |
| &#9733; red | Predicted strike |
| &#10010; red | Actual paddle contact |
| Blue trail | True ball trajectory |

---

## Quick start

### Install

```bash
conda create -n table_tennis python=3.10 -y
conda activate table_tennis
conda install -c conda-forge pyqt -y      # only needed for the matplotlib viewer
pip install -r requirements.txt
```

> SSH users need X11 forwarding (`ssh -X`) for the matplotlib viewer. The GUI/headless
> entry points work without it.

### Run

```bash
# Single-player benchmark, headless, five games
python one_player_game.py --player robot --episodes 5

# Two-robot match
python two_player_game.py --player1 robot --player2 robot --episodes 3

# Same, but with the PyBullet GUI
python one_player_game.py --player robot --gui --episodes 1

# Mix in the simple baseline
python two_player_game.py --player1 robot --player2 simple --episodes 3
```

### Generate the demo GIFs

```bash
python visualize_full_game.py --mode single --output assets/single_game.gif
python visualize_full_game.py --mode double --output assets/double_game.gif
```

Tunable flags: `--seed`, `--stride` (capture every N&thinsp;th sim step; default 12),
`--fps`, `--width`, `--height`.
Playback speed = `(240 / stride) / fps`; the defaults render slightly slowed for
clarity.

---

## Algorithm overview

```
            +----------------+      no       +-------------------+
            |  observation   | ------------> |  hold ready pose  |
            |  b, v, θ       |               +-------------------+
            +-------+--------+
                    | "coming to us"?
                    v yes
            +----------------+
            | build swing    |   predict bounce  -> strike pose
            | plan (once     |   then windup &  follow-through
            | per rally)     |   joint targets via mirror-IK
            +-------+--------+
                    |
                    v
            +----------------+
            | schedule swing |   fire at t_arrive - T_swing/2
            | (T_swing=0.22s)|
            +-------+--------+
                    |
                    v
            +----------------+
            | execute joint  |
            | target         |
            +----------------+
```

The controller plans a swing **once** as soon as the ball is heading toward it, then
executes the plan open-loop in joint space. Three components carry the load:

1. **Trajectory prediction.** Closed-form quadratic for the first table bounce, then
   propagation through Newtonian restitution (e&thinsp;=&thinsp;0.87) to the fixed
   strike plane.
2. **Strike-pose & swing geometry.** Paddle face tilted by &phi;&nbsp;=&nbsp;0.65&nbsp;rad
   toward the opponent, with a small windup behind the strike point and a slightly
   lifted follow-through past it. The joint controller's position mode sweeps the
   paddle through the strike point with non-zero velocity, imparting return momentum.
3. **Inverse kinematics.** Damped least squares with a rest-pose null-space bias, with
   a *mirror trick* for the right-hand robot to keep both arms on the same joint
   branch.

A full derivation is in the accompanying [`report.pdf`](../ECE275_Nussair_Hroub_TermProject/report.pdf).

---

## Results

All numbers below use the **default** `TableTennisEnv` configuration
(max joint velocity 12&thinsp;rad/s, paddle restitution 0.93, no table-robot collisions).
No simulator parameters were changed for evaluation.

### Single-player benchmark — 5 episodes, won every game

| Episode | Returns | Serves |
| ---:| ---:| ---:|
| 1 | 11 | 13 |
| 2 | 11 | 15 |
| 3 | 11 | 15 |
| 4 | 11 | 14 |
| 5 | 11 | 17 |
| **Total** | **55** | **74** |
| **Return rate** | **74.3 %** | |

### Two-player matches — 3 episodes per matchup

| Matchup | Episode 1 | Episode 2 | Episode 3 |
| --- | --- | --- | --- |
| robot vs robot | 9–11 (P2) | 11–6 (P1) | 11–8 (P1) |
| robot vs simple | 11–7 (P1) | 11–3 (P1) | 11–4 (P1) |
| simple vs robot | 5–11 (P2) | 4–11 (P2) | 4–11 (P2) |

The controller beats the `simple` baseline 6–0 from either side and is roughly even
against itself.

### Mirror-IK ablation

| Robot | Paddle-contact rate (before) | Paddle-contact rate (after) |
| --- | ---:| ---:|
| Player 0 (yaw 0 base) | 87.5 % | 87.5 % |
| Player 1 (yaw &pi; base) | **11.8 %** | **86.3 %** |

The mirror-IK fix is a one-line change conceptually (mirror the world target through the
origin, run IK on the well-behaved robot, apply the resulting joints to the rotated
robot) and is the single largest improvement during development.

---

## Repository layout

```
pybullet_table_tennis_environment_v2/
├── one_player_game.py        # single-player entry point + RobotPlayer
├── two_player_game.py        # two-player entry point + RobotPlayer
├── visualize_full_game.py    # script that renders the demo GIFs
├── param_search.py           # parallel parameter sweep coordinator
├── param_search_runner.py    # per-worker evaluator
├── tuning_legacy_env.json    # legacy-env tuning (kept for reference)
├── table_tennis_env.py       # physics environment (provided, not modified)
├── simulation_manager.py     # game rules (provided, not modified)
├── player.py                 # Player base class (provided, not modified)
├── utils.py                  # matplotlib viewer (provided, not modified)
├── assets/
│   ├── single_game.gif       # demo: full single-player game
│   ├── double_game.gif       # demo: full two-robot match
│   └── simple_env.png
└── README.md
```

Per the project brief, only `one_player_game.py` and `two_player_game.py` contain
new code; the simulator files are unmodified. `visualize_full_game.py` and the
parameter-search scripts are development tooling, not part of the submitted solution.

---

## Implementing a custom player

Subclass `Player` and implement `get_action(obs)` and `reset()`:

```python
from player import Player

class MyPlayer(Player):
    def __init__(self, player_id: int, env):
        self.player_id = player_id
        self.env = env

    def get_action(self, obs: dict) -> list:
        # obs keys: ball_position, ball_velocity, joint_positions,
        #           joint_velocities, table_info, player_side
        return [0.0] * 7      # 7 target joint angles in radians

    def reset(self) -> None:
        pass                  # called at the start of each point
```

Register it in `build_player()` inside either entry-point script and run:

```bash
python one_player_game.py --player myplayer
```

Returned joint angles are clipped to KUKA hardware limits before being sent to the
simulator.

---

## Visualisation tools

`visualize_full_game.py` (this repo) overlays the planner's predictions on the
PyBullet camera image and exports a GIF of the full game. Useful for debugging the
controller and for the figures in the accompanying [`report.pdf`](../ECE275_Nussair_Hroub_TermProject/report.pdf).

```bash
# Default (slightly slowed for clarity):
python visualize_full_game.py --mode single  --output assets/single_game.gif
python visualize_full_game.py --mode double  --output assets/double_game.gif

# 2× slow motion:
python visualize_full_game.py --mode single --stride 8 --fps 15 --output single_slow.gif

# Real time, smaller file:
python visualize_full_game.py --mode single --stride 12 --fps 20 --output single_rt.gif
```

---

## Acknowledgements

This project was developed for **ECE 275 — Robot Planning and Control**
(Prof. Shinkyu Park, KAUST, Spring 2026). The PyBullet simulator and the
`Player` / `SimulationManager` skeleton were provided by the course staff.

| Course | Role | Contact |
| --- | --- | --- |
| David Alvear | TA (simulator) | <david.alvear@kaust.edu.sa> |
| Sushil Samuel Dinesh | TA (simulator) | <sushilsamuel.dinesh@kaust.edu.sa> |

Author: **Nussair Hroub** &mdash; <nussair.hroub@kaust.edu.sa>
