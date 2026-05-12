"""
visualize_full_game.py
----------------------
Run a FULL game (until one side reaches 11 with a 2-point lead) and render
an overlay GIF that shows, for each rally:
  * the PyBullet camera image as background,
  * the active player's planned trajectory (orange dashed),
  * the planner's predicted bounce / hit (green X / red star),
  * the actual bounce / hit (green + / red +).

Two modes:
  --mode single   plays one_player_game and tracks Player 0's plan only.
  --mode double   plays two_player_game (robot vs robot) and tracks whichever
                  player has the most recent active plan.

Usage
-----
    python visualize_full_game.py --mode single  --output single_game.gif
    python visualize_full_game.py --mode double  --output double_game.gif
"""

from __future__ import annotations
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import numpy as np
import pybullet as p

from table_tennis_env import TableTennisEnv
from simulation_manager import SimulationManager, GameState
from one_player_game import RobotPlayer


G = 9.81
BALL_REST = 0.87
TABLE_Z = 0.76

CAM_DISTANCE = 3.0
CAM_YAW = 0
CAM_PITCH = -30
CAM_TARGET = [0.0, 0.0, 0.8]


# ──────────────────────────────────────────────────────────────────────────

class LoggingRobotPlayer(RobotPlayer):
    """RobotPlayer that exposes its most-recent swing plan to the outside."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.last_prediction = None

    def reset(self) -> None:
        super().reset()
        self.last_prediction = None

    def _build_swing_plan(self, bx, by, bz, vx, vy, vz, table_z):
        plan = super()._build_swing_plan(bx, by, bz, vx, vy, vz, table_z)
        if plan is not None:
            self.last_prediction = {
                "ball_pos": [bx, by, bz],
                "ball_vel": [vx, vy, vz],
                "hit_pos": list(plan["strike"]),
                "table_z": float(table_z),
            }
        return plan


# ──────────────────────────────────────────────────────────────────────────

def predicted_path_3d(pred):
    """Sample the planner's projectile-bounce trajectory from origin → bounce → hit."""
    x0, y0, z0 = pred["ball_pos"]
    vx, vy, vz = pred["ball_vel"]
    table_z = pred["table_z"]
    hit_xyz = pred["hit_pos"]

    pts: list[tuple[float, float, float]] = []

    a, b, c = 0.5 * G, -vz, table_z - z0
    disc = b * b - 4 * a * c
    if disc < 0:
        return np.empty((0, 3)), None, tuple(hit_xyz)
    sd = disc ** 0.5
    ts = sorted([(-b - sd) / (2 * a), (-b + sd) / (2 * a)])
    pos_ts = [t for t in ts if t > 0]
    if not pos_ts:
        return np.empty((0, 3)), None, tuple(hit_xyz)
    t_bounce = pos_ts[0]
    for t in np.linspace(0, t_bounce, 24):
        pts.append((x0 + vx * t, y0 + vy * t, z0 + vz * t - 0.5 * G * t * t))

    bx = x0 + vx * t_bounce
    by = y0 + vy * t_bounce
    bounce_xyz = (bx, by, table_z)
    vz_impact = vz - G * t_bounce
    vz_after = abs(vz_impact) * BALL_REST

    target_z = hit_xyz[2]
    a2, b2, c2 = 0.5 * G, -vz_after, target_z - table_z
    disc2 = b2 * b2 - 4 * a2 * c2
    if disc2 >= 0:
        sd2 = disc2 ** 0.5
        ts2 = sorted([(-b2 - sd2) / (2 * a2), (-b2 + sd2) / (2 * a2)])
        pos_ts2 = [t for t in ts2 if t > 0]
        if pos_ts2:
            t_hit = pos_ts2[0]
            for t in np.linspace(0, t_hit, 24):
                pts.append((
                    bx + vx * t,
                    by + vy * t,
                    table_z + vz_after * t - 0.5 * G * t * t,
                ))

    return np.asarray(pts), bounce_xyz, tuple(hit_xyz)


def project_to_image(points_3d, view_mat, proj_mat, img_w, img_h):
    if len(points_3d) == 0:
        return np.zeros((0, 2))
    V = np.asarray(view_mat).reshape(4, 4, order="F")
    P = np.asarray(proj_mat).reshape(4, 4, order="F")
    PV = P @ V
    homo = np.hstack([points_3d, np.ones((len(points_3d), 1))])
    clip = (PV @ homo.T).T
    w = clip[:, 3:4]
    w = np.where(np.abs(w) < 1e-9, 1e-9, w)
    ndc = clip[:, :3] / w
    px = (ndc[:, 0] + 1.0) * 0.5 * img_w
    py = (1.0 - ndc[:, 1]) * 0.5 * img_h
    return np.column_stack([px, py])


def capture_camera_image(client_id, img_w, img_h):
    view_mat = p.computeViewMatrixFromYawPitchRoll(
        cameraTargetPosition=CAM_TARGET, distance=CAM_DISTANCE,
        yaw=CAM_YAW, pitch=CAM_PITCH, roll=0, upAxisIndex=2,
        physicsClientId=client_id,
    )
    proj_mat = p.computeProjectionMatrixFOV(
        fov=60, aspect=img_w / img_h, nearVal=0.1, farVal=20,
        physicsClientId=client_id,
    )
    _, _, rgba, _, _ = p.getCameraImage(
        img_w, img_h, viewMatrix=view_mat, projectionMatrix=proj_mat,
        renderer=p.ER_TINY_RENDERER, physicsClientId=client_id,
    )
    frame = np.array(rgba, dtype=np.uint8).reshape(img_h, img_w, 4)[:, :, :3]
    return frame, view_mat, proj_mat


# ──────────────────────────────────────────────────────────────────────────

def collect_full_game(mode: str, seed: int, stride: int,
                       img_w: int, img_h: int, max_total_steps: int):
    """
    Roll the simulator until GAME_OVER (or max_total_steps), capturing a
    frame every `stride` simulator steps.
    """
    two_player = (mode == "double")
    env = TableTennisEnv(gui=False, one_robot_mode=not two_player)
    p1 = LoggingRobotPlayer(player_id=0, env=env)
    p2 = LoggingRobotPlayer(player_id=1, env=env) if two_player else None
    mgr = SimulationManager(
        player1=p1, player2=p2, gui=False, real_time=False,
        seed=seed, env=env, one_robot_mode=not two_player,
    )

    frames = []
    mgr.reset()
    p1.reset()
    if p2: p2.reset()
    mgr._do_serve()

    rally_idx = 0
    total_steps = 0
    locked = {0: None, 1: None}        # locked prediction per player
    actual_bounce = {0: None, 1: None} # per-player actual bounce on their side
    actual_hit = {0: None, 1: None}    # per-player actual paddle hit
    saw_bounce = {0: False, 1: False}
    saw_hit = {0: False, 1: False}

    # Track active player for THIS rally (the one currently receiving)
    active_pid = None

    while mgr.game_state != GameState.GAME_OVER:
        mgr.step_once()
        total_steps += 1
        if total_steps > max_total_steps:
            print(f"  hit max_total_steps={max_total_steps}, stopping early")
            break

        # Lock the FIRST prediction we see this rally for each player
        for pid, player in ((0, p1), (1, p2)):
            if player is None:
                continue
            if locked[pid] is None and player.last_prediction is not None:
                locked[pid] = dict(player.last_prediction)
                if active_pid is None:
                    active_pid = pid

        # Track per-player actual bounce on own side / paddle contact
        tbl = bool(env.get_contact_with_table())
        bxyz = tuple(env.get_ball_state()["position"])
        if tbl:
            # Attribute the bounce to whichever side it's on
            side = 0 if bxyz[0] < 0 else 1
            if not saw_bounce[side]:
                saw_bounce[side] = True
                actual_bounce[side] = bxyz

        for pid in (0, 1):
            if p2 is None and pid == 1:
                continue
            if not saw_hit[pid]:
                if env.get_contact_with_paddle(pid):
                    saw_hit[pid] = True
                    actual_hit[pid] = bxyz

        if total_steps % stride == 0:
            img, V, P = capture_camera_image(env.physics_client, img_w, img_h)
            ball_pix = project_to_image(np.asarray([bxyz]), V, P, img_w, img_h)[0]

            pid_show = active_pid if active_pid is not None else 0
            pred = locked[pid_show]
            frame = {
                "image": img,
                "rally": rally_idx,
                "score": tuple(mgr.scores),
                "ball_pix": ball_pix,
                "active_pid": pid_show,
            }
            if pred is not None:
                pts3d, bounce_xyz, hit_xyz = predicted_path_3d(pred)
                if len(pts3d) > 0:
                    frame["pred_pix"] = project_to_image(pts3d, V, P, img_w, img_h)
                if bounce_xyz is not None:
                    frame["bounce_pix"] = project_to_image(
                        np.asarray([bounce_xyz]), V, P, img_w, img_h)[0]
                if hit_xyz is not None:
                    frame["hit_pix"] = project_to_image(
                        np.asarray([hit_xyz]), V, P, img_w, img_h)[0]
                frame["origin_pix"] = project_to_image(
                    np.asarray([pred["ball_pos"]]), V, P, img_w, img_h)[0]

            ab = actual_bounce[pid_show]
            ah = actual_hit[pid_show]
            if ab is not None:
                frame["actual_bounce_pix"] = project_to_image(
                    np.asarray([ab]), V, P, img_w, img_h)[0]
            if ah is not None:
                frame["actual_hit_pix"] = project_to_image(
                    np.asarray([ah]), V, P, img_w, img_h)[0]

            frames.append(frame)

        # End-of-rally bookkeeping
        if mgr.game_state == GameState.POINT_SCORED:
            rally_idx += 1
            mgr.game_state = GameState.WAITING_SERVE
            p1.reset()
            if p2: p2.reset()
            env.reset()
            mgr._reset_rally_state()
            locked = {0: None, 1: None}
            actual_bounce = {0: None, 1: None}
            actual_hit = {0: None, 1: None}
            saw_bounce = {0: False, 1: False}
            saw_hit = {0: False, 1: False}
            active_pid = None
            if mgr.game_state != GameState.GAME_OVER:
                mgr._do_serve()
            # Add a marker frame so the title shows "Serving..."
            frames.append({
                "gap": True, "rally": rally_idx,
                "score": tuple(mgr.scores),
            })

    final_score = tuple(mgr.scores)
    env.close()
    return frames, final_score


# ──────────────────────────────────────────────────────────────────────────

def render_gif(frames, output_path: str, mode: str, final_score,
               fps: int, img_w: int, img_h: int, trail_len: int = 80):
    fig = plt.figure(figsize=(img_w / 100, img_h / 100), dpi=100)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")
    ax.set_xlim(0, img_w)
    ax.set_ylim(img_h, 0)

    blank = np.zeros((img_h, img_w, 3), dtype=np.uint8)
    img_artist = ax.imshow(blank)

    actual_line, = ax.plot([], [], color="#1f77b4", lw=2.0)
    actual_dot,  = ax.plot([], [], "o", color="#1f77b4", markersize=10,
                            markeredgecolor="white", markeredgewidth=1.5)
    pred_line,   = ax.plot([], [], "--", color="#ff7f0e", lw=2.2)
    origin_pt,   = ax.plot([], [], "o", color="#ff7f0e", markersize=9,
                            markerfacecolor="none", mew=2)
    bounce_pt,   = ax.plot([], [], "x", color="#2ca02c", markersize=14, mew=3)
    actual_bounce_pt, = ax.plot([], [], "P", color="#2ca02c", markersize=12,
                                 markerfacecolor="none", mew=2)
    hit_pt,      = ax.plot([], [], "*", color="#d62728", markersize=18,
                            markeredgecolor="white", markeredgewidth=1.0)
    actual_hit_pt, = ax.plot([], [], "P", color="#d62728", markersize=12,
                              markerfacecolor="none", mew=2)
    title = ax.text(
        10, 22, "", color="white", fontsize=11,
        bbox=dict(facecolor="black", alpha=0.55, pad=4),
    )

    # Legend
    ax.plot([], [], "--", color="#ff7f0e", lw=2.0, label="predicted path")
    ax.plot([], [], "o", color="#ff7f0e", markersize=8, markerfacecolor="none",
            mew=1.8, label="prediction origin")
    ax.plot([], [], "x", color="#2ca02c", markersize=10, mew=2, label="predicted bounce")
    ax.plot([], [], "P", color="#2ca02c", markersize=10, markerfacecolor="none",
            mew=1.8, label="actual bounce")
    ax.plot([], [], "*", color="#d62728", markersize=12, label="predicted hit")
    ax.plot([], [], "P", color="#d62728", markersize=10, markerfacecolor="none",
            mew=1.8, label="actual hit")
    leg = ax.legend(loc="upper right", fontsize=8, framealpha=0.7,
                    facecolor="black", labelcolor="white", ncol=2)
    for txt in leg.get_texts():
        txt.set_color("white")

    trail = []
    current_rally = -1
    artists = (actual_line, actual_dot, pred_line, origin_pt,
               bounce_pt, actual_bounce_pt, hit_pt, actual_hit_pt)

    def score_text(score):
        if mode == "single":
            return f"Returns: {score[0]}"
        return f"Score: P1 {score[0]}—{score[1]} P2"

    def init():
        img_artist.set_data(blank)
        for art in artists:
            art.set_data([], [])
        title.set_text("")
        return (img_artist,) + artists + (title,)

    def update(i):
        nonlocal trail, current_rally
        f = frames[i]

        if f.get("gap"):
            trail = []
            current_rally = f["rally"]
            for art in artists:
                art.set_data([], [])
            title.set_text(
                f"Rally {f['rally'] + 1}: serving…   {score_text(f['score'])}"
            )
            return (img_artist,) + artists + (title,)

        if f["rally"] != current_rally:
            trail = []
            current_rally = f["rally"]

        img_artist.set_data(f["image"])

        bp = f["ball_pix"]
        trail.append((float(bp[0]), float(bp[1])))
        if len(trail) > trail_len:
            trail = trail[-trail_len:]
        actual_line.set_data([t[0] for t in trail], [t[1] for t in trail])
        actual_dot.set_data([bp[0]], [bp[1]])

        for key, art in (("pred_pix", pred_line), ("origin_pix", origin_pt),
                         ("bounce_pix", bounce_pt),
                         ("actual_bounce_pix", actual_bounce_pt),
                         ("hit_pix", hit_pt),
                         ("actual_hit_pix", actual_hit_pt)):
            v = f.get(key)
            if v is None:
                art.set_data([], [])
            elif key == "pred_pix":
                art.set_data(v[:, 0], v[:, 1])
            else:
                art.set_data([v[0]], [v[1]])

        msg = (f"Rally {f['rally'] + 1}  •  frame {i + 1}/{len(frames)}"
               f"   |   {score_text(f['score'])}")
        title.set_text(msg)
        return (img_artist,) + artists + (title,)

    anim = FuncAnimation(fig, update, frames=len(frames), init_func=init,
                          interval=1000 / fps, blit=False)
    print(f"Writing {len(frames)} frames → {output_path} @ {fps} fps")
    anim.save(output_path, writer=PillowWriter(fps=fps))
    plt.close(fig)
    print(f"Done. Final {score_text(final_score)}")


# ──────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("single", "double"), required=True)
    ap.add_argument("--output", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--stride", type=int, default=12,
                    help="Capture every Nth simulator step (240 Hz).")
    ap.add_argument("--width",  type=int, default=800)
    ap.add_argument("--height", type=int, default=450)
    ap.add_argument("--max-total-steps", type=int, default=120_000,
                    help="Safety cap on total simulator steps (≈8 min sim).")
    args = ap.parse_args()

    output = args.output or f"game_{args.mode}.gif"
    print(f"Collecting full {args.mode}-player game (seed={args.seed}, "
          f"stride={args.stride}, {args.width}x{args.height})...")
    frames, final = collect_full_game(
        args.mode, args.seed, args.stride,
        args.width, args.height, args.max_total_steps,
    )
    print(f"Captured {len(frames)} frames; final score {final}.")
    render_gif(frames, output, args.mode, final,
               fps=args.fps, img_w=args.width, img_h=args.height)


if __name__ == "__main__":
    main()
