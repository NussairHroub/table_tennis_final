"""
visualize_prediction.py
-----------------------
Run a few rollouts with RobotPlayer and produce a GIF comparing:
  • the actual ball trajectory (what the simulator does)
  • the planner's predicted trajectory (what RobotPlayer thinks will happen)

Two rendering modes:
  --mode side     Side view (x–z plane). Light schematic with table & net.
  --mode overlay  PyBullet camera frame as background, prediction overlaid.

The dashed line is the prediction emitted by RobotPlayer._update_plan each
step: forward-integrated projectile motion from the current ball state
through the predicted bounce to the planned hit point.

Usage
-----
    python visualize_prediction.py --rallies 3 --output prediction.gif
    python visualize_prediction.py --mode overlay --rallies 2 --output overlay.gif
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


# Physics constants (match the simulator)
G = 9.81
BALL_REST = 0.87
TABLE_Z = 0.76
TABLE_HALF_LEN = 1.37
NET_TOP_Z = 0.9125

# Camera (matches utils.run_simulation_with_matplotlib)
CAM_DISTANCE = 3.0
CAM_YAW = 0
CAM_PITCH = -30
CAM_TARGET = [0.0, 0.0, 0.8]
IMG_W, IMG_H = 960, 540


# ──────────────────────────────────────────────────────────────────────────
# RobotPlayer that exposes its plan to the outside

class LoggingRobotPlayer(RobotPlayer):
    """Records the latest prediction snapshot each time _update_plan runs."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.last_prediction = None

    def reset(self) -> None:
        super().reset()
        self.last_prediction = None

    def _update_plan(self, ball_pos, ball_vel, table_z, net_top):
        super()._update_plan(ball_pos, ball_vel, table_z, net_top)
        if self._hit_pos is not None:
            self.last_prediction = {
                "ball_pos": list(ball_pos),
                "ball_vel": list(ball_vel),
                "bounced": bool(self._bounced),
                "hit_pos": list(self._hit_pos),
                "table_z": float(table_z),
            }


# ──────────────────────────────────────────────────────────────────────────
# Forward-integrate the planner's projectile-bounce model (3D)

def predicted_path_3d(pred):
    """
    Return (points_3d, bounce_xyz, hit_xyz) where points_3d is an (N, 3)
    array sampling the planner's predicted ball trajectory from the current
    state through the bounce to the hit point.
    """
    x0, y0, z0 = pred["ball_pos"]
    vx, vy, vz = pred["ball_vel"]
    table_z = pred["table_z"]
    hit_xyz = pred["hit_pos"]

    pts: list[tuple[float, float, float]] = []
    bounce_xyz = None

    if not pred["bounced"]:
        a, b, c = 0.5 * G, -vz, table_z - z0
        disc = b * b - 4 * a * c
        if disc < 0:
            return np.asarray(pts).reshape(-1, 3), None, None
        sd = disc ** 0.5
        ts = sorted([(-b - sd) / (2 * a), (-b + sd) / (2 * a)])
        pos_ts = [t for t in ts if t > 0]
        if not pos_ts:
            return np.asarray(pts).reshape(-1, 3), None, None
        t_bounce = pos_ts[0]

        for t in np.linspace(0, t_bounce, 24):
            pts.append((x0 + vx * t, y0 + vy * t, z0 + vz * t - 0.5 * G * t * t))

        bx = x0 + vx * t_bounce
        by = y0 + vy * t_bounce
        bounce_xyz = (bx, by, table_z)
        vz_impact = vz - G * t_bounce
        vz_after = abs(vz_impact) * BALL_REST
        start_x, start_y, start_z = bx, by, table_z
    else:
        start_x, start_y, start_z = x0, y0, z0
        vz_after = vz

    # Post-bounce to hit_z
    target_z = hit_xyz[2]
    a, b, c = 0.5 * G, -vz_after, target_z - start_z
    disc = b * b - 4 * a * c
    if disc >= 0:
        sd = disc ** 0.5
        ts = sorted([(-b - sd) / (2 * a), (-b + sd) / (2 * a)])
        pos_ts = [t for t in ts if t > 0]
        if pos_ts:
            t_hit = pos_ts[0]
            for t in np.linspace(0, t_hit, 24):
                pts.append((
                    start_x + vx * t,
                    start_y + vy * t,
                    start_z + vz_after * t - 0.5 * G * t * t,
                ))

    return np.asarray(pts).reshape(-1, 3), bounce_xyz, tuple(hit_xyz)


# ──────────────────────────────────────────────────────────────────────────
# Project 3D world points to 2D pixel coords using PyBullet camera matrices

def project_to_image(points_3d, view_mat, proj_mat, img_w, img_h):
    """Project Nx3 world points to Nx2 image-pixel coords."""
    if len(points_3d) == 0:
        return np.zeros((0, 2))
    V = np.asarray(view_mat).reshape(4, 4, order="F")
    P = np.asarray(proj_mat).reshape(4, 4, order="F")
    PV = P @ V
    homo = np.hstack([points_3d, np.ones((len(points_3d), 1))])
    clip = (PV @ homo.T).T
    w = clip[:, 3:4]
    # avoid divide-by-zero
    w = np.where(np.abs(w) < 1e-9, 1e-9, w)
    ndc = clip[:, :3] / w
    pixel_x = (ndc[:, 0] + 1.0) * 0.5 * img_w
    pixel_y = (1.0 - ndc[:, 1]) * 0.5 * img_h
    return np.column_stack([pixel_x, pixel_y])


def capture_camera_image(client_id):
    """Render the current scene from the fixed side camera."""
    view_mat = p.computeViewMatrixFromYawPitchRoll(
        cameraTargetPosition=CAM_TARGET,
        distance=CAM_DISTANCE,
        yaw=CAM_YAW,
        pitch=CAM_PITCH,
        roll=0,
        upAxisIndex=2,
        physicsClientId=client_id,
    )
    proj_mat = p.computeProjectionMatrixFOV(
        fov=60, aspect=IMG_W / IMG_H,
        nearVal=0.1, farVal=20,
        physicsClientId=client_id,
    )
    _, _, rgba, _, _ = p.getCameraImage(
        IMG_W, IMG_H,
        viewMatrix=view_mat,
        projectionMatrix=proj_mat,
        renderer=p.ER_TINY_RENDERER,
        physicsClientId=client_id,
    )
    frame = np.array(rgba, dtype=np.uint8).reshape(IMG_H, IMG_W, 4)[:, :, :3]
    return frame, view_mat, proj_mat


# ──────────────────────────────────────────────────────────────────────────
# Roll out the simulator and record per-step data

def collect_frames(n_rallies: int, seed: int, mode: str,
                   max_steps_per_rally: int = 1500, stride: int = 4,
                   live_predict: bool = False):
    """
    Run rollouts and collect data needed for rendering.

    live_predict=False (default): lock the FIRST prediction we see in each rally
        and re-use it for every frame of that rally. This is the honest visual
        test: does the trajectory the planner committed to at moment t actually
        play out? Differences reveal model error (spin, restitution mismatch,
        non-instant paddle impacts, etc.).

    live_predict=True: re-compute the prediction every step using the current
        ball state. This always overlays exactly on the ground truth because
        both share the same starting point and the same physics model — only
        useful for sanity-checking the projection math.
    """
    env = TableTennisEnv(gui=False, one_robot_mode=True)
    player = LoggingRobotPlayer(player_id=0, env=env)
    mgr = SimulationManager(
        player1=player, player2=None, gui=False, real_time=False,
        seed=seed, env=env, one_robot_mode=True,
    )

    frames = []
    mgr.reset()
    player.reset()
    mgr._do_serve()
    rally_idx = 0
    step_in_rally = 0
    total_steps = 0
    locked_pre = None        # pre-bounce prediction frozen on first sight
    actual_bounce_xyz = None # where the ball actually first touched the table
    actual_hit_xyz = None    # where the ball actually first touched the paddle
    saw_bounce_contact = False
    saw_paddle_contact = False

    while rally_idx < n_rallies:
        mgr.step_once()
        step_in_rally += 1
        total_steps += 1

        # Lock the prediction at the FIRST pre-bounce frame and never update it.
        # We deliberately keep showing this pre-bounce prediction even after the
        # real ball bounces, so any divergence between the planner's model
        # (gravity + restitution 0.87, no friction) and the simulator's actual
        # contact dynamics becomes visible.
        if not live_predict and player.last_prediction is not None:
            if locked_pre is None and not player.last_prediction["bounced"]:
                locked_pre = dict(player.last_prediction)

        # Capture actual bounce / paddle-hit positions on the rising edge of contact
        tbl_now = bool(env.get_contact_with_table())
        pad_now = bool(env.get_contact_with_paddle(0))
        if tbl_now and not saw_bounce_contact:
            saw_bounce_contact = True
            actual_bounce_xyz = tuple(env.get_ball_state()["position"])
        if pad_now and not saw_paddle_contact:
            saw_paddle_contact = True
            actual_hit_xyz = tuple(env.get_ball_state()["position"])

        should_capture = (total_steps % stride == 0)
        if should_capture:
            ball_state = env.get_ball_state()
            bxyz = tuple(ball_state["position"])

            frame = {
                "rally": rally_idx,
                "ball_3d": bxyz,
                "actual_bounce": actual_bounce_xyz,
                "actual_hit": actual_hit_xyz,
            }

            # Choose the prediction snapshot used for THIS frame.
            # In live mode we always grab the latest; in locked mode we always
            # show the very first pre-bounce prediction.
            pred_source = player.last_prediction if live_predict else locked_pre

            pred_pts = None
            bounce_xyz = None
            hit_xyz = None
            origin_xyz = None
            if pred_source is not None:
                pred_pts, bounce_xyz, hit_xyz = predicted_path_3d(pred_source)
                origin_xyz = tuple(pred_source["ball_pos"])

            if mode == "overlay":
                img, view_mat, proj_mat = capture_camera_image(env.physics_client)
                frame["image"] = img
                if pred_pts is not None and len(pred_pts) > 0:
                    pix = project_to_image(pred_pts, view_mat, proj_mat, IMG_W, IMG_H)
                    frame["pred_pix"] = pix
                if bounce_xyz is not None:
                    frame["bounce_pix"] = project_to_image(
                        np.asarray([bounce_xyz]), view_mat, proj_mat, IMG_W, IMG_H,
                    )[0]
                if hit_xyz is not None:
                    frame["hit_pix"] = project_to_image(
                        np.asarray([hit_xyz]), view_mat, proj_mat, IMG_W, IMG_H,
                    )[0]
                if origin_xyz is not None:
                    frame["origin_pix"] = project_to_image(
                        np.asarray([origin_xyz]), view_mat, proj_mat, IMG_W, IMG_H,
                    )[0]
                if actual_bounce_xyz is not None:
                    frame["actual_bounce_pix"] = project_to_image(
                        np.asarray([actual_bounce_xyz]), view_mat, proj_mat, IMG_W, IMG_H,
                    )[0]
                if actual_hit_xyz is not None:
                    frame["actual_hit_pix"] = project_to_image(
                        np.asarray([actual_hit_xyz]), view_mat, proj_mat, IMG_W, IMG_H,
                    )[0]
                frame["ball_pix"] = project_to_image(
                    np.asarray([bxyz]), view_mat, proj_mat, IMG_W, IMG_H,
                )[0]
            else:
                frame["predicted_xz"] = (
                    (pred_pts[:, 0].tolist(), pred_pts[:, 2].tolist())
                    if pred_pts is not None and len(pred_pts) > 0 else None
                )
                frame["bounce_xz"] = (
                    (bounce_xyz[0], bounce_xyz[2]) if bounce_xyz else None
                )
                frame["hit_xz"] = (
                    (hit_xyz[0], hit_xyz[2]) if hit_xyz else None
                )
                frame["origin_xz"] = (
                    (origin_xyz[0], origin_xyz[2]) if origin_xyz else None
                )
                frame["actual_bounce_xz"] = (
                    (actual_bounce_xyz[0], actual_bounce_xyz[2])
                    if actual_bounce_xyz else None
                )
                frame["actual_hit_xz"] = (
                    (actual_hit_xyz[0], actual_hit_xyz[2])
                    if actual_hit_xyz else None
                )
                frame["ball_xz"] = (bxyz[0], bxyz[2])

            frames.append(frame)

        if mgr.game_state == GameState.POINT_SCORED:
            rally_idx += 1
            if rally_idx >= n_rallies:
                break
            mgr.game_state = GameState.WAITING_SERVE
            player.reset()
            env.reset()
            mgr._reset_rally_state()
            mgr._do_serve()
            step_in_rally = 0
            locked_pre = None
            actual_bounce_xyz = None
            actual_hit_xyz = None
            saw_bounce_contact = False
            saw_paddle_contact = False
            frames.append({"rally": rally_idx, "gap": True})
        elif step_in_rally >= max_steps_per_rally:
            rally_idx += 1
            if rally_idx >= n_rallies:
                break
            mgr.game_state = GameState.WAITING_SERVE
            player.reset()
            env.reset()
            mgr._reset_rally_state()
            mgr._do_serve()
            step_in_rally = 0
            locked_pre = None
            actual_bounce_xyz = None
            actual_hit_xyz = None
            saw_bounce_contact = False
            saw_paddle_contact = False

    env.close()
    return frames


# ──────────────────────────────────────────────────────────────────────────
# Render: side-view schematic

def render_gif_side(frames, output_path: str, fps: int = 30,
                    trail_len: int = 80):
    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    ax.set_xlim(-2.2, 2.2)
    ax.set_ylim(0.6, 2.2)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("z (m)")
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)

    ax.plot([-TABLE_HALF_LEN, TABLE_HALF_LEN], [TABLE_Z, TABLE_Z],
            color="black", lw=2, label="table")
    ax.plot([0, 0], [TABLE_Z, NET_TOP_Z], color="dimgray", lw=2, label="net")
    ax.plot([-1.9], [0.66], marker="s", color="black", markersize=10)
    ax.annotate("robot", xy=(-1.9, 0.66), xytext=(-2.1, 0.78), fontsize=8)

    actual_line, = ax.plot([], [], color="#1f77b4", lw=2.0, label="actual ball")
    actual_dot,  = ax.plot([], [], "o", color="#1f77b4", markersize=8)
    pred_line,   = ax.plot([], [], "--", color="#ff7f0e", lw=1.6, label="predicted path")
    origin_pt,   = ax.plot([], [], "o", color="#ff7f0e", markersize=7,
                            markerfacecolor="none", mew=1.5,
                            label="prediction origin")
    bounce_pt,   = ax.plot([], [], "x", color="#2ca02c", markersize=10, mew=2,
                            label="predicted bounce")
    actual_bounce_pt, = ax.plot([], [], "P", color="#2ca02c", markersize=10,
                                 markerfacecolor="none", mew=1.5,
                                 label="actual bounce")
    hit_pt,      = ax.plot([], [], "*", color="#d62728", markersize=14,
                            label="predicted hit")
    actual_hit_pt, = ax.plot([], [], "P", color="#d62728", markersize=10,
                              markerfacecolor="none", mew=1.5,
                              label="actual hit")
    title = ax.set_title("")
    ax.legend(loc="upper right", fontsize=7, framealpha=0.85, ncol=2)

    trail_x: list[float] = []
    trail_z: list[float] = []
    current_rally = -1

    artists = (actual_line, actual_dot, pred_line, origin_pt,
               bounce_pt, actual_bounce_pt, hit_pt, actual_hit_pt)

    def init():
        for art in artists:
            art.set_data([], [])
        return artists + (title,)

    def update(i):
        nonlocal trail_x, trail_z, current_rally
        f = frames[i]

        if f.get("gap"):
            trail_x, trail_z = [], []
            current_rally = f["rally"]
            for art in artists:
                art.set_data([], [])
            title.set_text(f"Rally {f['rally'] + 1}: serving…")
            return artists + (title,)

        if f["rally"] != current_rally:
            trail_x, trail_z = [], []
            current_rally = f["rally"]

        bx, bz = f["ball_xz"]
        trail_x.append(bx)
        trail_z.append(bz)
        if len(trail_x) > trail_len:
            trail_x = trail_x[-trail_len:]
            trail_z = trail_z[-trail_len:]

        actual_line.set_data(trail_x, trail_z)
        actual_dot.set_data([bx], [bz])

        if f.get("predicted_xz"):
            xs, zs = f["predicted_xz"]
            pred_line.set_data(xs, zs)
        else:
            pred_line.set_data([], [])

        if f.get("origin_xz"):
            origin_pt.set_data([f["origin_xz"][0]], [f["origin_xz"][1]])
        else:
            origin_pt.set_data([], [])

        if f.get("bounce_xz"):
            bounce_pt.set_data([f["bounce_xz"][0]], [f["bounce_xz"][1]])
        else:
            bounce_pt.set_data([], [])

        if f.get("hit_xz"):
            hit_pt.set_data([f["hit_xz"][0]], [f["hit_xz"][1]])
        else:
            hit_pt.set_data([], [])

        if f.get("actual_bounce_xz"):
            actual_bounce_pt.set_data([f["actual_bounce_xz"][0]],
                                      [f["actual_bounce_xz"][1]])
        else:
            actual_bounce_pt.set_data([], [])

        if f.get("actual_hit_xz"):
            actual_hit_pt.set_data([f["actual_hit_xz"][0]],
                                   [f["actual_hit_xz"][1]])
        else:
            actual_hit_pt.set_data([], [])

        # Compute prediction error (x-z plane only)
        msg = f"Rally {f['rally'] + 1}  •  frame {i + 1}/{len(frames)}"
        if f.get("bounce_xz") and f.get("actual_bounce_xz"):
            dx = f["bounce_xz"][0] - f["actual_bounce_xz"][0]
            dz = f["bounce_xz"][1] - f["actual_bounce_xz"][1]
            msg += f"   |   bounce err = {(dx*dx + dz*dz)**0.5:.3f} m"
        if f.get("hit_xz") and f.get("actual_hit_xz"):
            dx = f["hit_xz"][0] - f["actual_hit_xz"][0]
            dz = f["hit_xz"][1] - f["actual_hit_xz"][1]
            msg += f"   |   hit err = {(dx*dx + dz*dz)**0.5:.3f} m"
        title.set_text(msg)
        return artists + (title,)

    anim = FuncAnimation(
        fig, update, frames=len(frames), init_func=init,
        interval=1000 / fps, blit=False,
    )
    print(f"Writing {len(frames)} frames to {output_path} at {fps} fps...")
    anim.save(output_path, writer=PillowWriter(fps=fps))
    plt.close(fig)
    print("Done.")


# ──────────────────────────────────────────────────────────────────────────
# Render: overlay on PyBullet camera frames

def render_gif_overlay(frames, output_path: str, fps: int = 30,
                       trail_len: int = 80):
    fig = plt.figure(figsize=(IMG_W / 100, IMG_H / 100), dpi=100)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")
    ax.set_xlim(0, IMG_W)
    ax.set_ylim(IMG_H, 0)  # image coords: y down

    # Placeholder image
    blank = np.zeros((IMG_H, IMG_W, 3), dtype=np.uint8)
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

    # Legend overlay
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

    trail_xy: list[tuple[float, float]] = []
    current_rally = -1

    artists = (actual_line, actual_dot, pred_line, origin_pt,
               bounce_pt, actual_bounce_pt, hit_pt, actual_hit_pt)

    def init():
        img_artist.set_data(blank)
        for art in artists:
            art.set_data([], [])
        title.set_text("")
        return (img_artist,) + artists + (title,)

    def update(i):
        nonlocal trail_xy, current_rally
        f = frames[i]

        if f.get("gap"):
            trail_xy = []
            current_rally = f["rally"]
            for art in artists:
                art.set_data([], [])
            title.set_text(f"Rally {f['rally'] + 1}: serving…")
            return (img_artist,) + artists + (title,)

        if f["rally"] != current_rally:
            trail_xy = []
            current_rally = f["rally"]

        img_artist.set_data(f["image"])

        ball_pix = f["ball_pix"]
        trail_xy.append((float(ball_pix[0]), float(ball_pix[1])))
        if len(trail_xy) > trail_len:
            trail_xy = trail_xy[-trail_len:]
        tx = [p[0] for p in trail_xy]
        ty = [p[1] for p in trail_xy]
        actual_line.set_data(tx, ty)
        actual_dot.set_data([ball_pix[0]], [ball_pix[1]])

        if f.get("pred_pix") is not None:
            pix = f["pred_pix"]
            pred_line.set_data(pix[:, 0], pix[:, 1])
        else:
            pred_line.set_data([], [])

        if f.get("origin_pix") is not None:
            op = f["origin_pix"]
            origin_pt.set_data([op[0]], [op[1]])
        else:
            origin_pt.set_data([], [])

        if f.get("bounce_pix") is not None:
            bp = f["bounce_pix"]
            bounce_pt.set_data([bp[0]], [bp[1]])
        else:
            bounce_pt.set_data([], [])

        if f.get("hit_pix") is not None:
            hp = f["hit_pix"]
            hit_pt.set_data([hp[0]], [hp[1]])
        else:
            hit_pt.set_data([], [])

        if f.get("actual_bounce_pix") is not None:
            ab = f["actual_bounce_pix"]
            actual_bounce_pt.set_data([ab[0]], [ab[1]])
        else:
            actual_bounce_pt.set_data([], [])

        if f.get("actual_hit_pix") is not None:
            ah = f["actual_hit_pix"]
            actual_hit_pt.set_data([ah[0]], [ah[1]])
        else:
            actual_hit_pt.set_data([], [])

        # 3D error: use the originals captured in collect_frames
        msg = f"Rally {f['rally'] + 1}  •  frame {i + 1}/{len(frames)}"
        title.set_text(msg)
        return (img_artist,) + artists + (title,)

    anim = FuncAnimation(
        fig, update, frames=len(frames), init_func=init,
        interval=1000 / fps, blit=False,
    )
    print(f"Writing {len(frames)} frames to {output_path} at {fps} fps...")
    anim.save(output_path, writer=PillowWriter(fps=fps))
    plt.close(fig)
    print("Done.")


# ──────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("side", "overlay"), default="side",
                        help="side: schematic side view; "
                             "overlay: PyBullet camera frame as background.")
    parser.add_argument("--rallies", type=int, default=3,
                        help="Number of rallies to record.")
    parser.add_argument("--seed", type=int, default=0,
                        help="Seed for the serve RNG.")
    parser.add_argument("--output", type=str, default=None,
                        help="Output GIF path. Default: prediction_<mode>.gif")
    parser.add_argument("--fps", type=int, default=30,
                        help="Output GIF frame rate.")
    parser.add_argument("--stride", type=int, default=4,
                        help="Capture every Nth simulator step "
                             "(simulator runs at 240 Hz).")
    parser.add_argument("--live-predict", action="store_true",
                        help="Re-predict every step (matches ground truth "
                             "trivially because both share starting state and "
                             "physics). Default: lock prediction once per rally "
                             "so divergence from ground truth is visible.")
    args = parser.parse_args()

    output = args.output or f"prediction_{args.mode}.gif"

    print(f"Collecting {args.rallies} rallies in mode={args.mode} "
          f"(seed={args.seed}, stride={args.stride}, "
          f"live_predict={args.live_predict})...")
    frames = collect_frames(args.rallies, args.seed, args.mode,
                             stride=args.stride,
                             live_predict=args.live_predict)
    print(f"Collected {len(frames)} captured frames.")

    if args.mode == "side":
        render_gif_side(frames, output, fps=args.fps)
    else:
        render_gif_overlay(frames, output, fps=args.fps)


if __name__ == "__main__":
    main()
