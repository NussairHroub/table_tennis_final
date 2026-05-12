from __future__ import annotations
import numpy as np
import pybullet as p
from simulation_manager import SimulationManager


# ---------------------------------------------------------------------------
# Visualization using matplotlib (works over SSH X11 forwarding)
def run_simulation_with_matplotlib(manager: SimulationManager, max_episodes: int, fps: float) -> list:
    # Camera parameters for the overhead-side view
    CAM_DISTANCE   = 3.0
    CAM_YAW        = 0
    CAM_PITCH      = -30
    CAM_TARGET     = [0.0, 0.0, 0.8]
    IMG_W, IMG_H   = 960, 540
    step_interval  = max(1, int(240 / fps))   # render every N physics steps
    
    import matplotlib
    matplotlib.use("Qt5Agg")          # works over X11 forwarding via PyQt5
    import matplotlib.pyplot as plt

    plt.ion()
    fig, ax = plt.subplots(figsize=(10, 5.6))
    fig.patch.set_facecolor("black")
    ax.axis("off")
    fig.tight_layout(pad=0)
    img_handle = None

    results = []
    env = manager.env
    client = env.physics_client

    for ep in range(max_episodes):
        manager.reset()
        step = 0

        while manager.game_state.name != "GAME_OVER":
            manager.step_once()
            step += 1

            if manager.game_state.name == "POINT_SCORED":
                if manager.one_robot_mode:
                    scored = manager.last_fault == "successful_return"
                    label = "Return" if scored else "Miss  "
                    print(f"  {label} ({manager.last_fault}) | Returns: {manager.scores[0]}")
                else:
                    print(
                        f"  Point! Fault: {manager.last_fault} | "
                        f"Score: {manager.scores[0]}-{manager.scores[1]}"
                    )
                manager.game_state = __import__(
                    "simulation_manager", fromlist=["GameState"]
                ).GameState.WAITING_SERVE
                manager.player1.reset()
                if manager.player2 is not None:
                    manager.player2.reset()
                env.reset()
                manager._reset_rally_state()

            # Render a frame every step_interval physics steps
            if step % step_interval == 0:
                view_mat = p.computeViewMatrixFromYawPitchRoll(
                    cameraTargetPosition=CAM_TARGET,
                    distance=CAM_DISTANCE,
                    yaw=CAM_YAW,
                    pitch=CAM_PITCH,
                    roll=0,
                    upAxisIndex=2,
                    physicsClientId=client,
                )
                proj_mat = p.computeProjectionMatrixFOV(
                    fov=60, aspect=IMG_W / IMG_H,
                    nearVal=0.1, farVal=20,
                    physicsClientId=client,
                )
                _, _, rgba, _, _ = p.getCameraImage(
                    IMG_W, IMG_H,
                    viewMatrix=view_mat,
                    projectionMatrix=proj_mat,
                    renderer=p.ER_TINY_RENDERER,
                    physicsClientId=client,
                )
                frame = np.array(rgba, dtype=np.uint8).reshape(IMG_H, IMG_W, 4)[:, :, :3]

                if img_handle is None:
                    img_handle = ax.imshow(frame)
                    title = ax.set_title("", color="white", fontsize=11)
                else:
                    img_handle.set_data(frame)

                if manager.one_robot_mode:
                    total_serves = manager.scores[0] + manager.scores[1]
                    title.set_text(
                        f"Episode {ep + 1}  |  "
                        f"Returns: {manager.scores[0]} / {total_serves}  |  "
                        f"Step {step}"
                    )
                else:
                    title.set_text(
                        f"Episode {ep + 1}  |  "
                        f"Score  P1: {manager.scores[0]}  P2: {manager.scores[1]}  |  "
                        f"Step {step}"
                    )
                plt.pause(0.001)

        winner = 0 if manager.scores[0] > manager.scores[1] else 1
        result = {"scores": list(manager.scores), "winner": winner,
                  "total_steps": manager.total_steps}
        results.append(result)
        if manager.one_robot_mode:
            total_serves = manager.scores[0] + manager.scores[1]
            print(
                f"Episode {ep + 1}: {manager.scores[0]} / {total_serves} returns "
                f"in {manager.total_steps} steps"
            )
        else:
            print(
                f"Episode {ep + 1}: Player {winner + 1} wins "
                f"{manager.scores[0]}-{manager.scores[1]} in {manager.total_steps} steps"
            )

    plt.ioff()
    plt.show()
    return results
