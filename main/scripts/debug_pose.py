from __future__ import annotations

import argparse
from pathlib import Path
import sys

import mujoco
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rl.env import DEFAULT_STAND, JOINT_NAMES


def run_debug(model_path: Path, sim_time: float, control_dt: float) -> None:
    model = mujoco.MjModel.from_xml_path(str(model_path))
    data = mujoco.MjData(model)
    n_substeps = max(1, int(round(control_dt / model.opt.timestep)))

    joint_qpos_adr = np.array(
        [model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)] for name in JOINT_NAMES]
    )
    joint_qvel_adr = np.array(
        [model.jnt_dofadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)] for name in JOINT_NAMES]
    )

    data.qpos[:] = 0.0
    data.qvel[:] = 0.0
    data.qpos[0:3] = np.array([0.0, 0.0, 0.22], dtype=np.float64)
    data.qpos[3:7] = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    data.qpos[joint_qpos_adr] = DEFAULT_STAND
    data.ctrl[:] = DEFAULT_STAND
    mujoco.mj_forward(model, data)

    n_steps = int(sim_time / control_dt)
    print(f"Loaded model: {model_path}")
    print(f"nq={model.nq}, nv={model.nv}, nu={model.nu}, dt={model.opt.timestep}")
    print(f"Running {sim_time:.2f}s debug rollout with stand pose.")

    for step in range(n_steps):
        data.ctrl[:] = DEFAULT_STAND
        for _ in range(n_substeps):
            mujoco.mj_step(model, data)
        if step % max(1, int(0.2 / control_dt)) == 0 or step == n_steps - 1:
            print(
                "t={:.2f}s z={:.3f} roll/pitch_rate=({:.3f}, {:.3f}) xvel={:.3f} max_joint_vel={:.3f}".format(
                    step * control_dt,
                    float(data.qpos[2]),
                    float(data.qvel[3]),
                    float(data.qvel[4]),
                    float(data.qvel[0]),
                    float(np.max(np.abs(data.qvel[joint_qvel_adr]))),
                )
            )

    print("Final base position:", np.array2string(data.qpos[0:3], precision=3))
    print("Final joint positions:", np.array2string(data.qpos[joint_qpos_adr], precision=3))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "models" / "pupper_like.xml",
    )
    parser.add_argument("--sim-time", type=float, default=3.0)
    parser.add_argument("--control-dt", type=float, default=0.02)
    args = parser.parse_args()
    run_debug(args.model, args.sim_time, args.control_dt)


if __name__ == "__main__":
    main()
