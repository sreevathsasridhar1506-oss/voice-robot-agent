"""
Standalone check (M1): does the MuJoCo Menagerie Franka Panda model load and step?

No agent, no voice, no viewer window here — just proves the MJCF/asset paths
are correct and the model is physically valid (mj_step doesn't NaN/crash).
"""

import mujoco
import numpy as np

SCENE_PATH = "assets_mujoco_menagerie/franka_emika_panda/scene.xml"


def main():
    print(f"Loading model from {SCENE_PATH} ...")
    model = mujoco.MjModel.from_xml_path(SCENE_PATH)
    data = mujoco.MjData(model)

    print(f"nq (position dims): {model.nq}")
    print(f"nu (actuators): {model.nu}")
    print(f"nbody: {model.nbody}")

    body_names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i) for i in range(model.nbody)]
    print(f"Bodies: {body_names}")

    actuator_names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i) for i in range(model.nu)]
    print(f"Actuators: {actuator_names}")

    print("Stepping physics 500 times...")
    for _ in range(500):
        mujoco.mj_step(model, data)

    if np.isnan(data.qpos).any():
        print("FAILED — qpos contains NaN after stepping.")
        return

    print(f"Final qpos: {np.round(data.qpos, 3)}")
    print("\nSUCCESS — model loads and simulates stably.")


if __name__ == "__main__":
    main()
