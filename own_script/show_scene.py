# -*- coding: utf-8 -*-
'''
@Author: captain_hq
@Date: 2025-12-17 13:12:28
@LastEditors: captain_hq tuanzhang_hc5090@outlook.com
@LastEditTime: 2025-12-24 11:33:56
@FilePath: /IsaacLab/own_script/show_scene.py
@Description:
'''

"""Launch Isaac Sim Simulator first."""


import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(
    description="Tutorial on using the interactive scene interface."
)
parser.add_argument(
    "--num_envs", type=int, default=2, help="Number of environments to spawn."
)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import torch
from isaaclab_tasks.direct.franka_cabinet.franka_cabinet_env_own import FrankaCabinetEnv, FrankaCabinetEnvCfg
from isaaclab_tasks.direct.humanoid.humanoid_env import HumanoidEnv, HumanoidEnvCfg


def main():
    """Main function."""
    # parse the arguments
    env_cfg = HumanoidEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = args_cli.device
    # setup base environment
    env = HumanoidEnv(cfg=env_cfg)

    # simulate physics
    count = 0
    while simulation_app.is_running():
        with torch.inference_mode():
            # reset
            if count % 300 == 0:
                count = 0
                env.reset()
                print("-" * 80)
                print("[INFO]: Resetting environment...")
            # sample random actions
            # joint_pos = torch.randn(
            #     (env.num_envs, env.cfg.action_space), device=env.device
            # )
            joint_pos =env.robot.data.default_joint_pos
            # step the environment
            obs, rew, terminated, truncated, info = env.step(joint_pos)
            # print current orientation of pole
            print("[Env 0]: Pole joint: ", obs["policy"][0][1].item())
            # update counter
            count += 1

    # close the environment
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
