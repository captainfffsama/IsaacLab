# -*- coding: utf-8 -*-
'''
@Author: captain_hq
@Date: 2025-12-18 13:29:07
@LastEditors: captain_hq tuanzhang_hc5090@outlook.com
@LastEditTime: 2025-12-18 13:40:27
@FilePath: /IsaacLab/source/isaaclab_tasks/isaaclab_tasks/direct/franka_cabinet/franka_cabinet_env_own.py
@Description:
'''

from __future__ import annotations

import torch
from  collections.abc import Sequence
from isaacsim.core.utils.torch.transformations import tf_combine, tf_inverse, tf_vector
from pxr import UsdGeom

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.sim.utils.stage import get_current_stage
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR
from isaaclab.utils.math import sample_uniform



@configclass
class FrankaCabinetEnvCfg(DirectRLEnvCfg):
    # env
    episode_length_s = 8.3333  # 500 timesteps
    decimation = 2
    action_space = 9
    observation_space = 23
    state_space = 0

    # simulation
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 120,
        render_interval=decimation,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
    )

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=4096, env_spacing=3.0, replicate_physics=True, clone_in_fabric=True
    )

    # robot
    robot = ArticulationCfg(
        prim_path="/World/envs/env_.*/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ISAACLAB_NUCLEUS_DIR}/Robots/FrankaEmika/panda_instanceable.usd",
            activate_contact_sensors=False,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                max_depenetration_velocity=5.0,# 穿模速度
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False, solver_position_iteration_count=12, solver_velocity_iteration_count=1
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            joint_pos={
                "panda_joint1": 1.157,
                "panda_joint2": -1.066,
                "panda_joint3": -0.155,
                "panda_joint4": -2.239,
                "panda_joint5": -1.841,
                "panda_joint6": 1.003,
                "panda_joint7": 0.469, #法兰位置
                "panda_finger_joint.*": 0.035,
            },
            pos=(1.0, 0.0, 0.0),
            rot=(0.0, 0.0, 0.0, 1.0),
        ),
        actuators={
            "panda_shoulder": ImplicitActuatorCfg(
                joint_names_expr=["panda_joint[1-4]"],
                effort_limit_sim=87.0, #最大输出扭矩nm
                stiffness=80.0,
                damping=4.0,
            ),
            "panda_forearm": ImplicitActuatorCfg(
                joint_names_expr=["panda_joint[5-7]"],
                effort_limit_sim=12.0,
                stiffness=80.0,
                damping=4.0,
            ),
            "panda_hand": ImplicitActuatorCfg(
                joint_names_expr=["panda_finger_joint.*"],
                effort_limit_sim=200.0,
                stiffness=2e3,
                damping=1e2,
            ),
        },
    )

    # cabinet
    cabinet = ArticulationCfg(
        prim_path="/World/envs/env_.*/Cabinet",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Sektion_Cabinet/sektion_cabinet_instanceable.usd",
            activate_contact_sensors=False,
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0, 0.4),
            rot=(0.1, 0.0, 0.0, 0.0),
            joint_pos={
                "door_left_joint": 0.0,
                "door_right_joint": 0.0,
                "drawer_bottom_joint": 0.0,
                "drawer_top_joint": 0.0,
            },
        ),
        actuators={
            "drawers": ImplicitActuatorCfg(
                joint_names_expr=["drawer_top_joint", "drawer_bottom_joint"],
                effort_limit_sim=87.0,
                stiffness=10.0,
                damping=1.0,
            ),
            "doors": ImplicitActuatorCfg(
                joint_names_expr=["door_left_joint", "door_right_joint"],
                effort_limit_sim=87.0,
                stiffness=10.0,
                damping=2.5,
            ),
        },
    )

    # ground plane
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1, # 定义碰撞组，这里是会和所有的物体碰撞
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",# 两个物体的摩擦力结合模式为乘法
            restitution_combine_mode="multiply",# 两个物体的弹性系数结合模式为乘法，反弹的方式
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
    )

    action_scale = 7.5
    dof_velocity_scale = 0.1

    # reward scales
    dist_reward_scale = 1.5
    rot_reward_scale = 1.5
    open_reward_scale = 10.0
    action_penalty_scale = 0.05
    finger_reward_scale = 2.0


class FrankaCabinetEnv(DirectRLEnv):
    # pre-physics step calls
    #   |-- _pre_physics_step(action)
    #   |-- _apply_action()
    # post-physics step calls
    #   |-- _get_dones()
    #   |-- _get_rewards()
    #   |-- _reset_idx(env_ids)
    #   |-- _get_observations()

    cfg: FrankaCabinetEnvCfg

    def __init__(self, cfg: FrankaCabinetEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        def get_env_local_pose(env_pos: torch.Tensor, xformable: UsdGeom.Xformable, device: str|torch.device):
            """Compute pose in env-local coordinates"""
            world_transform = xformable.ComputeLocalToWorldTransform(0)
            world_pos = world_transform.ExtractTranslation()
            world_quat = world_transform.ExtractRotationQuat()

            px = world_pos[0] - env_pos[0]
            py = world_pos[1] - env_pos[1]
            pz = world_pos[2] - env_pos[2]
            qx = world_quat.imaginary[0]
            qy = world_quat.imaginary[1]
            qz = world_quat.imaginary[2]
            qw = world_quat.real

            return torch.tensor([px, py, pz, qw, qx, qy, qz], device=device)

        # 计算控制时间步长
        self.dt = self.cfg.sim.dt * self.cfg.decimation

        # 创建用于计算应用动作、观测和奖励的辅助变量
        self.robot_dof_lower_limits = self._robot.data.soft_joint_pos_limits[0, :, 0].to(device=self.device)
        self.robot_dof_upper_limits = self._robot.data.soft_joint_pos_limits[0, :, 1].to(device=self.device)

        self.robot_dof_speed_scales = torch.ones_like(self.robot_dof_lower_limits)
        self.robot_dof_speed_scales[self._robot.find_joints("panda_finger_joint1")[0]] = 0.1
        self.robot_dof_speed_scales[self._robot.find_joints("panda_finger_joint2")[0]] = 0.1

        self.robot_dof_targets = torch.zeros((self.num_envs, self._robot.num_joints), device=self.device)

        stage = get_current_stage()

        #获取env0下的手，手指环境坐标系
        #NOTE: 这和b里的坐标的关系？
        hand_pose = get_env_local_pose(
            self.scene.env_origins[0],
            UsdGeom.Xformable(stage.GetPrimAtPath("/World/envs/env_0/Robot/panda_link7")),
            self.device,
        )
        lfinger_pose = get_env_local_pose(
            self.scene.env_origins[0],
            UsdGeom.Xformable(stage.GetPrimAtPath("/World/envs/env_0/Robot/panda_leftfinger")),
            self.device,
        )
        rfinger_pose = get_env_local_pose(
            self.scene.env_origins[0],
            UsdGeom.Xformable(stage.GetPrimAtPath("/World/envs/env_0/Robot/panda_rightfinger")),
            self.device,
        )

        finger_pose = torch.zeros(7, device=self.device)
        finger_pose[0:3] = (lfinger_pose[0:3] + rfinger_pose[0:3]) / 2.0
        finger_pose[3:7] = lfinger_pose[3:7]
        # 用于计算手腕到env的坐标变换
        hand_pose_inv_rot, hand_pose_inv_pos = tf_inverse(hand_pose[3:7], hand_pose[0:3])

        # 计算手指相对手腕的变换
        robot_local_grasp_pose_rot, robot_local_pose_pos = tf_combine(
            hand_pose_inv_rot, hand_pose_inv_pos, finger_pose[3:7], finger_pose[0:3]
        )
        robot_local_pose_pos += torch.tensor([0, 0.04, 0], device=self.device)
        self.robot_local_grasp_pos = robot_local_pose_pos.repeat((self.num_envs, 1))
        self.robot_local_grasp_rot = robot_local_grasp_pose_rot.repeat((self.num_envs, 1))

        #柜子抽屉把手的抓取点相对于抽屉的局部坐标系位置
        drawer_local_grasp_pose = torch.tensor([0.3, 0.01, 0.0, 1.0, 0.0, 0.0, 0.0], device=self.device)
        self.drawer_local_grasp_pos = drawer_local_grasp_pose[0:3].repeat((self.num_envs, 1))
        self.drawer_local_grasp_rot = drawer_local_grasp_pose[3:7].repeat((self.num_envs, 1))

        self.gripper_forward_axis = torch.tensor([0, 0, 1], device=self.device, dtype=torch.float32).repeat(
            (self.num_envs, 1)
        )
        self.drawer_inward_axis = torch.tensor([-1, 0, 0], device=self.device, dtype=torch.float32).repeat(
            (self.num_envs, 1)
        )
        self.gripper_up_axis = torch.tensor([0, 1, 0], device=self.device, dtype=torch.float32).repeat(
            (self.num_envs, 1)
        )
        self.drawer_up_axis = torch.tensor([0, 0, 1], device=self.device, dtype=torch.float32).repeat(
            (self.num_envs, 1)
        )

        self.hand_link_idx = self._robot.find_bodies("panda_link7")[0][0]
        self.left_finger_link_idx = self._robot.find_bodies("panda_leftfinger")[0][0]
        self.right_finger_link_idx = self._robot.find_bodies("panda_rightfinger")[0][0]
        self.drawer_link_idx = self._cabinet.find_bodies("drawer_top")[0][0]

        self.robot_grasp_rot = torch.zeros((self.num_envs, 4), device=self.device)
        self.robot_grasp_pos = torch.zeros((self.num_envs, 3), device=self.device)
        self.drawer_grasp_rot = torch.zeros((self.num_envs, 4), device=self.device)
        self.drawer_grasp_pos = torch.zeros((self.num_envs, 3), device=self.device)

    def _setup_scene(self):
        self._robot = Articulation(self.cfg.robot)
        self._cabinet = Articulation(self.cfg.cabinet)
        self.scene.articulations["robot"] = self._robot
        self.scene.articulations["cabinet"] = self._cabinet

        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)

        # clone and replicate
        self.scene.clone_environments(copy_from_source=False)
        # we need to explicitly filter collisions for CPU simulation
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[self.cfg.terrain.prim_path])

        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    # pre-physics step calls

    def _pre_physics_step(self, actions: torch.Tensor):
        self.actions = actions.clone().clamp(-1.0, 1.0)
        targets = self.robot_dof_targets + self.robot_dof_speed_scales * self.dt * self.actions * self.cfg.action_scale
        # #DEBUG:
        # self._targets_debug=targets.clone()
        self.robot_dof_targets[:] = torch.clamp(targets, self.robot_dof_lower_limits, self.robot_dof_upper_limits)
        print("self.robot_dof_targets:", self.robot_dof_targets)

    def _apply_action(self):
        self._robot.set_joint_position_target(self.robot_dof_targets)
        # pass

    # post-physics step calls

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        terminated = self._cabinet.data.joint_pos[:, 3] > 0.39
        truncated = self.episode_length_buf >= self.max_episode_length - 1
        return terminated, truncated

    def _get_rewards(self) -> torch.Tensor:
        # Refresh the intermediate values after the physics steps
        self._compute_intermediate_values()
        robot_left_finger_pos = self._robot.data.body_pos_w[:, self.left_finger_link_idx]
        robot_right_finger_pos = self._robot.data.body_pos_w[:, self.right_finger_link_idx]

        return self._compute_rewards(
            self.actions,
            self._cabinet.data.joint_pos,
            self.robot_grasp_pos,
            self.drawer_grasp_pos,
            self.robot_grasp_rot,
            self.drawer_grasp_rot,
            robot_left_finger_pos,
            robot_right_finger_pos,
            self.gripper_forward_axis,
            self.drawer_inward_axis,
            self.gripper_up_axis,
            self.drawer_up_axis,
            self.num_envs,
            self.cfg.dist_reward_scale,
            self.cfg.rot_reward_scale,
            self.cfg.open_reward_scale,
            self.cfg.action_penalty_scale,
            self.cfg.finger_reward_scale,
            self._robot.data.joint_pos,
        )


    def _reset_idx(self, env_ids: Sequence[int]):
        super()._reset_idx(env_ids)
        # robot state
        joint_pos = self._robot.data.default_joint_pos[env_ids] + sample_uniform(
            -0.125,
            0.125,
            (len(env_ids), self._robot.num_joints),
            self.device,
        )
        #DEBUG:
        # print("default_joint_pos:", self._robot.data.default_joint_pos[env_ids])
        # joint_pos = self._robot.data.default_joint_pos[env_ids]

        joint_pos = torch.clamp(joint_pos, self.robot_dof_lower_limits, self.robot_dof_upper_limits)
        joint_vel = torch.zeros_like(joint_pos)
        self._robot.set_joint_position_target(joint_pos, env_ids=env_ids)
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)

        # cabinet state
        zeros = torch.zeros((len(env_ids), self._cabinet.num_joints), device=self.device)
        self._cabinet.write_joint_state_to_sim(zeros, zeros, env_ids=env_ids)

        # Need to refresh the intermediate values so that _get_observations() can use the latest values
        self._compute_intermediate_values(env_ids)


    def _get_observations(self) -> dict:
        # NOTE: 这里关节做了归一化处理
        dof_pos_scaled = (
            2.0
            * (self._robot.data.joint_pos - self.robot_dof_lower_limits)
            / (self.robot_dof_upper_limits - self.robot_dof_lower_limits)
            - 1.0
        )
        to_target = self.drawer_grasp_pos - self.robot_grasp_pos

        obs = torch.cat(
            (
                dof_pos_scaled,
                self._robot.data.joint_vel * self.cfg.dof_velocity_scale,
                to_target,
                self._cabinet.data.joint_pos[:, 3].unsqueeze(-1),
                self._cabinet.data.joint_vel[:, 3].unsqueeze(-1),
            ),
            dim=-1,
        )
        return {"policy": torch.clamp(obs, -5.0, 5.0)}

    # auxiliary methods

    def _compute_intermediate_values(self, env_ids: torch.Tensor | None = None):
        """计算世界坐标系下抓取点位置和指尖位置

        Args:
            env_ids (torch.Tensor | None, optional): _description_. Defaults to None.
        """
        if env_ids is None:
            env_ids = self._robot._ALL_INDICES

        hand_pos = self._robot.data.body_pos_w[env_ids, self.hand_link_idx]
        hand_rot = self._robot.data.body_quat_w[env_ids, self.hand_link_idx]
        drawer_pos = self._cabinet.data.body_pos_w[env_ids, self.drawer_link_idx]
        drawer_rot = self._cabinet.data.body_quat_w[env_ids, self.drawer_link_idx]
        (
            self.robot_grasp_rot[env_ids],
            self.robot_grasp_pos[env_ids],
            self.drawer_grasp_rot[env_ids],
            self.drawer_grasp_pos[env_ids],
        ) = self._compute_grasp_transforms(
            hand_rot,
            hand_pos,
            self.robot_local_grasp_rot[env_ids],
            self.robot_local_grasp_pos[env_ids],
            drawer_rot,
            drawer_pos,
            self.drawer_local_grasp_rot[env_ids],
            self.drawer_local_grasp_pos[env_ids],
        )

        #  # 调试代码
        # if env_ids is None or 0 in env_ids:
        #     left_pos = self._robot.data.body_pos_w[0, self.left_finger_link_idx]
        #     right_pos = self._robot.data.body_pos_w[0, self.right_finger_link_idx]
        #     finger_avg = (left_pos + right_pos) / 2

        #     print(f"手指平均位置: {finger_avg}")
        #     print(f"理想抓取点:   {self.robot_grasp_pos[0]}")
        #     print(f"差异:        {self.robot_grasp_pos[0] - finger_avg}")

    def _compute_rewards(
        self,
        actions,
        cabinet_dof_pos,
        franka_grasp_pos,
        drawer_grasp_pos,
        franka_grasp_rot,
        drawer_grasp_rot,
        franka_lfinger_pos,
        franka_rfinger_pos,
        gripper_forward_axis,
        drawer_inward_axis,
        gripper_up_axis,
        drawer_up_axis,
        num_envs,
        dist_reward_scale,
        rot_reward_scale,
        open_reward_scale,
        action_penalty_scale,
        finger_reward_scale,
        joint_positions,
    ):
        """_summary_

        Args:
            actions (_type_): _description_
            cabinet_dof_pos (_type_): 柜子关节位置
            franka_grasp_pos (torch.Tensor): 指尖在w下位置
            drawer_grasp_pos (torch.Tensor): 把手在w下位置
            franka_grasp_rot (torch.Tensor): _description_
            drawer_grasp_rot (torch.Tensor): _description_
            franka_lfinger_pos (torch.Tensor): 左指几何中心在w下位置
            franka_rfinger_pos (torch.Tensor): 右指几何中心在w下位置
            gripper_forward_axis (torch.Tensor): 夹爪朝前轴
            drawer_inward_axis (torch.Tensor): 抽屉朝前轴
            gripper_up_axis (torch.Tensor): 夹爪朝上轴
            drawer_up_axis (torch.Tensor): 抽屉朝上轴
            num_envs (_type_): _description_
            dist_reward_scale (_type_): _description_
            rot_reward_scale (_type_): _description_
            open_reward_scale (_type_): _description_
            action_penalty_scale (_type_): _description_
            finger_reward_scale (_type_): _description_
            joint_positions (_type_): _description_

        Returns:
            _type_: _description_
        """
        #NOTE: 无存活reward
        # distance from hand to the drawer
        d = torch.norm(franka_grasp_pos - drawer_grasp_pos, p=2, dim=-1)
        dist_reward = 1.0 / (1.0 + d**2)
        dist_reward *= dist_reward
        #NOTE: 改分段之后，让整体的函数距离近0时更加平滑，但是实际两者在0.02处差异很小，reward差异仅仅少了0.001
        # 距离在0.2~1.6之间成45度斜率的线性递减
        dist_reward = torch.where(d <= 0.02, dist_reward * 2, dist_reward)

        # 全转到世界坐标下
        axis1 = tf_vector(franka_grasp_rot, gripper_forward_axis)
        axis2 = tf_vector(drawer_grasp_rot, drawer_inward_axis)
        axis3 = tf_vector(franka_grasp_rot, gripper_up_axis)
        axis4 = tf_vector(drawer_grasp_rot, drawer_up_axis)

        dot1 = (
            torch.bmm(axis1.view(num_envs, 1, 3), axis2.view(num_envs, 3, 1)).squeeze(-1).squeeze(-1)
        )  # alignment of forward axis for gripper
        dot2 = (
            torch.bmm(axis3.view(num_envs, 1, 3), axis4.view(num_envs, 3, 1)).squeeze(-1).squeeze(-1)
        )  # alignment of up axis for gripper
        # reward for matching the orientation of the hand to the drawer (fingers wrapped)
        rot_reward = 0.5 * (torch.sign(dot1) * dot1**2 + torch.sign(dot2) * dot2**2)

        # regularization on the actions (summed for each environment)
        action_penalty = torch.sum(actions**2, dim=-1)

        # how far the cabinet has been opened out
        open_reward = cabinet_dof_pos[:, 3]  # drawer_top_joint

        # penalty for distance of each finger from the drawer handle
        lfinger_dist = franka_lfinger_pos[:, 2] - drawer_grasp_pos[:, 2]
        rfinger_dist = drawer_grasp_pos[:, 2] - franka_rfinger_pos[:, 2]
        finger_dist_penalty = torch.zeros_like(lfinger_dist)
        finger_dist_penalty += torch.where(lfinger_dist < 0, lfinger_dist, torch.zeros_like(lfinger_dist))
        finger_dist_penalty += torch.where(rfinger_dist < 0, rfinger_dist, torch.zeros_like(rfinger_dist))

        ## reward scales
        #dist_reward_scale = 1.5
        #rot_reward_scale = 1.5
        #open_reward_scale = 10.0
        #action_penalty_scale = 0.05
        #finger_reward_scale = 2.0
        ###################
        rewards = (
            dist_reward_scale * dist_reward
            + rot_reward_scale * rot_reward
            + open_reward_scale * open_reward
            + finger_reward_scale * finger_dist_penalty
            - action_penalty_scale * action_penalty
        )

        self.extras["log"] = {
            "dist_reward": (dist_reward_scale * dist_reward).mean(),
            "rot_reward": (rot_reward_scale * rot_reward).mean(),
            "open_reward": (open_reward_scale * open_reward).mean(),
            "action_penalty": (-action_penalty_scale * action_penalty).mean(),
            "left_finger_distance_reward": (finger_reward_scale * lfinger_dist).mean(),
            "right_finger_distance_reward": (finger_reward_scale * rfinger_dist).mean(),
            "finger_dist_penalty": (finger_reward_scale * finger_dist_penalty).mean(),
        }

        # bonus for opening drawer properly
        rewards = torch.where(cabinet_dof_pos[:, 3] > 0.01, rewards + 0.25, rewards)
        rewards = torch.where(cabinet_dof_pos[:, 3] > 0.2, rewards + 0.25, rewards)
        rewards = torch.where(cabinet_dof_pos[:, 3] > 0.35, rewards + 0.25, rewards)

        return rewards

    def _compute_grasp_transforms(
        self,
        hand_rot, # 世界坐标系下手腕旋转四元数
        hand_pos,
        franka_local_grasp_rot, # 手腕坐标系下手指
        franka_local_grasp_pos,
        drawer_rot,
        drawer_pos,
        drawer_local_grasp_rot,
        drawer_local_grasp_pos,
    ):
        """
        计算世界坐标系下的指尖位置和抓取位置
        """
        global_franka_rot, global_franka_pos = tf_combine(
            hand_rot, hand_pos, franka_local_grasp_rot, franka_local_grasp_pos
        )
        global_drawer_rot, global_drawer_pos = tf_combine(
            drawer_rot, drawer_pos, drawer_local_grasp_rot, drawer_local_grasp_pos
        )

        return global_franka_rot, global_franka_pos, global_drawer_rot, global_drawer_pos



