# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import subtract_frame_transforms

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def object_position_in_robot_root_frame(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """The position of the object in the robot's root frame."""
    robot: RigidObject = env.scene[robot_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]
    object_pos_w = object.data.root_pos_w[:, :3]
    object_pos_b, _ = subtract_frame_transforms(robot.data.root_pos_w, robot.data.root_quat_w, object_pos_w)
    return object_pos_b


def franka_state(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """
    Franka robot state observation.
    
    Returns:
        torch.Tensor: State vector of shape (num_envs, 8) containing:
            - First 7 dimensions: Franka joint angles in radians
            - Last dimension: Gripper state binary value where -1.0 is closed, 1.0 is open
    """
    robot: Articulation = env.scene[robot_cfg.name]
    
    # Get joint positions for the 7 arm joints (panda_joint1 to panda_joint7)
    arm_joint_pos = robot.data.joint_pos[:, :7]  # First 7 joints are arm joints
    
    # Get gripper joint positions (panda_finger_joint1 and panda_finger_joint2)
    # Franka gripper has two finger joints, we'll use the average
    finger_joint_1 = robot.data.joint_pos[:, -2]  # panda_finger_joint1
    finger_joint_2 = robot.data.joint_pos[:, -1]  # panda_finger_joint2
    
    # Average the two finger joints
    # Franka gripper range is approximately [0, 0.04] meters
    # 0.0 means fully closed, 0.04 means fully open
    gripper_avg = (finger_joint_1 + finger_joint_2) / 2.0
    
    # Binary classification: -1.0 for open (> 0.02), 1.0 for closed (<= 0.02)
    # Using 0.02 as threshold (half of max opening)
    gripper_binary = torch.where(gripper_avg > 0.035, 1.0, -1.0)
    
    # Combine arm joints and gripper state
    state = torch.cat([arm_joint_pos, gripper_binary.unsqueeze(-1)], dim=-1)
    
    return state
