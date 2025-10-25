# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def arm_joint_pos(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
):
    robot = env.scene[asset_cfg.name]
    joint_names = [
        "panda_joint1",
        "panda_joint2",
        "panda_joint3",
        "panda_joint4",
        "panda_joint5",
        "panda_joint6",
        "panda_joint7",
    ]
    # get joint inidices
    joint_indices = [
        i for i, name in enumerate(robot.data.joint_names) if name in joint_names
    ]
    joint_pos = robot.data.joint_pos[0, joint_indices]
    return joint_pos


def gripper_pos(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
):
    robot = env.scene[asset_cfg.name]
    joint_names = ["finger_joint"]
    joint_indices = [
        i for i, name in enumerate(robot.data.joint_names) if name in joint_names
    ]
    joint_pos = robot.data.joint_pos[0, joint_indices]

    # rescale
    joint_pos = joint_pos / (torch.pi / 4)

    return joint_pos


def object_positions(
    env: ManagerBasedRLEnv,
    object_1_cfg: SceneEntityCfg = SceneEntityCfg("object_1"),
    object_2_cfg: SceneEntityCfg = SceneEntityCfg("object_2"),
) -> torch.Tensor:
    """Get the positions of the objects.

    Args:
        env: The RL environment instance.
        object_1_cfg: Configuration for object_1 (the object being placed).
        object_2_cfg: Configuration for object_2 (the target/container object).

    Returns:
        Tensor containing the positions of the objects.
    """

    # Get object entities from the scene
    object_1: RigidObject = env.scene[object_1_cfg.name]
    object_2: RigidObject = env.scene[object_2_cfg.name]

    # Get positions relative to environment origin
    object_1_pos = object_1.data.root_pos_w - env.scene.env_origins
    object_2_pos = object_2.data.root_pos_w - env.scene.env_origins

    print(f"Object 1 position: {object_1_pos}")
    print(f"Object 2 position: {object_2_pos}")

    # relative positions
    relative_x = torch.abs(object_1_pos[:, 0] - object_2_pos[:, 0])
    relative_y = torch.abs(object_1_pos[:, 1] - object_2_pos[:, 1])
    relative_z = object_1_pos[:, 2] - object_2_pos[:, 2]

    print(f"Relative x: {relative_x}")
    print(f"Relative y: {relative_y}")
    print(f"Relative z: {relative_z}")

    return torch.cat((object_1_pos, object_2_pos), dim=1)
