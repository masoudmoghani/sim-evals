# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Common functions that can be used to activate certain terminations for the lift task.

The functions can be passed to the :class:`isaaclab.managers.TerminationTermCfg` object to enable
the termination introduced by the function.
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def task_done(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_1_cfg: SceneEntityCfg = SceneEntityCfg("object_1"),
    object_2_cfg: SceneEntityCfg = SceneEntityCfg("object_2"),
    max_relative_x: float = 0.05,
    max_relative_y: float = 0.05,
    min_relative_z: float = -0.07,
    max_relative_z: float = 0.03,
) -> torch.Tensor:
    """Determine if the placement task is complete.

    This function checks whether object_1 is placed correctly into/onto object_2
    by verifying that object_1 is within the specified position bounds relative to object_2.

    Note on Z-axis convention:
        relative_z = object_1_z - object_2_z
        - Positive relative_z: object_1 is ABOVE object_2
        - Negative relative_z: object_1 is BELOW object_2
        - To ensure object_1 is below object_2, set max_relative_z <= 0.0

    Args:
        env: The RL environment instance.
        object_1_cfg: Configuration for object_1 (the object being placed).
        object_2_cfg: Configuration for object_2 (the target/container object).
        max_relative_x: Maximum absolute x distance between object_1 and object_2 for task completion.
        max_relative_y: Maximum absolute y distance between object_1 and object_2 for task completion.
        min_relative_z: Minimum z distance of object_1 relative to object_2 (prevents too far below).
        max_relative_z: Maximum z distance of object_1 relative to object_2 (set to 0.0 to require below).

    Returns:
        Boolean tensor indicating which environments have completed the task.
    """

    # Access the current scene name
    scene_name = env.cfg.scene_name

    # Modify the termination condition based on the scene name
    if scene_name == 1:
        max_relative_x = 0.03
        max_relative_y = 0.03
        min_relative_z = -0.01
        max_relative_z = 0.03
    elif scene_name == 2:
        max_relative_x = 0.01
        max_relative_y = 0.01
        min_relative_z = -0.01
        max_relative_z = 0.085
    elif scene_name == 3:
        max_relative_x = 0.055
        max_relative_y = 0.055
        min_relative_z = -0.075
        max_relative_z = 0.01
    else:
        raise ValueError(f"Invalid scene name: {scene_name}")

    # Get object entities from the scene
    robot = env.scene[robot_cfg.name]
    object_1: RigidObject = env.scene[object_1_cfg.name]
    object_2: RigidObject = env.scene[object_2_cfg.name]

    # Get positions relative to environment origin
    object_1_pos = object_1.data.root_pos_w - env.scene.env_origins
    object_2_pos = object_2.data.root_pos_w - env.scene.env_origins

    # relative positions
    relative_x = torch.abs(object_1_pos[:, 0] - object_2_pos[:, 0])
    relative_y = torch.abs(object_1_pos[:, 1] - object_2_pos[:, 1])
    relative_z = object_1_pos[:, 2] - object_2_pos[:, 2]

    # check gripper open
    joint_names = ["finger_joint"]
    joint_indices = [
        i for i, name in enumerate(robot.data.joint_names) if name in joint_names
    ]
    joint_pos = robot.data.joint_pos[0, joint_indices]

    # rescale
    joint_pos = joint_pos / (torch.pi / 4)
    
    gripper_open = joint_pos < 0.01

    done = relative_x < max_relative_x
    done = torch.logical_and(done, relative_y < max_relative_y)
    done = torch.logical_and(done, relative_z > min_relative_z)
    done = torch.logical_and(done, relative_z < max_relative_z)
    done = torch.logical_and(done, gripper_open)

    return done

def contact(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("contact"),
    threshold: float = 0.05,
) -> torch.Tensor:
    """Terminate when contact force exceeds threshold.
    
    This termination checks if the contact sensor detects any forces above the specified
    threshold.
    
    Args:
        env: The RL environment instance.
        asset_cfg: Configuration for the contact sensor entity in the scene.
        threshold: Minimum force magnitude (in Newtons) to register as contact.
    
    Returns:
        Boolean tensor of shape (num_envs,) indicating which environments should terminate
        due to contact detection.
    """
    # Get contact force matrix from sensor
    # Shape: (num_envs, num_bodies, num_filter_bodies, 3) or (num_envs, num_bodies, 3)
    force_matrix = env.scene[asset_cfg.name].data.force_matrix_w
    
    # Compute force magnitude for each contact point
    # Result shape: (num_envs, num_bodies, [num_filter_bodies])
    force_magnitude = torch.norm(force_matrix, dim=-1)
    
    # Find maximum force across all bodies and filter bodies for each environment
    # Shape: (num_envs,)
    max_force_per_env = force_magnitude.amax(dim=tuple(range(1, force_magnitude.ndim)))
    
    # Check if any force exceeds threshold
    has_contact = max_force_per_env > threshold

    print(f"Contact detected: {has_contact}")
    print(f"Max force magnitude: {max_force_per_env.max()}")
    
    return has_contact
