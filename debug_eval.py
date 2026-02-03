import tyro
import argparse
import gymnasium as gym
import torch
import cv2
import mediapy
import numpy as np
from datetime import datetime
from pathlib import Path
from tqdm import tqdm
from openpi_client import image_tools

from src.inference.droid_jointpos import Client as DroidJointPosClient


def main(
        episodes:int = 1,
        headless: bool = True,
        scene: int = 1,
        port: int = 8000,
        ):
    # launch omniverse app with arguments (inside function to prevent overriding tyro)
    from isaaclab.app import AppLauncher
    parser = argparse.ArgumentParser(description="Tutorial on creating an empty stage.")
    AppLauncher.add_app_launcher_args(parser)
    args_cli, _ = parser.parse_known_args()
    args_cli.enable_cameras = True
    args_cli.headless = headless
    app_launcher = AppLauncher(args_cli)
    simulation_app = app_launcher.app

    # All IsaacLab dependent modules should be imported after the app is launched
    import src.environments # noqa: F401
    from isaaclab_tasks.utils import parse_env_cfg


    # Initialize the env
    env_cfg = parse_env_cfg(
        "DROID",
        device=args_cli.device,
        num_envs=1,
        use_fabric=True,
    )
    instruction = None
    match scene:
        case 1:
            instruction = "put the cube in the bowl"
        case 2:
            instruction = "put the can in the mug"
        case 3:
            instruction = "put banana in the bin"
        case _:
            raise ValueError(f"Scene {scene} not supported")

    env_cfg.set_scene(scene)
    env = gym.make("DROID", cfg=env_cfg)

    obs, _ = env.reset()
    obs, _ = env.reset() # need second render cycle to get correctly loaded materials

    video_dir = Path("runs") / datetime.now().strftime("%Y-%m-%d") / datetime.now().strftime("%H-%M-%S")
    video_dir.mkdir(parents=True, exist_ok=True)
    video = []
    ep = 0
    max_steps = env.env.max_episode_length
    with torch.no_grad():
        for _ in tqdm(range(max_steps), desc=f"Episode {ep+1}/{episodes}"):

            right_image = obs["policy"]["external_cam"][0].clone().detach().cpu().numpy()
            wrist_image = obs["policy"]["wrist_cam"][0].clone().detach().cpu().numpy()

            img1 = image_tools.resize_with_pad(right_image, 224, 224)
            img2 = image_tools.resize_with_pad(wrist_image, 224, 224)

            video.append(np.concatenate([img1, img2], axis=1))
            action = torch.tensor([[0.0] * 8])
            obs, _, term, trunc, _ = env.step(action)
            if term or trunc:
                break

        mediapy.write_video(
            video_dir / f"episode_{ep}.mp4",
            video,
            fps=15,
        )
        video = []

    env.close()
    simulation_app.close()

if __name__ == "__main__":
    args = tyro.cli(main)
