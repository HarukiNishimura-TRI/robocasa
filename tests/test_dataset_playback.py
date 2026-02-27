import unittest
from termcolor import colored
import argparse
import os

from robocasa.scripts.playback_dataset import playback_dataset
from robocasa.utils.dataset_registry import get_ds_path

# Environments to test. Extend this list as new tasks are onboarded.
TARGET_ENVS = [
    "OpenSingleDoor",
    "CoffeePressButton",
    "CoffeeServeMug",
    "PnPSinkToCounter",
]


class TestDatasetPlayback(unittest.TestCase):
    def test_dataset_playback(self):
        """
        Tests dataset playback for TARGET_ENVS. For each task, looks up the
        human_raw dataset path, plays back 5 episodes using action replay,
        and saves a video to ~/tmp/playback_videos/.
        """

        for task_i, task in enumerate(TARGET_ENVS):
            human_path = get_ds_path(
                task=task, ds_type="human_im"
            )  # human image dataset path
            print(f"Dataset path: {human_path}")
            print(
                colored(
                    f"Playing back {task} environment [{task_i+1}/{len(TARGET_ENVS)}]...",
                    "green",
                )
            )

            args = argparse.ArgumentParser()
            args.dataset = human_path
            args.filter_key = None
            args.n = 5
            args.use_obs = False
            args.use_actions = True
            args.use_abs_actions = False
            args.render = False
            video_folder = os.path.expanduser("~/tmp/playback_videos")
            if os.path.exists(video_folder) is False:
                os.makedirs(video_folder)
            args.video_path = os.path.join(video_folder, f"{task}.mp4")
            args.video_skip = 5
            args.render_image_names = [
                "robot0_agentview_left",
                "robot0_agentview_right",
                "robot0_eye_in_hand",
            ]
            args.first = False
            args.extend_states = False
            args.verbose = False
            args.camera_height = 224
            args.camera_width = 224

            playback_dataset(args)


if __name__ == "__main__":
    unittest.main()
