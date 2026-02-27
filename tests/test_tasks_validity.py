import unittest
from termcolor import colored
import traceback

from robocasa.utils.env_utils import create_env, run_random_rollouts

DEFAULT_SEED = 0
NUM_ROLLOUTS = 10
NUM_STEPS = 20

# Environments to test. Extend this list as new tasks are onboarded.
TARGET_ENVS = [
    "OpenSingleDoor",
    "CoffeePressButton",
    "CoffeeServeMug",
    "PnPSinkToCounter",
]


class TestTasksValidity(unittest.TestCase):
    def test_tasks_validity(self):
        """
        Tests that TARGET_ENVS run error-free. Creates each environment and runs
        NUM_ROLLOUTS test episodes per scene. Prints successful and failed tasks
        with their errors at the end.
        """

        successful = []
        unsucessful = []
        error_dict = {}

        for i, env_name in enumerate(TARGET_ENVS):
            print(
                colored(
                    f"Testing {env_name} environment [{i+1}/{len(TARGET_ENVS)}]...",
                    "green",
                )
            )

            completed = True
            try:
                env = create_env(env_name)
                run_random_rollouts(
                    env,
                    num_rollouts=NUM_ROLLOUTS,
                    num_steps=NUM_STEPS,
                    video_path="/tmp/test.mp4",
                )
            except KeyboardInterrupt:
                print(colored(f"Exiting Test Early.", "yellow"))
                break
            except Exception:
                print(colored(f"Test {env_name} Failed.", "red"))
                if env_name not in error_dict.keys():
                    error_dict[env_name] = []
                error_dict[env_name].append(traceback.format_exc())
                completed = False

            if completed:
                successful.append(env_name)
            else:
                unsucessful.append(env_name)

        print(
            colored(f"The following tests ran successfully:\n{successful}\n", "green")
        )
        print(colored(f"The following tests ran unsuccessfully:\n{unsucessful}", "red"))
        if error_dict:
            print(colored(f"With the following errors:", "red"))
            for key in error_dict.keys():
                print(colored(f"{key}:", "red"))
                for i, error in enumerate(error_dict[key]):
                    print(colored(f"Error #{i+1}:", "red"))
                    print(colored(f"{error}\n", "red"))


if __name__ == "__main__":
    unittest.main()
