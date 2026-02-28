"""Test which layout_and_style_ids combinations work for each RoboCasa task.

RoboCasa kitchen layouts place fixtures on different walls, changing joint
names (e.g. ``microwave_main_group_microjoint`` vs
``microwave_front_group_microjoint``).  Not all 10 layouts x 12 styles = 120
combinations are compatible with every task.  This script systematically tests
each combination by calling ``env.reset()`` and reports which ones succeed.

Requires the ``use-robocasa`` virtual environment.

Example usage::

    # Quick test: 1 task, 1 seed, subset of layouts
    python /workspace/robocasa/tests/test_layout_compatibility.py \
        --tasks OpenSingleDoor --num_seeds 1 --layout_ids 0 1 2

    # Full run with YAML output
    python /workspace/robocasa/tests/test_layout_compatibility.py \
        --output_path /workspace/robocasa/tests/robocasa_layout_compatibility.yaml
"""

import argparse
import traceback
import yaml
import numpy as np

import robosuite
import robocasa  # registers RoboCasa envs


TARGET_TASKS = [
    "OpenSingleDoor",
    "CoffeePressButton",
    "CoffeeServeMug",
    "PnPSinkToCounter",
]

NUM_LAYOUTS = 10   # layout IDs 0-9
NUM_STYLES = 12    # style IDs 0-11

LAYOUT_NAMES = [
    "ONE_WALL_SMALL",
    "ONE_WALL_LARGE",
    "L_SHAPED_SMALL",
    "L_SHAPED_LARGE",
    "GALLEY",
    "U_SHAPED_SMALL",
    "U_SHAPED_LARGE",
    "G_SHAPED_SMALL",
    "G_SHAPED_LARGE",
    "WRAPAROUND",
]


def create_env(task_name, layout_and_style_ids, seed):
    """Create a RoboCasa env matching RobocasaImageRunner's configuration.

    Uses the same robot (PandaMobile), controller (None → default composite),
    and obj_instance_split ("B") as ``create_robocasa_env()`` in
    ``robocasa_image_runner.py``, but with rendering disabled to save GPU memory.
    """
    return robosuite.make(
        env_name=task_name,
        robots="PandaMobile",
        controller_configs=None,
        camera_names=[
            "robot0_agentview_left",
            "robot0_agentview_right",
            "robot0_eye_in_hand",
        ],
        camera_widths=128,
        camera_heights=128,
        has_renderer=False,
        has_offscreen_renderer=False,
        ignore_done=True,
        use_object_obs=True,
        use_camera_obs=False,
        camera_depths=False,
        seed=seed,
        obj_instance_split="B",
        translucent_robot=False,
        layout_and_style_ids=layout_and_style_ids,
    )


def test_combination(task_name, layout_id, style_id, num_seeds, start_seed):
    """Test a single task x layout x style combination across multiple seeds.

    Matches the RobocasaImageRunner reset flow: env is created with seed=0,
    then each reset is preceded by ``np.random.seed(seed)`` (as done by
    ``RobocasaImageWrapper.reset()`` for seed-based test rollouts).

    Returns (num_failed, first_error_message).
    """
    num_failed = 0
    first_error = None

    for seed in range(start_seed, start_seed + num_seeds):
        env = None
        try:
            env = create_env(
                task_name,
                layout_and_style_ids=[(layout_id, style_id)],
                seed=0,
            )
            np.random.seed(seed=seed)
            env.reset()
            env.close()
        except Exception:
            num_failed += 1
            if first_error is None:
                # Keep only the last line of the traceback (the error message)
                first_error = traceback.format_exc().strip().split("\n")[-1]
            try:
                if env is not None:
                    env.close()
            except Exception:
                pass

    return num_failed, first_error


def main():
    parser = argparse.ArgumentParser(
        description="Test RoboCasa layout/style compatibility for target tasks."
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=TARGET_TASKS,
        help="Task names to test (default: all 4 target tasks)",
    )
    parser.add_argument(
        "--num_seeds",
        type=int,
        default=3,
        help="Number of seeds to test per combination (default: 3)",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=None,
        help="Optional path to write YAML results",
    )
    parser.add_argument(
        "--layout_ids",
        nargs="+",
        type=int,
        default=None,
        help="Subset of layout IDs to test (default: all 0-9)",
    )
    parser.add_argument(
        "--style_ids",
        nargs="+",
        type=int,
        default=None,
        help="Subset of style IDs to test (default: all 0-11)",
    )
    parser.add_argument(
        "--start_seed",
        type=int,
        default=10000,
        help="First seed to test (default: 10000, matching RobocasaImageRunner test_start_seed)",
    )
    args = parser.parse_args()

    layout_ids = args.layout_ids if args.layout_ids is not None else list(range(NUM_LAYOUTS))
    style_ids = args.style_ids if args.style_ids is not None else list(range(NUM_STYLES))
    num_combos = len(layout_ids) * len(style_ids)

    results = {}

    for task_name in args.tasks:
        print(f"\n{'='*60}")
        print(f"Task: {task_name}")
        print(f"{'='*60}")

        compatible = []
        incompatible = []
        tested = 0

        for layout_id in layout_ids:
            for style_id in style_ids:
                tested += 1
                layout_name = LAYOUT_NAMES[layout_id] if layout_id < len(LAYOUT_NAMES) else str(layout_id)
                print(
                    f"  [{tested}/{num_combos}] layout={layout_id} ({layout_name}), "
                    f"style={style_id} ... ",
                    end="",
                    flush=True,
                )

                num_failed, first_error = test_combination(
                    task_name, layout_id, style_id, args.num_seeds, args.start_seed
                )

                if num_failed == 0:
                    print("OK")
                    compatible.append([layout_id, style_id])
                else:
                    print(f"FAIL ({num_failed}/{args.num_seeds} seeds)")
                    incompatible.append({
                        "layout_style": [layout_id, style_id],
                        "error": first_error,
                        "seeds_tested": args.num_seeds,
                        "seeds_failed": num_failed,
                    })

        results[task_name] = {
            "compatible": compatible,
            "incompatible": incompatible,
        }

        # Print summary for this task
        print(f"\n  Summary: {len(compatible)} compatible, "
              f"{len(incompatible)} incompatible out of {num_combos} tested")
        if incompatible:
            # Summarise by layout
            failed_layouts = set(entry["layout_style"][0] for entry in incompatible)
            for lid in sorted(failed_layouts):
                entries = [e for e in incompatible if e["layout_style"][0] == lid]
                layout_name = LAYOUT_NAMES[lid] if lid < len(LAYOUT_NAMES) else str(lid)
                failed_styles = [e["layout_style"][1] for e in entries]
                if len(failed_styles) == len(style_ids):
                    print(f"    layout {lid} ({layout_name}): ALL styles incompatible")
                else:
                    print(f"    layout {lid} ({layout_name}): styles {failed_styles} incompatible")

    # Print overall summary
    print(f"\n{'='*60}")
    print("OVERALL SUMMARY")
    print(f"{'='*60}")
    for task_name, data in results.items():
        n_ok = len(data["compatible"])
        n_fail = len(data["incompatible"])
        print(f"  {task_name}: {n_ok} compatible, {n_fail} incompatible")

    # Write YAML output
    if args.output_path:
        with open(args.output_path, "w") as f:
            yaml.dump(results, f, default_flow_style=False, sort_keys=False)
        print(f"\nResults written to {args.output_path}")


if __name__ == "__main__":
    main()
