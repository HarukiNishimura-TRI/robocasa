"""Replay expert demonstrations from RoboCasa HDF5 datasets.

For each demo in the dataset, reset the simulation to the episode's
initial condition (MuJoCo XML + state vector), replay the recorded 12D
actions step-by-step, render video from the specified camera(s), and
report per-demo success via ``env._check_success()``.

The environment is created using the exact ``env_args`` stored in the
HDF5 file (including ``controller_configs``), matching the official
``robocasa.scripts.playback_dataset`` approach.

Examples
--------
Sequential (single env, default)::

    python replay_robocasa_demos.py \\
        --dataset_path /path/to/demo.hdf5 \\
        -o /path/to/output

Parallel (4 worker processes)::

    python replay_robocasa_demos.py \\
        --dataset_path /path/to/demo.hdf5 \\
        -o /path/to/output \\
        --num_envs 4

Multiple camera views (tiled horizontally)::

    python replay_robocasa_demos.py \\
        --dataset_path /path/to/demo.hdf5 \\
        -o /path/to/output \\
        --render_cameras robot0_agentview_left robot0_eye_in_hand
"""

import argparse
import json
import multiprocessing as mp
import os
from typing import Dict, List, Optional

import h5py
import imageio.v2 as imageio
import numpy as np
import yaml
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Environment metadata & creation
# ---------------------------------------------------------------------------


def get_env_metadata_from_dataset(dataset_path: str) -> dict:
    """Read ``env_args`` from the HDF5 file's ``data`` group attributes.

    This matches ``robocasa.scripts.playback_dataset.get_env_metadata_from_dataset``.

    Returns:
        dict with keys ``env_name`` and ``env_kwargs`` (including
        ``controller_configs`` used during data collection).
    """
    dataset_path = os.path.expanduser(dataset_path)
    with h5py.File(dataset_path, "r") as f:
        env_meta = json.loads(f["data"].attrs["env_args"])
    return env_meta


def create_replay_env(dataset_path: str):
    """Create a robosuite/robocasa environment from HDF5 env metadata.

    Uses the exact ``env_kwargs`` (including ``controller_configs``)
    stored in the HDF5 file, matching the official
    ``robocasa.scripts.playback_dataset`` approach.  Only
    rendering-related kwargs are overridden.
    """
    import robosuite
    import robocasa  # noqa: F401 — registers RoboCasa envs

    env_meta = get_env_metadata_from_dataset(dataset_path)
    env_kwargs = env_meta["env_kwargs"]
    env_kwargs["env_name"] = env_meta["env_name"]
    env_kwargs["has_renderer"] = False
    env_kwargs["renderer"] = "mjviewer"
    env_kwargs["has_offscreen_renderer"] = True
    env_kwargs["use_camera_obs"] = False

    env = robosuite.make(**env_kwargs)
    return env, env_meta


# ---------------------------------------------------------------------------
# Environment reset
# ---------------------------------------------------------------------------


def reset_env_to_demo(env, init_state: dict):
    """Reset the base robosuite env to a demo's initial condition.

    Uses the HDF5 demo's ``model_file`` XML, ``ep_meta`` JSON, and
    ``states[0]`` vector to reconstruct the exact initial scene.

    Follows the same protocol as the official
    ``robocasa.scripts.playback_dataset.reset_to()``.
    """
    import robosuite

    # Set episode metadata (layout, style, fixture refs, object configs).
    ep_meta_raw = init_state.get("ep_meta", None)
    if ep_meta_raw is not None:
        ep_meta = (
            json.loads(ep_meta_raw)
            if isinstance(ep_meta_raw, str)
            else ep_meta_raw
        )
    else:
        ep_meta = {}

    # Use proper API methods when available (matching official reset_to).
    if hasattr(env, "set_attrs_from_ep_meta"):
        env.set_attrs_from_ep_meta(ep_meta)
    elif hasattr(env, "set_ep_meta"):
        env.set_ep_meta(ep_meta)
    elif hasattr(env, "_ep_meta"):
        env._ep_meta = ep_meta

    # Reset rebuilds the MuJoCo model.
    env.reset()

    # Load the exact model XML from the demo.
    robosuite_version_id = int(robosuite.__version__.split(".")[1])
    if robosuite_version_id <= 3:
        from robosuite.utils.mjcf_utils import postprocess_model_xml

        xml = postprocess_model_xml(init_state["model"])
    else:
        xml = env.edit_model_xml(init_state["model"])
    env.reset_from_xml_string(xml)
    env.sim.reset()

    # Set simulator state (qpos, qvel, etc.)
    if "states" in init_state:
        env.sim.set_state_from_flattened(init_state["states"])
        env.sim.forward()

    # Update visual sites and internal state.
    if hasattr(env, "update_sites"):
        env.update_sites()
    if hasattr(env, "update_state"):
        env.update_state()


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def render_frame(
    env, camera_names: List[str], width: int, height: int
) -> np.ndarray:
    """Render frame(s) from specified camera(s).

    If multiple cameras are given, tiles them horizontally.
    Returns uint8 ``(H, W*N, 3)`` array.
    """
    frames = []
    for cam in camera_names:
        frame = env.sim.render(
            height=height, width=width, camera_name=cam
        )[::-1]
        frames.append(frame)
    if len(frames) == 1:
        return frames[0]
    return np.concatenate(frames, axis=1)


# ---------------------------------------------------------------------------
# Single-demo replay
# ---------------------------------------------------------------------------


def replay_single_demo(
    env,
    demo_key: str,
    demo_data: dict,
    render_cameras: List[str],
    camera_width: int,
    camera_height: int,
    video_fps: int,
    output_dir: str,
) -> dict:
    """Replay one demo: reset -> step through actions -> save MP4.

    Returns a result dict with success info and video path.
    """
    reset_env_to_demo(env, demo_data)

    video_path = os.path.join(output_dir, f"{demo_key}_replay.mp4")
    writer = imageio.get_writer(video_path, fps=video_fps, macro_block_size=1)

    # Render initial frame (before any action).
    frame = render_frame(env, render_cameras, camera_width, camera_height)
    writer.append_data(frame)

    any_success = False
    first_success_step = None
    actions = demo_data["actions"]

    for t in range(len(actions)):
        env.step(actions[t])
        frame = render_frame(env, render_cameras, camera_width, camera_height)
        writer.append_data(frame)

        if env._check_success():
            if not any_success:
                any_success = True
                first_success_step = t + 1

    writer.close()

    return {
        "demo_key": demo_key,
        "success": any_success,
        "first_success_step": first_success_step,
        "total_steps": len(actions),
        "video_path": video_path,
    }


# ---------------------------------------------------------------------------
# Parallel worker
# ---------------------------------------------------------------------------


def _worker_fn(worker_args: tuple) -> List[dict]:
    """Worker function for parallel replay (runs in a spawned process)."""
    (
        dataset_path,
        render_cameras,
        camera_width,
        camera_height,
        video_fps,
        output_dir,
        demo_items,
    ) = worker_args

    env, _env_meta = create_replay_env(dataset_path)

    results = []
    for demo_key, demo_data in demo_items:
        result = replay_single_demo(
            env,
            demo_key,
            demo_data,
            render_cameras,
            camera_width,
            camera_height,
            video_fps,
            output_dir,
        )
        results.append(result)

    env.close()
    return results


# ---------------------------------------------------------------------------
# HDF5 loading
# ---------------------------------------------------------------------------


def _demo_sort_key(key: str):
    """Sort demo keys numerically (demo_0, demo_1, ..., demo_10, ...)."""
    try:
        return int(key.split("_")[-1])
    except ValueError:
        return key


def load_demos_from_hdf5(
    dataset_path: str,
    mask: str = "train",
    demo_indices: Optional[List[int]] = None,
    max_demos: Optional[int] = None,
) -> Dict[str, dict]:
    """Load demo data from an HDF5 file.

    Returns dict mapping demo_key -> {actions, states, model, ep_meta}.
    """
    demos = {}

    with h5py.File(dataset_path, "r") as f:
        if demo_indices is not None:
            demo_keys = [f"demo_{i}" for i in demo_indices]
        else:
            mask_data = f[f"mask/{mask}"][:]
            demo_keys = sorted(
                [k.decode() if isinstance(k, bytes) else k for k in mask_data],
                key=_demo_sort_key,
            )

        for key in demo_keys:
            h5_key = f"data/{key}"
            if h5_key not in f:
                print(f"Warning: {key} not found in HDF5 file, skipping")
                continue

            grp = f[h5_key]
            demo = {
                "actions": grp["actions"][:],
                "states": grp["states"][0],
                "model": grp.attrs["model_file"],
            }
            ep_meta = grp.attrs.get("ep_meta", None)
            if ep_meta is not None:
                demo["ep_meta"] = ep_meta
            demos[key] = demo

    if max_demos is not None and len(demos) > max_demos:
        keys = sorted(demos.keys(), key=_demo_sort_key)[:max_demos]
        demos = {k: demos[k] for k in keys}

    return demos


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Replay RoboCasa expert demos from HDF5 datasets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        required=True,
        help="Path to HDF5 dataset file",
    )
    parser.add_argument(
        "-o",
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save videos and summary",
    )
    parser.add_argument(
        "--num_envs",
        type=int,
        default=1,
        help="1 = sequential (default), N > 1 = parallel with N workers",
    )
    parser.add_argument(
        "--render_cameras",
        type=str,
        nargs="+",
        default=["robot0_agentview_left"],
        help=(
            "Camera(s) for video rendering. Multiple cameras are tiled "
            "horizontally. (default: robot0_agentview_left)"
        ),
    )
    parser.add_argument(
        "--camera_width",
        type=int,
        default=256,
        help="Render width per camera (default: 256)",
    )
    parser.add_argument(
        "--camera_height",
        type=int,
        default=256,
        help="Render height per camera (default: 256)",
    )
    parser.add_argument(
        "--video_fps",
        type=int,
        default=20,
        help="Video frame rate (default: 20, matches robosuite sim rate)",
    )
    parser.add_argument(
        "--mask",
        type=str,
        default="train",
        choices=["train", "valid"],
        help="HDF5 mask to select demos (default: train)",
    )
    parser.add_argument(
        "--demo_indices",
        type=int,
        nargs="+",
        default=None,
        help="Specific demo indices (overrides --mask)",
    )
    parser.add_argument(
        "--max_demos",
        type=int,
        default=None,
        help="Max number of demos to replay (for quick testing)",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # ---- Load env metadata from HDF5 ----
    print(f"Reading env metadata from {args.dataset_path}")
    env_meta = get_env_metadata_from_dataset(args.dataset_path)
    env_name = env_meta["env_name"]
    controller_configs = env_meta["env_kwargs"].get("controller_configs", None)
    print(f"  env_name: {env_name}")
    print(f"  controller_configs: {controller_configs}")

    # ---- Load demos ----
    print(f"Loading demos from {args.dataset_path}")
    demos = load_demos_from_hdf5(
        args.dataset_path, args.mask, args.demo_indices, args.max_demos
    )
    if not demos:
        print("No demos to replay. Exiting.")
        return
    print(f"Loaded {len(demos)} demos")

    demo_items = sorted(demos.items(), key=lambda x: _demo_sort_key(x[0]))

    if args.num_envs <= 1:
        # ---- Sequential mode ----
        print("Running in sequential mode (single env)")
        env, _ = create_replay_env(args.dataset_path)

        results = []
        for demo_key, demo_data in tqdm(demo_items, desc="Replaying demos"):
            result = replay_single_demo(
                env,
                demo_key,
                demo_data,
                args.render_cameras,
                args.camera_width,
                args.camera_height,
                args.video_fps,
                args.output_dir,
            )
            results.append(result)
            status = "SUCCESS" if result["success"] else "FAIL"
            step_info = (
                f", first success at step {result['first_success_step']}"
                if result["success"]
                else ""
            )
            tqdm.write(
                f"  {demo_key}: {status} "
                f"({result['total_steps']} steps{step_info})"
            )

        env.close()
    else:
        # ---- Parallel mode ----
        n_workers = min(args.num_envs, len(demo_items))
        print(f"Running in parallel mode with {n_workers} workers")

        # Distribute demos across workers (round-robin).
        chunks: List[list] = [[] for _ in range(n_workers)]
        for i, item in enumerate(demo_items):
            chunks[i % n_workers].append(item)

        worker_args_list = [
            (
                args.dataset_path,
                args.render_cameras,
                args.camera_width,
                args.camera_height,
                args.video_fps,
                args.output_dir,
                chunk,
            )
            for chunk in chunks
            if chunk
        ]

        ctx = mp.get_context("spawn")
        with ctx.Pool(n_workers) as pool:
            results_nested = list(
                tqdm(
                    pool.imap_unordered(_worker_fn, worker_args_list),
                    total=len(worker_args_list),
                    desc="Worker batches",
                )
            )

        results = []
        for r_list in results_nested:
            results.extend(r_list)

        # Print per-demo results.
        for r in sorted(results, key=lambda x: _demo_sort_key(x["demo_key"])):
            status = "SUCCESS" if r["success"] else "FAIL"
            step_info = (
                f", first success at step {r['first_success_step']}"
                if r["success"]
                else ""
            )
            print(
                f"  {r['demo_key']}: {status} "
                f"({r['total_steps']} steps{step_info})"
            )

    # ---- Summary ----
    n_success = sum(1 for r in results if r["success"])
    n_total = len(results)
    success_rate = n_success / n_total if n_total > 0 else 0.0

    print()
    print(f"Results: {n_success}/{n_total} demos succeeded ({100 * success_rate:.1f}%)")

    summary = {
        "dataset_path": args.dataset_path,
        "env_name": env_name,
        "controller_configs": controller_configs,
        "render_cameras": args.render_cameras,
        "camera_width": args.camera_width,
        "camera_height": args.camera_height,
        "video_fps": args.video_fps,
        "num_demos": n_total,
        "num_success": n_success,
        "success_rate": float(success_rate),
        "per_demo": {
            r["demo_key"]: {
                "success": r["success"],
                "first_success_step": r["first_success_step"],
                "total_steps": r["total_steps"],
                "video_path": r["video_path"],
            }
            for r in sorted(results, key=lambda x: _demo_sort_key(x["demo_key"]))
        },
    }
    summary_path = os.path.join(args.output_dir, "replay_summary.yaml")
    with open(summary_path, "w") as f:
        yaml.safe_dump(summary, f, sort_keys=False)
    print(f"Summary saved to {summary_path}")


if __name__ == "__main__":
    main()
