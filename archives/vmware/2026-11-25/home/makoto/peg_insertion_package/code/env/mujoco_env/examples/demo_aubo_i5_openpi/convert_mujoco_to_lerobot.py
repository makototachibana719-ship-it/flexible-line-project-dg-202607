"""
Convert Mujoco collected data -> LeRobot local dataset layout (HF cache style), robust for OpenPI/LeRobot.

Usage:
  python convert_mujoco_to_lerobot.py \
    --data-dir path/to/your/data \
    --repo-id <repo-id>/mujoco_aubo_data \
    --robot-type aubo_i5 \
    --fps 50 \
    --image-size 224 \
    --chunk-size 1000 \
    --overwrite
"""

from __future__ import annotations

import json
import math
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import tyro
import imageio.v2 as imageio

# LeRobot cache root (same place you看到的 ~/.cache/huggingface/lerobot/...)
try:
    from lerobot.datasets.lerobot_dataset import HF_LEROBOT_HOME
except Exception:
    # fallback: typical default
    HF_LEROBOT_HOME = Path.home() / ".cache" / "huggingface" / "lerobot"


# -------------------------
# math helpers
# -------------------------
def quat2axisangle(quat_wxyz: np.ndarray) -> np.ndarray:
    """quat [qw,qx,qy,qz] -> axis-angle (3,)"""
    q = np.asarray(quat_wxyz, dtype=np.float64)
    q = q / (np.linalg.norm(q) + 1e-12)
    w, x, y, z = q
    if abs(w) > 0.999999:
        return np.zeros(3, dtype=np.float32)
    angle = 2.0 * np.arccos(np.clip(w, -1.0, 1.0))
    sin_half = math.sqrt(max(1.0 - w * w, 1e-12))
    axis = np.array([x, y, z], dtype=np.float64) / sin_half
    aa = axis * angle
    return aa.astype(np.float32)


def convert_state_to_lerobot(state8: np.ndarray) -> np.ndarray:
    """
    Input:  [x,y,z, qw,qx,qy,qz, gripper]  (8,)
    Output: [x,y,z, ax,ay,az, gripper,gripper] (8,)
    """
    s = np.asarray(state8, dtype=np.float32)
    pos = s[:3]
    quat = s[3:7]
    grip = float(s[7])
    aa = quat2axisangle(quat)
    out = np.concatenate([pos.astype(np.float32), aa, np.array([grip, grip], dtype=np.float32)], axis=0)
    assert out.shape == (8,)
    return out


def ensure_uint8_rgb(img: np.ndarray, image_size: int) -> np.ndarray:
    """Ensure (H,W,3) uint8"""
    arr = np.asarray(img)
    assert arr.shape == (image_size, image_size, 3), f"bad image shape: {arr.shape}"
    if arr.dtype != np.uint8:
        # if float [0,1] -> uint8
        if np.issubdtype(arr.dtype, np.floating):
            arr = np.clip(arr, 0.0, 1.0)
            arr = (arr * 255.0).astype(np.uint8)
        else:
            arr = arr.astype(np.uint8)
    return arr


def compute_stats_1d(x: np.ndarray) -> Dict[str, object]:
    """
    Return stats dict with keys: min/max/mean/std/count/q01/q10/q50/q90/q99
    Values use python scalars / lists for json.
    """
    x = np.asarray(x)
    x = x.astype(np.float64)

    def q(p: float):
        return np.quantile(x, p, axis=0)

    stats = {
        "min": np.min(x, axis=0),
        "max": np.max(x, axis=0),
        "mean": np.mean(x, axis=0),
        "std": np.std(x, axis=0),
        "count": np.array([x.shape[0]], dtype=np.int64),
        "q01": q(0.01),
        "q10": q(0.10),
        "q50": q(0.50),
        "q90": q(0.90),
        "q99": q(0.99),
    }

    def to_jsonable(v):
        if isinstance(v, np.ndarray):
            return v.tolist()
        if isinstance(v, (np.floating, np.integer)):
            return v.item()
        return v

    return {k: to_jsonable(v) for k, v in stats.items()}


# -------------------------
# IO helpers
# -------------------------
def load_episode(episode_dir: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
    """
    episode_dir contains:
      states.npy (N,8)
      actions.npy (N,7)
      images.npz (optional; contains main_images, wrist_images)
      OR main_images.npy + wrist_images.npy (legacy)
      metadata.json (optional)
    """
    states = np.load(episode_dir / "states.npy")
    actions = np.load(episode_dir / "actions.npy")

    meta_path = episode_dir / "metadata.json"
    metadata = {}
    if meta_path.exists():
        metadata = json.loads(meta_path.read_text(encoding="utf-8"))

    # Prefer compressed images.npz if present
    npz_path = episode_dir / "images.npz"
    if npz_path.exists():
        d = np.load(npz_path)
        # your collector used keys: main_images / wrist_images
        if "main_images" in d and "wrist_images" in d:
            main_images = d["main_images"]
            wrist_images = d["wrist_images"]
        else:
            raise ValueError(f"{episode_dir}: images.npz missing keys main_images/wrist_images. keys={list(d.keys())}")
        return states, actions, main_images, wrist_images, metadata

    # Legacy fallback
    main_images_path = episode_dir / "main_images.npy"
    wrist_images_path = episode_dir / "wrist_images.npy"
    if main_images_path.exists() and wrist_images_path.exists():
        main_images = np.load(main_images_path)
        wrist_images = np.load(wrist_images_path)
        return states, actions, main_images, wrist_images, metadata

    raise FileNotFoundError(f"{episode_dir}: no images found (expected images.npz or main_images.npy+wrist_images.npy)")


def write_images(
    root: Path,
    episode_index: int,
    main_images: np.ndarray,
    wrist_images: np.ndarray,
    image_size: int,
) -> Tuple[List[str], List[str]]:
    """
    Write frames to:
      root/images/image/frame-XXXXXX.png
      root/images/wrist_image/frame-XXXXXX.png
    Return absolute paths lists (len N).
    """
    out_image = root / "images" / "image"
    out_wrist = root / "images" / "wrist_image"
    out_image.mkdir(parents=True, exist_ok=True)
    out_wrist.mkdir(parents=True, exist_ok=True)

    N = main_images.shape[0]
    image_paths: List[str] = []
    wrist_paths: List[str] = []

    # Use global naming frame-000000.png ... (same as你后面 patch 的逻辑)
    # If you want per-episode prefix, change here, but then parquet paths也要一致。
    for i in range(N):
        img = ensure_uint8_rgb(main_images[i], image_size)
        wimg = ensure_uint8_rgb(wrist_images[i], image_size)

        p1 = out_image / f"frame-{(episode_index*1_000_000 + i):06d}.png"
        p2 = out_wrist / f"frame-{(episode_index*1_000_000 + i):06d}.png"

        imageio.imwrite(p1, img)
        imageio.imwrite(p2, wimg)

        image_paths.append(str(p1))
        wrist_paths.append(str(p2))

    return image_paths, wrist_paths


def write_parquet_no_metadata(path: Path, table: pa.Table) -> None:
    """Write parquet stripping schema metadata (avoid datasets feature parsing issues)."""
    table = table.replace_schema_metadata(None)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".parquet.tmp")
    pq.write_table(table, tmp, compression="zstd")
    tmp.replace(path)


# -------------------------
# main conversion
# -------------------------
@dataclass
class Args:
    data_dir: str
    repo_id: str = "wangrui/mujoco_aubo_data"
    robot_type: str = "aubo_i5"
    fps: int = 200
    image_size: int = 512
    chunk_size: int = 1000
    overwrite: bool = False


def main(args: Args) -> None:
    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(f"data_dir not found: {data_dir}")

    # output root: ~/.cache/huggingface/lerobot/<repo_id>
    out_root = Path(HF_LEROBOT_HOME) / args.repo_id
    meta_dir = out_root / "meta"
    data_out_dir = out_root / "data"

    if out_root.exists():
        if not args.overwrite:
            raise RuntimeError(
                f"Output exists: {out_root}\n"
                f"Use --overwrite to remove and recreate."
            )
        shutil.rmtree(out_root)

    # discover episodes
    episode_dirs = sorted([p for p in data_dir.glob("episode_*") if p.is_dir()])
    if not episode_dirs:
        raise RuntimeError(f"No episode_* folders under: {data_dir}")

    out_root.mkdir(parents=True, exist_ok=True)
    meta_dir.mkdir(parents=True, exist_ok=True)

    # task registry: task string -> task_index
    task_to_index: Dict[str, int] = {}
    tasks_list: List[Dict[str, object]] = []

    episodes_jsonl: List[Dict[str, object]] = []
    episodes_stats_jsonl: List[Dict[str, object]] = []

    global_frame_index = 0
    total_frames = 0

    for ep_i, ep_dir in enumerate(episode_dirs):
        try:
            states, actions, main_imgs, wrist_imgs, metadata = load_episode(ep_dir)

            N = int(states.shape[0])
            if not (states.shape == (N, 8)):
                raise ValueError(f"{ep_dir}: states shape {states.shape} != (N,8)")
            if not (actions.shape == (N, 7)):
                raise ValueError(f"{ep_dir}: actions shape {actions.shape} != (N,7)")
            # image size per-episode: prefer metadata, fallback to args
            ep_image_size = args.image_size
            ms = metadata.get("image_size", None)
            if isinstance(ms, list) and len(ms) >= 2:
                ep_image_size = int(ms[0])

            if not (main_imgs.shape == (N, ep_image_size, ep_image_size, 3)):
                raise ValueError(f"{ep_dir}: main_images shape {main_imgs.shape} != (N,{ep_image_size},{ep_image_size},3)")
            if not (wrist_imgs.shape == (N, ep_image_size, ep_image_size, 3)):
                raise ValueError(f"{ep_dir}: wrist_images shape {wrist_imgs.shape} != (N,{ep_image_size},{ep_image_size},3)")


        except Exception as e:
            print(f"Skipping episode {ep_dir.name} due to error: {e}")
            continue

        task = metadata.get("task", "default task")
        if not isinstance(task, str):
            task = str(task)

        if task not in task_to_index:
            tid = len(task_to_index)
            task_to_index[task] = tid
            tasks_list.append({"task_index": tid, "task": task})
        task_index = task_to_index[task]

        # write images to disk, store absolute paths
        img_paths, wrist_paths = write_images(out_root, ep_i, main_imgs, wrist_imgs, args.image_size)

        # build columns
        # state convert
        lerobot_states = np.stack([convert_state_to_lerobot(states[i]) for i in range(N)], axis=0).astype(np.float32)
        actions = np.asarray(actions, dtype=np.float32)

        ep_fps = int(metadata.get("fps", args.fps))
        timestamp = np.arange(N, dtype=np.float32) / float(ep_fps)

        frame_index = np.arange(N, dtype=np.int64)
        episode_index = np.full((N,), ep_i, dtype=np.int64)
        task_index_col = np.full((N,), task_index, dtype=np.int64)
        index_col = np.arange(global_frame_index, global_frame_index + N, dtype=np.int64)

        # arrow schema: use string for images; fixed_size_list for vectors
        # fixed_size_list in pyarrow: pa.list_(value_type, list_size) isn't fixed; use pa.fixed_size_list
        state_type = pa.list_(pa.float32(), 8)
        action_type = pa.list_(pa.float32(), 7)

        table = pa.table(
            {
                "image": pa.array(img_paths, type=pa.string()),
                "wrist_image": pa.array(wrist_paths, type=pa.string()),
                "state": pa.array(lerobot_states.tolist(), type=state_type),
                "actions": pa.array(actions.tolist(), type=action_type),
                "timestamp": pa.array(timestamp.tolist(), type=pa.float32()),
                "frame_index": pa.array(frame_index.tolist(), type=pa.int64()),
                "episode_index": pa.array(episode_index.tolist(), type=pa.int64()),
                "index": pa.array(index_col.tolist(), type=pa.int64()),
                "task_index": pa.array(task_index_col.tolist(), type=pa.int64()),
            }
        )

        # decide parquet path by chunking episodes (not frames)
        episode_chunk = ep_i // args.chunk_size
        episode_file_index = ep_i % args.chunk_size
        parquet_path = data_out_dir / f"chunk-{episode_chunk:03d}" / f"file-{episode_file_index:03d}.parquet"
        write_parquet_no_metadata(parquet_path, table)

        # episodes.jsonl record (LeRobot expects tasks as list of strings)
        ep_from = global_frame_index
        ep_to = global_frame_index + N
        episodes_jsonl.append(
            {
                "episode_index": ep_i,
                "tasks": [task],
                "length": N,
                "data/chunk_index": int(episode_chunk),
                "data/file_index": int(episode_file_index),
                "dataset_from_index": int(ep_from),
                "dataset_to_index": int(ep_to),
            }
        )

        # episodes_stats.jsonl (must have {"episode_index":..., "stats": {...}})
        # Only compute numeric stats (no image stats)
        stats_dict: Dict[str, object] = {}

        # state stats -> keys like "state/min", etc.
        st = compute_stats_1d(lerobot_states)  # returns min/max/...
        for k, v in st.items():
            stats_dict[f"state/{k}"] = v

        ac = compute_stats_1d(actions)
        for k, v in ac.items():
            stats_dict[f"actions/{k}"] = v

        ts = compute_stats_1d(timestamp.reshape(-1, 1))
        for k, v in ts.items():
            stats_dict[f"timestamp/{k}"] = v

        fi = compute_stats_1d(frame_index.reshape(-1, 1))
        for k, v in fi.items():
            stats_dict[f"frame_index/{k}"] = v

        ii = compute_stats_1d(index_col.reshape(-1, 1))
        for k, v in ii.items():
            stats_dict[f"index/{k}"] = v

        ti = compute_stats_1d(task_index_col.reshape(-1, 1))
        for k, v in ti.items():
            stats_dict[f"task_index/{k}"] = v

        ei = compute_stats_1d(episode_index.reshape(-1, 1))
        for k, v in ei.items():
            stats_dict[f"episode_index/{k}"] = v

        episodes_stats_jsonl.append({"episode_index": ep_i, "stats": stats_dict})

        global_frame_index += N
        total_frames += N

        print(f"[{ep_i+1}/{len(episode_dirs)}] wrote episode {ep_dir.name} -> {parquet_path} | frames={N} | task_index={task_index}")

    # write tasks.jsonl
    tasks_jsonl_path = meta_dir / "tasks.jsonl"
    with tasks_jsonl_path.open("w", encoding="utf-8") as f:
        for item in tasks_list:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    # write tasks.parquet (for compatibility)
    tasks_parquet_path = meta_dir / "tasks.parquet"
    tasks_table = pa.table(
        {
            "task_index": pa.array([int(x["task_index"]) for x in tasks_list], type=pa.int64()),
            "task": pa.array([str(x["task"]) for x in tasks_list], type=pa.string()),
        }
    )
    write_parquet_no_metadata(tasks_parquet_path, tasks_table)

    # write episodes.jsonl
    episodes_jsonl_path = meta_dir / "episodes.jsonl"
    with episodes_jsonl_path.open("w", encoding="utf-8") as f:
        for item in episodes_jsonl:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    # write episodes_stats.jsonl
    episodes_stats_path = meta_dir / "episodes_stats.jsonl"
    with episodes_stats_path.open("w", encoding="utf-8") as f:
        for item in episodes_stats_jsonl:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    # write global stats.json (optional but nice)
    # aggregate from episodes_stats by just taking overall frame-level stats is more work;
    # keep a minimal file to satisfy "meta/stats.json" presence like你之前的目录。
    (meta_dir / "stats.json").write_text(json.dumps({"note": "global stats not precomputed; use compute_norm_stats.py"}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    # write info.json
    info = {
        "codebase_version": "v3.0",
        "robot_type": args.robot_type,
        "total_episodes": len(episode_dirs),
        "total_frames": total_frames,
        "total_tasks": len(tasks_list),
        "chunks_size": args.chunk_size,
        "data_files_size_in_mb": None,
        "video_files_size_in_mb": None,
        "fps": args.fps,
        "splits": {"train": f"0:{len(episode_dirs)}"},
        "data_path": "data/chunk-{episode_chunk:03d}/file-{episode_index:03d}.parquet",
        "video_path": "videos/{video_key}/chunk-{episode_chunk:03d}/file-{episode_index:03d}.mp4",
        # IMPORTANT: images stored as string paths (not HF Image struct)
        "features": {
            "image": {"dtype": "string", "shape": [1], "names": None},
            "wrist_image": {"dtype": "string", "shape": [1], "names": None},
            "state": {"dtype": "float32", "shape": [8], "names": ["state"]},
            "actions": {"dtype": "float32", "shape": [7], "names": ["actions"]},
            "timestamp": {"dtype": "float32", "shape": [1], "names": None},
            "frame_index": {"dtype": "int64", "shape": [1], "names": None},
            "episode_index": {"dtype": "int64", "shape": [1], "names": None},
            "index": {"dtype": "int64", "shape": [1], "names": None},
            "task_index": {"dtype": "int64", "shape": [1], "names": None},
        },
    }
    (meta_dir / "info.json").write_text(json.dumps(info, ensure_ascii=False, indent=4) + "\n", encoding="utf-8")

    print("\nDONE")
    print("out_root:", out_root)
    print("meta files:", meta_dir)
    print("sample:")
    print("  ", tasks_jsonl_path)
    print("  ", episodes_jsonl_path)
    print("  ", episodes_stats_path)
    print("  ", meta_dir / "info.json")


if __name__ == "__main__":
    main(tyro.cli(Args))
