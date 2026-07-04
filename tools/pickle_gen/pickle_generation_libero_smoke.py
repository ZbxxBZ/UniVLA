import argparse
import os
import pathlib
import pickle
import re
import sys

import numpy as np
from tqdm import tqdm

PROJECT_ROOT = os.environ.get("PROJECT_ROOT", str(pathlib.Path(__file__).resolve().parents[2]))
sys.path.append(PROJECT_ROOT)

from train.dataset.normalize_pi0 import RunningStats, save


def natural_key(path):
    stem = pathlib.Path(path).stem
    return [int(token) if token.isdigit() else token.lower() for token in re.split(r"(\d+)", stem)]


def files_by_stem(directory, suffix):
    return {
        path.stem: path
        for path in sorted(pathlib.Path(directory).glob(f"*{suffix}"), key=natural_key)
    }


def build_samples(processed_dir, vq_dir, gripper_vq_dir, min_frames):
    processed_dir = pathlib.Path(processed_dir)
    vq_dir = pathlib.Path(vq_dir)
    gripper_vq_dir = pathlib.Path(gripper_vq_dir)

    samples = []
    skipped = 0

    episodes = sorted([path for path in processed_dir.iterdir() if path.is_dir()])
    for episode in tqdm(episodes, desc="Loading LIBERO smoke episodes"):
        instr_file = episode / "instruction.txt"
        action_dir = episode / "actions"
        image_dir = vq_dir / episode.name
        gripper_dir = gripper_vq_dir / episode.name

        if not instr_file.exists() or not action_dir.exists() or not image_dir.exists() or not gripper_dir.exists():
            skipped += 1
            continue

        text = instr_file.read_text(encoding="utf-8").strip()
        if not text:
            skipped += 1
            continue

        image_files = files_by_stem(image_dir, ".npy")
        gripper_files = files_by_stem(gripper_dir, ".npy")
        action_files = files_by_stem(action_dir, ".npy")

        stems = sorted(set(image_files) & set(gripper_files) & set(action_files), key=natural_key)
        if len(stems) < min_frames:
            skipped += 1
            continue

        actions = np.stack([np.load(action_files[stem]) for stem in stems], axis=0).astype(np.float32)
        samples.append(
            {
                "text": text,
                "image": [str(image_files[stem]) for stem in stems],
                "gripper_image": [str(gripper_files[stem]) for stem in stems],
                "action": actions,
            }
        )

    return samples, skipped


def normalize_actions(samples, normalizer_path, normalizer_key):
    normalizer = RunningStats()
    action_data = np.concatenate([sample["action"] for sample in samples], axis=0)
    normalizer.update(action_data)
    stats = normalizer.get_statistics()

    for sample in samples:
        action = sample["action"]
        normalized = 2 * (action - stats.q01) / (stats.q99 - stats.q01 + 1e-8) - 1
        sample["action"] = np.clip(normalized, -1, 1).astype(np.float32)

    save(normalizer_path, {normalizer_key: stats})
    return stats


def main(args):
    samples, skipped = build_samples(
        processed_dir=args.processed_dir,
        vq_dir=args.vq_dir,
        gripper_vq_dir=args.gripper_vq_dir,
        min_frames=args.min_frames,
    )
    if not samples:
        raise ValueError("No valid LIBERO smoke policy samples found.")

    stats = normalize_actions(samples, args.normalizer_path, args.normalizer_key)

    output = pathlib.Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("wb") as f:
        pickle.dump(samples, f)

    print(f"Saved normalized LIBERO smoke policy data to {output}")
    print(f"Saved normalizer statistics to {args.normalizer_path}")
    print(f"Total valid samples: {len(samples)}; skipped: {skipped}")
    print(f"First sample lengths: image={len(samples[0]['image'])}, gripper_image={len(samples[0]['gripper_image'])}, action={samples[0]['action'].shape}")
    print("Mean:", stats.mean)
    print("Std:", stats.std)
    print("Q01:", stats.q01)
    print("Q99:", stats.q99)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate a normalized LIBERO smoke policy pickle from processed images, VQ codes, and actions."
    )
    parser.add_argument("--processed_dir", default="datasets/processed_data/libero_smoke")
    parser.add_argument("--vq_dir", default="datasets/processed_data/libero_smoke_codes_200")
    parser.add_argument("--gripper_vq_dir", default="datasets/processed_data/libero_smoke_gripper_codes_200")
    parser.add_argument("--output", default="datasets/processed_data/meta/libero_smoke_policy_norm.pkl")
    parser.add_argument("--normalizer_path", default="configs/normalizer_libero_smoke")
    parser.add_argument("--normalizer_key", default="libero")
    parser.add_argument("--min_frames", type=int, default=20)
    main(parser.parse_args())
