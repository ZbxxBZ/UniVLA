import argparse
import os
import re
import shutil
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm
import tensorflow as tf
import tensorflow_datasets as tfds

os.environ.setdefault('CUDA_VISIBLE_DEVICES', '-1')

DEFAULT_SUITES = [
    'libero_10_no_noops',
    'libero_goal_no_noops',
    'libero_object_no_noops',
    'libero_spatial_no_noops',
]


def sanitize(name: str) -> str:
    return re.sub(r'[^A-Za-z0-9_.=-]+', '_', name)


def scalar_to_text(value):
    if hasattr(value, 'numpy'):
        value = value.numpy()
    if isinstance(value, bytes):
        return value.decode('utf-8')
    return str(value)


def get_episode_name(episode, suite: str, local_idx: int) -> str:
    try:
        file_path = scalar_to_text(episode['episode_metadata']['file_path'])
        parts = [p for p in file_path.split('/') if p]
        stem = '__'.join(parts[-2:]).replace('.hdf5', '').replace('.tfrecord', '')
    except Exception:
        stem = f'episode_{local_idx:06d}'
    return sanitize(f'{suite}__{stem}__{local_idx:06d}')


def process_suite(raw_root: Path, output_dir: Path, suite: str, episodes_per_suite: int, min_steps: int, max_steps: int) -> int:
    dataset_dir = raw_root / suite / '1.0.0'
    if not dataset_dir.exists():
        raise FileNotFoundError(f'Missing dataset dir: {dataset_dir}')

    builder = tfds.builder_from_directory(str(dataset_dir))
    ds = builder.as_dataset(split='train')

    saved = 0
    seen = 0
    pbar = tqdm(ds, desc=f'Processing {suite}', unit='episode')
    for episode in pbar:
        if saved >= episodes_per_suite:
            break
        episode_name = get_episode_name(episode, suite, seen)
        seen += 1
        episode_dir = output_dir / episode_name
        image_dir = episode_dir / 'images'
        gripper_dir = episode_dir / 'gripper_images'
        action_dir = episode_dir / 'actions'

        if (episode_dir / 'instruction.txt').exists() and len(list(image_dir.glob('*.jpg'))) >= min_steps:
            saved += 1
            continue

        tmp_dir = output_dir / f'.tmp_{episode_name}'
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir)
        (tmp_dir / 'images').mkdir(parents=True, exist_ok=True)
        (tmp_dir / 'gripper_images').mkdir(parents=True, exist_ok=True)
        (tmp_dir / 'actions').mkdir(parents=True, exist_ok=True)

        language = None
        count = 0
        try:
            for i, step in enumerate(episode['steps']):
                if max_steps and count >= max_steps:
                    break
                obs = step['observation']
                image = Image.fromarray(obs['image'].numpy())
                wrist = Image.fromarray(obs['wrist_image'].numpy())
                action = step['action'].numpy()
                if language is None:
                    language = scalar_to_text(step['language_instruction'])

                image.save(tmp_dir / 'images' / f'{count}.jpg')
                wrist.save(tmp_dir / 'gripper_images' / f'{count}.jpg')
                np.save(tmp_dir / 'actions' / f'{count}.npy', action)
                count += 1
        except Exception:
            shutil.rmtree(tmp_dir, ignore_errors=True)
            raise

        if count < min_steps or not language:
            shutil.rmtree(tmp_dir, ignore_errors=True)
            continue

        with open(tmp_dir / 'instruction.txt', 'w', encoding='utf-8') as f:
            f.write(language)

        if episode_dir.exists():
            shutil.rmtree(episode_dir)
        tmp_dir.rename(episode_dir)
        saved += 1
        pbar.set_postfix(saved=saved, frames=count)

    return saved


def main():
    parser = argparse.ArgumentParser(description='Prepare a tiny LIBERO processed_data subset for UniVLA smoke tests.')
    parser.add_argument('--raw-root', default='datasets/raw/modified_libero_rlds')
    parser.add_argument('--output-dir', default='datasets/processed_data/libero_smoke')
    parser.add_argument('--suites', nargs='+', default=DEFAULT_SUITES)
    parser.add_argument('--episodes-per-suite', type=int, default=8)
    parser.add_argument('--min-steps', type=int, default=60)
    parser.add_argument('--max-steps-per-episode', type=int, default=80)
    args = parser.parse_args()

    raw_root = Path(args.raw_root).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    total = 0
    for suite in args.suites:
        total += process_suite(raw_root, output_dir, suite, args.episodes_per_suite, args.min_steps, args.max_steps_per_episode)

    print(f'Prepared {total} LIBERO smoke episodes in {output_dir}')


if __name__ == '__main__':
    main()
