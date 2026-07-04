import argparse
import os
import re
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoImageProcessor, AutoModel


def natural_key(path: Path):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r'(\d+)', path.stem)]


def load_batch(paths, size):
    images = []
    for path in paths:
        with Image.open(path) as img:
            images.append(img.convert('RGB').resize((size, size)))
    return images


def main():
    parser = argparse.ArgumentParser(description='Extract Emu3 VisionVQ codes for LIBERO static camera images.')
    parser.add_argument('--model-path', default='pretrain/Emu3-VisionVQ')
    parser.add_argument('--input-dir', default='datasets/processed_data/libero_smoke')
    parser.add_argument('--output-dir', default='datasets/processed_data/libero_smoke_codes_200')
    parser.add_argument('--image-subdir', default='images')
    parser.add_argument('--size', type=int, default=200)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--min-pixels', type=int, default=128 * 128)
    args = parser.parse_args()

    model_path = Path(args.model_path).resolve()
    input_dir = Path(args.input_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    processor = AutoImageProcessor.from_pretrained(str(model_path), trust_remote_code=True)
    processor.min_pixels = args.min_pixels
    model = AutoModel.from_pretrained(str(model_path), trust_remote_code=True).eval().cuda()

    episodes = sorted([p for p in input_dir.iterdir() if p.is_dir()])
    for episode in tqdm(episodes, desc='VQ episodes', unit='episode'):
        image_dir = episode / args.image_subdir
        if not image_dir.exists():
            continue
        image_paths = sorted(image_dir.glob('*.jpg'), key=natural_key)
        save_dir = output_dir / episode.name
        existing = list(save_dir.glob('*.npy')) if save_dir.exists() else []
        if len(existing) >= len(image_paths) and image_paths:
            continue
        save_dir.mkdir(parents=True, exist_ok=True)

        for start in range(0, len(image_paths), args.batch_size):
            batch_paths = image_paths[start:start + args.batch_size]
            images = load_batch(batch_paths, args.size)
            inputs = processor(images, return_tensors='pt')['pixel_values'].cuda()
            with torch.no_grad():
                codes = model.encode(inputs)
            for path, code in zip(batch_paths, codes):
                # Keep the leading batch-like dimension expected by train/datasets.py.
                np.save(save_dir / f'{path.stem}.npy', code.detach().cpu().unsqueeze(0).numpy())

    print(f'Saved LIBERO static-camera VQ codes to {output_dir}')


if __name__ == '__main__':
    main()
