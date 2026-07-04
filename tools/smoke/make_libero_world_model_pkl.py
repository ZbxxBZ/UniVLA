import argparse
import pickle
import re
from pathlib import Path


def natural_key(path: Path):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r'(\d+)', path.stem)]


def main():
    parser = argparse.ArgumentParser(description='Create a world-model-only LIBERO smoke pickle.')
    parser.add_argument('--processed-dir', default='datasets/processed_data/libero_smoke')
    parser.add_argument('--vq-dir', default='datasets/processed_data/libero_smoke_codes_200')
    parser.add_argument('--output', default='datasets/post_train_data/meta/libero_world_model_smoke.pkl')
    parser.add_argument('--min-frames', type=int, default=60)
    args = parser.parse_args()

    processed_dir = Path(args.processed_dir).resolve()
    vq_dir = Path(args.vq_dir).resolve()
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    samples = []
    skipped = 0
    for episode in sorted([p for p in processed_dir.iterdir() if p.is_dir()]):
        instr = episode / 'instruction.txt'
        codes_dir = vq_dir / episode.name
        if not instr.exists() or not codes_dir.exists():
            skipped += 1
            continue
        code_files = sorted(codes_dir.glob('*.npy'), key=natural_key)
        if len(code_files) < args.min_frames:
            skipped += 1
            continue
        text = instr.read_text(encoding='utf-8').strip()
        if not text:
            skipped += 1
            continue
        samples.append({
            'text': text,
            'image': [str(p) for p in code_files],
            'dataset': 'libero',
        })

    if not samples:
        raise RuntimeError('No valid LIBERO world-model samples were generated.')

    with open(output, 'wb') as f:
        pickle.dump(samples, f)

    print(f'Saved {len(samples)} samples to {output}; skipped={skipped}')
    print('Example:', samples[0]['text'], len(samples[0]['image']), samples[0]['image'][0])


if __name__ == '__main__':
    main()
