import os
import os.path as osp
import pickle
from tqdm import tqdm
import numpy as np
import sys
import argparse
import pathlib
# ====================== Config ======================

# Project-specific paths
PROJECT_ROOT = os.environ.get("PROJECT_ROOT", str(pathlib.Path(__file__).resolve().parents[2]))
sys.path.append(PROJECT_ROOT)

# Import normalization utility
from train.dataset.normalize_pi0 import RunningStats, save

parser = argparse.ArgumentParser(description="Generate a normalized CALVIN policy pickle.")
parser.add_argument("--dataset_name", type=str, default=os.environ.get("DATASET_NAME", "calvin"), help="Dataset directory name under processed_data.")
parser.add_argument("--dataset_path", type=str, default=os.environ.get("DATASET_PATH"), help="Root processed_data path.")
parser.add_argument("--vq_dir", type=str, default=os.environ.get("VQ_DIR"), help="Static-camera VQ token directory.")
parser.add_argument("--gripper_vq_dir", type=str, default=os.environ.get("GRIPPER_VQ_DIR"), help="Gripper-camera VQ token directory.")
parser.add_argument("--output_path", type=str, default=os.environ.get("OUTPUT_PATH"), help="Directory to save the pickle.")
parser.add_argument("--normalizer_path", type=str, default=os.environ.get("NORMALIZER_PATH"), help="Directory to save normalization stats.")
parser.add_argument("--output_filename", type=str, default=os.environ.get("OUTPUT_FILENAME"), help="Output pickle filename.")
parser.add_argument("--min_frame_count", type=int, default=int(os.environ.get("MIN_FRAME_COUNT", "8")))
parser.add_argument("--use_raw_images", action="store_true", default=os.environ.get("USE_RAW_IMAGES", "False").lower() in ("1", "true", "yes"))
args = parser.parse_args()

# Input paths
DATASET_NAME = args.dataset_name  # Options: calvin, calvin_d, calvin_abc
DATA_ROOT = os.environ.get("DATA_ROOT", osp.join(PROJECT_ROOT, "datasets"))
dataset_path = args.dataset_path or osp.join(DATA_ROOT, "processed_data")
language_dir = osp.join(dataset_path, DATASET_NAME)
vq_dir = args.vq_dir or osp.join(dataset_path, f"{DATASET_NAME}_codes")
gripper_vq_dir = args.gripper_vq_dir or osp.join(dataset_path, f"{DATASET_NAME}_gripper_codes")

# Output paths
output_path = args.output_path or osp.join(dataset_path, "meta")
normalizer_path = args.normalizer_path or osp.join(PROJECT_ROOT, "configs", "normalizer_calvin")
output_pkl_file = osp.join(output_path, args.output_filename or f"{DATASET_NAME}_norm.pkl")

# Settings
interval = 1           # Not currently used but may be useful
min_frame_count = args.min_frame_count    # Minimum frame count per scene
use_raw_images = args.use_raw_images # If True, use raw RGB images instead of VQ codes

# ====================================================

# Ensure output dirs exist
os.makedirs(normalizer_path, exist_ok=True)
os.makedirs(output_path, exist_ok=True)

# Load and process dataset
result_file = []
for scene in tqdm(os.listdir(language_dir), desc="Processing scenes"):
    instr_file = osp.join(language_dir, scene, "instruction.txt")
    if not osp.exists(instr_file):
        print(f"Warning: Missing instruction file in {scene}")
        continue

    with open(instr_file, "r") as f:
        text = f.read().strip()

    # Load action data
    action_folder = osp.join(language_dir, scene, "actions")
    if not osp.exists(action_folder):
        print(f"Warning: Missing action folder in {scene}")
        continue

    action_files = sorted([
        osp.join(action_folder, fname)
        for fname in os.listdir(action_folder)
        if fname.endswith(".npz")
    ])

    try:
        actions = [np.load(f)["rel_actions"] for f in action_files]
    except Exception as e:
        print(f"Error loading actions for {scene}: {e}")
        continue

    # Load image paths
    if use_raw_images:
        img_dir = osp.join(language_dir, scene, "rgb_static")
        gripper_dir = osp.join(language_dir, scene, "rgb_gripper")
    else:
        img_dir = osp.join(vq_dir, scene)
        gripper_dir = osp.join(gripper_vq_dir, scene)

    try:
        img_files = sorted([
            osp.join(img_dir, fname) for fname in os.listdir(img_dir)
        ])
        gripper_img_files = sorted([
            osp.join(gripper_dir, fname) for fname in os.listdir(gripper_dir)
        ])
    except FileNotFoundError:
        print(f"Warning: Missing VQ images in {scene}")
        continue

    # Skip scenes with too few frames
    if len(img_files) < min_frame_count:
        continue

    result_file.append({
        "text": text,
        "image": img_files,
        "action": actions,
        "gripper_image": gripper_img_files,
    })

print(f"\nTotal valid scenes: {len(result_file)}")

# Initialize and compute normalization statistics
if not result_file:
    raise RuntimeError("No valid scenes found. Exiting.")

normalizer = RunningStats()
action_data = np.concatenate([scene["action"] for scene in result_file])
normalizer.update(action_data)
stats = normalizer.get_statistics()

# Print stats
print("Normalization statistics:")
print("  Mean:", stats.mean)
print("  Std:", stats.std)
print("  Q01:", stats.q01)
print("  Q99:", stats.q99)

# Save normalization parameters
save(normalizer_path, {DATASET_NAME: stats})

# Normalize all actions
for scene in result_file:
    action = scene["action"]
    normalized = 2 * (action - stats.q01) / (stats.q99 - stats.q01 + 1e-8) - 1
    scene["action"] = np.clip(normalized, -1, 1)

# Save result
with open(output_pkl_file, "wb") as f:
    pickle.dump(result_file, f)

print(f"\nProcessed dataset saved to: {output_pkl_file}")
