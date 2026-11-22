import os
import numpy as np
from PIL import Image
from tqdm import tqdm

DAVIS_ROOT = "DAVIS"
IMG_SIZE = 64
NUM_SEGS = 15  # 0 is background, 1..14 are foreground object instances

def read_split(split_file):
    with open(split_file, "r") as f:
        return [line.strip() for line in f if line.strip()]

def prepare_train(videos, out_root="davis_train"):
    os.makedirs(out_root, exist_ok=True)
    for vid in tqdm(videos, desc="Preparing DAVIS Train"):
        src_dir = os.path.join(DAVIS_ROOT, "JPEGImages", "480p", vid)
        dst_dir = os.path.join(out_root, vid)
        os.makedirs(dst_dir, exist_ok=True)
        frames = sorted(os.listdir(src_dir))
        for idx, fname in enumerate(frames):
            img = Image.open(os.path.join(src_dir, fname)).convert("RGB")
            img = img.resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
            img.save(os.path.join(dst_dir, f"{idx:08d}.png"))

def prepare_eval(videos, out_root="davis_eval"):
    os.makedirs(out_root, exist_ok=True)
    for vid in tqdm(videos, desc="Preparing DAVIS Eval"):
        img_dir = os.path.join(DAVIS_ROOT, "JPEGImages", "480p", vid)
        ann_dir = os.path.join(DAVIS_ROOT, "Annotations", "480p", vid)
        dst_dir = os.path.join(out_root, vid)
        os.makedirs(dst_dir, exist_ok=True)

        frames = sorted(os.listdir(img_dir))
        for idx, fname in enumerate(frames):
            # Save RGB frame
            img = Image.open(os.path.join(img_dir, fname)).convert("RGB")
            img = img.resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
            img.save(os.path.join(dst_dir, f"{idx:08d}_image.png"))

            # Load indexed annotation mask (0=BG, 1..N=objects)
            ann_name = os.path.splitext(fname)[0] + ".png"
            ann = Image.open(os.path.join(ann_dir, ann_name))
            ann = ann.resize((IMG_SIZE, IMG_SIZE), Image.NEAREST)
            ann_arr = np.array(ann)

            # Save binary masks: mask_00 = background, mask_01..14 = object instances
            for seg_id in range(NUM_SEGS):
                bin_mask = (ann_arr == seg_id).astype(np.uint8) * 255
                Image.fromarray(bin_mask, mode="L").save(
                    os.path.join(dst_dir, f"{idx:08d}_mask_{seg_id:02d}.png")
                )

if __name__ == "__main__":
    train_videos = read_split(os.path.join(DAVIS_ROOT, "ImageSets", "2017", "train.txt"))
    val_videos = read_split(os.path.join(DAVIS_ROOT, "ImageSets", "2017", "val.txt"))
    print(f"Found {len(train_videos)} train videos and {len(val_videos)} val videos.")
    prepare_train(train_videos)
    prepare_eval(val_videos)
    print("DAVIS 2017 preparation complete!")
