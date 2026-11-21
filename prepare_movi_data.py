import os
import argparse
import torch
import torch.nn.functional as F
import tensorflow_datasets as tfds
import torchvision.utils as vutils
from torchvision import transforms


def download_movi_subset(out_path, level, split, target_size, max_videos, save_masks=False, max_num_objs=25):
    """
    Streams a subset of the MOVi dataset (128x128) from Google Cloud TFDS,
    resizes frames/masks to target_size (64x64), and saves them in STEVE's format.
    """
    os.makedirs(out_path, exist_ok=True)
    # Kubric TFDS hosts 128x128 and 256x256; we load 128x128 and resize to 64x64
    dataset_name = f"movi_{level}/128x128:1.0.0"
    print(f"Streaming {dataset_name} [{split}] -> {out_path} (Resizing to {target_size}x{target_size}, Max videos: {max_videos})...")

    ds = tfds.load(dataset_name, data_dir="gs://kubric-public/tfds", split=split)
    data_iter = iter(tfds.as_numpy(ds))
    to_tensor = transforms.ToTensor()
    resize_rgb = transforms.Resize((target_size, target_size), antialias=True)

    count = 0
    for record in data_iter:
        if count >= max_videos:
            break

        video = record["video"]  # Shape: (T, 128, 128, 3)
        T = video.shape[0]
        vid_dir = os.path.join(out_path, f"{count:08d}")
        os.makedirs(vid_dir, exist_ok=True)

        masks = record.get("segmentations", None) if save_masks else None

        for t in range(T):
            img_tensor = resize_rgb(to_tensor(video[t]))
            vutils.save_image(img_tensor, os.path.join(vid_dir, f"{t:08d}_image.png"))

            if save_masks and masks is not None:
                for n in range(max_num_objs):
                    mask = (masks[t] == n).astype(float)
                    mask_tensor = torch.tensor(mask, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
                    mask_resized = F.interpolate(mask_tensor, size=(target_size, target_size), mode="nearest").squeeze(0)
                    vutils.save_image(mask_resized, os.path.join(vid_dir, f"{t:08d}_mask_{n:02d}.png"))

        count += 1
        if count % 50 == 0:
            print(f"  Saved {count}/{max_videos} videos...")

    print(f"Completed! Total saved in {out_path}: {count} videos.\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download bounded MOVi-C dataset and resize to 64x64 for STEVE.")
    parser.add_argument("--level", type=str, default="c", help="MOVi difficulty level (a, b, c, d, e)")
    parser.add_argument("--image_size", type=int, default=64, help="Target frame resolution (64 for Phase 1)")
    parser.add_argument("--train_videos", type=int, default=300, help="Number of training videos to download")
    parser.add_argument("--eval_videos", type=int, default=60, help="Number of validation videos with masks")
    args = parser.parse_args()

    # Download training videos (RGB frames only)
    download_movi_subset(
        out_path="data/movi_c_64",
        level=args.level,
        split="train",
        target_size=args.image_size,
        max_videos=args.train_videos,
        save_masks=False,
    )

    # Download evaluation videos (RGB frames + ground-truth masks for FG-ARI)
    download_movi_subset(
        out_path="eval_data/movi_c_64",
        level=args.level,
        split="validation",
        target_size=args.image_size,
        max_videos=args.eval_videos,
        save_masks=True,
        max_num_objs=25,
    )
