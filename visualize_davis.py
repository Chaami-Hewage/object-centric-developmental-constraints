import os
import argparse
import numpy as np
import torch
from PIL import Image
from steve import STEVE
from data import GlobVideoDatasetWithMasks

# Distinct colors for up to 15 slots
PALETTE = np.array([
    [0, 0, 0],       [230, 25, 75],   [60, 180, 75],   [255, 225, 25],
    [0, 130, 200],   [245, 130, 48],  [145, 30, 180],  [70, 240, 240],
    [240, 50, 230],  [210, 245, 60],  [250, 190, 212], [0, 128, 128],
    [220, 190, 255], [170, 110, 40],  [255, 250, 200], [128, 0, 0]
], dtype=np.uint8)

def colorize_mask(mask_2d):
    return PALETTE[mask_2d % len(PALETTE)]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint_path', default='logs_phase2/best_model_davis.pt')
    parser.add_argument('--data_path', default='davis_eval/*')
    parser.add_argument('--out_dir', default='results/phase2')
    parser.add_argument('--image_size', type=int, default=64)
    parser.add_argument('--img_channels', type=int, default=3)
    parser.add_argument('--ep_len', type=int, default=6)
    parser.add_argument('--data_num_segs_per_frame', type=int, default=15)
    parser.add_argument('--num_iterations', type=int, default=2)
    parser.add_argument('--num_slots', type=int, default=15)
    parser.add_argument('--cnn_hidden_size', type=int, default=64)
    parser.add_argument('--slot_size', type=int, default=192)
    parser.add_argument('--mlp_hidden_size', type=int, default=192)
    parser.add_argument('--num_predictor_blocks', type=int, default=1)
    parser.add_argument('--num_predictor_heads', type=int, default=4)
    parser.add_argument('--predictor_dropout', type=float, default=0.0)
    parser.add_argument('--vocab_size', type=int, default=4096)
    parser.add_argument('--num_decoder_blocks', type=int, default=8)
    parser.add_argument('--num_decoder_heads', type=int, default=4)
    parser.add_argument('--d_model', type=int, default=192)
    parser.add_argument('--dropout', type=float, default=0.1)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    dataset = GlobVideoDatasetWithMasks(
        root=args.data_path, img_size=args.image_size,
        num_segs=args.data_num_segs_per_frame, ep_len=args.ep_len,
        img_glob='????????_image.png'
    )

    model = STEVE(args)
    ckpt = torch.load(args.checkpoint_path, map_location='cpu')
    model.load_state_dict(ckpt['model'] if 'model' in ckpt else ckpt)
    model = model.cuda().eval()

    # Pick 4 evenly spaced episodes across DAVIS validation videos
    sample_indices = np.linspace(0, len(dataset) - 1, 4, dtype=int)

    with torch.no_grad():
        for idx_num, ep_idx in enumerate(sample_indices):
            video, true_masks = dataset[ep_idx]  # (T, C, H, W), (T, K, 1, H, W)
            _, _, pred_masks = model.encode(video.unsqueeze(0).cuda())
            pred_hard = pred_masks[0].squeeze(2).argmax(dim=1).cpu().numpy()  # (T, H, W)
            gt_hard = true_masks.squeeze(2).argmax(dim=1).numpy()              # (T, H, W)

            rows = []
            for t in range(args.ep_len):
                rgb = (video[t].permute(1, 2, 0).numpy() * 255).astype(np.uint8)
                gt_vis = colorize_mask(gt_hard[t])
                pr_vis = colorize_mask(pred_hard[t])
                # Upscale 2x (128x128) for clearer viewing in README
                col = np.concatenate([rgb, gt_vis, pr_vis], axis=0)
                rows.append(col)

            grid = np.concatenate(rows, axis=1)
            grid_img = Image.fromarray(grid).resize((grid.shape[1] * 2, grid.shape[0] * 2), Image.NEAREST)
            out_path = os.path.join(args.out_dir, f"davis_sample_{idx_num+1}.png")
            grid_img.save(out_path)
            print(f"Saved {out_path}")

if __name__ == '__main__':
    main()
