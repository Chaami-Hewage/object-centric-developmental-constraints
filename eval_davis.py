import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
from steve import STEVE
from data import GlobVideoDatasetWithMasks
from ari import evaluate_ari

def compute_mbo(true_masks, pred_masks):
    """
    true_masks: (B, T, K_true, C, H, W) binary masks, channel 0 is background
    pred_masks: (B, T, K_pred, C, H, W) soft/hard slot masks
    Computes Mean Best Overlap (mBO) across all foreground ground-truth objects.
    """
    B, T, K_true, _, H, W = true_masks.shape
    K_pred = pred_masks.shape[2]

    # Convert predicted masks to hard slot assignments per pixel
    pred_hard = pred_masks.squeeze(3).argmax(dim=2)  # (B, T, H, W)

    best_ious = []
    for b in range(B):
        for t in range(T):
            for k in range(1, K_true):  # skip background (k=0)
                gt = true_masks[b, t, k, 0] > 0.5
                if gt.sum() == 0:
                    continue
                max_iou = 0.0
                for s in range(K_pred):
                    pr = (pred_hard[b, t] == s)
                    inter = (gt & pr).sum().item()
                    union = (gt | pr).sum().item()
                    if union > 0:
                        iou = inter / union
                        if iou > max_iou:
                            max_iou = iou
                best_ious.append(max_iou)
    return 100.0 * np.mean(best_ious) if best_ious else 0.0

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--image_size', type=int, default=64)
    parser.add_argument('--img_channels', type=int, default=3)
    parser.add_argument('--ep_len', type=int, default=6)
    parser.add_argument('--checkpoint_path', type=str, required=True)
    parser.add_argument('--data_path', type=str, default='davis_eval/*')
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

    torch.manual_seed(args.seed)
    eval_dataset = GlobVideoDatasetWithMasks(
        root=args.data_path,
        img_size=args.image_size,
        num_segs=args.data_num_segs_per_frame,
        ep_len=args.ep_len,
        img_glob='????????_image.png'
    )
    eval_loader = DataLoader(
        eval_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True, drop_last=False
    )

    model = STEVE(args)
    ckpt = torch.load(args.checkpoint_path, map_location='cpu')
    if 'model' in ckpt:
        model.load_state_dict(ckpt['model'])
    else:
        model.load_state_dict(ckpt)
    model = model.cuda()
    model.eval()

    fgaris, mbos = [], []
    with torch.no_grad():
        for video, true_masks in eval_loader:
            video = video.cuda()
            _, _, pred_masks = model.encode(video)

            fgari = 100.0 * evaluate_ari(
                true_masks.permute(0, 2, 1, 3, 4, 5)[:, 1:].flatten(start_dim=2),
                pred_masks.permute(0, 2, 1, 3, 4, 5).flatten(start_dim=2)
            )
            mbo = compute_mbo(true_masks.cpu(), pred_masks.cpu())
            fgaris.append(fgari)
            mbos.append(mbo)

    print(f"Checkpoint: {args.checkpoint_path}")
    print(f"DAVIS 2017 Val -> FG-ARI: {np.mean(fgaris):.2f}% | mBO: {np.mean(mbos):.2f}%")

if __name__ == '__main__':
    main()
