import math
import os
import argparse
import torch
from torch.optim import Adam
from torch.nn import DataParallel as DP
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader

from steve import STEVE
from data import GlobVideoDataset
from utils import cosine_anneal, linear_warmup

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--num_workers', type=int, default=2)
parser.add_argument('--image_size', type=int, default=64)
parser.add_argument('--img_channels', type=int, default=3)
parser.add_argument('--ep_len', type=int, default=3)

parser.add_argument('--pretrained_weights', default='phase1_best_model.pt')
parser.add_argument('--train_path', default='davis_train/*')
parser.add_argument('--val_path', default='davis_eval/*')
parser.add_argument('--log_path', default='logs_phase2')

parser.add_argument('--lr_dvae', type=float, default=1e-4)
parser.add_argument('--lr_enc', type=float, default=5e-5)
parser.add_argument('--lr_dec', type=float, default=1e-4)
parser.add_argument('--lr_warmup_steps', type=int, default=500)
parser.add_argument('--lr_half_life', type=int, default=50000)
parser.add_argument('--clip', type=float, default=0.05)
parser.add_argument('--epochs', type=int, default=25)

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

parser.add_argument('--tau_start', type=float, default=0.5)
parser.add_argument('--tau_final', type=float, default=0.1)
parser.add_argument('--tau_steps', type=int, default=3000)
parser.add_argument('--hard', action='store_true')
args = parser.parse_args()

torch.manual_seed(args.seed)
os.makedirs(args.log_path, exist_ok=True)

train_dataset = GlobVideoDataset(root=args.train_path, phase='all', img_size=args.image_size, ep_len=args.ep_len, img_glob='*.png')
val_dataset = GlobVideoDataset(root=args.val_path, phase='all', img_size=args.image_size, ep_len=args.ep_len, img_glob='*_image.png')

print(f"Train episodes: {len(train_dataset)} | Val episodes: {len(val_dataset)}")

train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True, drop_last=True)
val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True, drop_last=False)

model = STEVE(args)
if os.path.isfile(args.pretrained_weights):
    state_dict = torch.load(args.pretrained_weights, map_location='cpu')
    model.load_state_dict(state_dict['model'] if 'model' in state_dict else state_dict)
    print(f"Loaded Phase 1 pretrained weights from {args.pretrained_weights}")

use_dp = torch.cuda.device_count() > 1
model = model.cuda()
if use_dp:
    model = DP(model)

optimizer = Adam([
    {'params': (x[1] for x in model.named_parameters() if 'dvae' in x[0]), 'lr': args.lr_dvae},
    {'params': (x[1] for x in model.named_parameters() if 'steve_encoder' in x[0]), 'lr': 0.0},
    {'params': (x[1] for x in model.named_parameters() if 'steve_decoder' in x[0]), 'lr': 0.0},
])

train_epoch_size = len(train_loader)
best_val_loss = math.inf

for epoch in range(args.epochs):
    model.train()
    train_loss = 0.0
    for batch, video in enumerate(train_loader):
        global_step = epoch * train_epoch_size + batch
        tau = cosine_anneal(global_step, args.tau_start, args.tau_final, 0, args.tau_steps)
        lr_warmup = linear_warmup(global_step, 0., 1.0, 0., args.lr_warmup_steps)
        lr_decay = math.exp(global_step / args.lr_half_life * math.log(0.5))

        optimizer.param_groups[0]['lr'] = args.lr_dvae
        optimizer.param_groups[1]['lr'] = lr_decay * lr_warmup * args.lr_enc
        optimizer.param_groups[2]['lr'] = lr_decay * lr_warmup * args.lr_dec

        video = video.cuda()
        optimizer.zero_grad()
        recon, cross_entropy, mse, attns = model(video, tau, args.hard)
        if use_dp:
            mse = mse.mean()
            cross_entropy = cross_entropy.mean()
        loss = mse + cross_entropy
        loss.backward()
        clip_grad_norm_(model.parameters(), args.clip, 'inf')
        optimizer.step()
        train_loss += loss.item()

    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for video in val_loader:
            video = video.cuda()
            recon, cross_entropy, mse, attns = model(video, args.tau_final, args.hard)
            if use_dp:
                mse = mse.mean()
                cross_entropy = cross_entropy.mean()
            val_loss += (mse + cross_entropy).item()

    train_loss /= len(train_loader)
    val_loss /= len(val_loader)
    print(f"Epoch [{epoch+1:02d}/{args.epochs}] | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        save_path = os.path.join(args.log_path, 'best_model_davis.pt')
        torch.save(model.module.state_dict() if use_dp else model.state_dict(), save_path)
        print(f"  --> Saved new best DAVIS model (Val Loss: {best_val_loss:.4f})")
