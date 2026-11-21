# Object-Centric Learning with Developmental Constraints

**Student Index Number:** `268215K`  
**Selected Algorithm:** STEVE (Slot-Transformer for Videos, Singh et al., NeurIPS 2022) — `64x64` Input Configuration  

---

## Repository Structure

* `train.py` — Official STEVE training script (dVAE + Slot Attention + Transformer Decoder).
* `steve.py` — Core STEVE model architecture.
* `dvae.py` — Discrete Variational Autoencoder (dVAE) visual tokenizer.
* `transformer.py` — Autoregressive Transformer decoder.
* `data.py` — Video dataset loaders (`GlobVideoDataset` and `GlobVideoDatasetWithMasks`).
* `prepare_movi_data.py` — Bounded downloader and `64x64` preprocessor for the MOVi-C dataset.
* `eval_fgari_video.py` — Official Foreground Adjusted Rand Index (FG-ARI) evaluation script.
* `ari.py` — Adjusted Rand Index computation utilities.

---

## Phase 1: Baseline Reproduction on MOVi-C (`64x64`)

### 1. Training Configuration
* **Dataset:** MOVi-C (`128x128` streamed from Kubric TFDS and resized to `64x64`)
  * Training split: 210 videos (70% of 300 downloaded training videos)
  * Validation split: 45 videos (15% of 300 downloaded training videos)
  * Evaluation split (with ground-truth masks): 60 videos from MOVi-C validation split
* **Input Resolution:** `64 x 64` (`--image_size 64`)
* **Episode Length:** `3` frames for training (`--ep_len 3`), `6` frames for FG-ARI evaluation (`--ep_len 6`)
* **Number of Slots:** `11` (`--num_slots 11`)
* **Batch Size:** `24` (training) / `20` (evaluation)
* **Learning Rates & Schedule:** `lr_dvae=3e-4`, `lr_enc=1e-4`, `lr_dec=3e-4`, `lr_warmup_steps=2000`, `tau_steps=3000`, `lr_half_life=15000`
* **Hardware:** Kaggle 2x NVIDIA Tesla T4 GPUs

### 2. Quantitative Results (Phase 1)

| Task | Dataset | Resolution | FG-ARI (%) |
| :--- | :--- | :---: | :---: |
| **Task 1 (Phase 1)** | MOVi-C (Validation Subset, 60 videos) | `64 x 64` | **39.67** |

### 3. Instructions to Replicate Phase 1

1. **Install Dependencies:**
   ```bash
   pip install torch torchvision einops tqdm tensorboard scipy scikit-learn opencv-python imageio moviepy tensorflow-datasets
   ```

2. **Download & Preprocess MOVi-C (`64x64`):**
   ```bash
   python prepare_movi_data.py --level c --image_size 64 --train_videos 300 --eval_videos 60
   ```

3. **Train STEVE (`64x64`):**
   ```bash
   python train.py --image_size 64 --batch_size 24 --num_workers 2 --ep_len 3 --num_slots 11 --epochs 250 --steps 17500 --lr_warmup_steps 2000 --tau_steps 3000 --lr_half_life 15000 --data_path 'data/movi_c_64/*' --log_path 'logs_phase1/'
   ```

4. **Evaluate FG-ARI:**
   ```bash
   python eval_fgari_video.py --image_size 64 --batch_size 20 --num_workers 2 --ep_len 6 --num_slots 11 --data_path 'eval_data/movi_c_64/*' --data_num_segs_per_frame 25 --trained_model_paths logs_phase1/<TIMESTAMP>/best_model.pt
   ```
