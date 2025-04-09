import gc
import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torch.amp import GradScaler, autocast
from torch.utils import tensorboard
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

# Adjust imports based on the new structure
from data.dataloader import get_dataloader # Removed src. prefix
from utils.kl_annealing import kl_annealing # Removed src. prefix
from .modules import (
    Decoder_Fusion,
    Gaussian_Predictor,
    Generator,
    Label_Encoder,
    RGB_Encoder,
)

def Generate_PSNR(img1, img2):
    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0:
        return torch.tensor(100.0)
    PIXEL_MAX = 1.0
    return 20 * torch.log10(PIXEL_MAX / torch.sqrt(mse))


class VAE_Model(nn.Module):
    def __init__(self, args):
        super(VAE_Model, self).__init__()
        self.args = args
        self.debug = getattr(args, 'debug', False) # Add default value for debug

        # Modules to transform image from RGB-domain to feature-domain
        self.frame_transformation = RGB_Encoder(3, args.F_dim)
        self.label_transformation = Label_Encoder(3, args.L_dim)

        # Conduct Posterior prediction in Encoder
        # Adjust Gaussian_Predictor input channels based on args
        gauss_in_chans = args.F_dim + args.L_dim
        self.Gaussian_Predictor = Gaussian_Predictor(
            in_chans=gauss_in_chans, out_chans=args.N_dim
        )

        # Adjust Decoder_Fusion input channels based on args
        decoder_in_chans = args.F_dim + args.L_dim + args.N_dim
        self.Decoder_Fusion = Decoder_Fusion(
            in_chans=decoder_in_chans, out_chans=args.D_out_dim
        )

        # Generative model
        self.Generator = Generator(input_nc=args.D_out_dim, output_nc=3)

        # Initialize optimizer based on args.optim
        if args.optim == "Adam":
            self.optim = optim.Adam(self.parameters(), lr=self.args.lr)
        elif args.optim == "AdamW":
            self.optim = optim.AdamW(self.parameters(), lr=self.args.lr)
        else:
            raise ValueError(f"Unsupported optimizer: {args.optim}")

        self.scheduler = optim.lr_scheduler.MultiStepLR(
            self.optim, milestones=self.args.milestones, gamma=self.args.gamma
        )
        # Initialize kl_annealing with current_epoch=0 initially
        self.kl_annealing = kl_annealing(args, current_epoch=0)
        self.mse_criterion = nn.MSELoss()
        self.current_epoch = 1 # Start from epoch 1

        # Teacher forcing arguments
        self.tfr = args.tfr
        self.tfr_d_step = args.tfr_d_step
        self.tfr_sde = args.tfr_sde

        self.train_vi_len = args.train_vi_len
        self.val_vi_len = args.val_vi_len
        self.batch_size = args.batch_size

        # Setup TensorBoard writer path
        log_dir_suffix = f"kl-type_{args.kl_anneal_type}_tfr_{args.tfr}_teacher-decay_{args.tfr_d_step}"
        self.log_dir = os.path.join(self.args.save_root, 'logs', log_dir_suffix)
        os.makedirs(self.log_dir, exist_ok=True)
        self.writer = tensorboard.SummaryWriter(self.log_dir)

        # Setup checkpoint save path
        self.checkpoint_dir = os.path.join(self.args.save_root, 'checkpoints', log_dir_suffix)
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        # AMP scaler
        self.use_amp = not getattr(args, 'no_use_amp', False) # Add default value
        self.scaler = GradScaler(init_scale=2.0**16) if self.use_amp else None

        # Track best PSNR
        self.best_psnr = 0.0

    # Note: Forward is not explicitly defined here, assuming it's implicitly handled by modules
    # Or should be implemented if VAE_Model itself needs a forward pass logic.

    def training_stage(self):
        for i in range(self.current_epoch -1, self.args.num_epoch): # Adjust loop range if resuming
            train_loader = self.train_dataloader()
            adapt_TeacherForcing = True if random.random() < self.tfr else False

            train_losses = []
            self.train() # Set model to training mode
            for img, label in (pbar := tqdm(train_loader, ncols=120)):

                img = img.to(self.args.device)
                label = label.to(self.args.device)
                if adapt_TeacherForcing:
                    loss = self.training_one_step_with_teacher_forcing(img, label)
                else:
                    loss = self.training_one_step_without_teacher_forcing(img, label)

                # Check for NaN loss
                if torch.isnan(loss):
                    print(f"Epoch {self.current_epoch}: NaN loss detected. Skipping step.")
                    # Optionally clear gradients and continue
                    self.optim.zero_grad()
                    continue

                train_losses.append(loss.item()) # Use .item() to avoid memory leak

                beta = self.kl_annealing.get_beta()

                tf_status = "ON" if adapt_TeacherForcing else "OFF"
                self.tqdm_bar(
                    f"train [TF: {tf_status}, {self.tfr:.1f}], beta: {beta:.4f}",
                    pbar,
                    loss.item(),
                    lr=self.scheduler.get_last_lr()[0],
                )

            mean_train_loss = np.mean([l for l in train_losses if not np.isnan(l)]) if train_losses else 0.0
            mean_val_loss, mean_psnr = self.eval_stage()

            # Save best model based on PSNR
            if mean_psnr > self.best_psnr:
                self.best_psnr = mean_psnr
                save_path = os.path.join(
                    self.checkpoint_dir, f"best_epoch-{self.current_epoch}_psnr-{mean_psnr:.4f}.ckpt"
                )
                self.save(save_path)
                print(f"Saved best model to {save_path} (PSNR: {mean_psnr:.4f})")

            # Save periodic checkpoint
            if self.current_epoch % self.args.per_save == 0:
                save_path = os.path.join(
                    self.checkpoint_dir, f"epoch-{self.current_epoch}.ckpt"
                )
                self.save(save_path)

            # Log metrics to TensorBoard
            self.writer.add_scalar("Loss/train", mean_train_loss, self.current_epoch)
            self.writer.add_scalar("Loss/val", mean_val_loss, self.current_epoch)
            self.writer.add_scalar("PSNR/val", mean_psnr, self.current_epoch)
            self.writer.add_scalar("beta", beta, self.current_epoch)
            self.writer.add_scalar("tfr", self.tfr, self.current_epoch)
            self.writer.add_scalar("learning_rate", self.scheduler.get_last_lr()[0], self.current_epoch)

            print(
                f"Epoch {self.current_epoch} | Train Loss: {mean_train_loss:.6f} | Val Loss: {mean_val_loss:.6f} | Val PSNR: {mean_psnr:.4f} | Beta: {beta:.4f} | TFR: {self.tfr:.2f}"
            )

            # Update learning rate, teacher forcing ratio, and KL annealing
            self.scheduler.step()
            self.teacher_forcing_ratio_update()
            self.kl_annealing.update()
            self.current_epoch += 1

            # Clear cache
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    @torch.no_grad()
    def eval_stage(self):
        self.eval()
        # self.train(False) # Use self.train(False) instead of recursive self.eval()
        val_loader = self.val_dataloader()
        PSNRS = []
        val_losses = []
        save_imgs = []
        first_batch = True
        val_demo_dir = os.path.join(self.args.save_root, 'val_demos', f"kl-type_{self.args.kl_anneal_type}_tfr_{self.args.tfr}_teacher-decay_{self.args.tfr_d_step}")
        os.makedirs(val_demo_dir, exist_ok=True)

        for img, label in tqdm(val_loader, desc="Evaluating", ncols=100):
            img = img.to(self.args.device)
            label = label.to(self.args.device)
            loss, psnr_list, generated_imgs = self.val_one_step(img, label)

            if not torch.isnan(loss):
                val_losses.append(loss.item())
                PSNRS.extend(psnr_list)
            else:
                print(f"Epoch {self.current_epoch}: NaN loss detected during validation. Skipping batch.")

            # Save images from the first batch for GIF generation
            if first_batch:
                # Ensure generated_imgs contains tensors
                save_imgs.extend([frame.cpu() for frame in generated_imgs if isinstance(frame, torch.Tensor)])
                first_batch = False

        mean_val_loss = np.mean([l for l in val_losses if not np.isnan(l)]) if val_losses else 0.0
        mean_psnr = np.mean([p for p in PSNRS if not np.isnan(p)]) if PSNRS else 0.0

        # --- Debug PSNR --- #
        if len(PSNRS) > 10:
            print(f"[Debug] eval: Final PSNRS list sample (first 5): {PSNRS[:5]}, (last 5): {PSNRS[-5:]} (len: {len(PSNRS)})")
        else:
            print(f"[Debug] eval: Final PSNRS list sample: {PSNRS} (len: {len(PSNRS)})") # Print all if less than 10
        print(f"[Debug] eval: Calculated mean_psnr: {mean_psnr:.4f}")
        # --- End Debug --- #

        # Save GIF periodically or if it's the best model
        should_save_gif = (self.current_epoch % self.args.per_save == 0) or (mean_psnr > self.best_psnr)
        if len(save_imgs) > 0 and should_save_gif:
            save_path = os.path.join(val_demo_dir, f"val_demo_epoch_{self.current_epoch}_psnr_{mean_psnr:.2f}.gif")
            try:
                self.make_gif(save_imgs, save_path)
                print(f"Saved validation demo GIF to {save_path} (PSNR: {mean_psnr:.4f})")
            except Exception as e:
                print(f"Error saving GIF: {e}")
                print(f"Number of frames: {len(save_imgs)}")
                if save_imgs:
                    print(f"First frame type: {type(save_imgs[0])}, shape: {save_imgs[0].shape if hasattr(save_imgs[0], 'shape') else 'N/A'}")


        return mean_val_loss, mean_psnr

    def training_one_step_without_teacher_forcing(self, img, label):
        batch_size, time_step, channel, height, width = img.shape
        f_dim, l_dim, n_dim = self.args.F_dim, self.args.L_dim, self.args.N_dim
        beta = self.kl_annealing.get_beta()

        self.optim.zero_grad()

        with autocast(device_type='cuda', dtype=torch.float16, enabled=self.use_amp):
            # Process labels once
            no_head_label = label[:, 1:].reshape(-1, channel, height, width)
            no_head_label_emb = self.label_transformation(no_head_label).view(
                batch_size, time_step - 1, l_dim, height, width
            )

            # Initialize previous frame and latent variable
            prev_frame = img[:, 0] # Shape: (batch_size, channel, height, width)
            prev_frame_emb = self.frame_transformation(prev_frame) # Shape: (batch_size, f_dim, height, width)
            prev_z = torch.randn(batch_size, n_dim, height, width, device=self.args.device)

            pred_no_head_img_list, mu_list, logvar_list = [], [], []
            total_mse = 0.0
            total_kld = 0.0

            for i in range(time_step - 1):
                current_label_emb = no_head_label_emb[:, i] # Shape: (batch_size, l_dim, height, width)

                # Decode step
                decoded_features = self.Decoder_Fusion(prev_frame_emb, current_label_emb, prev_z)
                img_hat = self.Generator(decoded_features) # Shape: (batch_size, channel, height, width)
                pred_no_head_img_list.append(img_hat)

                # Calculate MSE for this step
                target_frame = img[:, i + 1]
                mse_step = self.mse_criterion(img_hat, target_frame)
                total_mse += mse_step

                # Encode the generated frame for the next step
                prev_frame_emb = self.frame_transformation(img_hat)

                # Predict latent variables for the next step
                z, mu, logvar = self.Gaussian_Predictor(prev_frame_emb, current_label_emb)
                prev_z = z # Use the predicted z for the next step's input
                mu_list.append(mu)
                logvar_list.append(logvar)

                # Calculate KLD for this step
                kld_step = self.kl_criterion(mu, logvar)
                total_kld += kld_step

            # Average losses over the sequence length
            mean_mse = total_mse / (time_step - 1)
            mean_kld = total_kld / (time_step - 1)

            # Check for NaN/Inf before calculating total loss
            if torch.isnan(mean_mse) or torch.isinf(mean_mse) or torch.isnan(mean_kld) or torch.isinf(mean_kld):
                print(f"Warning: NaN/Inf detected in losses (no TF) - mse: {mean_mse.item()}, kld: {mean_kld.item()}")
                return torch.tensor(float('nan'), device=self.args.device, requires_grad=True) # Return NaN to skip step

            loss = mean_mse + beta * mean_kld

        if self.scaler:
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optim) # Unscale before clipping
            torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
            self.scaler.step(self.optim)
            self.scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
            self.optim.step()

        # Log step losses (optional, can be verbose)
        # self.writer.add_scalar("StepLoss/mse_noTF", mean_mse.item(), self.global_step) # Requires global step tracking
        # self.writer.add_scalar("StepLoss/kld_noTF", mean_kld.item(), self.global_step)

        return loss

    def training_one_step_with_teacher_forcing(self, img, label):
        batch_size, time_step, channel, height, width = img.shape
        f_dim, l_dim, n_dim = self.args.F_dim, self.args.L_dim, self.args.N_dim
        beta = self.kl_annealing.get_beta()

        self.optim.zero_grad()

        with autocast(device_type='cuda', dtype=torch.float16, enabled=self.use_amp):
            # Process all frames and labels at once
            frame_emb = self.frame_transformation(img.view(-1, channel, height, width)).view(
                batch_size, time_step, f_dim, height, width
            )
            no_head_label = label[:, 1:].reshape(-1, channel, height, width)
            no_head_label_emb = self.label_transformation(no_head_label) # Shape: (batch*(t-1), l_dim, h, w)

            # Get embeddings for prediction (t=1 to T)
            no_head_frame_emb = frame_emb[:, 1:].reshape(-1, f_dim, height, width)
            # Get embeddings for decoding input (t=0 to T-1)
            no_tail_frame_emb = frame_emb[:, :-1].reshape(-1, f_dim, height, width)

            # Predict latent variables z, mu, logvar
            # Input uses target frame embedding (no_head_frame_emb)
            z, mu, logvar = self.Gaussian_Predictor(no_head_frame_emb, no_head_label_emb)

            # Decode using previous frame embedding (no_tail_frame_emb)
            decoded_features = self.Decoder_Fusion(no_tail_frame_emb, no_head_label_emb, z)
            img_hat = self.Generator(decoded_features) # Shape: (batch*(t-1), c, h, w)

            # Calculate MSE Loss
            target_img = img[:, 1:].reshape(-1, channel, height, width)
            mse = self.mse_criterion(img_hat, target_img)

            # Calculate KL Divergence Loss
            kld = self.kl_criterion(mu, logvar)

            # Check for NaN/Inf
            if torch.isnan(mse) or torch.isinf(mse) or torch.isnan(kld) or torch.isinf(kld):
                print(f"Warning: NaN/Inf detected in losses (TF) - mse: {mse.item()}, kld: {kld.item()}")
                return torch.tensor(float('nan'), device=self.args.device, requires_grad=True)

            loss = mse + beta * kld

        if self.scaler:
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optim)
            torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
            self.scaler.step(self.optim)
            self.scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
            self.optim.step()

        # Log step losses (optional)
        # self.writer.add_scalar("StepLoss/mse_TF", mse.item(), self.global_step)
        # self.writer.add_scalar("StepLoss/kld_TF", kld.item(), self.global_step)

        return loss

    @torch.no_grad()
    def val_one_step(self, img, label):
        # self.eval() # Ensure model is in eval mode
        batch_size, time_step, channel, height, width = img.shape
        f_dim, l_dim, n_dim = self.args.F_dim, self.args.L_dim, self.args.N_dim
        # Note: beta is not typically used directly in validation loss calculation
        # but we might need it if logging KLD specifically for validation
        # beta = self.kl_annealing.get_beta() # Get current beta if needed for logging

        psnr_list = []
        generated_frames = [] # Store combined label+prediction frames for GIF
        pred_img_list = [] # Store only predicted frames for loss calculation

        with autocast(device_type='cuda', dtype=torch.float16, enabled=self.use_amp):
            # Process labels once
            no_head_label = label[:, 1:].reshape(-1, channel, height, width)
            no_head_label_emb = self.label_transformation(no_head_label).view(
                batch_size, time_step - 1, l_dim, height, width
            )

            # Initialize
            prev_frame = img[:, 0]
            prev_frame_emb = self.frame_transformation(prev_frame)
            # Use a fixed random seed or ensure consistent random state if needed for reproducibility
            prev_z = torch.randn(batch_size, n_dim, height, width, device=self.args.device)

            # Add the first frame (label + ground truth) to the GIF list (only for the first item in batch)
            if batch_size == 1:
                first_frame_display = torch.cat([label[0, 0].float(), prev_frame[0].float()], dim=1) # Stack vertically
                generated_frames.append(torch.clamp(first_frame_display, 0.0, 1.0))

            total_mse = 0.0
            total_kld = 0.0 # Calculate KLD for validation monitoring

            for i in range(time_step - 1):
                current_label_emb = no_head_label_emb[:, i]

                # Decode
                decoded_features = self.Decoder_Fusion(prev_frame_emb, current_label_emb, prev_z)
                img_hat = self.Generator(decoded_features)
                img_hat_clamped = torch.clamp(img_hat, 0.0, 1.0)
                pred_img_list.append(img_hat_clamped) # Store for loss calculation

                # Calculate PSNR for this step (compared to ground truth)
                target_frame = img[:, i + 1]
                current_psnr = Generate_PSNR(img_hat_clamped, target_frame)
                # --- Debug PSNR Calculation --- #
                if i < 3: # Print for first few frames
                    print(f"[Debug Frame {i+1}] Target min/max: {target_frame.min():.3f}/{target_frame.max():.3f}, Pred min/max: {img_hat_clamped.min():.3f}/{img_hat_clamped.max():.3f}, PSNR: {current_psnr.item():.4f}")
                # --- End Debug ---
                if not torch.isnan(current_psnr):
                    psnr_list.append(current_psnr.item())

                # Add combined frame (label + prediction) to GIF list (only for the first item in batch)
                if batch_size == 1:
                   combined_frame_display = torch.cat([label[0, i + 1].float(), img_hat_clamped[0].float()], dim=1) # Stack vertically
                   generated_frames.append(combined_frame_display)

                # Prepare for next step: Encode generated frame and predict next z
                prev_frame_emb = self.frame_transformation(img_hat) # Use generated frame (not clamped for internal state)
                z, mu, logvar = self.Gaussian_Predictor(prev_frame_emb, current_label_emb)
                prev_z = z # Use predicted z

                # Accumulate KLD for monitoring
                kld_step = self.kl_criterion(mu, logvar)
                total_kld += kld_step

            # Calculate overall MSE loss
            if pred_img_list:
                pred_no_head_img_tensor = torch.stack(pred_img_list, dim=1) # Shape: (batch, t-1, c, h, w)
                target_no_head_img = img[:, 1:]
                mse = self.mse_criterion(pred_no_head_img_tensor, target_no_head_img)
            else:
                mse = torch.tensor(0.0, device=self.args.device)

            mean_kld = total_kld / (time_step - 1)

            # Validation loss typically only includes MSE, but KLD can be monitored
            loss = mse # + beta * mean_kld # Decide if beta*KLD should be part of reported val loss

        # Log validation KLD and MSE separately if needed
        self.writer.add_scalar("kld/val", mean_kld.item(), self.current_epoch)
        self.writer.add_scalar("mse/val", mse.item(), self.current_epoch)

        # --- Debug PSNR --- #
        if not psnr_list:
            print("[Debug] val_one_step: psnr_list is empty!")
        else:
            print(f"[Debug] val_one_step: psnr_list sample: {psnr_list[:5]} (len: {len(psnr_list)})")
        # --- End Debug --- #

        return loss, psnr_list, generated_frames # Return list of PSNRs per frame

    def make_gif(self, images_list, img_name):
        if not images_list:
            print("Warning: No images provided for GIF generation.")
            return
        try:
            pil_images = []
            for img_tensor in images_list:
                if img_tensor.ndim == 4: # Remove batch dimension if present
                    img_tensor = img_tensor.squeeze(0)
                # Ensure tensor is on CPU and detach from graph
                img_tensor_cpu = img_tensor.detach().cpu()
                # Convert to PIL Image
                pil_img = transforms.ToPILImage()(img_tensor_cpu)
                pil_images.append(pil_img)

            if not pil_images:
                print("Warning: Could not convert any tensors to PIL Images.")
                return

            pil_images[0].save(
                img_name,
                format="GIF",
                append_images=pil_images[1:],
                save_all=True,
                duration=100, # Adjust duration as needed (milliseconds per frame)
                loop=0, # Loop indefinitely
            )
        except Exception as e:
            print(f"Error creating GIF {img_name}: {e}")

    def train_dataloader(self):
        # Logic moved to data.dataloader.get_dataloader
        # Ensure args needed by get_dataloader are available (e.g., args.DR, args.frame_H, etc.)
        partial = self.args.fast_partial if self.args.fast_train else self.args.partial
        # Handle potential change in fast_train status
        if self.current_epoch > self.args.fast_train_epoch and self.args.fast_train:
            print(f"Epoch {self.current_epoch}: Disabling fast_train mode.")
            self.args.fast_train = False
            partial = self.args.partial # Update partial value accordingly

        return get_dataloader(
            self.args, "train", self.train_vi_len, partial, self.batch_size
        )

    def val_dataloader(self):
        # Logic moved to data.dataloader.get_dataloader
        # Force num_workers=0 for validation to debug potential multiprocessing issues
        args_copy = self.args
        original_num_workers = getattr(args_copy, 'num_workers', 4)
        args_copy.num_workers = 0 # Temporarily set workers to 0 for validation
        print(f"[Debug] Forcing num_workers=0 for validation dataloader (original was {original_num_workers}).")
        loader = get_dataloader(args_copy, "val", self.val_vi_len, 1.0, batch_size=1) # Val typically uses batch_size=1
        args_copy.num_workers = original_num_workers # Restore original value (important if args object is shared)
        return loader

    def teacher_forcing_ratio_update(self):
        # Decrease TFR only if it's positive and the condition is met
        if self.tfr > 0 and self.current_epoch >= self.tfr_sde and (self.current_epoch - self.tfr_sde) % self.args.tfr_decay_every == 0:
             # Add tfr_decay_every parameter to args or set a default
             # self.args.tfr_decay_every = getattr(self.args, 'tfr_decay_every', 1)
             self.tfr = max(0.0, self.tfr - self.tfr_d_step)
             print(f"Epoch {self.current_epoch}: Updated TFR to {self.tfr:.4f}")


    def tqdm_bar(self, mode, pbar, loss, lr):
        pbar.set_description(f"({mode}) Epoch {self.current_epoch}")
        pbar.set_postfix(loss=f"{loss:.4f}", lr=f"{lr:.1e}", refresh=False)
        # pbar.refresh() # Refresh might not be needed if tqdm updates automatically

    def save(self, path):
        # Ensure the directory exists
        os.makedirs(os.path.dirname(path), exist_ok=True)
        save_obj = {
            "state_dict": self.state_dict(),
            "optimizer_state_dict": self.optim.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "kl_annealing_state": self.kl_annealing.__dict__,
            "tfr": self.tfr,
            "current_epoch": self.current_epoch, # Save next epoch to start from
            "best_psnr": self.best_psnr,
            "scaler_state_dict": self.scaler.state_dict() if self.scaler else None,
            # Include args for reproducibility, be cautious with large objects in args
            # "args": vars(self.args) # Convert argparse.Namespace to dict
        }
        try:
            torch.save(save_obj, path)
            print(f"Saved checkpoint to {path}")
        except Exception as e:
            print(f"Error saving checkpoint to {path}: {e}")

    def load_checkpoint(self, ckpt_path=None):
        load_path = ckpt_path if ckpt_path else self.args.ckpt_path
        if load_path and os.path.isfile(load_path):
            try:
                print(f"Loading checkpoint from {load_path}")
                checkpoint = torch.load(load_path, map_location=self.args.device)

                # Load model state
                self.load_state_dict(checkpoint["state_dict"])

                # Load optimizer and scheduler states
                if "optimizer_state_dict" in checkpoint:
                    self.optim.load_state_dict(checkpoint["optimizer_state_dict"])
                if "scheduler_state_dict" in checkpoint:
                    self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

                # Load KL annealing state
                if "kl_annealing_state" in checkpoint:
                    self.kl_annealing.__dict__.update(checkpoint["kl_annealing_state"])

                # Load training progress state
                self.current_epoch = checkpoint.get("current_epoch", 1)
                self.tfr = checkpoint.get("tfr", self.args.tfr)
                self.best_psnr = checkpoint.get("best_psnr", 0.0)

                # Load AMP scaler state
                if self.scaler and "scaler_state_dict" in checkpoint and checkpoint["scaler_state_dict"] is not None:
                    self.scaler.load_state_dict(checkpoint["scaler_state_dict"])

                print(f"Checkpoint loaded. Resuming from epoch {self.current_epoch}, TFR {self.tfr:.4f}, Best PSNR {self.best_psnr:.4f}")

            except Exception as e:
                print(f"Error loading checkpoint from {load_path}: {e}")
                print("Starting training from scratch.")
                self.current_epoch = 1 # Ensure starting from epoch 1 if load fails
        else:
            print("No checkpoint path provided or file not found. Starting training from scratch.")
            self.current_epoch = 1 # Ensure starting from epoch 1

    def kl_criterion(self, mu, logvar):
        # KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=[1, 2, 3]) # Sum over spatial and channel dims
        # KLD = torch.mean(KLD) # Average over batch dimension
        # Original calculation using mean across all dimensions:
        KLD = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        KLD = torch.clamp(KLD, min=0.0) # Ensure non-negative KLD
        return KLD
