import argparse
import os
import re # Import re module
from glob import glob
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch import stack
from torch.utils.data import Dataset as torchData, DataLoader
from torchvision import transforms
from torchvision.datasets.folder import default_loader as imgloader
from PIL import Image # Added PIL import
from tqdm import tqdm
from torch.cuda.amp import autocast

# Assuming VAE_Model and its components are correctly structured in src
from models.cvae import VAE_Model
from models.modules import (
    Decoder_Fusion,
    Gaussian_Predictor,
    Generator,
    Label_Encoder,
    RGB_Encoder,
)

# Helper function from dataloader.py (could also be imported)
def get_key(fp):
    filename = os.path.basename(fp)
    filename = os.path.splitext(filename)[0].replace("frame", "")
    try:
        return int(filename)
    except ValueError:
        print(f"Warning: Could not convert filename {filename} to int. Returning 0.")
        return 0


# Specific Dataset for the test/prediction phase based on the provided code
class TestDataset_Dance(torchData):
    def __init__(self, root, transform, video_len=630): # video_len seems fixed for test
        super().__init__()
        self.img_folder = []
        self.label_folder = []
        self.transform = transform
        self.video_len = video_len

        test_img_root = os.path.join(root, "test/test_img")
        test_label_root = os.path.join(root, "test/test_label")

        # Assuming test data is structured in subfolders 0, 1, 2, ...
        data_indices = sorted([d for d in os.listdir(test_img_root) if os.path.isdir(os.path.join(test_img_root, d))])
        # data_num = len(glob(os.path.join(root, f"test/test_img/*"))) # Original glob might be less robust
        data_num = len(data_indices)
        print(f"Found {data_num} test sequences.")

        for i_str in data_indices:
            img_seq_path = os.path.join(test_img_root, i_str)
            label_seq_path = os.path.join(test_label_root, i_str)

            img_files = sorted(glob(os.path.join(img_seq_path, "*.png")), key=get_key)
            label_files = sorted(glob(os.path.join(label_seq_path, "*.png")), key=get_key)

            if not img_files:
                print(f"Warning: No images found in {img_seq_path}. Skipping.")
                continue
            if not label_files:
                print(f"Warning: No labels found in {label_seq_path}. Skipping.")
                continue

            # The original test code seems to load only the *first* image
            # and all labels for prediction. Let's replicate that logic.
            self.img_folder.append(img_files[0]) # Store only the first frame path
            self.label_folder.append(label_files) # Store list of all label paths

    def __len__(self):
        return len(self.img_folder) # Number of test sequences

    def __getitem__(self, index):
        first_frame_path = self.img_folder[index]
        label_seq_paths = self.label_folder[index]

        # Load the first image
        try:
            first_img = self.transform(imgloader(first_frame_path))
        except Exception as e:
            print(f"Error loading first image {first_frame_path}: {e}. Returning dummy data.")
            # Need consistent dummy shape based on transform
            # Infer shape from transforms if possible, otherwise use args
            h = getattr(self.transform, 'transforms', [None])[0].size[0] if isinstance(self.transform.transforms[0], transforms.Resize) else 32
            w = getattr(self.transform, 'transforms', [None])[0].size[1] if isinstance(self.transform.transforms[0], transforms.Resize) else 64
            first_img = torch.zeros((3, h, w))

        # Load all label images
        labels = []
        for label_path in label_seq_paths:
            try:
                labels.append(self.transform(imgloader(label_path)))
            except Exception as e:
                 print(f"Error loading label image {label_path}: {e}. Appending dummy label.")
                 labels.append(torch.zeros_like(first_img)) # Append dummy with same shape

        # Pad labels if necessary (though test length seems fixed at 630)
        while len(labels) < self.video_len:
            print(f"Warning: Sequence {index} has fewer than {self.video_len} labels. Padding last label.")
            if labels: labels.append(labels[-1])
            else: labels.append(torch.zeros_like(first_img))

        # Ensure exactly video_len labels
        labels = labels[:self.video_len]

        # Return the single first image and the stack of labels
        # Shape: [1, C, H, W], [video_len, C, H, W] - need to match model input expectation
        # The original code feeds [B=1, 1, C, H, W] and [B=1, T, C, H, W]
        return first_img.unsqueeze(0), stack(labels).unsqueeze(0)


# Model specifically for testing, inheriting VAE_Model but overriding eval/val steps
class Test_model(VAE_Model):
    def __init__(self, args):
        # Call VAE_Model's init but only setup necessary components for inference
        super(VAE_Model, self).__init__() # Use base nn.Module init first
        self.args = args

        # Modules needed for inference
        self.frame_transformation = RGB_Encoder(3, args.F_dim).to(args.device)
        self.label_transformation = Label_Encoder(3, args.L_dim).to(args.device)
        # Adjust Gaussian_Predictor input channels
        gauss_in_chans = args.F_dim + args.L_dim
        self.Gaussian_Predictor = Gaussian_Predictor(
            in_chans=gauss_in_chans, out_chans=args.N_dim
        ).to(args.device)
        # Adjust Decoder_Fusion input channels
        decoder_in_chans = args.F_dim + args.L_dim + args.N_dim
        self.Decoder_Fusion = Decoder_Fusion(
            in_chans=decoder_in_chans, out_chans=args.D_out_dim
        ).to(args.device)
        self.Generator = Generator(input_nc=args.D_out_dim, output_nc=3).to(args.device)

        # No optimizer, scheduler, loss, tensorboard writer, kl_annealing needed for inference
        self.val_vi_len = args.val_vi_len # Need this for dataloader
        self.batch_size = args.batch_size # Usually 1 for testing sequence by sequence
        self.current_epoch = 0 # Not relevant but avoids potential errors if accessed


    @torch.no_grad()
    def run_prediction(self):
        self.train(False) # Use self.train(False) instead of recursive self.eval()
        all_pred_numpy = []
        test_loader = self.test_dataloader()

        for idx, (img, label) in enumerate(test_loader):
            print(f"Predicting sequence {idx+1}/{len(test_loader)}...")
            # Ensure tensors are on the correct device *before* passing
            img = img.to(self.args.device)
            label = label.to(self.args.device)

            # Assuming batch size is 1 and an extra dim exists, squeeze the first two dims
            # Pass the first image (img needs squeeze too) and the full label sequence
            squeezed_img = img.squeeze(0).squeeze(0) # Shape [C, H, W]
            squeezed_label_seq = label.squeeze(0).squeeze(0) # Shape [T, C, H, W]

            pred_frames_tensor = self.predict_one_sequence(squeezed_img, squeezed_label_seq, idx)
            # pred_frames_tensor shape should be [T, C, H, W]

            # Append the numpy array version for saving later
            all_pred_numpy.append(pred_frames_tensor.cpu().numpy())

        print("Prediction finished. Concatenating results for submission...")
        # Concatenate predictions from all sequences
        if all_pred_numpy:
            all_preds_np = np.concatenate(all_pred_numpy, axis=0)
             # Convert predictions to integer format [0, 255]
            all_preds_np = all_preds_np.reshape(all_preds_np.shape[0], -1)

            pred_to_int = np.clip(np.rint(all_preds_np * 255), 0, 255).astype(np.uint8)
            

            # Create DataFrame for submission.csv
            df = pd.DataFrame(pred_to_int)
            df.insert(0, "id", range(len(df))) # Add ID column

            # Define submission file path
            submission_path = os.path.join(self.args.save_root, "submission.csv")
            df.to_csv(submission_path, header=True, index=False)
            print(f"Submission file saved to: {submission_path}")
        else:
            print("No predictions were generated.")


    @torch.no_grad()
    def predict_one_sequence(self, first_img, label_seq, seq_idx=0):
        time_step, channel, height, width = label_seq.shape
        f_dim, l_dim, n_dim = self.args.F_dim, self.args.L_dim, self.args.N_dim
        pred_frames = []
        generated_frames_for_gif = []
        generated_frames_for_csv = []

        # Use simpler torch.amp.autocast syntax for CUDA
        with torch.amp.autocast('cuda', dtype=torch.float32, enabled=False):
            # # Remove: Process the entire label sequence at once (causes padding error)
            # print(f"[Debug Predict] Shape of label_seq before label_transformation: {label_seq.shape}")
            # label_seq_emb = self.label_transformation(label_seq) # Shape: [T, L_dim, H, W]

            # Initialize first frame and latent variable
            prev_frame = first_img.to(self.args.device)
            prev_frame_emb = self.frame_transformation(prev_frame.unsqueeze(0)).squeeze(0) # Add/remove batch dim
            prev_z = torch.randn(1, n_dim, height, width, device=self.args.device).squeeze(0) # Remove batch dim for loop logic

            # Add the first frame (label + ground truth) to GIF list
            if self.args.make_gif:
                first_frame_display = torch.cat([label_seq[0].float(), prev_frame.float()], dim=2) # Stack horizontally
                generated_frames_for_gif.append(torch.clamp(first_frame_display, 0.0, 1.0))

            for i in tqdm(range(time_step), desc=f"Predicting frames for seq {seq_idx+1}", ncols=100):
                # Get embedding for the *current* label frame
                current_label = label_seq[i] # Shape [C, H, W]
                # Add batch dim for transformation, then remove it
                current_label_emb = self.label_transformation(current_label.unsqueeze(0)).squeeze(0) # Shape [L_dim, H, W]

                # Decode step
                # Ensure all inputs to Decoder_Fusion have a batch dimension [1, Channels, H, W]
                decoded_features = self.Decoder_Fusion(
                    prev_frame_emb.unsqueeze(0),    # Add batch dim. Should be [1, F_dim, H, W]
                    current_label_emb.unsqueeze(0), # Add batch dim [1, L_dim, H, W]
                    prev_z.unsqueeze(0)             # Add batch dim [1, N_dim, H, W]
                )
                # Output decoded_features likely has shape [1, D_out_dim, H, W]
                img_hat = self.Generator(decoded_features) # Input shape [1, D_out_dim, H, W]
                img_hat_clamped = torch.clamp(img_hat.squeeze(0), 0.0, 1.0) # Squeeze batch dim *after* Generator
                pred_frames.append(img_hat_clamped.cpu()) # Store predicted frame (already squeezed)
                generated_frames_for_csv.append(img_hat_clamped.cpu().numpy()) # Store for CSV

                # Add combined frame (label + prediction) to GIF list
                if self.args.make_gif:
                    # Squeeze img_hat for concatenation
                    combined_frame_display = torch.cat([current_label.float(), img_hat.squeeze(0).float()], dim=2)
                    generated_frames_for_gif.append(combined_frame_display)

                # Prepare for next step: Encode generated frame and predict next z
                prev_frame_emb = self.frame_transformation(img_hat) # Keep batch dim. Shape [1, F_dim, H, W]
                # Need to add batch dim for Gaussian Predictor's other input
                z, _, _ = self.Gaussian_Predictor(prev_frame_emb, current_label_emb.unsqueeze(0))
                prev_z = z.squeeze(0) # Remove batch dim for next iteration's loop logic
                prev_frame_emb = prev_frame_emb.squeeze(0)

        # Save GIF if requested
        if generated_frames_for_gif and self.args.make_gif:
            gif_path = os.path.join(self.args.save_root, f"pred_seq{seq_idx}.gif")
            self.make_gif(generated_frames_for_gif, gif_path)
            print(f"Saved prediction GIF to {gif_path}")

        # Concatenate flattened frames for CSV output
        output_tensor = torch.stack(pred_frames, dim=0) # Shape [T, C, H, W]
        print(f'Output tensor shape for CSV: {output_tensor.shape}')

        # Original code had an assert for shape (1, 630, 3, 32, 64)
        # Let's check the shape before reshaping for CSV
        expected_frames = self.args.val_vi_len # Should match label length
        assert output_tensor.shape[0] == expected_frames, \
               f"Expected {expected_frames} frames, but generated {output_tensor.shape[0]}"
        assert output_tensor.shape[1] == channel, \
               f"Unexpected flattened dimension size."

        return output_tensor

    def make_gif(self, images_list, img_name):
        os.makedirs(os.path.dirname(img_name), exist_ok=True)
        pil_images = []
        for img_tensor in images_list:
            try:
                # Ensure tensor is detached, on CPU, and correct type/range
                img_tensor_cpu = img_tensor.detach().float().cpu()
                pil_img = transforms.ToPILImage()(img_tensor_cpu)
                pil_images.append(pil_img)
            except Exception as e:
                print(f"Error converting tensor to PIL image for GIF: {e}")
                return # Stop GIF creation if conversion fails

        if pil_images:
            try:
                pil_images[0].save(
                    img_name,
                    format="GIF",
                    append_images=pil_images[1:],
                    save_all=True,
                    duration=100, # Adjust duration (ms)
                    loop=0,
                )
            except Exception as e:
                print(f"Error saving GIF {img_name}: {e}")
        else:
             print("No valid frames to save in GIF.")

    # Dataloader specific to the test set structure
    def test_dataloader(self):
        transform = transforms.Compose([
            transforms.Resize((self.args.frame_H, self.args.frame_W)),
            transforms.ToTensor(),
        ])
        try:
            dataset = TestDataset_Dance(
                root=self.args.DR,
                transform=transform,
                video_len=self.args.val_vi_len # Use val_vi_len for test sequence length
            )
        except FileNotFoundError as e:
             print(f"Error: Test dataset not found at {self.args.DR}. {e}")
             raise
        except Exception as e:
             print(f"Error loading test dataset: {e}")
             raise

        if len(dataset) == 0:
            print("Error: Test dataset is empty!")
            # Handle appropriately - maybe raise an error or return None
            raise ValueError("Test dataset is empty or could not be loaded.")

        test_loader = DataLoader(
            dataset,
            batch_size=1, # Process one sequence at a time
            num_workers=self.args.num_workers,
            shuffle=False, # Do not shuffle test data
            pin_memory=True
        )
        print(f"Test DataLoader created with {len(dataset)} sequences.")
        return test_loader

    # Override load_checkpoint to only load state_dict
    def load_checkpoint(self, ckpt_path=None):
        load_path = ckpt_path if ckpt_path else self.args.ckpt_path
        if load_path and os.path.isfile(load_path):
            print(f"Loading checkpoint for prediction from {load_path}")
            try:
                # Explicitly set weights_only=False to load older checkpoints
                checkpoint = torch.load(load_path, map_location=self.args.device, weights_only=False)
                # Load only the model's state dictionary
                if "state_dict" in checkpoint:
                    self.load_state_dict(checkpoint["state_dict"])
                    print("Model state_dict loaded successfully.")
                else:
                    # Attempt to load the entire checkpoint if state_dict key is missing
                    print("Warning: 'state_dict' key not found in checkpoint. Attempting to load entire object as state_dict.")
                    self.load_state_dict(checkpoint)
                    print("Assumed entire checkpoint was state_dict and loaded.")
            except Exception as e:
                print(f"Error loading state_dict from checkpoint {load_path}: {e}")
                raise # Re-raise error as prediction requires a loaded model
        else:
            raise FileNotFoundError(f"Checkpoint file not found at {load_path}. Cannot proceed with prediction.")


def main(args):
    # Ensure save directory exists
    os.makedirs(args.save_root, exist_ok=True)

    # Determine device
    if args.device == 'auto':
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {args.device}")

    # Initialize the Test_model
    model = Test_model(args) # Device assignment happens inside __init__ for modules

    # Load the checkpoint (strict loading, error if missing/invalid)
    model.load_checkpoint(args.ckpt_path)

    # Run the prediction process
    model.run_prediction()

    print("Prediction script finished.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run CVAE Model Prediction")

    # --- Essential Arguments --- (Match names used in Test_model and TestDataset_Dance)
    parser.add_argument("--DR", type=str, required=True, help="Root directory for the dataset (containing test/ subfolder)")
    parser.add_argument("--save_root", type=str, default=None, help="Directory to save prediction results (submission.csv, gifs). Defaults based on ckpt_path if not set.")
    parser.add_argument("--ckpt_path", type=str, required=True, help="Path to the trained model checkpoint (.ckpt)")

    # --- Model Hyperparameters (Must match the trained model's architecture) ---
    parser.add_argument("--F_dim", type=int, default=128, help="Dimension of frame features (MATCH TRAINED MODEL)")
    parser.add_argument("--L_dim", type=int, default=32, help="Dimension of label features (MATCH TRAINED MODEL)")
    parser.add_argument("--N_dim", type=int, default=12, help="Dimension of latent variables (z) (MATCH TRAINED MODEL)")
    parser.add_argument("--D_out_dim", type=int, default=192, help="Output dimension of Decoder Fusion module (MATCH TRAINED MODEL)")

    # --- Data/Prediction Settings ---
    parser.add_argument("--val_vi_len", type=int, default=630, help="Length of video sequences for prediction (fixed based on test data)")
    parser.add_argument("--frame_H", type=int, default=32, help="Frame height (MATCH TRAINED MODEL'S INPUT)")
    parser.add_argument("--frame_W", type=int, default=64, help="Frame width (MATCH TRAINED MODEL'S INPUT)")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for prediction (should generally be 1)")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of workers for dataloader")
    parser.add_argument("--make_gif", default=True, action='store_true', help="Generate GIF visualization for each predicted sequence")

    # --- System Settings ---
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"], help="Device to use for prediction")
    # parser.add_argument("--gpu", type=int, default=0, help="GPU ID to use if device is cuda") # Use CUDA_VISIBLE_DEVICES instead if needed

    args = parser.parse_args()

    # Set default save_root based on ckpt_path if not provided
    if args.save_root is None:
        if args.ckpt_path:
            ckpt_filename = os.path.basename(args.ckpt_path)
            # Extract epoch number using regex
            match = re.search(r'epoch-(\d+)', ckpt_filename)
            epoch_number = match.group(1) if match else "unknown_epoch"

            ckpt_dir = os.path.dirname(args.ckpt_path)
            checkpoints_parent_dir = os.path.dirname(ckpt_dir) # e.g., ./output/experiment_4/checkpoints
            experiment_dir = os.path.dirname(checkpoints_parent_dir) # e.g., ./output/experiment_4
            # Include epoch number in the save path
            args.save_root = os.path.join(experiment_dir, "predictions_test", f"epoch_{epoch_number}")
            print(f"Save root not specified. Defaulting to: {args.save_root}")
        else:
            # This case should ideally not happen because ckpt_path is required,
            # but handle it defensively.
            raise ValueError("Cannot determine default save_root because --ckpt_path is missing.")

    # Some arguments from the original test script might not be needed for prediction
    # e.g., lr, optim, epoch-related, KL annealing, teacher forcing args are irrelevant here.

    main(args)
