import argparse
import os
import torch
import random
import numpy as np

# Adjust import path based on project structure
from models.cvae import VAE_Model
# Assuming other necessary utilities might be in src.utils
# from utils import some_utility_function

def set_seed(seed):
    """Sets the seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # The two lines below might slow down training, but ensure reproducibility
        # torch.backends.cudnn.deterministic = True
        # torch.backends.cudnn.benchmark = False

def main(args):
    """Main training function."""
    set_seed(args.seed)

    # Determine device
    if args.device == 'auto':
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {args.device}")

    # --- Data Preparation (Handled by the model's dataloader methods) ---
    # No explicit dataloader creation here, model handles it internally.

    # --- Model Initialization ---
    print("Initializing model...")
    model = VAE_Model(args).to(args.device)
    print(model)
    # Print number of parameters (optional)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of trainable parameters: {num_params / 1e6:.2f} M")

    # --- Checkpoint Loading ---
    if args.ckpt_path:
        model.load_checkpoint(args.ckpt_path)
    else:
        print("No checkpoint path provided, starting training from scratch.")

    # --- Training Loop ---
    print("Starting training...")
    try:
        model.training_stage()
    except KeyboardInterrupt:
        print("Training interrupted by user.")
        # Optionally save a final checkpoint upon interruption
        interrupt_save_path = os.path.join(model.checkpoint_dir, f"interrupt_epoch-{model.current_epoch}.ckpt")
        print(f"Saving final state to {interrupt_save_path}...")
        model.save(interrupt_save_path)
    except Exception as e:
        print(f"An error occurred during training: {e}")
        # Potentially save checkpoint on other errors too
        error_save_path = os.path.join(model.checkpoint_dir, f"error_epoch-{model.current_epoch}.ckpt")
        print(f"Saving state due to error to {error_save_path}...")
        model.save(error_save_path)
        raise # Re-raise the exception after saving

    print("Training finished.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train CVAE Model")

    # Paths and Directories
    parser.add_argument("--DR", type=str, default="./data/dummy_data", help="Root directory for the dataset")
    parser.add_argument("--save_root", type=str, default="./output/experiment_1", help="Root directory to save logs, checkpoints, and demos")
    parser.add_argument("--ckpt_path", type=str, default=None, help="Path to checkpoint to resume training")

    # Model Hyperparameters
    parser.add_argument("--F_dim", type=int, default=24, help="Dimension of frame features")
    parser.add_argument("--L_dim", type=int, default=24, help="Dimension of label features")
    parser.add_argument("--N_dim", type=int, default=48, help="Dimension of latent variables (z)")
    parser.add_argument("--D_out_dim", type=int, default=96, help="Output dimension of Decoder Fusion module")

    # Training Settings
    parser.add_argument("--num_epoch", type=int, default=50, help="Total number of training epochs")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for training")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--optim", type=str, default="AdamW", choices=["Adam", "AdamW"], help="Optimizer type")
    parser.add_argument("--milestones", type=int, nargs='+', default=[10, 20, 30, 40], help="Epoch milestones for learning rate scheduler") # Example milestones
    parser.add_argument("--gamma", type=float, default=0.5, help="Learning rate decay factor for scheduler")
    parser.add_argument("--per_save", type=int, default=5, help="Save checkpoint every N epochs")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"], help="Device to use for training")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of workers for dataloader")
    parser.add_argument("--no_use_amp", action='store_true', help="Disable Automatic Mixed Precision (AMP)")
    parser.add_argument("--debug", action='store_true', help="Enable debug prints in the model")


    # KL Annealing Settings
    parser.add_argument("--kl_anneal_type", type=str, default="Cyclical", choices=["Cyclical", "Monotonic", "Without"], help="Type of KL annealing")
    parser.add_argument("--kl_anneal_cycle", type=int, default=4, help="Number of cycles for Cyclical KL annealing")
    parser.add_argument("--kl_anneal_ratio", type=float, default=0.5, help="Ratio of epoch to reach target beta for KL annealing")

    # Teacher Forcing Settings
    parser.add_argument("--tfr", type=float, default=1.0, help="Initial teacher forcing ratio")
    parser.add_argument("--tfr_d_step", type=float, default=0.01, help="Teacher forcing ratio decay step")
    parser.add_argument("--tfr_sde", type=int, default=5, help="Start decaying teacher forcing ratio after this many epochs")
    parser.add_argument("--tfr_decay_every", type=int, default=1, help="Decay teacher forcing ratio every N epochs after tfr_sde")

    # Data Settings
    parser.add_argument("--train_vi_len", type=int, default=16, help="Length of video sequences for training")
    parser.add_argument("--val_vi_len", type=int, default=630, help="Length of video sequences for validation")
    parser.add_argument("--frame_H", type=int, default=64, help="Frame height")
    parser.add_argument("--frame_W", type=int, default=64, help="Frame width")
    parser.add_argument("--use_random_crop", action='store_true', help="Use random resized crop for training augmentation")
    parser.add_argument("--partial", type=float, default=1.0, help="Use a partial dataset (fraction from 0.0 to 1.0)")
    parser.add_argument("--fast_train", action='store_true', help="Use fast_partial for the first few epochs")
    parser.add_argument("--fast_partial", type=float, default=0.1, help="Partial dataset fraction for fast training")
    parser.add_argument("--fast_train_epoch", type=int, default=5, help="Number of epochs to use fast training")
    parser.add_argument("--persistent_workers", action=argparse.BooleanOptionalAction, default=True, help="Use persistent workers in DataLoader")


    args = parser.parse_args()

    # Create save_root directory if it doesn't exist
    os.makedirs(args.save_root, exist_ok=True)

    # Save args to a file in the save_root for reference
    args_save_path = os.path.join(args.save_root, 'training_args.txt')
    with open(args_save_path, 'w') as f:
        for key, value in vars(args).items():
            f.write(f"{key}: {value}\n")
    print(f"Saved training arguments to {args_save_path}")

    main(args)
