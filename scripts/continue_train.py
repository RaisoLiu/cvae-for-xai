import argparse
import os
import sys
import glob
from argparse import Namespace
import ast # For safely evaluating literals like lists

# Add project root to sys.path to allow importing train
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Now import the main function from train.py
try:
    from scripts.train import main as train_main
    from scripts.train import set_seed # Import set_seed if needed
except ImportError as e:
    print(f"Error importing train script: {e}")
    print("Ensure continue_train.py is in the 'scripts' directory and train.py exists.")
    sys.exit(1)

def find_latest_checkpoint(save_root):
    """Finds the latest checkpoint file in the save_root directory."""
    checkpoint_dir = os.path.join(save_root, 'checkpoints')
    if not os.path.isdir(checkpoint_dir):
        print(f"Error: Checkpoint directory not found: {checkpoint_dir}")
        return None

    # Search recursively for .ckpt files within the checkpoints directory
    ckpt_files = glob.glob(os.path.join(checkpoint_dir, '**', '*.ckpt'), recursive=True)

    if not ckpt_files:
        print(f"Error: No .ckpt files found in {checkpoint_dir} or its subdirectories.")
        return None

    # Find the latest file based on modification time
    try:
        latest_ckpt = max(ckpt_files, key=os.path.getmtime)
        print(f"Found latest checkpoint: {latest_ckpt}")
        return latest_ckpt
    except Exception as e:
        print(f"Error finding latest checkpoint: {e}")
        return None

def load_training_args(args_filepath):
    """Loads training arguments from the training_args.txt file."""
    if not os.path.exists(args_filepath):
        print(f"Error: training_args.txt not found at {args_filepath}")
        return None

    args_dict = {}
    try:
        with open(args_filepath, 'r') as f:
            for line in f:
                line = line.strip()
                if ':' in line:
                    key, value = line.split(':', 1)
                    key = key.strip()
                    value = value.strip()

                    # Attempt to infer type (basic types, bool, None, list)
                    if value == 'None':
                        args_dict[key] = None
                    elif value == 'True':
                        args_dict[key] = True
                    elif value == 'False':
                        args_dict[key] = False
                    elif value.startswith('[') and value.endswith(']'):
                        try:
                            # Safely evaluate list literals
                            args_dict[key] = ast.literal_eval(value)
                        except (ValueError, SyntaxError):
                             # Fallback for simple space-separated lists if literal_eval fails
                            args_dict[key] = value.strip('[]').split()
                            # Try converting elements to int/float if possible (basic case)
                            try:
                                args_dict[key] = [int(item) for item in args_dict[key]]
                            except ValueError:
                                try:
                                     args_dict[key] = [float(item) for item in args_dict[key]]
                                except ValueError:
                                     pass # Keep as strings if conversion fails
                    else:
                        try:
                            # Try int, then float, then keep as string
                            args_dict[key] = int(value)
                        except ValueError:
                            try:
                                args_dict[key] = float(value)
                            except ValueError:
                                args_dict[key] = value # Keep as string

    except Exception as e:
        print(f"Error reading or parsing {args_filepath}: {e}")
        return None

    # Convert dict to Namespace
    return Namespace(**args_dict)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Continue training CVAE Model from a checkpoint.")
    parser.add_argument("--save_root", type=str, required=True,
                        help="Root directory of the previous experiment to continue training from.")

    args = parser.parse_args()

    # 1. Validate save_root
    if not os.path.isdir(args.save_root):
        print(f"Error: Provided save_root directory does not exist: {args.save_root}")
        sys.exit(1)

    # 2. Find the latest checkpoint
    latest_ckpt_path = find_latest_checkpoint(args.save_root)
    if latest_ckpt_path is None:
        sys.exit(1)

    # 3. Load original training arguments
    args_filepath = os.path.join(args.save_root, 'training_args.txt')
    original_args = load_training_args(args_filepath)
    if original_args is None:
        sys.exit(1)

    # 4. Prepare arguments for continuing training
    # Start with original args and override necessary ones
    continue_args = original_args
    continue_args.ckpt_path = latest_ckpt_path
    continue_args.save_root = args.save_root # Crucially, use the *provided* save_root
    continue_args.is_resuming = True

    # Optional: Reset starting epoch or learning rate?
    # Depending on how train.py handles loading checkpoints, you might
    # not need to adjust epoch count here. If train.py doesn't restore
    # epoch/optimizer state correctly, you might need to add logic here
    # or modify train.py. For now, assume train.py handles it.
    # Example: continue_args.start_epoch = loaded_epoch + 1

    print("--- Arguments for Continuing Training ---")
    for key, value in vars(continue_args).items():
        print(f"{key}: {value}")
    # Explicitly print relevant KL args
    print("--- KL Annealing Parameters Loaded ---")
    print(f"kl_anneal_type: {getattr(continue_args, 'kl_anneal_type', 'Not Found')}")
    print(f"kl_anneal_cycle: {getattr(continue_args, 'kl_anneal_cycle', 'Not Found')}")
    print(f"kl_anneal_ratio: {getattr(continue_args, 'kl_anneal_ratio', 'Not Found')}")
    print("-----------------------------------------")

    # Set seed using the loaded args before starting training
    if hasattr(continue_args, 'seed'):
        print(f"Setting seed: {continue_args.seed}")
        set_seed(continue_args.seed)
    else:
        print("Warning: Seed not found in original arguments.")


    # 5. Call the main training function
    print(f"\nStarting continued training from checkpoint: {continue_args.ckpt_path}\n")
    try:
        train_main(continue_args)
    except Exception as e:
        print(f"\nAn error occurred during continued training: {e}")
        # Potentially save a final checkpoint here if desired
        sys.exit(1)

    print("\nContinued training finished.") 