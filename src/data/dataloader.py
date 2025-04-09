import os
from glob import glob
import torch
from torch import stack
from torch.utils.data import Dataset as torchData, DataLoader
from torchvision import transforms
from torchvision.datasets.folder import default_loader as imgloader
from PIL import Image

# Helper function to extract frame number
def get_key(fp):
    filename = os.path.basename(fp) # Use os.path.basename for cross-platform compatibility
    filename = os.path.splitext(filename)[0].replace("frame", "")
    try:
        return int(filename)
    except ValueError:
        # Handle cases where filename might not be convertible to int
        print(f"Warning: Could not convert filename {filename} to int. Returning 0.")
        return 0


class Dataset_Dance(torchData):
    """
    Args:
        root (str)      : The path of your Dataset (should contain train/ and val/ subdirs)
        transform       : Transformation to your dataset
        mode (str)      : 'train' or 'val'
        video_len (int) : Length of the video sequence clips
        partial (float) : Percentage of your Dataset to use (0.0 to 1.0)
    """

    def __init__(self, root, transform, mode="train", video_len=7, partial=1.0):
        super().__init__()
        if mode not in ["train", "val"]:
             raise ValueError("Mode must be 'train' or 'val'")

        self.root = root
        self.transform = transform
        self.mode = mode
        self.video_len = video_len
        self.partial = partial

        self.img_folder_path = os.path.join(root, f"{mode}/{mode}_img")
        self.label_folder_path = os.path.join(root, f"{mode}/{mode}_label")

        # Find all image files and sort them
        self.img_files = sorted(
            glob(os.path.join(self.img_folder_path, "*.png")), key=get_key
        )

        if not self.img_files:
            raise FileNotFoundError(f"No PNG images found in {self.img_folder_path}")

        # Calculate the number of sequences based on video length and partial dataset use
        num_total_frames = len(self.img_files)
        num_available_sequences = num_total_frames // self.video_len
        self.num_sequences = int(num_available_sequences * self.partial)

        if self.num_sequences == 0:
             print(f"Warning: Calculated number of sequences is 0. Check video_len ({self.video_len}) and dataset size/partial ({self.partial}).")


    def __len__(self):
        return self.num_sequences

    def __getitem__(self, index):
        # Calculate the starting frame index for the sequence
        start_frame_idx = index * self.video_len

        imgs = []
        labels = []
        for i in range(self.video_len):
            current_frame_idx = start_frame_idx + i
            if current_frame_idx >= len(self.img_files):
                # Handle edge case if the last sequence is incomplete (shouldn't happen with __len__ calculation)
                print(f"Warning: Attempting to access frame index {current_frame_idx} beyond available frames ({len(self.img_files)}). Skipping frame.")
                continue

            img_path = self.img_files[current_frame_idx]

            # Construct the corresponding label path
            img_filename = os.path.basename(img_path)
            label_path = os.path.join(self.label_folder_path, img_filename)

            if not os.path.exists(label_path):
                 # Fallback or error handling if label is missing
                 print(f"Warning: Label file not found at {label_path}. Skipping frame pair.")
                 continue # Or handle differently, e.g., load a placeholder

            try:
                 img = imgloader(img_path)
                 label = imgloader(label_path)
                 imgs.append(self.transform(img))
                 labels.append(self.transform(label))
            except Exception as e:
                 print(f"Error loading or transforming image/label pair: {img_path}, {label_path}. Error: {e}")
                 # Handle error, e.g., skip this pair or return None/raise exception
                 continue

        if not imgs or not labels or len(imgs) != self.video_len:
             print(f"Warning: Could not load complete sequence for index {index}. Returning None or handling error.")
             # Handle incomplete sequence - this might require adjusting __len__ or skipping in DataLoader collate_fn
             # For now, returning None which might cause issues downstream. Best to ensure data integrity.
             # As a placeholder, return dummy data of the correct shape if possible:
             if len(imgs) > 0: # Try to pad
                 while len(imgs) < self.video_len: imgs.append(imgs[-1])
                 while len(labels) < self.video_len: labels.append(labels[-1])
             else: # Cannot even load one frame
                 dummy_img = torch.zeros((self.video_len, 3, self.transform.transforms[-2].size[0], self.transform.transforms[-2].size[1])) # Assuming ToTensor and Resize are last
                 dummy_label = torch.zeros_like(dummy_img)
                 print(f"Returning dummy data for index {index}")
                 return dummy_img, dummy_label


        return stack(imgs), stack(labels)


def _get_image_transforms(args, mode="train"):
    transform_list = []
    # Use getattr for safe access to args attributes with defaults
    frame_h = getattr(args, 'frame_H', 32) # Adjusted default based on test script args
    frame_w = getattr(args, 'frame_W', 64) # Adjusted default based on test script args
    use_random_crop = getattr(args, 'use_random_crop', False)

    if mode == "train" and use_random_crop:
        transform_list.extend([
            transforms.RandomResizedCrop(
                (frame_h, frame_w),
                scale=(0.8, 1.2), # Slightly wider scale range example
                ratio=(0.9, 1.1)
            ),
            transforms.RandomHorizontalFlip(p=0.5), # Increased probability example
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2), # Added color jitter
        ])
    else:
        # Resize for validation or if random crop is not used/specified
        transform_list.append(
            transforms.Resize((frame_h, frame_w))
        )

    transform_list.append(transforms.ToTensor())
    # Add normalization if desired (example using ImageNet stats)
    # transform_list.append(transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))

    return transforms.Compose(transform_list)

def get_dataloader(args, mode, video_len, partial, batch_size):
    transform = _get_image_transforms(args, mode)
    # Use getattr for safe access to args attributes
    data_root = getattr(args, 'DR', './data') # Use args.DR for data root
    num_workers = getattr(args, 'num_workers', 4)

    try:
        dataset = Dataset_Dance(
            root=data_root,
            transform=transform,
            mode=mode,
            video_len=video_len,
            partial=partial,
        )
    except FileNotFoundError as e:
        print(f"Error initializing dataset: {e}")
        print(f"Please ensure the dataset exists at {data_root} with {mode}/{mode}_img subdirectories.")
        raise e
    except Exception as e:
        print(f"An unexpected error occurred during dataset initialization: {e}")
        raise e


    if len(dataset) == 0:
        print(f"Warning: The {mode} dataset contains 0 sequences. Check dataset path, video_len, and partial settings.")
        # Return an empty loader or handle as appropriate
        return DataLoader(dataset, batch_size=batch_size) # Return empty loader


    shuffle = (mode == "train")
    drop_last = (mode == "train") # Drop last incomplete batch only during training
    persistent_workers = getattr(args, 'persistent_workers', True) and num_workers > 0

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        drop_last=drop_last,
        shuffle=shuffle,
        pin_memory=True, # Enable pin_memory for faster GPU transfer if using CUDA
        persistent_workers=persistent_workers,
    )
    print(f"Successfully created DataLoader for {mode} mode with {len(dataset)} sequences.")
    return dataloader
