import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torch
import numpy as np


class ImageClassificationDataset(Dataset):
    """
    Custom Dataset for image classification.
    
    Args:
        csv_file (str): Path to the CSV file with image filenames and labels
        root_dir (str): Directory where images are stored
        transform (callable, optional): Optional transform to be applied on images
        class_to_idx (dict, optional): Mapping from class names to indices
        target_size (tuple): Target size for all images as (width, height) (default: (512, 512))
        grayscale (bool): Whether to convert images to grayscale (1 channel) (default: True)
        maintain_aspect_ratio (bool): Whether to maintain aspect ratio when resizing (default: True)
    """
    
    def __init__(self, csv_file, root_dir, transform=None, class_to_idx=None, 
                 target_size=(512, 512), grayscale=True, maintain_aspect_ratio=True):
        """
        Initialize the dataset.
        """
        self.annotations = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        self.target_width, self.target_height = target_size
        self.grayscale = grayscale
        self.maintain_aspect_ratio = maintain_aspect_ratio
        
        # Store original sizes for statistics
        self.original_sizes = self._collect_size_statistics()
        
        # Get unique classes and create mapping
        self.classes = sorted(self.annotations['category'].unique())
        
        # Use provided class_to_idx or create new one
        if class_to_idx is not None:
            self.class_to_idx = class_to_idx
            # Ensure all classes in data are in class_to_idx
            for cls in self.classes:
                if cls not in self.class_to_idx:
                    raise ValueError(f"Class '{cls}' not found in provided class_to_idx")
        else:
            self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        
        # Create reverse mapping for convenience
        self.idx_to_class = {idx: cls for cls, idx in self.class_to_idx.items()}
        
        # Map labels to indices
        self.annotations['label_idx'] = self.annotations['category'].map(self.class_to_idx)
    
    def _collect_size_statistics(self):
        """
        Collect size statistics from a sample of images.
        """
        sizes = []
        print("Collecting image size statistics...")
        
        # Sample up to 100 images for statistics
        sample_size = min(100, len(self.annotations))
        sample_indices = np.linspace(0, len(self.annotations) - 1, sample_size, dtype=int)
        
        for idx in sample_indices:
            img_name = os.path.join(self.root_dir, self.annotations.iloc[idx, 0])
            
            try:
                # Load image to get original size
                with Image.open(img_name) as img:
                    sizes.append(img.size)
            except Exception as e:
                print(f"Warning: Could not process {img_name}: {e}")
                continue
        
        print(f"Collected size statistics from {len(sizes)} images")
        return sizes
    
    def _correct_orientation(self, img):
        """
        Rotate image 90 degrees clockwise if height > width.
        Ensures all images are landscape (width >= height).
        """
        width, height = img.size
        if height > width:
            # Rotate 90 degrees clockwise
            img = img.rotate(-90, expand=True)
        return img
    
    def _resize_with_aspect_ratio(self, img, target_width, target_height):
        """
        Resize image while maintaining aspect ratio and pad if necessary.
        
        Args:
            img: PIL Image
            target_width: Target width
            target_height: Target height
            
        Returns:
            Resized and padded PIL Image
        """
        # Get original dimensions
        original_width, original_height = img.size
        
        if self.maintain_aspect_ratio:
            # Calculate scaling factor to fit within target dimensions
            width_ratio = target_width / original_width
            height_ratio = target_height / original_height
            
            # Use the smaller ratio to ensure the image fits
            ratio = min(width_ratio, height_ratio)
            
            # Calculate new dimensions
            new_width = int(original_width * ratio)
            new_height = int(original_height * ratio)
            
            # Resize image
            img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
            
            # Create new image with target size and white background
            if img.mode == 'L':  # Grayscale
                background_color = 255  # White
            else:  # RGB
                background_color = (255, 255, 255)  # White
            
            new_img = Image.new(img.mode, (target_width, target_height), background_color)
            
            # Calculate padding to center the image
            left = (target_width - new_width) // 2
            top = (target_height - new_height) // 2
            
            # Paste the resized image onto the padded canvas
            new_img.paste(img, (left, top))
            return new_img
        else:
            # Simple resize without maintaining aspect ratio
            return img.resize((target_width, target_height), Image.Resampling.LANCZOS)
    
    def _preprocess_image(self, img):
        """
        Apply preprocessing steps:
        1. Convert to grayscale if specified
        2. Correct orientation (rotate if height > width)
        3. Resize to target size while maintaining aspect ratio
        4. Pad if necessary
        """
        # Step 0: Convert to grayscale if specified
        if self.grayscale and img.mode != 'L':
            img = img.convert('L')
        
        # Step 1: Correct orientation (ensure landscape)
        img = self._correct_orientation(img)
        
        # Step 2 & 3: Resize with aspect ratio preservation and padding
        img = self._resize_with_aspect_ratio(img, self.target_width, self.target_height)
        
        return img
    
    def __len__(self):
        """
        Return the total number of samples in the dataset.
        """
        return len(self.annotations)
    
    def __getitem__(self, idx):
        """
        Get a sample from the dataset.
        
        Args:
            idx (int): Index of the sample to retrieve
            
        Returns:
            tuple: (image, label) where label is the class index
        """
        # Get image filename and label
        img_name = os.path.join(self.root_dir, self.annotations.iloc[idx, 0])
        label = self.annotations.iloc[idx]['label_idx']
        
        # Load and preprocess image
        try:
            image = Image.open(img_name)
            image = self._preprocess_image(image)
        except Exception as e:
            print(f"Error loading image {img_name}: {e}")
            # Return a blank image as fallback
            if self.grayscale:
                image = Image.new('L', (self.target_width, self.target_height), color=255)
            else:
                image = Image.new('RGB', (self.target_width, self.target_height), color='white')
        
        # Apply additional transformations if specified
        if self.transform:
            image = self.transform(image)
        else:
            # Default to tensor conversion
            image = transforms.ToTensor()(image)
        
        return image, label
    
    def get_class_distribution(self):
        """
        Get the distribution of classes in the dataset.
        
        Returns:
            dict: Dictionary with class names as keys and counts as values
        """
        return dict(self.annotations['category'].value_counts())
    
    def get_size_statistics(self):
        """
        Get statistics about image sizes.
        
        Returns:
            dict: Statistics about image sizes or None if no sizes collected
        """
        if not self.original_sizes or len(self.original_sizes) == 0:
            return None
        
        widths = [w for w, h in self.original_sizes]
        heights = [h for w, h in self.original_sizes]
        
        return {
            'original_widths': {
                'min': min(widths),
                'max': max(widths),
                'mean': np.mean(widths),
                'std': np.std(widths),
                'median': np.median(widths)
            },
            'original_heights': {
                'min': min(heights),
                'max': max(heights),
                'mean': np.mean(heights),
                'std': np.std(heights),
                'median': np.median(heights)
            },
            'aspect_ratios': {
                'min': min(h/w for w, h in self.original_sizes),
                'max': max(h/w for w, h in self.original_sizes),
                'mean': np.mean([h/w for w, h in self.original_sizes]),
                'std': np.std([h/w for w, h in self.original_sizes])
            },
            'sample_count': len(self.original_sizes),
            'target_size': (self.target_width, self.target_height),
            'grayscale': self.grayscale,
            'maintain_aspect_ratio': self.maintain_aspect_ratio
        }
    
    def visualize_sample(self, idx, save_path=None):
        """
        Visualize a sample image with preprocessing steps.
        
        Args:
            idx (int): Index of the sample to visualize
            save_path (str, optional): Path to save the visualization
            
        Returns:
            PIL.Image: The preprocessed image
        """
        # Get image filename
        img_name = os.path.join(self.root_dir, self.annotations.iloc[idx, 0])
        label_idx = self.annotations.iloc[idx]['label_idx']
        class_name = self.idx_to_class[label_idx]
        
        try:
            # Load original image
            original_img = Image.open(img_name)
            original_size = original_img.size
            
            # Apply preprocessing
            preprocessed_img = self._preprocess_image(original_img.copy())
            
            print(f"\nSample {idx}: {os.path.basename(img_name)}")
            print(f"  Class: {class_name} (index: {label_idx})")
            print(f"  Original size: {original_size[0]}x{original_size[1]} ({original_img.mode})")
            print(f"  Preprocessed size: {preprocessed_img.size[0]}x{preprocessed_img.size[1]} ({preprocessed_img.mode})")
            print(f"  Grayscale: {self.grayscale}")
            print(f"  Maintain aspect ratio: {self.maintain_aspect_ratio}")
            
            if save_path:
                # Save the preprocessed image
                preprocessed_img.save(save_path, "PNG")
                print(f"  Saved to: {save_path}")
            
            return preprocessed_img
            
        except Exception as e:
            print(f"Error visualizing sample {idx}: {e}")
            return None


def get_default_transforms(augment=True, grayscale=True):
    """
    Get default transformations for training and validation.
    
    Args:
        augment (bool): Whether to apply data augmentation for training
        grayscale (bool): Whether images are grayscale (1 channel)
    
    Returns:
        transforms.Compose: Transformations
    """
    # Use appropriate normalization for grayscale or RGB
    if grayscale:
        # Grayscale normalization
        mean = [0.5]  # Middle gray for documents
        std = [0.5]   # Standard deviation for normalization
    else:
        # RGB normalization (standard ImageNet)
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
    
    if augment:
        # Training transformations with data augmentation
        transform_list = [
            transforms.ToTensor(),
        ]
        
        # Add data augmentation for grayscale
        if grayscale:
            # Document-specific augmentations
            transform_list.extend([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(5),  # Smaller rotation for documents
                # Adjust brightness and contrast for documents
                transforms.ColorJitter(brightness=0.2, contrast=0.2),
                # Add random affine transformations
                transforms.RandomAffine(degrees=0, translate=(0.05, 0.05), scale=(0.95, 1.05)),
            ])
        else:
            # RGB augmentations
            transform_list.extend([
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(10),
            ])
        
        # Add normalization at the end
        transform_list.append(transforms.Normalize(mean=mean, std=std))
        
        transform = transforms.Compose(transform_list)
    else:
        # Validation/Test transformations (no augmentation)
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])
    
    return transform


def create_data_loaders(train_csv, train_dir, test_csv, test_dir, 
                       batch_size=32, num_workers=2, pin_memory=True,
                       target_size=(512, 512), augment=True, grayscale=True,
                       maintain_aspect_ratio=True):
    """
    Create DataLoaders for training and testing.
    
    Args:
        train_csv (str): Path to training CSV file
        train_dir (str): Directory containing training images
        test_csv (str): Path to testing CSV file
        test_dir (str): Directory containing testing images
        batch_size (int): Batch size for DataLoader
        num_workers (int): Number of workers for DataLoader
        pin_memory (bool): Whether to pin memory for GPU transfer
        target_size (tuple): Target size for images as (width, height)
        augment (bool): Whether to apply data augmentation for training
        grayscale (bool): Whether to convert images to grayscale
        maintain_aspect_ratio (bool): Whether to maintain aspect ratio
        
    Returns:
        tuple: (train_loader, test_loader, class_to_idx, idx_to_class, train_dataset)
    """
    # Get transformations
    train_transform = get_default_transforms(augment=augment, grayscale=grayscale)
    val_transform = get_default_transforms(augment=False, grayscale=grayscale)
    
    # Create training dataset
    print("Creating training dataset...")
    train_dataset = ImageClassificationDataset(
        csv_file=train_csv,
        root_dir=train_dir,
        transform=train_transform,
        target_size=target_size,
        grayscale=grayscale,
        maintain_aspect_ratio=maintain_aspect_ratio
    )
    
    # Use same class_to_idx for test dataset
    print("Creating test dataset...")
    test_dataset = ImageClassificationDataset(
        csv_file=test_csv,
        root_dir=test_dir,
        transform=val_transform,
        class_to_idx=train_dataset.class_to_idx,
        target_size=target_size,
        grayscale=grayscale,
        maintain_aspect_ratio=maintain_aspect_ratio
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True  # Drop last incomplete batch
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,  # Don't shuffle for evaluation
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    return train_loader, test_loader, train_dataset.class_to_idx, train_dataset.idx_to_class, train_dataset


if __name__ == "__main__":
    """
    Example usage and testing of the dataset.
    """
    import os
    from PIL import Image
    
    # Example paths - replace with your actual paths
    TRAIN_CSV = "/home/nery/datasets/data_scientist_AI_QL/train.csv"
    TRAIN_DIR = "/home/nery/datasets/data_scientist_AI_QL/train"
    TEST_CSV = "/home/nery/datasets/data_scientist_AI_QL/test.csv"
    TEST_DIR = "/home/nery/datasets/data_scientist_AI_QL/test"
    
    # Create output directory for saved images
    output_dir = "dataset_samples_512"
    os.makedirs(output_dir, exist_ok=True)
    
    print("Creating datasets with 512x512 preprocessing...")
    print("Target size: 512x512 pixels")
    print("Grayscale: True")
    print("Maintain aspect ratio: True")
    
    # Create data loaders with 512x512
    train_loader, test_loader, class_to_idx, idx_to_class, train_dataset = create_data_loaders(
        train_csv=TRAIN_CSV,
        train_dir=TRAIN_DIR,
        test_csv=TEST_CSV,
        test_dir=TEST_DIR,
        batch_size=8,  # Larger batch size possible with smaller images
        num_workers=0,  # Set to 0 for debugging
        target_size=(512, 512),
        augment=False,  # No augmentation for visualization
        grayscale=True,   # Convert to grayscale
        maintain_aspect_ratio=True
    )
    
    print(f"\nDataset Information:")
    print(f"Number of classes: {len(class_to_idx)}")
    print(f"Class to index mapping: {class_to_idx}")
    print(f"Index to class mapping: {idx_to_class}")
    print(f"Number of training samples: {len(train_dataset)}")
    print(f"Number of testing samples: {len(test_loader.dataset)}")
    print(f"Number of training batches: {len(train_loader)}")
    print(f"Number of testing batches: {len(test_loader)}")
    print(f"Target size: {train_dataset.target_width}x{train_dataset.target_height}")
    print(f"Grayscale: {train_dataset.grayscale}")
    print(f"Maintain aspect ratio: {train_dataset.maintain_aspect_ratio}")
    
    # Get statistics
    stats = train_dataset.get_size_statistics()
    if stats:
        print(f"\nOriginal image size statistics (sampled from {stats['sample_count']} images):")
        print(f"  Widths: min={stats['original_widths']['min']:.0f}, "
              f"max={stats['original_widths']['max']:.0f}, "
              f"mean={stats['original_widths']['mean']:.0f}, "
              f"median={stats['original_widths']['median']:.0f}")
        print(f"  Heights: min={stats['original_heights']['min']:.0f}, "
              f"max={stats['original_heights']['max']:.0f}, "
              f"mean={stats['original_heights']['mean']:.0f}, "
              f"median={stats['original_heights']['median']:.0f}")
        print(f"  Aspect ratios (h/w): min={stats['aspect_ratios']['min']:.2f}, "
              f"max={stats['aspect_ratios']['max']:.2f}, "
              f"mean={stats['aspect_ratios']['mean']:.2f}")
    else:
        print("\nWarning: Could not collect size statistics")
    
    # Get one batch to test
    images, labels = next(iter(train_loader))
    print(f"\nBatch shape - Images: {images.shape}, Labels: {labels.shape}")
    print(f"Note: Images have {images.shape[1]} channel(s) (grayscale)")
    print(f"Labels in batch: {labels}")
    
    # Show class distribution
    print(f"\nTraining class distribution: {train_dataset.get_class_distribution()}")
    
    def tensor_to_pil_image(tensor, grayscale=True):
        """
        Convert a PyTorch tensor to PIL Image.
        
        Args:
            tensor (torch.Tensor): Tensor of shape (C, H, W)
            grayscale (bool): Whether the tensor is grayscale
            
        Returns:
            PIL.Image: Grayscale or RGB image
        """
        # Convert tensor to numpy array
        if grayscale:
            # Grayscale: (1, H, W) -> (H, W)
            img = tensor.numpy().squeeze(0)
            # Denormalize for grayscale
            mean = np.array([0.5])
            std = np.array([0.5])
            img = std * img + mean
        else:
            # RGB: (3, H, W) -> (H, W, 3)
            img = tensor.numpy().transpose((1, 2, 0))
            # Denormalize for RGB
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            img = std * img + mean
        
        # Clip to valid range and convert to 0-255
        img = np.clip(img, 0, 1)
        img = (img * 255).astype(np.uint8)
        
        # Convert to PIL Image
        if grayscale:
            return Image.fromarray(img, mode='L')
        else:
            return Image.fromarray(img, mode='RGB')
    
    # Save images to PNG files
    print(f"\nSaving preprocessed images to '{output_dir}/' directory...")
    for i in range(min(12, len(images))):
        # Convert tensor to PIL Image
        pil_img = tensor_to_pil_image(images[i], grayscale=True)
        
        # Get label and corresponding class name
        label_idx = labels[i].item()
        class_name = idx_to_class[label_idx]
        
        # Create filename
        filename = f"sample_{i+1:03d}_class_{class_name}_label_{label_idx}.png"
        filepath = os.path.join(output_dir, filename)
        
        # Save image
        pil_img.save(filepath, "PNG")
        print(f"Saved: {filename} ({pil_img.size[0]}x{pil_img.size[1]} pixels, mode: {pil_img.mode})")
    
    # Also save some metadata
    metadata_path = os.path.join(output_dir, "metadata.txt")
    with open(metadata_path, 'w') as f:
        f.write(f"Dataset Information\n")
        f.write(f"===================\n\n")
        f.write(f"Target size: {train_dataset.target_width}x{train_dataset.target_height}\n")
        f.write(f"Number of classes: {len(class_to_idx)}\n")
        f.write(f"Grayscale: {train_dataset.grayscale}\n")
        f.write(f"Maintain aspect ratio: {train_dataset.maintain_aspect_ratio}\n")
        f.write(f"\nClass to index mapping:\n")
        for cls, idx in class_to_idx.items():
            f.write(f"  {cls}: {idx}\n")
        
        f.write(f"\nIndex to class mapping:\n")
        for idx, cls in idx_to_class.items():
            f.write(f"  {idx}: {cls}\n")
        
        f.write(f"\nTraining class distribution:\n")
        dist = train_dataset.get_class_distribution()
        for cls, count in dist.items():
            f.write(f"  {cls}: {count} samples\n")
        
        if stats:
            f.write(f"\nImage size statistics (sampled from {stats['sample_count']} images):\n")
            f.write(f"  Original widths: min={stats['original_widths']['min']:.0f}, "
                    f"max={stats['original_widths']['max']:.0f}, "
                    f"mean={stats['original_widths']['mean']:.0f}, "
                    f"median={stats['original_widths']['median']:.0f}\n")
            f.write(f"  Original heights: min={stats['original_heights']['min']:.0f}, "
                    f"max={stats['original_heights']['max']:.0f}, "
                    f"mean={stats['original_heights']['mean']:.0f}, "
                    f"median={stats['original_heights']['median']:.0f}\n")
            f.write(f"  Aspect ratios (h/w): min={stats['aspect_ratios']['min']:.2f}, "
                    f"max={stats['aspect_ratios']['max']:.2f}, "
                    f"mean={stats['aspect_ratios']['mean']:.2f}\n")
        
        f.write(f"\nBatch information:\n")
        f.write(f"  Batch size: {images.shape[0]}\n")
        f.write(f"  Image shape: {images.shape[1:]} (CxHxW)\n")
        f.write(f"  Labels in saved batch: {labels.tolist()}\n")
    
    print(f"\nSaved metadata to: {metadata_path}")
    
    # Create thumbnails for easier viewing
    print(f"\nCreating thumbnails for easier viewing...")
    thumbnails_dir = os.path.join(output_dir, "thumbnails")
    os.makedirs(thumbnails_dir, exist_ok=True)
    
    for i in range(min(12, len(images))):
        # Load the saved image
        filename = f"sample_{i+1:03d}_class_{idx_to_class[labels[i].item()]}_label_{labels[i].item()}.png"
        img_path = os.path.join(output_dir, filename)
        img = Image.open(img_path)
        
        # Create thumbnail (already small, but make a grid)
        thumbnail = img.copy()
        
        # Save thumbnail
        thumbnail_path = os.path.join(thumbnails_dir, f"thumbnail_{filename}")
        thumbnail.save(thumbnail_path, "PNG")
        print(f"Created thumbnail: thumbnail_{filename}")
    
    print(f"Saved thumbnails to: {thumbnails_dir}")
    
    # Visualize a few samples with their preprocessing steps
    print(f"\nVisualizing sample preprocessing steps...")
    samples_dir = os.path.join(output_dir, "sample_preprocessing")
    os.makedirs(samples_dir, exist_ok=True)
    
    for i in range(min(3, len(train_dataset.annotations))):
        sample_idx = i
        save_path = os.path.join(samples_dir, f"sample_{i+1:03d}_preprocessed.png")
        train_dataset.visualize_sample(sample_idx, save_path)
    
    # Create a grid of all saved samples
    print(f"\nCreating sample grid...")
    grid_size = 4
    num_images = min(16, len(images))
    
    # Calculate grid dimensions
    cols = min(grid_size, num_images)
    rows = (num_images + cols - 1) // cols
    
    # Create blank grid
    img_width, img_height = train_dataset.target_width, train_dataset.target_height
    grid_width = cols * img_width
    grid_height = rows * img_height
    
    grid_img = Image.new('L', (grid_width, grid_height), color=255)
    
    for i in range(num_images):
        # Load the saved image
        filename = f"sample_{i+1:03d}_class_{idx_to_class[labels[i].item()]}_label_{labels[i].item()}.png"
        img_path = os.path.join(output_dir, filename)
        img = Image.open(img_path)
        
        # Calculate position
        row = i // cols
        col = i % cols
        x = col * img_width
        y = row * img_height
        
        # Paste image
        grid_img.paste(img, (x, y))
    
    # Save grid
    grid_path = os.path.join(output_dir, "sample_grid.png")
    grid_img.save(grid_path, "PNG")
    print(f"Saved sample grid to: {grid_path}")
    
    print(f"\nDone! All files saved in '{output_dir}/' directory.")