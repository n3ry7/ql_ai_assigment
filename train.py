import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
import numpy as np
import time
import os
import argparse
import json
from datetime import datetime
from tqdm import tqdm
import matplotlib.pyplot as plt

# Import your modules
from dataset import create_data_loaders
from model import create_model


def ensure_dir(path):
    """Ensure directory exists, create if it doesn't."""
    os.makedirs(path, exist_ok=True)
    return path


class Trainer:
    """
    Trainer class for image classification.
    """
    
    def __init__(self, model, train_loader, val_loader, device, config):
        """
        Initialize trainer.
        
        Args:
            model: PyTorch model
            train_loader: Training DataLoader
            val_loader: Validation DataLoader
            device: Device to train on
            config: Configuration dictionary
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.config = config
        
        # Loss function
        self.criterion = nn.CrossEntropyLoss()
        
        # Optimizer
        if config['optimizer'] == 'adam':
            self.optimizer = optim.Adam(
                model.parameters(),
                lr=config['learning_rate'],
                weight_decay=config['weight_decay']
            )
        elif config['optimizer'] == 'sgd':
            self.optimizer = optim.SGD(
                model.parameters(),
                lr=config['learning_rate'],
                momentum=config['momentum'],
                weight_decay=config['weight_decay'],
                nesterov=True
            )
        else:
            raise ValueError(f"Unknown optimizer: {config['optimizer']}")
        
        # Learning rate scheduler
        if config['scheduler'] == 'plateau':
            self.scheduler = ReduceLROnPlateau(
                self.optimizer,
                mode='max',  # Monitor accuracy
                factor=config['lr_factor'],
                patience=config['lr_patience']
            )
        elif config['scheduler'] == 'cosine':
            self.scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=config['epochs'],
                eta_min=config['min_lr']
            )
        else:
            self.scheduler = None
        
        # Training history
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'learning_rates': []
        }
        
        # Best model tracking
        self.best_val_acc = 0.0
        self.best_model_state = None
        
        # Create output directory
        self.create_output_dir()
        
        # Move model to device
        self.model.to(self.device)
        
        # Print model info
        self.print_model_info()
    
    def create_output_dir(self):
        """Create output directory for saving models and results."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        model_name = self.config['model_type'].replace('/', '_')
        
        if self.config['output_dir'] is None:
            self.config['output_dir'] = os.path.join(
                'experiments',
                f"{model_name}_{timestamp}"
            )
        
        # Ensure directory exists
        ensure_dir(self.config['output_dir'])
        
        # Save config
        config_path = os.path.join(self.config['output_dir'], 'config.json')
        with open(config_path, 'w') as f:
            json.dump(self.config, f, indent=2)
        
        print(f"Output directory: {self.config['output_dir']}")
    
    def print_model_info(self):
        """Print model information."""
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        print("\n" + "="*50)
        print("MODEL INFORMATION")
        print("="*50)
        print(f"Model: {self.config['model_type']}")
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        print(f"Device: {self.device}")
        print(f"Training samples: {len(self.train_loader.dataset):,}")
        print(f"Validation samples: {len(self.val_loader.dataset):,}")
        print(f"Batch size: {self.config['batch_size']}")
        print("="*50 + "\n")
    
    def train_epoch(self):
        """
        Train for one epoch.
        
        Returns:
            tuple: (average_loss, accuracy)
        """
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        # Progress bar
        pbar = tqdm(self.train_loader, desc='Training', leave=False)
        
        for batch_idx, (inputs, targets) in enumerate(pbar):
            # Move data to device
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            
            # Zero the parameter gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            
            # Backward pass and optimize
            loss.backward()
            
            # Gradient clipping
            if self.config['gradient_clip'] > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config['gradient_clip']
                )
            
            self.optimizer.step()
            
            # Statistics
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'acc': f"{100.*correct/total:.2f}%"
            })
        
        # Calculate epoch statistics
        avg_loss = running_loss / len(self.train_loader)
        accuracy = 100. * correct / total
        
        return avg_loss, accuracy
    
    def validate(self):
        """
        Validate the model.
        
        Returns:
            tuple: (average_loss, accuracy)
        """
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
        # Disable gradient computation
        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc='Validation', leave=False)
            
            for inputs, targets in pbar:
                # Move data to device
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                
                # Forward pass
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                
                # Statistics
                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
                
                # Update progress bar
                pbar.set_postfix({
                    'loss': f"{loss.item():.4f}",
                    'acc': f"{100.*correct/total:.2f}%"
                })
        
        # Calculate validation statistics
        avg_loss = running_loss / len(self.val_loader)
        accuracy = 100. * correct / total
        
        return avg_loss, accuracy
    
    def train(self):
        """
        Main training loop.
        """
        print("\n" + "="*50)
        print("STARTING TRAINING")
        print("="*50)
        
        start_time = time.time()
        
        for epoch in range(self.config['epochs']):
            print(f"\nEpoch {epoch+1}/{self.config['epochs']}")
            print("-" * 30)
            
            # Train for one epoch
            train_loss, train_acc = self.train_epoch()
            
            # Validate
            val_loss, val_acc = self.validate()
            
            # Update learning rate scheduler
            current_lr = self.optimizer.param_groups[0]['lr']
            
            if self.scheduler is not None:
                if isinstance(self.scheduler, ReduceLROnPlateau):
                    self.scheduler.step(val_acc)
                    # Print learning rate update if it changed
                    new_lr = self.optimizer.param_groups[0]['lr']
                    if new_lr != current_lr:
                        print(f"Learning rate reduced to: {new_lr:.6f}")
                else:
                    self.scheduler.step()
            
            # Save history
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            self.history['learning_rates'].append(current_lr)
            
            # Print epoch results
            print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
            print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
            print(f"Learning Rate: {current_lr:.6f}")
            
            # Save best model
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.best_model_state = self.model.state_dict().copy()
                
                # Save best model
                self.save_checkpoint(
                    filename='best_model.pth',
                    epoch=epoch,
                    val_acc=val_acc
                )
                
                print(f"New best model saved with accuracy: {val_acc:.2f}%")
            
            # Save regular checkpoint
            if (epoch + 1) % self.config['checkpoint_interval'] == 0:
                self.save_checkpoint(
                    filename=f'checkpoint_epoch_{epoch+1}.pth',
                    epoch=epoch,
                    val_acc=val_acc
                )
        
        # Training complete
        end_time = time.time()
        training_time = end_time - start_time
        
        print("\n" + "="*50)
        print("TRAINING COMPLETE")
        print("="*50)
        print(f"Total training time: {training_time:.2f} seconds")
        print(f"Best validation accuracy: {self.best_val_acc:.2f}%")
        
        # Save final model
        self.save_checkpoint(
            filename='final_model.pth',
            epoch=self.config['epochs'] - 1,
            val_acc=self.history['val_acc'][-1]
        )
        
        # Save training history
        self.save_history()
        
        # Plot training curves
        self.plot_training_curves()
        
        # Load best model for final evaluation
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
        
        return self.history
    
    def save_checkpoint(self, filename, epoch, val_acc):
        """
        Save model checkpoint.
        
        Args:
            filename: Checkpoint filename
            epoch: Current epoch
            val_acc: Current validation accuracy
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'val_acc': val_acc,
            'history': self.history,
            'config': self.config
        }
        
        checkpoint_path = os.path.join(self.config['output_dir'], filename)
        torch.save(checkpoint, checkpoint_path)
    
    def save_history(self):
        """Save training history to JSON file."""
        history_path = os.path.join(self.config['output_dir'], 'training_history.json')
        
        # Convert numpy arrays to lists for JSON serialization
        history_serializable = {}
        for key, value in self.history.items():
            if isinstance(value, list) and len(value) > 0 and isinstance(value[0], np.ndarray):
                history_serializable[key] = [v.tolist() if hasattr(v, 'tolist') else v for v in value]
            elif isinstance(value, np.ndarray):
                history_serializable[key] = value.tolist()
            else:
                history_serializable[key] = value
        
        with open(history_path, 'w') as f:
            json.dump(history_serializable, f, indent=2)
    
    def plot_training_curves(self):
        """Plot and save training curves."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Plot training/validation loss
        epochs = range(1, len(self.history['train_loss']) + 1)
        
        axes[0, 0].plot(epochs, self.history['train_loss'], 'b-', label='Training Loss')
        axes[0, 0].plot(epochs, self.history['val_loss'], 'r-', label='Validation Loss')
        axes[0, 0].set_title('Training and Validation Loss')
        axes[0, 0].set_xlabel('Epochs')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot training/validation accuracy
        axes[0, 1].plot(epochs, self.history['train_acc'], 'b-', label='Training Accuracy')
        axes[0, 1].plot(epochs, self.history['val_acc'], 'r-', label='Validation Accuracy')
        axes[0, 1].set_title('Training and Validation Accuracy')
        axes[0, 1].set_xlabel('Epochs')
        axes[0, 1].set_ylabel('Accuracy (%)')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot learning rate
        axes[1, 0].plot(epochs, self.history['learning_rates'], 'g-', label='Learning Rate')
        axes[1, 0].set_title('Learning Rate Schedule')
        axes[1, 0].set_xlabel('Epochs')
        axes[1, 0].set_ylabel('Learning Rate')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].set_yscale('log')
        
        # Plot accuracy difference
        acc_diff = [val - train for train, val in zip(self.history['train_acc'], self.history['val_acc'])]
        axes[1, 1].plot(epochs, acc_diff, 'm-', label='Val Acc - Train Acc')
        axes[1, 1].axhline(y=0, color='k', linestyle='--', alpha=0.3)
        axes[1, 1].set_title('Accuracy Difference (Validation - Training)')
        axes[1, 1].set_xlabel('Epochs')
        axes[1, 1].set_ylabel('Accuracy Difference (%)')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save figure
        plot_path = os.path.join(self.config['output_dir'], 'training_curves.png')
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Training curves saved to: {plot_path}")


def get_default_config():
    """
    Get default configuration.
    
    Returns:
        dict: Default configuration
    """
    return {
        # Model configuration
        'model_type': 'resnet50',
        'num_classes': 3,
        'input_channels': 1,  # Grayscale
        'pretrained': False,  # Can't use pretrained with 1 channel
        
        # Dataset configuration
        'target_size': (512, 512),
        'grayscale': True,
        'maintain_aspect_ratio': True,
        
        # Training configuration
        'epochs': 50,
        'batch_size': 32,
        'learning_rate': 0.001,
        'weight_decay': 1e-4,
        'momentum': 0.9,
        'optimizer': 'adam',  # 'adam' or 'sgd'
        'scheduler': 'plateau',  # 'plateau' or 'cosine' or None
        
        # Scheduler configuration
        'lr_factor': 0.5,
        'lr_patience': 5,
        'min_lr': 1e-6,
        
        # Regularization
        'gradient_clip': 1.0,
        'dropout_rate': 0.5,
        
        # Data augmentation
        'augment': True,
        
        # Checkpointing
        'checkpoint_interval': 5,
        
        # System
        'num_workers': 4,
        'pin_memory': True,
        
        # Output
        'output_dir': None  # Will be auto-generated
    }


def main():
    """
    Main training function.
    """
    parser = argparse.ArgumentParser(description='Train ResNet50 on document images')
    
    # Dataset arguments
    parser.add_argument('--train_csv', type=str, required=True,
                       help='Path to training CSV file')
    parser.add_argument('--train_dir', type=str, required=True,
                       help='Directory containing training images')
    parser.add_argument('--test_csv', type=str, required=True,
                       help='Path to testing CSV file')
    parser.add_argument('--test_dir', type=str, required=True,
                       help='Directory containing testing images')
    
    # Model arguments
    parser.add_argument('--model', type=str, default='resnet50',
                       choices=['resnet18', 'resnet34', 'resnet50', 'resnet101', 'efficientnet_b3', 'document_cnn'],
                       help='Model architecture')
    parser.add_argument('--num_classes', type=int, default=3,
                       help='Number of output classes')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for training')
    parser.add_argument('--lr', type=float, default=0.001,
                       help='Learning rate')
    parser.add_argument('--optimizer', type=str, default='adam',
                       choices=['adam', 'sgd'],
                       help='Optimizer to use')
    parser.add_argument('--scheduler', type=str, default='plateau',
                       choices=['plateau', 'cosine', 'none'],
                       help='Learning rate scheduler')
    
    # Data arguments
    parser.add_argument('--image_size', type=int, default=512,
                       help='Image size (square)')
    parser.add_argument('--no_augment', action='store_true',
                       help='Disable data augmentation')
    parser.add_argument('--no_grayscale', action='store_true',
                       help='Use RGB instead of grayscale')
    parser.add_argument('--no_aspect_ratio', action='store_true',
                       help='Do not maintain aspect ratio')
    
    # System arguments
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of data loading workers')
    parser.add_argument('--device', type=str, default='cuda',
                       choices=['cuda', 'cpu'],
                       help='Device to train on')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')
    
    # Output arguments
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Output directory for models and logs')
    
    args = parser.parse_args()
    
    # Set random seeds for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
    
    # Set device
    device = torch.device(args.device if torch.cuda.is_available() and args.device == 'cuda' else 'cpu')
    print(f"Using device: {device}")
    
    # Load configuration
    config = get_default_config()
    
    # Update config with command line arguments
    config.update({
        'model_type': args.model,
        'num_classes': args.num_classes,
        'target_size': (args.image_size, args.image_size),
        'grayscale': not args.no_grayscale,
        'maintain_aspect_ratio': not args.no_aspect_ratio,
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'learning_rate': args.lr,
        'optimizer': args.optimizer,
        'scheduler': args.scheduler if args.scheduler != 'none' else None,
        'augment': not args.no_augment,
        'num_workers': args.num_workers,
        'output_dir': args.output_dir
    })
    
    # Create data loaders
    print("Creating data loaders...")
    train_loader, val_loader, class_to_idx, idx_to_class, _ = create_data_loaders(
        train_csv=args.train_csv,
        train_dir=args.train_dir,
        test_csv=args.test_csv,
        test_dir=args.test_dir,
        batch_size=config['batch_size'],
        num_workers=config['num_workers'],
        pin_memory=config['pin_memory'],
        target_size=config['target_size'],
        augment=config['augment'],
        grayscale=config['grayscale'],
        maintain_aspect_ratio=config['maintain_aspect_ratio']
    )
    
    # Create model
    print(f"Creating {config['model_type']} model...")
    model = create_model(
        model_type=config['model_type'],
        num_classes=config['num_classes'],
        input_channels=1 if config['grayscale'] else 3,
        pretrained=config['pretrained'],
        freeze_backbone=False,  # Train from scratch for grayscale
        device=device
    )
    
    # Create trainer (this will create the output directory)
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        config=config
    )
    
    # Save class mappings AFTER trainer creates the output directory
    if config['output_dir']:
        ensure_dir(config['output_dir'])
        mappings_path = os.path.join(config['output_dir'], 'class_mappings.json')
        with open(mappings_path, 'w') as f:
            json.dump({
                'class_to_idx': class_to_idx,
                'idx_to_class': idx_to_class
            }, f, indent=2)
        print(f"Class mappings saved to: {mappings_path}")
    
    # Start training
    history = trainer.train()
    
    # Final evaluation
    print("\n" + "="*50)
    print("FINAL EVALUATION")
    print("="*50)
    
    # Load best model for final evaluation
    if trainer.best_model_state is not None:
        model.load_state_dict(trainer.best_model_state)
    
    # Evaluate on validation set
    model.eval()
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for inputs, targets in tqdm(val_loader, desc='Final Evaluation'):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, preds = outputs.max(1)
            
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
    
    # Calculate metrics
    from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
    
    accuracy = accuracy_score(all_targets, all_preds) * 100
    print(f"\nFinal Validation Accuracy: {accuracy:.2f}%")
    
    # Classification report
    print("\nClassification Report:")
    print(classification_report(
        all_targets, 
        all_preds, 
        target_names=[idx_to_class[i] for i in range(config['num_classes'])]
    ))
    
    # Confusion matrix
    cm = confusion_matrix(all_targets, all_preds)
    print("\nConfusion Matrix:")
    print(cm)
    
    # Save final metrics
    metrics = {
        'final_accuracy': float(accuracy),
        'best_accuracy': float(trainer.best_val_acc),
        'confusion_matrix': cm.tolist(),
        'classification_report': classification_report(
            all_targets, 
            all_preds, 
            target_names=[idx_to_class[i] for i in range(config['num_classes'])],
            output_dict=True
        )
    }
    
    metrics_path = os.path.join(config['output_dir'], 'final_metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"\nFinal metrics saved to: {metrics_path}")
    print(f"All outputs saved to: {config['output_dir']}")


if __name__ == "__main__":
    main()
