import torch
import torch.nn as nn
from torchvision import transforms
import numpy as np
import os
import json
import argparse
from PIL import Image
import glob
from tqdm import tqdm
import pandas as pd
import warnings
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import seaborn as sns

warnings.filterwarnings('ignore')

# Import your modules
from dataset import ImageClassificationDataset
from model import create_model


class ImageClassifier:
    """
    Image classifier for inference and evaluation.
    """
    
    def __init__(self, model_path, config_path=None, device=None):
        """
        Initialize the classifier.
        
        Args:
            model_path: Path to the trained model checkpoint
            config_path: Path to the config file (optional, will look for it if not provided)
            device: Device to run inference on (cuda/cpu)
        """
        # Set device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        print(f"Using device: {self.device}")
        
        # Load checkpoint
        print(f"Loading model from: {model_path}")
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # Load config
        if config_path is None:
            # Try to find config in the same directory as model
            model_dir = os.path.dirname(model_path)
            config_path = os.path.join(model_dir, 'config.json')
            
            if not os.path.exists(config_path):
                # Try class mappings as fallback
                config_path = os.path.join(model_dir, 'class_mappings.json')
                if not os.path.exists(config_path):
                    raise FileNotFoundError(
                        f"Could not find config.json or class_mappings.json in {model_dir}. "
                        f"Please provide config_path explicitly."
                    )
        
        print(f"Loading config from: {config_path}")
        with open(config_path, 'r') as f:
            config_data = json.load(f)
        
        # Get config from checkpoint if available, otherwise from file
        if 'config' in checkpoint:
            self.config = checkpoint['config']
            print("Using config from checkpoint")
        else:
            self.config = config_data
            print("Using config from file")
        
        # Load class mappings
        self.class_mappings = self._load_class_mappings(model_path)
        
        # Create model
        print(f"Creating {self.config.get('model_type', 'model')}...")
        self.model = create_model(
            model_type=self.config.get('model_type', 'resnet50'),
            num_classes=len(self.class_mappings['class_to_idx']),
            input_channels=self.config.get('input_channels', 1),
            pretrained=False,
            freeze_backbone=False,
            device=self.device
        )
        
        # Load model weights
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        print(f"Model loaded successfully")
        
        # Get image size from config
        self.image_size = self.config.get('target_size', (512, 512))
        if isinstance(self.image_size, list):
            self.image_size = tuple(self.image_size)
        
        # Get grayscale setting
        self.grayscale = self.config.get('grayscale', True)
        
        # Setup transforms
        self.transform = self._get_transforms()
        
        # Print info
        self.print_info()
    
    def _load_class_mappings(self, model_path):
        """
        Load class mappings from various possible locations.
        """
        model_dir = os.path.dirname(model_path)
        possible_paths = [
            os.path.join(model_dir, 'class_mappings.json'),
            os.path.join(model_dir, 'config.json'),
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                with open(path, 'r') as f:
                    data = json.load(f)
                
                # Check if this is class_mappings or config
                if 'class_to_idx' in data:
                    return data
                elif 'class_to_idx' in data.get('config', {}):
                    return data['config']
        
        # If no mappings found, try to extract from checkpoint
        print("Warning: Could not find class mappings file. Creating default mappings.")
        return {
            'class_to_idx': {'class_0': 0, 'class_1': 1, 'class_2': 2},
            'idx_to_class': {0: 'class_0', 1: 'class_1', 2: 'class_2'}
        }
    
    def _get_transforms(self):
        """Get the same transforms used during training."""
        # Use appropriate normalization
        if self.grayscale:
            mean = [0.5]
            std = [0.5]
        else:
            mean = [0.485, 0.456, 0.406]
            std = [0.229, 0.224, 0.225]
        
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])
        
        return transform
    
    def print_info(self):
        """Print model information."""
        print("\n" + "="*50)
        print("MODEL INFORMATION")
        print("="*50)
        print(f"Model: {self.config.get('model_type', 'Unknown')}")
        print(f"Classes: {len(self.class_mappings['class_to_idx'])}")
        print(f"Class mappings: {self.class_mappings['class_to_idx']}")
        print(f"Image size: {self.image_size}")
        print(f"Grayscale: {self.grayscale}")
        print(f"Device: {self.device}")
        print("="*50 + "\n")
    
    def _preprocess_image(self, image_path):
        """
        Preprocess a single image.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Preprocessed image tensor
        """
        try:
            # Load image
            img = Image.open(image_path)
            
            # Convert to grayscale if needed
            if self.grayscale and img.mode != 'L':
                img = img.convert('L')
            
            # Resize while maintaining aspect ratio (same as dataset)
            original_width, original_height = img.size
            
            # Ensure landscape orientation (width >= height)
            if original_height > original_width:
                img = img.rotate(-90, expand=True)
                original_width, original_height = img.size
            
            # Calculate scaling factor to fit within target dimensions
            target_width, target_height = self.image_size
            width_ratio = target_width / original_width
            height_ratio = target_height / original_height
            ratio = min(width_ratio, height_ratio)
            
            # Calculate new dimensions
            new_width = int(original_width * ratio)
            new_height = int(original_height * ratio)
            
            # Resize image
            img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
            
            # Create new image with target size and white background
            if self.grayscale:
                background_color = 255  # White for grayscale
                new_img = Image.new('L', (target_width, target_height), background_color)
            else:
                background_color = (255, 255, 255)  # White for RGB
                new_img = Image.new('RGB', (target_width, target_height), background_color)
            
            # Calculate padding to center the image
            left = (target_width - new_width) // 2
            top = (target_height - new_height) // 2
            
            # Paste the resized image onto the padded canvas
            new_img.paste(img, (left, top))
            
            # Apply transforms
            img_tensor = self.transform(new_img)
            
            return img_tensor, True
            
        except Exception as e:
            print(f"Error preprocessing image {image_path}: {e}")
            return None, False
    
    def predict(self, image_path, return_probabilities=False):
        """
        Predict class for a single image.
        
        Args:
            image_path: Path to the image file
            return_probabilities: Whether to return probabilities
            
        Returns:
            dict: Prediction results
        """
        # Preprocess image
        img_tensor, success = self._preprocess_image(image_path)
        if not success:
            return None
        
        # Add batch dimension and move to device
        img_tensor = img_tensor.unsqueeze(0).to(self.device)
        
        # Predict
        with torch.no_grad():
            outputs = self.model(img_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            predicted_prob, predicted_class = torch.max(probabilities, 1)
        
        # Convert to Python types
        predicted_class = predicted_class.item()
        predicted_prob = predicted_prob.item()
        class_name = self.class_mappings['idx_to_class'].get(str(predicted_class), 
                                                             self.class_mappings['idx_to_class'].get(predicted_class, f"class_{predicted_class}"))
        
        # Get all probabilities
        all_probs = probabilities[0].cpu().numpy()
        class_probs = {}
        for idx, prob in enumerate(all_probs):
            class_name_idx = self.class_mappings['idx_to_class'].get(str(idx), 
                                                                    self.class_mappings['idx_to_class'].get(idx, f"class_{idx}"))
            class_probs[class_name_idx] = float(prob)
        
        result = {
            'image_path': image_path,
            'predicted_class': class_name,
            'predicted_class_idx': predicted_class,
            'confidence': predicted_prob,
            'all_probabilities': class_probs if return_probabilities else None
        }
        
        return result
    
    def predict_batch(self, image_paths, batch_size=32, return_probabilities=False):
        """
        Predict classes for multiple images.
        
        Args:
            image_paths: List of image paths
            batch_size: Batch size for inference
            return_probabilities: Whether to return probabilities
            
        Returns:
            list: List of prediction results
        """
        results = []
        
        # Process in batches
        for i in tqdm(range(0, len(image_paths), batch_size), desc="Processing images"):
            batch_paths = image_paths[i:i+batch_size]
            batch_tensors = []
            valid_paths = []
            
            # Preprocess batch
            for img_path in batch_paths:
                img_tensor, success = self._preprocess_image(img_path)
                if success:
                    batch_tensors.append(img_tensor)
                    valid_paths.append(img_path)
            
            if not batch_tensors:
                continue
            
            # Stack tensors and move to device
            batch_tensor = torch.stack(batch_tensors).to(self.device)
            
            # Predict batch
            with torch.no_grad():
                outputs = self.model(batch_tensor)
                probabilities = torch.nn.functional.softmax(outputs, dim=1)
                predicted_probs, predicted_classes = torch.max(probabilities, 1)
            
            # Process results
            for idx, img_path in enumerate(valid_paths):
                predicted_class = predicted_classes[idx].item()
                predicted_prob = predicted_probs[idx].item()
                
                class_name = self.class_mappings['idx_to_class'].get(str(predicted_class), 
                                                                     self.class_mappings['idx_to_class'].get(predicted_class, f"class_{predicted_class}"))
                
                # Get all probabilities if requested
                all_probs = None
                if return_probabilities:
                    all_probs = {}
                    probs = probabilities[idx].cpu().numpy()
                    for class_idx, prob in enumerate(probs):
                        cls_name = self.class_mappings['idx_to_class'].get(str(class_idx), 
                                                                          self.class_mappings['idx_to_class'].get(class_idx, f"class_{class_idx}"))
                        all_probs[cls_name] = float(prob)
                
                results.append({
                    'image_path': img_path,
                    'predicted_class': class_name,
                    'predicted_class_idx': predicted_class,
                    'confidence': predicted_prob,
                    'all_probabilities': all_probs
                })
        
        return results
    
    def predict_directory(self, directory_path, extensions=None, recursive=True, 
                         batch_size=32, return_probabilities=False):
        """
        Predict classes for all images in a directory.
        
        Args:
            directory_path: Path to directory
            extensions: List of image extensions to look for (default: common image formats)
            recursive: Whether to search recursively
            batch_size: Batch size for inference
            return_probabilities: Whether to return probabilities
            
        Returns:
            list: List of prediction results
        """
        if extensions is None:
            extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff', '*.tif', '*.gif']
        
        # Find all image files
        image_paths = []
        for ext in extensions:
            pattern = os.path.join(directory_path, '**', ext) if recursive else os.path.join(directory_path, ext)
            image_paths.extend(glob.glob(pattern, recursive=recursive))
        
        # Remove duplicates and sort
        image_paths = sorted(list(set(image_paths)))
        
        if not image_paths:
            print(f"No images found in {directory_path}")
            return []
        
        print(f"Found {len(image_paths)} images in {directory_path}")
        return self.predict_batch(image_paths, batch_size, return_probabilities)

    def evaluate_on_csv(self, csv_file, image_dir, batch_size=32):
        """
        Evaluate model performance on a dataset defined by CSV file.
        
        Args:
            csv_file: Path to CSV file with columns: filename, category
            image_dir: Directory containing images
            batch_size: Batch size for evaluation
            
        Returns:
            dict: Evaluation metrics
        """
        print(f"\n{'='*60}")
        print("MODEL EVALUATION")
        print('='*60)
        
        # Load CSV file
        try:
            df = pd.read_csv(csv_file)
            if 'filename' not in df.columns or 'category' not in df.columns:
                raise ValueError("CSV file must contain 'filename' and 'category' columns")
        except Exception as e:
            print(f"Error loading CSV file: {e}")
            return None
        
        # Prepare data
        image_paths = []
        true_labels = []
        true_label_indices = []
        
        for _, row in df.iterrows():
            img_path = os.path.join(image_dir, row['filename'])
            if os.path.exists(img_path):
                image_paths.append(img_path)
                true_labels.append(row['category'])
                # Convert class name to index
                true_label_idx = self.class_mappings['class_to_idx'].get(row['category'])
                if true_label_idx is None:
                    print(f"Warning: Class '{row['category']}' not found in class mappings")
                    true_label_idx = -1
                true_label_indices.append(true_label_idx)
            else:
                print(f"Warning: Image not found: {img_path}")
        
        if not image_paths:
            print("No valid images found for evaluation")
            return None
        
        print(f"Evaluating on {len(image_paths)} images...")
        
        # Get predictions
        all_predictions = []
        all_probs = []
        
        # Process in batches
        for i in tqdm(range(0, len(image_paths), batch_size), desc="Evaluating"):
            batch_paths = image_paths[i:i+batch_size]
            batch_tensors = []
            valid_indices = []
            
            # Preprocess batch
            for idx, img_path in enumerate(batch_paths):
                img_tensor, success = self._preprocess_image(img_path)
                if success:
                    batch_tensors.append(img_tensor)
                    valid_indices.append(i + idx)
            
            if not batch_tensors:
                continue
            
            # Stack tensors and move to device
            batch_tensor = torch.stack(batch_tensors).to(self.device)
            
            # Predict batch
            with torch.no_grad():
                outputs = self.model(batch_tensor)
                probabilities = torch.nn.functional.softmax(outputs, dim=1)
                predicted_probs, predicted_classes = torch.max(probabilities, 1)
            
            # Store results
            for j, idx in enumerate(valid_indices):
                all_predictions.append(predicted_classes[j].item())
                all_probs.append(predicted_probs[j].item())
        
        # Calculate metrics
        print("\n" + "="*60)
        print("EVALUATION RESULTS")
        print("="*60)
        
        # Accuracy
        accuracy = accuracy_score(true_label_indices, all_predictions)
        print(f"\nOverall Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
        
        # Precision, Recall, F1 (weighted average)
        precision = precision_score(true_label_indices, all_predictions, average='weighted', zero_division=0)
        recall = recall_score(true_label_indices, all_predictions, average='weighted', zero_division=0)
        f1 = f1_score(true_label_indices, all_predictions, average='weighted', zero_division=0)
        
        print(f"Weighted Precision: {precision:.4f}")
        print(f"Weighted Recall: {recall:.4f}")
        print(f"Weighted F1-Score: {f1:.4f}")
        
        # Detailed classification report
        print("\n" + "-"*60)
        print("DETAILED CLASSIFICATION REPORT")
        print("-"*60)
        
        # Handle both string and integer keys in idx_to_class
        idx_to_class = self.class_mappings['idx_to_class']
        num_classes = len(self.class_mappings['class_to_idx'])
        
        # Try to get class names, handling both string and integer keys
        class_names = []
        for i in range(num_classes):
            # Try integer key first, then string key
            class_name = idx_to_class.get(i, idx_to_class.get(str(i), f"class_{i}"))
            class_names.append(class_name)
        
        # Convert class names to list for sklearn
        unique_true_labels = sorted(list(set(true_label_indices)))
        target_names = [class_names[i] for i in unique_true_labels]
        
        report = classification_report(
            true_label_indices, 
            all_predictions, 
            target_names=target_names,
            output_dict=False,
            zero_division=0
        )
        print(report)
        
        # Confusion matrix
        print("\n" + "-"*60)
        print("CONFUSION MATRIX")
        print("-"*60)
        
        cm = confusion_matrix(true_label_indices, all_predictions)
        
        # Create labels for confusion matrix
        cm_labels = []
        for i in range(num_classes):
            if i < len(class_names):
                cm_labels.append(class_names[i])
            else:
                cm_labels.append(f"class_{i}")
        
        cm_df = pd.DataFrame(cm, 
                            index=[f"True {name}" for name in cm_labels[:num_classes]],
                            columns=[f"Pred {name}" for name in cm_labels[:num_classes]])
        print(cm_df)
        
        # Create visualization
        self._plot_confusion_matrix(cm, cm_labels[:num_classes])
        
        # Per-class metrics
        print("\n" + "-"*60)
        print("PER-CLASS METRICS")
        print("-"*60)
        
        report_dict = classification_report(
            true_label_indices, 
            all_predictions, 
            target_names=target_names,
            output_dict=True,
            zero_division=0
        )
        
        per_class_metrics = []
        for class_name in target_names:
            if class_name in report_dict:
                metrics = report_dict[class_name]
                per_class_metrics.append({
                    'class': class_name,
                    'precision': metrics['precision'],
                    'recall': metrics['recall'],
                    'f1-score': metrics['f1-score'],
                    'support': metrics['support']
                })
        
        metrics_df = pd.DataFrame(per_class_metrics)
        print(metrics_df.to_string(index=False))
        
        # Calculate confidence statistics
        avg_confidence = np.mean(all_probs)
        confidence_std = np.std(all_probs)
        print(f"\nAverage Prediction Confidence: {avg_confidence:.4f} ± {confidence_std:.4f}")
        
        # Misclassified examples
        misclassified_indices = [i for i in range(len(all_predictions)) 
                                if all_predictions[i] != true_label_indices[i]]
        
        if misclassified_indices:
            print(f"\nMisclassified Examples: {len(misclassified_indices)}/{len(all_predictions)} "
                  f"({len(misclassified_indices)/len(all_predictions)*100:.2f}%)")
            
            # Show some misclassified examples
            print("\nTop 5 misclassified examples:")
            for i, idx in enumerate(misclassified_indices[:5]):
                true_class_idx = true_label_indices[idx]
                pred_class_idx = all_predictions[idx]
                # Get class names, handling both string and integer indices
                true_class = class_names[true_class_idx] if true_class_idx < len(class_names) else f"class_{true_class_idx}"
                pred_class = class_names[pred_class_idx] if pred_class_idx < len(class_names) else f"class_{pred_class_idx}"
                confidence = all_probs[idx]
                filename = os.path.basename(image_paths[idx])
                print(f"  {filename}: True={true_class}, Pred={pred_class}, Conf={confidence:.4f}")
        
        # Save results to file
        results = {
            'overall_accuracy': float(accuracy),
            'weighted_precision': float(precision),
            'weighted_recall': float(recall),
            'weighted_f1': float(f1),
            'average_confidence': float(avg_confidence),
            'confidence_std': float(confidence_std),
            'total_samples': len(all_predictions),
            'misclassified_samples': len(misclassified_indices),
            'misclassification_rate': len(misclassified_indices) / len(all_predictions),
            'per_class_metrics': per_class_metrics,
            'confusion_matrix': cm.tolist(),
            'confusion_matrix_labels': cm_labels[:num_classes],
            'classification_report': classification_report(
                true_label_indices, 
                all_predictions, 
                target_names=target_names,
                output_dict=True,
                zero_division=0
            )
        }
        
        # Save to JSON
        output_dir = os.path.dirname(self.config.get('output_dir', '.'))
        if output_dir == '':
            output_dir = '.'
        eval_file = os.path.join(output_dir, 'evaluation_results.json')
        with open(eval_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nEvaluation results saved to: {eval_file}")
        
        return results
    
    
    def _plot_confusion_matrix(self, cm, class_names, save_path=None):
        """
        Plot and save confusion matrix.
        
        Args:
            cm: Confusion matrix array
            class_names: List of class names
            save_path: Path to save the plot (optional)
        """
        plt.figure(figsize=(10, 8))
        
        # Create heatmap
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=class_names, yticklabels=class_names)
        
        plt.title('Confusion Matrix', fontsize=16, fontweight='bold')
        plt.xlabel('Predicted Label', fontsize=12)
        plt.ylabel('True Label', fontsize=12)
        plt.tight_layout()
        
        if save_path is None:
            output_dir = os.path.dirname(self.config.get('output_dir', '.'))
            save_path = os.path.join(output_dir, 'confusion_matrix.png')
        
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Confusion matrix plot saved to: {save_path}")
    
    def print_result(self, result, verbose=False):
        """
        Print a single prediction result.
        
        Args:
            result: Prediction result dictionary
            verbose: Whether to print all probabilities
        """
        if result is None:
            print("Prediction failed")
            return
        
        print("\n" + "="*50)
        print(f"Image: {os.path.basename(result['image_path'])}")
        print(f"Predicted class: {result['predicted_class']} (index: {result['predicted_class_idx']})")
        print(f"Confidence: {result['confidence']:.4f}")
        print("-" * 30)
        
        if verbose and result['all_probabilities']:
            print("All probabilities:")
            for class_name, prob in sorted(result['all_probabilities'].items(), key=lambda x: x[1], reverse=True):
                print(f"  {class_name}: {prob:.4f}")
        print("="*50)
    
    def save_results(self, results, output_path):
        """
        Save prediction results to a CSV file.
        
        Args:
            results: List of prediction results
            output_path: Path to save CSV file
        """
        if not results:
            print("No results to save")
            return
        
        # Convert to DataFrame
        df_data = []
        for result in results:
            row = {
                'image_path': result['image_path'],
                'predicted_class': result['predicted_class'],
                'predicted_class_idx': result['predicted_class_idx'],
                'confidence': result['confidence']
            }
            
            # Add individual class probabilities
            if result['all_probabilities']:
                for class_name, prob in result['all_probabilities'].items():
                    row[f'prob_{class_name}'] = prob
            
            df_data.append(row)
        
        df = pd.DataFrame(df_data)
        
        # Save to CSV
        df.to_csv(output_path, index=False)
        print(f"Results saved to: {output_path}")
        
        # Print summary
        print(f"\nSummary:")
        print(f"Total images processed: {len(results)}")
        class_counts = df['predicted_class'].value_counts()
        for class_name, count in class_counts.items():
            print(f"  {class_name}: {count} images ({count/len(results)*100:.1f}%)")


def main():
    """
    Main inference function.
    """
    parser = argparse.ArgumentParser(description='Run inference or evaluation with trained model')
    
    # Model arguments
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to trained model checkpoint (.pth file)')
    parser.add_argument('--config_path', type=str, default=None,
                       help='Path to config file (optional, will be auto-detected)')
    
    # Mode selection
    parser.add_argument('--mode', type=str, default='inference',
                       choices=['inference', 'evaluate'],
                       help='Mode: inference for predictions, evaluate for performance metrics')
    
    # Input arguments for inference mode
    parser.add_argument('--image', type=str, default=None,
                       help='Path to single image file (inference mode)')
    parser.add_argument('--directory', type=str, default=None,
                       help='Path to directory containing images (inference mode)')
    parser.add_argument('--image_list', type=str, default=None,
                       help='Path to text file containing list of image paths (inference mode)')
    
    # Input arguments for evaluation mode
    parser.add_argument('--eval_csv', type=str, default=None,
                       help='Path to CSV file for evaluation (evaluation mode)')
    parser.add_argument('--eval_dir', type=str, default=None,
                       help='Directory containing images for evaluation (evaluation mode)')
    
    # Inference arguments
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for inference/evaluation')
    parser.add_argument('--no_probabilities', action='store_true',
                       help='Do not compute class probabilities (inference mode)')
    parser.add_argument('--verbose', action='store_true',
                       help='Print verbose output including all probabilities')
    
    # Output arguments
    parser.add_argument('--output_csv', type=str, default=None,
                       help='Path to save results as CSV file (inference mode)')
    parser.add_argument('--device', type=str, default=None,
                       choices=['cuda', 'cpu'],
                       help='Device to run inference on')
    
    args = parser.parse_args()
    
    # Initialize classifier
    classifier = ImageClassifier(
        model_path=args.model_path,
        config_path=args.config_path,
        device=args.device
    )
    
    if args.mode == 'evaluate':
        # Evaluation mode
        if not args.eval_csv or not args.eval_dir:
            parser.error("Evaluation mode requires --eval_csv and --eval_dir arguments")
        
        if not os.path.exists(args.eval_csv):
            print(f"Error: Evaluation CSV file not found: {args.eval_csv}")
            return
        
        if not os.path.exists(args.eval_dir):
            print(f"Error: Evaluation directory not found: {args.eval_dir}")
            return
        
        # Run evaluation
        classifier.evaluate_on_csv(
            csv_file=args.eval_csv,
            image_dir=args.eval_dir,
            batch_size=args.batch_size
        )
        
    else:
        # Inference mode
        if not any([args.image, args.directory, args.image_list]):
            parser.error("Inference mode requires at least one of --image, --directory, or --image_list")
        
        # Get image paths
        image_paths = []
        
        if args.image:
            if not os.path.exists(args.image):
                print(f"Error: Image file not found: {args.image}")
                return
            image_paths.append(args.image)
        
        if args.directory:
            if not os.path.exists(args.directory):
                print(f"Error: Directory not found: {args.directory}")
                return
            # Will process directory separately
        
        if args.image_list:
            if not os.path.exists(args.image_list):
                print(f"Error: Image list file not found: {args.image_list}")
                return
            with open(args.image_list, 'r') as f:
                for line in f:
                    path = line.strip()
                    if path and os.path.exists(path):
                        image_paths.append(path)
                    elif path:
                        print(f"Warning: Image not found, skipping: {path}")
        
        # Run inference
        results = []
        
        if args.directory:
            # Process directory
            dir_results = classifier.predict_directory(
                directory_path=args.directory,
                batch_size=args.batch_size,
                return_probabilities=not args.no_probabilities
            )
            results.extend(dir_results)
        
        if image_paths:
            # Process individual images
            if len(image_paths) == 1 and not args.directory:
                # Single image
                result = classifier.predict(
                    image_path=image_paths[0],
                    return_probabilities=not args.no_probabilities
                )
                if result:
                    classifier.print_result(result, verbose=args.verbose)
                    results.append(result)
            else:
                # Multiple images
                batch_results = classifier.predict_batch(
                    image_paths=image_paths,
                    batch_size=args.batch_size,
                    return_probabilities=not args.no_probabilities
                )
                results.extend(batch_results)
                
                # Print first few results
                for i, result in enumerate(batch_results[:3]):  # Print first 3
                    print(f"\nResult {i+1}:")
                    classifier.print_result(result, verbose=args.verbose)
                
                if len(batch_results) > 3:
                    print(f"\n... and {len(batch_results) - 3} more results")
        
        # Save results if requested
        if args.output_csv and results:
            classifier.save_results(results, args.output_csv)
        
        # Print summary
        if results:
            print(f"\n{'='*50}")
            print(f"INFERENCE COMPLETE")
            print(f"{'='*50}")
            print(f"Total images processed: {len(results)}")
            
            # Group by class
            from collections import Counter
            class_counts = Counter([r['predicted_class'] for r in results])
            
            print("\nClass distribution:")
            for class_name, count in class_counts.most_common():
                percentage = count / len(results) * 100
                print(f"  {class_name}: {count} ({percentage:.1f}%)")


if __name__ == "__main__":
    main()
