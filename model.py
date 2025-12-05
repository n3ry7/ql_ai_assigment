import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import ResNet18_Weights, ResNet34_Weights, ResNet50_Weights, ResNet101_Weights


class ImageClassifier(nn.Module):
    """
    Image classifier using ResNet architecture modified for 1-channel input.
    
    Args:
        num_classes (int): Number of output classes
        model_name (str): Name of the ResNet model to use
        input_channels (int): Number of input channels (1 for grayscale, 3 for RGB)
        pretrained (bool): Whether to use pretrained weights (only for 3-channel)
        freeze_backbone (bool): Whether to freeze the backbone layers for transfer learning
        dropout_rate (float): Dropout rate for regularization
    """
    
    def __init__(self, num_classes=3, model_name='resnet18', input_channels=1, 
                 pretrained=True, freeze_backbone=False, dropout_rate=0.5):
        super(ImageClassifier, self).__init__()
        
        self.num_classes = num_classes
        self.model_name = model_name
        self.input_channels = input_channels
        self.pretrained = pretrained and input_channels == 3  # Only pretrained for 3 channels
        
        # Load the specified ResNet model
        self.backbone = self._get_resnet_model(model_name, self.pretrained)
        
        # Modify first conv layer for 1-channel input if needed
        if input_channels != 3:
            self._modify_first_conv_layer()
        
        # Freeze backbone layers if specified
        if freeze_backbone:
            self._freeze_backbone()
        
        # Get the number of features in the final layer
        num_features = self.backbone.fc.in_features
        
        # Add dropout for regularization
        self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity()
        
        # Replace the final fully connected layer
        self.backbone.fc = nn.Sequential(
            self.dropout,
            nn.Linear(num_features, num_classes)
        )
        
        # Initialize the new fc layer
        self._initialize_fc_layer()
        
        # Initialize the modified first conv layer if needed
        if input_channels != 3:
            self._initialize_first_conv_layer()
    
    def _get_resnet_model(self, model_name, pretrained):
        """Get the specified ResNet model with optional pretrained weights."""
        model_weights = {
            'resnet18': ResNet18_Weights.DEFAULT if pretrained else None,
            'resnet34': ResNet34_Weights.DEFAULT if pretrained else None,
            'resnet50': ResNet50_Weights.DEFAULT if pretrained else None,
            'resnet101': ResNet101_Weights.DEFAULT if pretrained else None,
        }
        
        if model_name not in model_weights:
            raise ValueError(f"Unsupported model: {model_name}. "
                           f"Choose from {list(model_weights.keys())}")
        
        # Get the model constructor
        model_constructor = getattr(models, model_name)
        
        # Create model with or without pretrained weights
        if pretrained:
            model = model_constructor(weights=model_weights[model_name])
        else:
            model = model_constructor()
        
        return model
    
    def _modify_first_conv_layer(self):
        """Modify the first convolutional layer to accept 1-channel input."""
        original_first_conv = self.backbone.conv1
        
        # Create new conv layer with appropriate number of input channels
        new_first_conv = nn.Conv2d(
            self.input_channels,
            original_first_conv.out_channels,
            kernel_size=original_first_conv.kernel_size,
            stride=original_first_conv.stride,
            padding=original_first_conv.padding,
            bias=original_first_conv.bias is not None
        )
        
        # Replace the first conv layer
        self.backbone.conv1 = new_first_conv
    
    def _initialize_first_conv_layer(self, init_method='kaiming'):
        """Initialize the modified first conv layer."""
        if init_method == 'kaiming':
            nn.init.kaiming_normal_(self.backbone.conv1.weight, mode='fan_out', nonlinearity='relu')
        elif init_method == 'xavier':
            nn.init.xavier_normal_(self.backbone.conv1.weight)
        elif init_method == 'normal':
            nn.init.normal_(self.backbone.conv1.weight, mean=0.0, std=0.01)
        
        if self.backbone.conv1.bias is not None:
            nn.init.constant_(self.backbone.conv1.bias, 0)
        
        print(f"Initialized first conv layer for {self.input_channels} channel input")
    
    def _freeze_backbone(self):
        """Freeze all backbone layers except the final fully connected layer."""
        for param in self.backbone.parameters():
            param.requires_grad = False
        
        # Unfreeze the final fc layer (will be replaced anyway)
        if hasattr(self.backbone, 'fc'):
            for param in self.backbone.fc.parameters():
                param.requires_grad = True
    
    def _initialize_fc_layer(self, init_method='kaiming'):
        """Initialize the new fully connected layer."""
        if init_method == 'kaiming':
            nn.init.kaiming_normal_(self.backbone.fc[1].weight, mode='fan_out', nonlinearity='relu')
        elif init_method == 'xavier':
            nn.init.xavier_normal_(self.backbone.fc[1].weight)
        elif init_method == 'normal':
            nn.init.normal_(self.backbone.fc[1].weight, mean=0.0, std=0.01)
        
        if self.backbone.fc[1].bias is not None:
            nn.init.constant_(self.backbone.fc[1].bias, 0)
    
    def forward(self, x):
        """Forward pass."""
        return self.backbone(x)
    
    def unfreeze_layers(self, num_layers=None):
        """
        Unfreeze specified number of layers for fine-tuning.
        
        Args:
            num_layers (int or list): Number of layers to unfreeze from the end,
                                     or list of layer names to unfreeze.
                                     If None, unfreeze all layers.
        """
        if num_layers is None:
            # Unfreeze all layers
            for param in self.parameters():
                param.requires_grad = True
            print("Unfroze all layers")
            return
        
        if isinstance(num_layers, list):
            # Unfreeze specific layers by name
            for name, param in self.named_parameters():
                if any(layer_name in name for layer_name in num_layers):
                    param.requires_grad = True
                    print(f"Unfroze layer: {name}")
            return
        
        # Unfreeze last 'num_layers' layers
        total_layers = len(list(self.backbone.parameters()))
        layers_unfrozen = 0
        
        for i, (name, param) in enumerate(reversed(list(self.backbone.named_parameters()))):
            if layers_unfrozen >= num_layers:
                break
            param.requires_grad = True
            layers_unfrozen += 1
            print(f"Unfroze layer: {name}")
        
        # Always make sure the fc layer is trainable
        for param in self.backbone.fc.parameters():
            param.requires_grad = True
    
    def get_trainable_parameters(self):
        """Get the number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def get_total_parameters(self):
        """Get the total number of parameters."""
        return sum(p.numel() for p in self.parameters())


class DocumentClassifier(nn.Module):
    """
    Custom CNN optimized for document images (1-channel input).
    This model is designed specifically for document classification.
    """
    
    def __init__(self, num_classes=3, input_channels=1, dropout_rate=0.5):
        super(DocumentClassifier, self).__init__()
        
        # Feature extractor with aggressive downsampling for large document images
        self.features = nn.Sequential(
            # Block 1: Initial downsampling
            nn.Conv2d(input_channels, 32, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            
            # Block 2: Feature extraction
            nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            
            # Block 3: Deeper features
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            
            # Block 4: More depth
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            
            # Block 5: Final features
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )
        
        # Adaptive pooling to handle any input size
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate/2),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate/4),
            nn.Linear(128, num_classes)
        )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights using He initialization for ReLU activations."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """Forward pass."""
        x = self.features(x)
        x = self.adaptive_pool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
    
    def get_trainable_parameters(self):
        """Get the number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def get_total_parameters(self):
        """Get the total number of parameters."""
        return sum(p.numel() for p in self.parameters())


class EfficientNetClassifier(nn.Module):
    """
    EfficientNet classifier modified for 1-channel input.
    """
    
    def __init__(self, num_classes=3, model_name='efficientnet_b0', input_channels=1,
                 pretrained=True, freeze_backbone=False, dropout_rate=0.3):
        super(EfficientNetClassifier, self).__init__()
        
        try:
            from torchvision.models import EfficientNet_B0_Weights, EfficientNet_B1_Weights, \
                                          EfficientNet_B2_Weights, EfficientNet_B3_Weights
            
            self.pretrained = pretrained and input_channels == 3  # Only pretrained for 3 channels
            
            model_weights = {
                'efficientnet_b0': EfficientNet_B0_Weights.DEFAULT if self.pretrained else None,
                'efficientnet_b1': EfficientNet_B1_Weights.DEFAULT if self.pretrained else None,
                'efficientnet_b2': EfficientNet_B2_Weights.DEFAULT if self.pretrained else None,
                'efficientnet_b3': EfficientNet_B3_Weights.DEFAULT if self.pretrained else None,
            }
            
            if model_name not in model_weights:
                raise ValueError(f"Unsupported EfficientNet model: {model_name}")
            
            # Get the model constructor
            model_constructor = getattr(models, model_name)
            
            # Create model with or without pretrained weights
            if self.pretrained:
                model = model_constructor(weights=model_weights[model_name])
            else:
                model = model_constructor()
            
            # Modify first conv layer if needed
            if input_channels != 3:
                original_first_conv = model.features[0][0]
                new_first_conv = nn.Conv2d(
                    input_channels,
                    original_first_conv.out_channels,
                    kernel_size=original_first_conv.kernel_size,
                    stride=original_first_conv.stride,
                    padding=original_first_conv.padding,
                    bias=original_first_conv.bias is not None
                )
                model.features[0][0] = new_first_conv
                
                # Initialize the new conv layer
                nn.init.kaiming_normal_(new_first_conv.weight, mode='fan_out', nonlinearity='relu')
                if new_first_conv.bias is not None:
                    nn.init.constant_(new_first_conv.bias, 0)
            
            # Get the number of features in the classifier
            num_features = model.classifier[1].in_features
            
            # Freeze backbone if specified
            if freeze_backbone:
                for param in model.parameters():
                    param.requires_grad = False
            
            # Replace the final classifier layer
            model.classifier = nn.Sequential(
                nn.Dropout(p=dropout_rate, inplace=True) if dropout_rate > 0 else nn.Identity(),
                nn.Linear(num_features, num_classes)
            )
            
            self.model = model
            
        except ImportError:
            print("EfficientNet requires torchvision >= 0.11.0")
            print("Falling back to custom DocumentClassifier")
            self.model = DocumentClassifier(
                num_classes=num_classes,
                input_channels=input_channels,
                dropout_rate=dropout_rate
            )
    
    def forward(self, x):
        return self.model(x)


def create_model(model_type='resnet18', num_classes=3, input_channels=1, 
                 pretrained=True, freeze_backbone=False, device=None):
    """
    Factory function to create a model.
    
    Args:
        model_type (str): Type of model to create
        num_classes (int): Number of output classes
        input_channels (int): Number of input channels (1 for grayscale)
        pretrained (bool): Whether to use pretrained weights (only for 3-channel)
        freeze_backbone (bool): Whether to freeze backbone layers
        device: Device to place the model on
        
    Returns:
        nn.Module: The created model
    """
    if model_type.startswith('resnet'):
        model = ImageClassifier(
            num_classes=num_classes,
            model_name=model_type,
            input_channels=input_channels,
            pretrained=pretrained and input_channels == 3,  # Only if 3 channels
            freeze_backbone=freeze_backbone,
            dropout_rate=0.5
        )
    elif model_type.startswith('efficientnet'):
        model = EfficientNetClassifier(
            num_classes=num_classes,
            model_name=model_type,
            input_channels=input_channels,
            pretrained=pretrained and input_channels == 3,  # Only if 3 channels
            freeze_backbone=freeze_backbone,
            dropout_rate=0.3
        )
    elif model_type == 'document_cnn':
        model = DocumentClassifier(
            num_classes=num_classes,
            input_channels=input_channels,
            dropout_rate=0.5
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    if device:
        model = model.to(device)
    
    return model


def print_model_summary(model, input_size=(1, 224, 224), device='cpu'):
    """
    Print a summary of the model architecture.
    
    Args:
        model: The model to summarize
        input_size: Input tensor size
        device: Device to use for summary
    """
    try:
        from torchsummary import summary
        model = model.to(device)
        summary(model, input_size=input_size)
    except ImportError:
        print("Install torchsummary for detailed model summary: pip install torchsummary")
        print(f"Model: {model.__class__.__name__}")
        if hasattr(model, 'get_trainable_parameters'):
            print(f"Trainable parameters: {model.get_trainable_parameters():,}")
            print(f"Total parameters: {model.get_total_parameters():,}")
        print(f"Input channels: {model.input_channels if hasattr(model, 'input_channels') else 'N/A'}")


if __name__ == "__main__":
    """
    Test the model creation and print summaries.
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='Test model creation for document classification')
    parser.add_argument('--model', type=str, default='resnet50',
                       choices=['resnet18', 'resnet34', 'resnet50', 'resnet101',
                                'efficientnet_b0', 'efficientnet_b1', 'efficientnet_b2',
                                'document_cnn'],
                       help='Model architecture to test')
    parser.add_argument('--num_classes', type=int, default=3,
                       help='Number of output classes')
    parser.add_argument('--input_channels', type=int, default=1,
                       choices=[1, 3],
                       help='Number of input channels (1 for grayscale, 3 for RGB)')
    parser.add_argument('--pretrained', action='store_true',
                       help='Use pretrained weights (only for 3-channel input)')
    parser.add_argument('--freeze', action='store_true',
                       help='Freeze backbone layers')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                       help='Device to use')
    
    args = parser.parse_args()
    
    print(f"Testing model: {args.model}")
    print(f"Device: {args.device}")
    print(f"Input channels: {args.input_channels}")
    print(f"Pretrained: {args.pretrained} (only applicable for 3-channel input)")
    print(f"Freeze backbone: {args.freeze}")
    print(f"Number of classes: {args.num_classes}")
    print("-" * 50)
    
    # Create model
    model = create_model(
        model_type=args.model,
        num_classes=args.num_classes,
        input_channels=args.input_channels,
        pretrained=args.pretrained,
        freeze_backbone=args.freeze,
        device=args.device
    )
    
    # Print summary
    if args.model == 'document_cnn':
        # For large document images
        print_model_summary(model, input_size=(args.input_channels, 2197, 2200), device=args.device)
    else:
        # For standard ImageNet size
        print_model_summary(model, input_size=(args.input_channels, 224, 224), device=args.device)
    
    # Test forward pass with dummy input
    print("\nTesting forward pass...")
    if args.model == 'document_cnn':
        dummy_input = torch.randn(2, args.input_channels, 2197, 2200).to(args.device)
    else:
        dummy_input = torch.randn(2, args.input_channels, 224, 224).to(args.device)
    
    with torch.no_grad():
        output = model(dummy_input)
        print(f"Input shape: {dummy_input.shape}")
        print(f"Output shape: {output.shape}")
        print(f"Output sample: {output[0]}")
    
    # Show trainable parameters
    if hasattr(model, 'get_trainable_parameters'):
        print(f"\nTrainable parameters: {model.get_trainable_parameters():,}")
        print(f"Total parameters: {model.get_total_parameters():,}")
    
    # Test unfreezing for transfer learning models
    if args.freeze and hasattr(model, 'unfreeze_layers'):
        print("\nTesting layer unfreezing...")
        model.unfreeze_layers(num_layers=5)
        print(f"Trainable parameters after unfreezing: {model.get_trainable_parameters():,}")