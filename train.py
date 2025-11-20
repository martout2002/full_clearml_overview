"""Training script for ResNet on CIFAR-10.

This script trains a ResNet model with full ClearML tracking and lineage.
It works out of the box with PyTorch and CIFAR-10 dataset.

automatically tracks:
- Git repository, branch, and commit
- Python environment and installed packages
- Console output
- Configuration parameters
- Metrics and plots
- Models and artifacts
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
from clearml import Task, Logger
import yaml
from pathlib import Path
from model import create_model
import matplotlib.pyplot as plt
import numpy as np


def load_config(config_path: str = "config.yaml"):
    """Load hyperparameters from config file.

    Args:
        config_path: Path to config YAML file.

    Returns:
        Configuration dictionary.
    """
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def get_data_loaders(config):
    """Create train and validation data loaders.

    Args:
        config: Configuration dictionary

    Returns:
        Tuple of (train_loader, val_loader)
    """
    data_config = config['data']
    batch_size = config['training']['batch_size']
    num_workers = data_config.get('num_workers', 2)

    # Data augmentation for training
    if data_config.get('augmentation', True):
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
    else:
        train_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

    # Validation transform (no augmentation)
    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    # Load CIFAR-10 dataset
    train_dataset = torchvision.datasets.CIFAR10(
        root=data_config.get('data_dir', './data'),
        train=True,
        download=True,
        transform=train_transform
    )

    val_dataset = torchvision.datasets.CIFAR10(
        root=data_config.get('data_dir', './data'),
        train=False,
        download=True,
        transform=val_transform
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return train_loader, val_loader


def train_epoch(model, train_loader, criterion, optimizer, device, epoch, logger, log_interval=10):
    """Train for one epoch.

    Args:
        model: PyTorch model
        train_loader: Training data loader
        criterion: Loss function
        optimizer: Optimizer
        device: Device to train on
        epoch: Current epoch number
        logger: ClearML logger
        log_interval: How often to log (in batches)

    Returns:
        Dictionary with average loss and accuracy
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)

        # Forward pass
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        # Backward pass
        loss.backward()
        optimizer.step()

        # Statistics
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        # Log batch metrics
        if (batch_idx + 1) % log_interval == 0:
            batch_loss = running_loss / (batch_idx + 1)
            batch_acc = 100. * correct / total
            print(f'  Batch [{batch_idx + 1}/{len(train_loader)}] '
                  f'Loss: {batch_loss:.3f} | Acc: {batch_acc:.2f}%')

    # Calculate epoch metrics
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100. * correct / total

    return {
        'loss': epoch_loss,
        'accuracy': epoch_acc
    }


def validate(model, val_loader, criterion, device):
    """Validate the model.

    Args:
        model: PyTorch model
        val_loader: Validation data loader
        criterion: Loss function
        device: Device to validate on

    Returns:
        Dictionary with average loss and accuracy
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            # Statistics
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    # Calculate validation metrics
    val_loss = running_loss / len(val_loader)
    val_acc = 100. * correct / total

    return {
        'loss': val_loss,
        'accuracy': val_acc
    }


def main():
    # Initialize Task with lineage tracking and auto-capture enabled
    # This automatically links to the previous experiment with the same name
    task = Task.current_task()
    if task is None:
        print("Creating new task with lineage tracking...")
        task = Task.init_with_lineage(
            project_name="resnet_testing_2",
            task_name="training",
            task_type=Task.TaskTypes.training,
            auto_connect_frameworks={
                'matplotlib': True,  # Auto-capture matplotlib plots
                'tensorboard': True,  # Auto-capture TensorBoard scalars
                'pytorch': True,      # Auto-capture PyTorch models
            }
        )
        print(f"Task created: {task.id}")
        if task.parent:
            print(f"Parent task: {task.parent}")
        else:
            print("No parent (first run)")
    
    # Enable automatic logging of all scalars, plots and debug samples
    from clearml import Logger
    # Note: Logger.set_default_upload_destination requires a fileserver URI
    # For default ClearML server, use: Logger.set_default_upload_destination('s3://your-bucket' or 'file://path')
    # For now, we'll rely on default server settings
    
    task.set_project_defaults(
        auto_connect_arg_parser=True,
        auto_connect_frameworks=True,
        auto_resource_monitoring=True
    )

    # Load configuration
    config = load_config()

    # Connect configuration for tracking
    task.connect(config)

    # Get logger for metrics
    logger = task.get_logger()

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")

    # Access hyperparameters
    learning_rate = config['training']['learning_rate']
    batch_size = config['training']['batch_size']
    num_epochs = config['training']['num_epochs']
    optimizer_name = config['training']['optimizer']
    weight_decay = config['training']['weight_decay']

    print(f"\nStarting training with:")
    print(f"  Model: {config['model']['architecture']}")
    print(f"  Pretrained: {config['model']['pretrained']}")
    print(f"  Learning rate: {learning_rate}")
    print(f"  Batch size: {batch_size}")
    print(f"  Epochs: {num_epochs}")
    print(f"  Optimizer: {optimizer_name}")
    print()

    # Create data loaders
    print("Loading CIFAR-10 dataset...")
    train_loader, val_loader = get_data_loaders(config)
    train_size = len(train_loader.dataset)
    val_size = len(val_loader.dataset)
    print(f"  Training samples: {train_size}")
    print(f"  Validation samples: {val_size}")

    # Log dataset information
    task.connect_configuration(name="dataset_info", configuration={
        "train_size": train_size,
        "val_size": val_size,
        "dataset": config['data']['dataset'],
        "data_dir": config['data']['data_dir']
    })

    # Create model
    print(f"\nCreating {config['model']['architecture']} model...")
    model = create_model(config)
    model = model.to(device)

    # Log model architecture
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    model_info = f"{config['model']['architecture']}\n"
    model_info += f"Total parameters: {total_params:,}\n"
    model_info += f"Trainable parameters: {trainable_params:,}"
    logger.report_text(model_info)
    print(model_info)

    # Create optimizer
    if optimizer_name.lower() == 'adam':
        optimizer = optim.Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
    elif optimizer_name.lower() == 'sgd':
        optimizer = optim.SGD(
            model.parameters(),
            lr=learning_rate,
            momentum=config['training'].get('momentum', 0.9),
            weight_decay=weight_decay
        )
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")

    # Loss function
    criterion = nn.CrossEntropyLoss()

    # Training loop
    print("\nStarting training...\n")
    best_val_acc = 0.0
    
    # Lists to track metrics for plotting
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []

    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")
        print("-" * 50)

        # Train
        metrics_train = train_epoch(
            model, train_loader, criterion, optimizer, device,
            epoch, logger, log_interval=config['logging']['log_interval']
        )

        # Validate
        metrics_val = validate(model, val_loader, criterion, device)

        # Print epoch summary
        print(f"  Train Loss: {metrics_train['loss']:.3f} | Train Acc: {metrics_train['accuracy']:.2f}%")
        print(f"  Val Loss: {metrics_val['loss']:.3f} | Val Acc: {metrics_val['accuracy']:.2f}%")

        # Track metrics for plotting
        train_losses.append(metrics_train['loss'])
        val_losses.append(metrics_val['loss'])
        train_accs.append(metrics_train['accuracy'])
        val_accs.append(metrics_val['accuracy'])
        
        # Log metrics to ClearML (auto-captured scalars)
        logger.report_scalar(
            title="Loss",
            series="train",
            value=metrics_train['loss'],
            iteration=epoch
        )
        logger.report_scalar(
            title="Loss",
            series="validation",
            value=metrics_val['loss'],
            iteration=epoch
        )
        logger.report_scalar(
            title="Accuracy",
            series="train",
            value=metrics_train['accuracy'],
            iteration=epoch
        )
        logger.report_scalar(
            title="Accuracy",
            series="validation",
            value=metrics_val['accuracy'],
            iteration=epoch
        )
        
        # Log learning rate
        current_lr = optimizer.param_groups[0]['lr']
        logger.report_scalar(
            title="Learning Rate",
            series="lr",
            value=current_lr,
            iteration=epoch
        )

        # Save best model
        if metrics_val['accuracy'] > best_val_acc:
            best_val_acc = metrics_val['accuracy']
            print(f"  New best validation accuracy: {best_val_acc:.2f}%")
        
        # Log debug sample images every 5 epochs
        if (epoch + 1) % 5 == 0:
            # Get a batch of validation images
            val_iter = iter(val_loader)
            sample_images, sample_labels = next(val_iter)
            sample_images = sample_images.to(device)
            
            # Get predictions
            model.eval()
            with torch.no_grad():
                sample_outputs = model(sample_images)
                _, sample_preds = sample_outputs.max(1)
            
            # Log first 8 images as debug samples
            class_names = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
            for i in range(min(8, len(sample_images))):
                img = sample_images[i].cpu().numpy().transpose(1, 2, 0)
                # Denormalize
                img = img * np.array([0.2023, 0.1994, 0.2010]) + np.array([0.4914, 0.4822, 0.4465])
                img = np.clip(img, 0, 1)
                
                title = f"True: {class_names[sample_labels[i]]} | Pred: {class_names[sample_preds[i]]}"
                logger.report_image(
                    title="Predictions",
                    series=f"epoch_{epoch+1}",
                    iteration=i,
                    image=img
                )

        print()

    print("Training completed!")
    
    # Create and log training plots
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # Loss plot
    epochs_range = range(1, num_epochs + 1)
    axes[0].plot(epochs_range, train_losses, 'b-', label='Train Loss')
    axes[0].plot(epochs_range, val_losses, 'r-', label='Val Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training and Validation Loss')
    axes[0].legend()
    axes[0].grid(True)
    
    # Accuracy plot
    axes[1].plot(epochs_range, train_accs, 'b-', label='Train Acc')
    axes[1].plot(epochs_range, val_accs, 'r-', label='Val Acc')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy (%)')
    axes[1].set_title('Training and Validation Accuracy')
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    logger.report_matplotlib_figure(
        title="Training Summary",
        series="metrics",
        figure=fig,
        iteration=num_epochs
    )
    plt.close()

    # Save and upload model
    Path(config['logging']['checkpoint_dir']).mkdir(exist_ok=True)
    model_path = f"{config['logging']['checkpoint_dir']}/model_final.pth"
    torch.save(model.state_dict(), model_path)
    task.upload_artifact("model_weights", artifact_object=model_path)
    print(f"Model saved to {model_path}")

    # Save final metrics as summary
    task.set_parameter("final_metrics/train_loss", metrics_train['loss'])
    task.set_parameter("final_metrics/val_loss", metrics_val['loss'])
    task.set_parameter("final_metrics/train_accuracy", metrics_train['accuracy'])
    task.set_parameter("final_metrics/val_accuracy", metrics_val['accuracy'])
    task.set_parameter("final_metrics/best_val_accuracy", best_val_acc)

    print(f"\nTask ID: {task.id}")
    print(f"View results at: {task.get_output_log_web_page()}")


if __name__ == "__main__":
    main()
