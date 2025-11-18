"""PyTorch Fashion-MNIST training script with ClearML tracking."""

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from clearml import Task
from sklearn.metrics import confusion_matrix
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms
from pathlib import Path
import yaml


def load_config(config_path: str = "config.yaml"):
    """Load hyperparameters from config file.

    Args:
        config_path: Path to config YAML file.

    Returns:
        Configuration dictionary.
    """
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def create_model(input_size: int, hidden_size: int, output_size: int):
    """Create the neural network model.
    
    Args:
        input_size: Size of input layer (784 for Fashion-MNIST)
        hidden_size: Size of hidden layer
        output_size: Number of output classes (10 for Fashion-MNIST)
    
    Returns:
        PyTorch sequential model
    """
    return nn.Sequential(
        nn.Flatten(),
        nn.Linear(input_size, hidden_size),
        nn.ReLU(),
        nn.Linear(hidden_size, output_size)
    )


def train_model(model, trainloader, criterion, optimizer, num_epochs, writer, task):
    """Train the model.
    
    Args:
        model: Neural network model
        trainloader: Training data loader
        criterion: Loss function
        optimizer: Optimizer
        num_epochs: Number of training epochs
        writer: TensorBoard writer
        task: ClearML task
    """
    print("Starting training...")
    for epoch in range(num_epochs):
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            # Log to TensorBoard
            writer.add_scalar('Loss/train', loss.item(), epoch * len(trainloader) + i)
            
            # Print statistics
            running_loss += loss.item()
            if i % 100 == 99:
                avg_loss = running_loss / 100
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {avg_loss:.3f}')
                running_loss = 0.0
    
    print('Finished training')


def evaluate_model(model, testloader):
    """Evaluate the model on test data.
    
    Args:
        model: Trained neural network model
        testloader: Test data loader
    
    Returns:
        Tuple of (y_true, y_pred, accuracy)
    """
    print("Evaluating model...")
    y_true = []
    y_pred = []
    
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            y_true.extend(labels.tolist())
            y_pred.extend(predicted.tolist())
    
    # Calculate accuracy
    accuracy = sum([1 for true, pred in zip(y_true, y_pred) if true == pred]) / len(y_true)
    print(f'Test Accuracy: {accuracy:.4f}')
    
    return y_true, y_pred, accuracy


def plot_confusion_matrix(y_true, y_pred, classes, task):
    """Create and log confusion matrix.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        classes: Class names
        task: ClearML task
    """
    confusion_mat = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(10, 8))
    plt.imshow(confusion_mat, cmap=plt.cm.Blues)
    plt.xticks(np.arange(len(classes)), classes, rotation=90)
    plt.yticks(np.arange(len(classes)), classes)
    plt.colorbar()
    plt.xlabel('Predicted label')
    plt.ylabel('True label')
    plt.title('Confusion matrix')
    plt.tight_layout()
    
    # Save and log to ClearML
    confusion_matrix_path = 'confusion_matrix.png'
    plt.savefig(confusion_matrix_path)
    task.upload_artifact('confusion_matrix', artifact_object=confusion_matrix_path)
    
    # Also log as matplotlib figure
    task.get_logger().report_matplotlib_figure(
        title="Confusion Matrix",
        series="",
        figure=plt.gcf(),
        iteration=0
    )
    
    plt.close()


def main():
    # Initialize ClearML Task
    task = Task.init(
        project_name="Full Overview",
        task_name="model_training"
    )
    
    # Load configuration
    config = load_config()
    
    # Connect configuration to ClearML for tracking
    config = task.connect(config)
    
    # Extract hyperparameters
    training_config = config['training']
    model_config = config['model']
    data_config = config['data']
    
    batch_size = training_config['batch_size']
    learning_rate = training_config['learning_rate']
    num_epochs = training_config['num_epochs']
    input_size = model_config['input_size']
    hidden_size = model_config['hidden_size']
    output_size = model_config['output_size']
    normalize_mean = data_config['normalize_mean']
    normalize_std = data_config['normalize_std']
    
    print(f"Starting training with:")
    print(f"  Learning rate: {learning_rate}")
    print(f"  Batch size: {batch_size}")
    print(f"  Epochs: {num_epochs}")
    print(f"  Hidden size: {hidden_size}")
    
    # Initialize TensorBoard writer
    writer = SummaryWriter()
    
    # Define the transformation
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((normalize_mean,), (normalize_std,))
    ])
    
    # Load training data
    trainset = datasets.FashionMNIST('data', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)
    
    # Load test data
    testset = datasets.FashionMNIST('data', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False)
    
    # Create model
    model = create_model(input_size, hidden_size, output_size)
    
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    
    # Train the model
    train_model(model, trainloader, criterion, optimizer, num_epochs, writer, task)
    
    # Evaluate the model
    classes = ('T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot')
    y_true, y_pred, accuracy = evaluate_model(model, testloader)
    
    # Log accuracy to ClearML
    task.get_logger().report_single_value('Test Accuracy', accuracy)
    
    # Create and log confusion matrix
    plot_confusion_matrix(y_true, y_pred, classes, task)
    
    # Save model
    model_path = 'fashion_mnist_model.pth'
    torch.save(model.state_dict(), model_path)
    task.upload_artifact('model_weights', artifact_object=model_path)
    
    print("Training completed!")
    
    # Close TensorBoard writer and ClearML task
    writer.close()
    task.close()


if __name__ == "__main__":
    main()