"""Training script template.

This is a basic template for training a machine learning model with ClearML tracking.
Customize this template based on your specific framework and requirements.
"""

from clearml import Task
import yaml
from pathlib import Path


def load_config(config_path: str = "config.yaml"):
    """Load hyperparameters from config file.

    Args:
        config_path: Path to config YAML file.

    Returns:
        Configuration dictionary.
    """
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def main():
    # Initialize ClearML Task
    task = Task.current_task()
    if task is None:
        print("Warning: No active ClearML task found. Starting in development mode...")
        task = Task.init(
            project_name="full_clearml_overview",
            task_name="development",
        )

    # Load configuration
    config = load_config()

    # Connect configuration to ClearML for tracking
    task.connect(config)

    # Access hyperparameters
    learning_rate = config['training']['learning_rate']
    batch_size = config['training']['batch_size']
    num_epochs = config['training']['num_epochs']

    print(f"Starting training with:")
    print(f"  Learning rate: {learning_rate}")
    print(f"  Batch size: {batch_size}")
    print(f"  Epochs: {num_epochs}")

    # TODO: Add your data loading code here
    # train_dataset = load_dataset(...)
    # val_dataset = load_dataset(...)

    # TODO: Add your model initialization here
    # model = create_model(...)

    # TODO: Add your training loop here
    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")

        # Training step
        # train_loss = train_one_epoch(model, train_dataset, ...)
        # val_loss = validate(model, val_dataset, ...)

        # Log metrics to ClearML
        # task.get_logger().report_scalar(
        #     title="Loss",
        #     series="train",
        #     value=train_loss,
        #     iteration=epoch
        # )
        # task.get_logger().report_scalar(
        #     title="Loss",
        #     series="validation",
        #     value=val_loss,
        #     iteration=epoch
        # )

    print("Training completed!")

    # TODO: Save model
    # torch.save(model.state_dict(), "model.pth")
    # task.upload_artifact("model", artifact_object="model.pth")


if __name__ == "__main__":
    main()
