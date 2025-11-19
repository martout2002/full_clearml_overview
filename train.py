"""Training script template.

This is a basic template for training a machine learning model with full tracking.
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
    # Initialize Task
    task = Task.current_task()
    if task is None:
        print("Warning: No active task found. Creating new task with lineage tracking...")
        task = Task.init_with_lineage(
            project_name="full_clearml_test",
            task_name="training",
            task_type=Task.TaskTypes.training,
        )
        print(f"Task created: {task.id}")
        if task.parent:
            print(f"Parent task: {task.parent}")
        else:
            print("No parent (first run)")

    # Load configuration
    config = load_config()

    # Connect configuration for tracking
    # This logs all hyperparameters and allows you to modify them in the UI
    task.connect(config)

    # Access hyperparameters
    learning_rate = config['training']['learning_rate']
    batch_size = config['training']['batch_size']
    num_epochs = config['training']['num_epochs']

    print(f"\nStarting training with:")
    print(f"  Learning rate: {learning_rate}")
    print(f"  Batch size: {batch_size}")
    print(f"  Epochs: {num_epochs}")
    print()

    # Get logger for metrics
    logger = task.get_logger()

    # TODO: Add your data loading code here
    # Example: train_dataset = load_dataset(...)
    # Example: val_dataset = load_dataset(...)
    train_size = 1000  # TODO: Replace with actual dataset size
    val_size = 200     # TODO: Replace with actual dataset size

    # Log dataset information
    task.connect_configuration(name="dataset_info", configuration={
        "train_size": train_size,
        "val_size": val_size,
        "data_path": "/path/to/data"  # TODO: Update with actual path
    })

    # TODO: Add your model initialization here
    # Example: model = create_model(...)
    model_info = "Model architecture not yet defined"  # TODO: Replace with str(model)

    # Log model architecture
    logger.report_text(model_info)

    # Training loop with metric logging
    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")

        # TODO: Replace these with your actual training code
        # Example training step:
        # train_loss = train_one_epoch(model, train_dataset, optimizer, ...)
        # train_accuracy = evaluate_accuracy(model, train_dataset, ...)
        # val_loss = validate(model, val_dataset, ...)
        # val_accuracy = evaluate_accuracy(model, val_dataset, ...)

        # Placeholder values - REPLACE WITH YOUR ACTUAL METRICS
        train_loss = 0.5 - (epoch * 0.05)  # Simulated decreasing loss
        val_loss = 0.6 - (epoch * 0.04)
        train_accuracy = 0.5 + (epoch * 0.05)  # Simulated increasing accuracy
        val_accuracy = 0.4 + (epoch * 0.04)

        # Log loss metrics
        logger.report_scalar(
            title="Loss",
            series="train",
            value=train_loss,
            iteration=epoch
        )
        logger.report_scalar(
            title="Loss",
            series="validation",
            value=val_loss,
            iteration=epoch
        )

        # Log accuracy metrics
        logger.report_scalar(
            title="Accuracy",
            series="train",
            value=train_accuracy,
            iteration=epoch
        )
        logger.report_scalar(
            title="Accuracy",
            series="validation",
            value=val_accuracy,
            iteration=epoch
        )

        # Optional: Log learning rate schedule if using scheduler
        # current_lr = optimizer.param_groups[0]['lr']
        # logger.report_scalar(
        #     title="Learning Rate",
        #     series="lr",
        #     value=current_lr,
        #     iteration=epoch
        # )

        # Optional: Log confusion matrix or other plots
        # import numpy as np
        # confusion_matrix = np.random.randint(0, 100, size=(10, 10))
        # logger.report_confusion_matrix(
        #     title="Confusion Matrix",
        #     series="validation",
        #     matrix=confusion_matrix,
        #     iteration=epoch
        # )

    print("\nTraining completed!")

    # Save and upload model (PyTorch)
    # TODO: Uncomment when you have a trained model
    import torch
    model_path = "model.pth"
    torch.save(model.state_dict(), model_path)
    task.upload_artifact("model_weights", artifact_object=model_path)
    #
    # # Or save entire model:
    # torch.save(model, "full_model.pth")
    # task.upload_artifact("full_model", artifact_object="full_model.pth")

    # Upload additional artifacts (uncomment as needed)
    # task.upload_artifact("training_history", artifact_object=history)
    # task.upload_artifact("preprocessor", artifact_object=preprocessor)

    # If using TensorFlow/Keras instead:
    # model_path = "saved_model"
    # model.save(model_path)
    # task.upload_artifact("saved_model", artifact_object=model_path)

    # If using scikit-learn instead:
    # import joblib
    # model_path = "model.pkl"
    # joblib.dump(model, model_path)
    # task.upload_artifact("model", artifact_object=model_path)

    # Save final metrics as summary
    task.set_parameter("final_metrics/train_loss", train_loss)
    task.set_parameter("final_metrics/val_loss", val_loss)
    task.set_parameter("final_metrics/train_accuracy", train_accuracy)
    task.set_parameter("final_metrics/val_accuracy", val_accuracy)

    print(f"\nTask ID: {task.id}")
    print(f"View results at: {task.get_output_log_web_page()}")


if __name__ == "__main__":
    main()
