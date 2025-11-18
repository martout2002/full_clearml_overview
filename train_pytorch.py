import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from clearml import Task
from sklearn.metrics import confusion_matrix
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms

# task = Task.init(
#     project_name="Full Overview",
#     task_name="model_training",
# )

task = Task.init_with_lineage(
    project_name="Full Overview",
    task_name="model_training",
)
# Connect hyperparameters as args
args = {
    'batch_size': 63,
    'learning_rate': 0.01,
    'epochs': 10,
    'hidden_size': 128,
    'input_size': 784,
    'output_size': 10,
    'optimizer': 'SGD',
    'normalize_mean': 0.5,
    'normalize_std': 0.5
}
args = task.connect(args)

# Writer will output to ./runs/ directory by default
writer = SummaryWriter()

# Define the transformation to apply to the data
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Download and load the training data
trainset = datasets.FashionMNIST('data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=args['batch_size'], shuffle=True)

# Define the neural network model
model = nn.Sequential(
    nn.Flatten(),
    nn.Linear(args['input_size'], args['hidden_size']),
    nn.ReLU(),
    nn.Linear(args['hidden_size'], args['output_size'])
)

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=args['learning_rate'])

# Train the model
for epoch in range(args['epochs']):
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # Get the inputs
        inputs, labels = data

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward + backward + optimize
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        # Tensorboard
        writer.add_scalar('Loss/train', loss, epoch * len(trainloader) + i)

        # Print statistics
        running_loss += loss.item()
        if i % 100 == 99:    # print every 100 mini-batches
            print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 100))
            running_loss = 0.0

print('Finished training')

# Download and load the test data
testset = datasets.FashionMNIST('data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False)

# Evaluate the model on the test data
y_true = []
y_pred = []
with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        y_true.extend(labels.tolist())
        y_pred.extend(predicted.tolist())

# Create and display the confusion matrix
classes = ('T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
           'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot')
confusion_mat = confusion_matrix(y_true, y_pred)
plt.imshow(confusion_mat, cmap=plt.cm.Blues)
plt.xticks(np.arange(len(classes)), classes, rotation=90)
plt.yticks(np.arange(len(classes)), classes)
plt.colorbar()
plt.xlabel('Predicted label')
plt.ylabel('True label')
plt.title('Confusion matrix')
# plt.show()

# Save and log artifacts
plt.savefig('confusion_matrix.png')
task.upload_artifact('confusion_matrix', artifact_object='confusion_matrix.png')

# Save and log model
torch.save(model.state_dict(), 'fashion_mnist_model.pth')
task.upload_artifact('model_weights', artifact_object='fashion_mnist_model.pth')

# Log test accuracy
accuracy = sum([1 for true, pred in zip(y_true, y_pred) if true == pred]) / len(y_true)
task.logger.report_single_value('Test Accuracy', accuracy)

# Close task properly
task.close()
print('Task closed and artifacts uploaded')