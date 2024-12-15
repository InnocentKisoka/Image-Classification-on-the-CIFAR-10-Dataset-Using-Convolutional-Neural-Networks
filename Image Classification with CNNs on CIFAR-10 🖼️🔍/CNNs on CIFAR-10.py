"INNOCENT KISOKA"

import torch

import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.optim import AdamW
import matplotlib.pyplot as plt
from torchvision import datasets, transforms, models

from math import floor
from PIL import Image, __version__

def out_dimensions(conv_layer, h_in, w_in):
    '''
    This function computes the output dimension of each convolutional layer in the most general way.
    '''
    h_out = floor((h_in + 2 * conv_layer.padding[0] - conv_layer.dilation[0] * (conv_layer.kernel_size[0] - 1) - 1) /
                  conv_layer.stride[0] + 1)
    w_out = floor((w_in + 2 * conv_layer.padding[1] - conv_layer.dilation[1] * (conv_layer.kernel_size[1] - 1) - 1) /
                  conv_layer.stride[1] + 1)
    return h_out, w_out

# CNN Model Definition
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        # Setting in_channels to 3 for RGB images
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=(3, 3)) 
        h_out, w_out = out_dimensions(self.conv1, 32, 32)  # Using 32x32 for CIFAR-10 images
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3))
        h_out, w_out = out_dimensions(self.conv2, h_out, w_out)
        
        # Pooling Layer
        self.pool1 = nn.MaxPool2d(2, 2)
        h_out, w_out = int(h_out / 2), int(w_out / 2)  # Divide dimensions after pooling
        
        # Fully Connected Layer
        self.fc1 = nn.Linear(32 * h_out * w_out, 10)  # 32 corresponds to the output channels of conv2
        self.dimensions_final = (32, h_out, w_out)

    def forward(self, x):
        # Apply each layer in sequence with ReLU activations
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool1(x)
        
        # Flatten for fully connected layer
        n_channels, h, w = self.dimensions_final
        x = x.view(-1, n_channels * h * w)
        x = self.fc1(x)
        return x

if __name__ == "__main__":
    print("Hello World!")

    # Set the seed for reproducibility
    manual_seed = 42
    torch.manual_seed(manual_seed)
    torch.cuda.manual_seed(manual_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    '''
    Q1 - Code
    '''
    # Cell 2: Generate a random number
    print(torch.randint(1, 10, (1, 1)))

    '''
    Q2 - Code
    '''
    transform = transforms.ToTensor()  # Move this above the dataset loading
    train_set = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    test_set = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

    # Display images per class
    classes = train_set.classes
    def show_images_per_class(dataset, classes, filename):
        plt.figure(figsize=(10, 5))
        for i, class_label in enumerate(classes):
            idx = [j for j, label in enumerate(dataset.targets) if label == i][0]
            image, _ = dataset[idx]
            plt.subplot(2, 5, i + 1)
            plt.imshow(image.permute(1, 2, 0))
            plt.title(class_label)
            plt.axis('off')
        plt.savefig(filename)  # Save the figure to the specified file
        plt.show()

    show_images_per_class(train_set, classes, "images_per_class.png")

    # Plot class distribution for training and test sets
    def plot_distribution(dataset, title, filename):
        labels = dataset.targets
        plt.hist(labels, bins=len(classes), edgecolor='black', rwidth=0.8)
        plt.xticks(range(len(classes)), classes, rotation=45)
        plt.title(title)
        plt.xlabel('Classes')
        plt.ylabel('Frequency')
        plt.savefig(filename)
        plt.show()

    plot_distribution(train_set, "Training Set Distribution", "training_set_distribution.png")
    plot_distribution(test_set, "Test Set Distribution", "test_set_distribution.png")

    '''
    ........
    '''
    '''
    Q3 - Code
    '''
   

# Define the transformation
    transform = transforms.ToTensor()

# Apply the transformation when loading the dataset
    dataset_train = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    print(type(train_set[0]))  # Expected output: tuple
    print(type(train_set[0][0]))  # Expected output: torch.Tensor
    print(train_set[0][0].dtype)  # Expected output: torch.float32
    print(train_set[0][0].shape)  # Expected output: torch.Size([3, 32, 32])
    '''
    Q4 - Code
    
    '''
    # Normalization values for CIFAR-10
    mean = (0.4914, 0.4822, 0.4465)
    std = (0.247, 0.243, 0.261)

  # Applying normalization
    transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])

  # Reload datasets with normalization
    train_set = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    test_set = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    test_set = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    '''
    Q5 - Code
    
    '''
    from torch.utils.data import random_split, DataLoader

# Split the test set into validation and test sets
    val_size = 2000
    test_size = len(test_set) - val_size
    validation_set, new_test_set = random_split(test_set, [val_size, test_size])

# Creating DataLoaders
    train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
    val_loader = DataLoader(validation_set, batch_size=64, shuffle=False)
    test_loader = DataLoader(new_test_set, batch_size=64, shuffle=False)
    
    '''
    Q6 - Code
    
    '''
    
    
batch_size = 32
learning_rate = 0.03
epochs = 4
n = 100  # Steps to print log

class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        # Layer definitions
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=0, stride=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=0, stride=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=0, stride=1)
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=0, stride=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Fully connected layers
        self.fc1 = nn.Linear(128 * 5 * 5, 256)  # 5x5 after the pooling layers
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 10)  # 10 classes for CIFAR-10

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool1(x)
        
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.pool2(x)
        
        x = x.view(-1, 128 * 5 * 5)  # Flatten
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        
        return x


model = ConvNet()
optimizer = optim.SGD(model.parameters(), lr=learning_rate)



    
'''
Q7 - Code
    
'''


# Hyperparameters
batch_size = 32
learning_rate = 0.03
epochs = 4
n = 100  # Steps to print log

# Data loaders, assuming dataset_train and dataset_valid are defined
train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
val_loader = DataLoader(validation_set, batch_size=64, shuffle=False)
test_set = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

# Model, loss, optimizer

criterion = nn.CrossEntropyLoss()



# Training Loop
train_losses, valid_losses = [], []
train_accuracies, valid_accuracies = [], []

for epoch in range(epochs):
    model.train()
    running_loss, correct, total = 0.0, 0, 0

    for i, (images, labels) in enumerate(train_loader):
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # Accumulate loss and accuracy
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        if (i + 1) % n == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}')
    
    train_loss = running_loss / len(train_loader)
    train_acc = correct / total * 100
    train_losses.append(train_loss)
    train_accuracies.append(train_acc)

    # Validation phase
    model.eval()
    val_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for images, labels in val_loader:
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    valid_loss = val_loss / len(val_loader)
    valid_acc = correct / total * 100
    valid_losses.append(valid_loss)
    valid_accuracies.append(valid_acc)
    
    

    print(f'Epoch [{epoch+1}/{epochs}] Training Loss: {train_loss:.4f}, Validation Loss: {valid_loss:.4f}')
    print(f'Training Accuracy: {train_acc:.2f}%, Validation Accuracy: {valid_acc:.2f}%')
    
    
    
with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
test_accuracy = correct / total * 100
print(f'Test Accuracy: {test_accuracy:.2f}%')
    

'''
Q8 - Code
    
'''
import matplotlib.pyplot as plt

plt.plot(train_losses, label='Training Loss')
plt.plot(valid_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.savefig("Train and Validation loss")
plt.show()    
    
    
'''
Q9 - Code
    
'''
class ImprovedConvNet(nn.Module):
    def __init__(self):
        super(ImprovedConvNet, self).__init__()
        # First Convolutional Block
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(2, 2)
        
        # Second Convolutional Block (Deeper Layers)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.pool2 = nn.MaxPool2d(2, 2)
        
        # Third Convolutional Block
        self.conv5 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm2d(256)
        self.conv6 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.bn6 = nn.BatchNorm2d(512)
        self.pool3 = nn.MaxPool2d(2, 2)
        
        # Fully Connected Layers
        self.fc1 = nn.Linear(512 * 4 * 4, 512)
        self.drop1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, 256)
        self.drop2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(256, 128)
        self.drop3 = nn.Dropout(0.5)
        self.fc4 = nn.Linear(128, 10)

    def forward(self, x):
        # First Convolutional Block
        x = self.pool1(F.gelu(self.bn2(self.conv2(F.gelu(self.bn1(self.conv1(x)))))))
        
        # Second Convolutional Block
        x = self.pool2(F.gelu(self.bn4(self.conv4(F.gelu(self.bn3(self.conv3(x)))))))
        
        # Third Convolutional Block
        x = self.pool3(F.gelu(self.bn6(self.conv6(F.gelu(self.bn5(self.conv5(x)))))))
        
        # Flatten the tensor for the fully connected layer
        x = x.view(-1, 512 * 4 * 4)
        
        # Fully Connected Layers with Dropout
        x = self.drop1(F.gelu(self.fc1(x)))
        x = self.drop2(F.gelu(self.fc2(x)))
        x = self.drop3(F.gelu(self.fc3(x)))
        x = self.fc4(x)
        return x

# Define the optimizer with weight decay for L2 regularization
model = ImprovedConvNet()
optimizer = AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)

# Hyperparameters
batch_size = 32
learning_rate = 0.001
epochs = 6
n = 100  # Steps to print log

# Data loaders, assuming dataset_train and dataset_valid are defined
train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
val_loader = DataLoader(validation_set, batch_size=64, shuffle=False)
test_set = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

# Model, loss, optimizer

criterion = nn.CrossEntropyLoss()



# Training Loop
train_losses, valid_losses = [], []
train_accuracies, valid_accuracies = [], []

for epoch in range(epochs):
    model.train()
    running_loss, correct, total = 0.0, 0, 0

    for i, (images, labels) in enumerate(train_loader):
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # Accumulate loss and accuracy
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        if (i + 1) % n == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}')
    
    train_loss = running_loss / len(train_loader)
    train_acc = correct / total * 100
    train_losses.append(train_loss)
    train_accuracies.append(train_acc)

    # Validation phase
    model.eval()
    val_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for images, labels in val_loader:
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    valid_loss = val_loss / len(val_loader)
    valid_acc = correct / total * 100
    valid_losses.append(valid_loss)
    valid_accuracies.append(valid_acc)
    
    

    print(f'Epoch [{epoch+1}/{epochs}] Training Loss: {train_loss:.4f}, Validation Loss: {valid_loss:.4f}')
    print(f'Training Accuracy: {train_acc:.2f}%, Validation Accuracy: {valid_acc:.2f}%')
    
    
    
with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
test_accuracy = correct / total * 100
print(f'Test Accuracy: {test_accuracy:.2f}%')
    

'''
Q8 - Code
    
'''
import matplotlib.pyplot as plt

plt.plot(train_losses, label='Training Loss')
plt.plot(valid_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.savefig("Train and Validation loss Improved")
plt.show()    
    

'''
   
   
 
Q10 - Code
    '''
for seed in range(5, 10):
    torch.manual_seed(seed)
    print("Seed equal to", torch.initial_seed())
    
    # Initialize model, loss function, and optimizer
    model = ConvNet().to(torch.device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Training loop
    for epoch in range(epochs):
        model.train()
        for images, labels in train_loader:
            images, labels = images.to(torch.device), labels.to(torch.device)
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}")

    # Evaluate the model
    accuracy = model.eval(model)
    print(f"Test Accuracy with seed {seed}: {accuracy:.2f}%\n")
    

    
   


