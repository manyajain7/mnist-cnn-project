import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

# -------------------
# 1. Dataset & Preprocessing
# -------------------
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)

# -------------------
# 2. Define CNN Model
# -------------------
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        
        # Dummy input to find flatten size
        self._to_linear = None
        self.convs = nn.Sequential(self.conv1, nn.ReLU(), self.conv2, nn.ReLU())
        self._get_flatten_size()
        
        self.fc1 = nn.Linear(self._to_linear, 128)
        self.fc2 = nn.Linear(128, 10)

    def _get_flatten_size(self):
        with torch.no_grad():
            x = torch.zeros(1, 1, 28, 28)  # dummy MNIST image
            x = self.convs(x)
            self._to_linear = x.numel()

    def forward(self, x):
        x = self.convs(x)
        x = torch.flatten(x, 1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

model = CNN()

# -------------------
# 3. Loss & Optimizer
# -------------------
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# -------------------
# 4. Training Loop
# -------------------
epochs = 3
for epoch in range(epochs):
    running_loss = 0.0
    for images, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f"Epoch {epoch+1}, Loss: {running_loss/len(train_loader):.4f}")

print("Training complete ✅")

# -------------------
# 5. Evaluation
# -------------------
correct = 0
total = 0
with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f"Accuracy on test set: {100 * correct / total:.2f}%")

# -------------------
# 6. Show Predictions
# -------------------
images, labels = next(iter(test_loader))
outputs = model(images)
_, predicted = torch.max(outputs, 1)

fig, axes = plt.subplots(3, 3, figsize=(6,6))
for i, ax in enumerate(axes.flat):
    ax.imshow(images[i].squeeze(), cmap="gray", interpolation="nearest")
    ax.set_title(f"True:{labels[i].item()} Pred:{predicted[i].item()}")
    ax.axis("off")

plt.tight_layout()
plt.show()

torch.save(model.state_dict(), "mnist_cnn.pth")
print("Model saved as mnist_cnn.pth ✅")

plt.tight_layout()
plt.savefig("predictions.png")
print("Predictions saved as predictions.png ✅")
# plt.show()   # you can comment this out if you don’t want the blocking window

plt.tight_layout()
plt.savefig("predictions.png")
print("Predictions saved as predictions.png ✅")
# plt.show()  # optional, comment this out to avoid freezing terminal

