import torch
import torchvision
import torchvision.transforms as transforms

# Transform: convert image to tensor & normalize
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Training dataset & loader
train_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)

print("Training samples:", len(train_dataset))
print("Testing samples:", len(test_dataset))

import matplotlib.pyplot as plt

# Show 25 sample images in a 5x5 grid
fig, axes = plt.subplots(5, 5, figsize=(6,6))
for i, ax in enumerate(axes.flat):
    image, label = train_dataset[i]
    ax.imshow(image.squeeze(), cmap="gray", interpolation="nearest")  # keep crisp
    ax.set_title(str(label))
    ax.axis("off")

plt.tight_layout()
plt.show()

