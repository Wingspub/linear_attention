from torchvision.datasets import MNIST

train_data = MNIST(root="./dataset/data", train=True, download=True)
valid_data = MNIST(root="./dataset/data", train=False, download=True)

print(train_data.train_data[0])