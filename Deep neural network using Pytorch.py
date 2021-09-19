import torch
import torchvision
from torchvision import transforms, datasets
import torch.optim as optim
import matplotlib.pyplot as plt
from collections import Counter
import torch.nn as nn
import torch.nn.functional as f

# Step 1 Loading the data

# Only in this specific instance are we using an available MNIST dataset within the torch library

# The use of "" has been done to indicate that the training data is being saved at the file path itself.

# If you want to save it at a specific file path you need to state the location within the ''

train = datasets.MNIST("", train=True, download=True, transform=transforms.Compose([transforms.ToTensor()]))

test = datasets.MNIST("", train=False, download=True, transform=transforms.Compose([transforms.ToTensor()]))

# Step 2 Accessing the loaded data

trainset = torch.utils.data.DataLoader(train, batch_size=10, shuffle=True)

testset = torch.utils.data.DataLoader(test, batch_size=10, shuffle=True)

total = 0

# The next section just checks if the data is balanced (the features or outputs are uniformly distributed)

counter_dict = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0}

for data in trainset:
    xs, ys = data
    for i in ys:
        counter_dict[int(i)] += 1
        total += 1

print(counter_dict)
print(total)

for i in counter_dict:
    print('{} is {}%'.format(i, counter_dict[i] / total * 100))

# Note that you could have used the Counter() from Collections to

# Data seems to be uniform

# Building the actual neural network

# The input is 784 because images are of the size 28x28 which upon flattening is 784

# The final layer should predict between 0-9 so 19 images so 10 outputs can be made


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 64)
        self.fc4 = nn.Linear(64, 10)

    def forward(self, x):
        x = f.relu(self.fc1(x))
        x = f.relu(self.fc2(x))
        x = f.relu(self.fc3(x))
        x = self.fc4(x)
        return f.log_softmax(x, dim=1)


net = Net()

optimizer = optim.Adam(net.parameters(), lr=0.01)

epochs = 10

for i in range(epochs):
    for data in trainset:
        X, y = data
        net.zero_grad()
        output = net(X.view(-1, 28*28))
        loss = f.nll_loss(output, y)
        loss.backward()
        optimizer.step()
    print(loss)

correct= 0
total = 0

with torch.no_grad():
    for data in trainset:
        X, y = data
        output = net(X.view(-1,28*28))
        for idx, i in enumerate(output):
            if torch.argmax(i) == y[idx]:
                correct += 1
            total += 1

print("Accuracy:", round((correct/total)*100, 3))

# Checking the accuracy of the model

plt.imshow(X[0].view(28,28))
plt.show()

print(torch.argmax(net(X[0].view(-1, 784))[0]))


