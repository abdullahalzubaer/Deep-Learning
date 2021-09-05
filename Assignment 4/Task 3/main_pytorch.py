import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

# Load the dataset
training_data = datasets.CIFAR10(
    root="data", train=True, download=True, transform=ToTensor()
)

test_data = datasets.CIFAR10(
    root="data", train=False, download=True, transform=ToTensor()
)


# DataLoaders is an iterable.
train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=64, shuffle=True)

device = "cuda" if torch.cuda.is_available() else "cpu"

# Create the model
class Net(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=128, kernel_size=2, padding='same')
        self.conv2 = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=2, padding='same')
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=2, padding='same')

        self.fc1 = nn.Linear(
            in_features=32*32*32, out_features=1024
        )  # in_features = out_channels * img_height * img_width (since same padding, img shape remain same)
        self.fc2 = nn.Linear(in_features=1024, out_features=10)

    def forward(self, x):

        x = nn.functional.relu(self.conv1(x))
        x = nn.functional.relu(self.conv2(x))
        x = nn.functional.relu(self.conv3(x))
        # input = x.view(64, -1)
        # print(input.shape)

        x = torch.flatten(x, 1)  # Flatten all the dimensions except batch dimensions
        x = nn.functional.relu(self.fc1(x))
        x = self.fc2(x)
        return x


net = Net().to(device)
# print(net)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(params=net.parameters(), lr=0.001)

loss_list = list()


def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # computing prediction error
        pred = model(X)
        # print(type(pred))
        loss = loss_fn(pred, y)

        # Backpropagation

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            loss_list.append(loss)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


epochs = 1
for t in range(epochs):
    print(f"Epoch {t+1}\n----------------------------------")
    train(train_dataloader, net, loss_fn, optimizer)


# print(len(loss_list))

plt.style.use("ggplot")
plt.plot(range(len(loss_list)), loss_list, "r")
plt.xlabel("#Batch")
plt.ylabel("BatchLoss")
plt.title("Loss vs Batch")
plt.show()
