import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import matplotlib.pyplot as plt
import numpy as np

from simple_encoder import SimpleAutoEncoder

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])

train_dataset = torchvision.datasets.MNIST(
    root="~/torch_datasets", train=True, transform=transform, download=True
)

test_dataset = torchvision.datasets.MNIST(
    root="~/torch_datasets", train=False, transform=transform, download=True
)

train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=128, shuffle=True, num_workers=4, pin_memory=True
)

test_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=32, shuffle=False, num_workers=4
)

images = [[] for i in range(10)]


def visualize(epoch):
    global images
    with torch.no_grad():
        truths = [test_dataset[i][0][0] for i in range(len(images))]

        for i, inp in enumerate(truths):
            flat_input = inp.view(-1, 784).to(device)
            flat_output = model(flat_input).cpu()
            output = flat_output.view(28, 28)
            images[i] += [[output, str(epoch)]]

        num_row = len(images)
        num_col = len(images[0]) + 1
        fig, axes = plt.subplots(num_row, num_col, figsize=(1.5 * num_col, 2 * num_row))

        for i, image_row in enumerate(images):
            for j, image in enumerate(image_row):
                axes[i, j].imshow(image[0], cmap='gray')
                axes[i, j].set_title(image[1])
            axes[i, j + 1].imshow(truths[i], cmap='gray')
            axes[i, j + 1].set_title('truth')

        fig.tight_layout()
        fig.show()


model = SimpleAutoEncoder(input_size=28 * 28, hidden_size=1024, encoded_size=20).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-5)
criterion = nn.MSELoss()

epochs = 1000
for epoch in range(epochs):
    loss = 0
    for batch_features, _ in train_loader:
        # reshape mini-batch data to [N, 784] matrix
        # load it to the active device
        batch_features = batch_features.view(-1, 28 * 28).to(device)

        # reset the gradients back to zero
        # PyTorch accumulates gradients on subsequent backward passes
        optimizer.zero_grad()

        # compute reconstructions
        outputs = model(batch_features)

        # compute training reconstruction loss
        train_loss = criterion(outputs, batch_features)

        # compute accumulated gradients
        train_loss.backward()

        # perform parameter update based on current gradients
        optimizer.step()

        # add the mini-batch training loss to epoch loss
        loss += train_loss.item()

    # compute the epoch training loss
    loss = loss / len(train_loader)

    # display the epoch training loss
    print("epoch : {}/{}, loss = {:.6f}".format(epoch + 1, epochs, loss))
    if epoch % (epochs / 10) == 0:
        visualize(epoch + 1)
