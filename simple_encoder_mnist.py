import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import matplotlib.pyplot as plt
from torchviz import make_dot

from simple_encoder import SimpleAutoEncoder

device = "cpu"  # torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])

print("Setting up train dataset...")
train_dataset = torchvision.datasets.FashionMNIST(
    root="~/torch_datasets", train=True, transform=transform, download=True
)

print("Setting up test dataset...")
test_dataset = torchvision.datasets.FashionMNIST(
    root="~/torch_datasets", train=False, transform=transform, download=True
)

print("Setting up train loader...")
train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=8, shuffle=True, num_workers=4, pin_memory=True
)

print("Setting up test loader...")
test_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=32, num_workers=4
)

images = [[] for i in range(10)]


def visualize(epoch_name):
    global images
    with torch.no_grad():
        truths = [test_dataset[i][0][0] for i in range(len(images))]

        for i, inp in enumerate(truths):
            flat_input = inp.view(-1, 784).to(device)
            flat_output = model(flat_input).cpu()
            output = flat_output.view(28, 28)
            images[i] += [[output, str(epoch_name)]]

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


print("Setting up model...")
model = SimpleAutoEncoder(input_size=28 * 28, hidden_dimensions=[512, 256, 128]).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.MSELoss()

# TODO: save best model

print("Starting training...")
saved_model_visualization = False
epochs = 10
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

        if not saved_model_visualization:
            make_dot(outputs, params=dict(model.named_parameters()))
            saved_model_visualization = True

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

    # TODO: test loss/train loss graph
    # display the epoch training loss
    print("epoch : {}/{}, loss = {:.6f}".format(epoch + 1, epochs, loss))
    if epoch % (epochs / 10) == 0:
        visualize(epoch + 1)
if(epochs > 10 and epochs % 10 > 0):
    visualize(epochs)
