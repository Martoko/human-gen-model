import matplotlib.pyplot as plt


def plot(batch_size, model_type, hidden_dimensions):
    losses_save_path = f"data/{model_type}_{'-'.join([str(x) for x in hidden_dimensions])}_batch-{batch_size}_losses.csv"
    train_losses = []
    test_losses = []
    with open(losses_save_path, "r") as file:
        for line in file.readlines()[1:]:
            train_losses += [[float(string) for string in line.split(",")][:3]]
            test_losses += [[float(string) for string in line.split(",")][3:]]

    # plt.plot([x for x, _, _ in train_losses], label="train_total_loss")
    # plt.plot([x for x, _, _ in test_losses], label="test_total_loss")
    plt.plot([x for _, x, _ in train_losses], label="Train MSE loss")
    plt.plot([x for _, x, _ in test_losses], label="Test MSE loss")
    if model_type == "vae":
        plt.plot([x for _, _, x in train_losses], label="Train KLD loss", linestyle="dashed")
        plt.plot([x for _, _, x in test_losses], label="Test KLD loss", linestyle="dashed")
    if model_type == "simple":
        name = "Autoencoder"
    if model_type == "vae":
        name = "Variational autoencoder"
    plt.title(f"{name}, layers {hidden_dimensions}, batch_size {batch_size}")
    plt.xlabel("epochs")
    plt.legend(loc='upper left')
    plt.ylim(0, 37)
    plt.savefig(f"data/{model_type}_{'-'.join([str(x) for x in hidden_dimensions])}_batch-{batch_size}_losses.png")
    plt.close()


model_type = "simple"
hidden_dimensions = [1024, 64]
batch_size = 128
plot(batch_size, model_type, hidden_dimensions)

model_type = "simple"
hidden_dimensions = [256, 256, 256, 256, 4]
batch_size = 8
plot(batch_size, model_type, hidden_dimensions)

model_type = "vae"
hidden_dimensions = [256, 256, 256, 256, 8]
batch_size = 8
plot(batch_size, model_type, hidden_dimensions)
