import torch
from sklearn.datasets import make_classification, load_iris, load_wine, load_digits, fetch_covtype, load_linnerud
from torchvision import datasets, transforms
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import src.forward_cluster_learning as fcl
import matplotlib.pyplot as plt
import numpy as np
import math

def profile_model(name, bp_model, bp_lr, fcl, num_training_batches, batch_size, x, y, scale, runs=5):
    batch_number_samples = list(range(0, num_training_batches, math.ceil(num_training_batches / 100)))
    bp_accuracy_runs = np.zeros((runs, len(batch_number_samples)))
    fcl_accuracy_runs = np.zeros((runs, len(batch_number_samples)))

    for run in range(runs):
        print(f"\n--- Run {run + 1}/{runs} ---")

        bp_model.apply(lambda m: m.reset_parameters() if hasattr(m, 'reset_parameters') else None)
        fcl.reset()

        bp_optimizer = torch.optim.Adam(bp_model.parameters(), lr=bp_lr)

        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

        if scale:
            scaler = StandardScaler()
            x_train = torch.tensor(scaler.fit_transform(x_train), dtype=torch.float32)
            x_test  = torch.tensor(scaler.transform(x_test),      dtype=torch.float32)
            y_train = torch.tensor(y_train, dtype=torch.long)
            y_test  = torch.tensor(y_test,  dtype=torch.long)

        sample_idx = 0
        for i in range(num_training_batches):
            batch_indices = torch.randperm(len(x_train))[:batch_size]
            batch_x = x_train[batch_indices]
            batch_y = y_train[batch_indices]

            bp_model.train()
            bp_optimizer.zero_grad()
            torch.nn.functional.cross_entropy(bp_model(batch_x), batch_y).backward()
            bp_optimizer.step()

            fcl.backward(batch_x, batch_y)

            if i % math.ceil(num_training_batches / 100) == 0:
                print(f"  Batch {i}/{num_training_batches}")

                bp_model.eval()
                with torch.no_grad():
                    bp_acc = (bp_model(x_test).argmax(dim=1) == y_test).float().mean().item()

                fcl_acc = (fcl.forward(x_test) == y_test).float().mean().item()

                bp_accuracy_runs[run, sample_idx]  = bp_acc
                fcl_accuracy_runs[run, sample_idx] = fcl_acc
                sample_idx += 1

    bp_mean  = bp_accuracy_runs.mean(axis=0)
    bp_std   = bp_accuracy_runs.std(axis=0)
    fcl_mean = fcl_accuracy_runs.mean(axis=0)
    fcl_std  = fcl_accuracy_runs.std(axis=0)

    plt.clf()
    plt.title(name, pad=20)
    plt.xlabel("Training Iterations")
    plt.ylabel("Accuracy")
    plt.ylim(0, 1)

    plt.plot(batch_number_samples, bp_mean,  label="Back Propagation")
    plt.fill_between(batch_number_samples, bp_mean - bp_std, bp_mean + bp_std, alpha=0.2)

    plt.plot(batch_number_samples, fcl_mean, label="Forward Cluster Learning")
    plt.fill_between(batch_number_samples, fcl_mean - fcl_std, fcl_mean + fcl_std, alpha=0.2)

    plt.legend(loc="lower right")
    plt.savefig(name)

def profile_synthetically_generated_clusters():
    n_features = 64
    hidden_width = 32
    n_classes = 5

    x, y = make_classification(
        n_samples=4000,
        n_features=n_features,
        n_classes=n_classes,
        n_informative=int(0.7 * n_features),
        n_clusters_per_class=5,
        class_sep=2.5,
        flip_y=0.04,
        random_state=0
    )

    profile_model(
        "Difficult Synthetically Generated Clusters",
        torch.nn.Sequential(
            torch.nn.Linear(n_features, hidden_width),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_width, hidden_width),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_width, hidden_width),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_width, n_classes)
        ),
        0.016,
        fcl.ForwardClusterLearning(
            torch.nn.Sequential(
                torch.nn.Linear(n_features, hidden_width),
                torch.nn.ReLU(),
                fcl.PredictionLayer(hidden_width, n_classes),
                torch.nn.Linear(hidden_width, hidden_width),
                torch.nn.ReLU(),
                fcl.PredictionLayer(hidden_width, n_classes),
                torch.nn.Linear(hidden_width, hidden_width),
                torch.nn.ReLU(),
                fcl.PredictionLayer(hidden_width, n_classes)
            ),
            torch.optim.Adam,
            { "lr": 0.016 }
        ),
        200,
        int(0.8 * 4000),
        x,
        y,
        True,
        runs=10
    )

def load_mnist():
    dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transforms.ToTensor())
    
    x = dataset.data.unsqueeze(1).float() / 255.0  # (70000, 1, 28, 28)
    y = torch.tensor(dataset.targets.numpy(), dtype=torch.long)

    return x, y

def profile_mnist_digit_cnn():
    x, y = load_mnist()

    profile_model(
        "MNIST Digit",
        torch.nn.Sequential(
            # Block 1: 1x28x28 -> 32x14x14
            torch.nn.Conv2d(1, 32, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(32, 32, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),          # 32x14x14

            # Block 2: 32x14x14 -> 64x7x7
            torch.nn.Conv2d(32, 64, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(64, 64, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),          # 64x7x7

            torch.nn.Flatten(),             # 64*7*7 = 3136
            torch.nn.Linear(64 * 7 * 7, 1024),
            torch.nn.ReLU(),

            torch.nn.Linear(1024, 128),
            torch.nn.ReLU(),

            torch.nn.Linear(128, 10)
        ),
        0.003,
        fcl.ForwardClusterLearning(
            torch.nn.Sequential(
                # Block 1: 1x28x28 -> 32x14x14
                torch.nn.Conv2d(1, 32, kernel_size=3, padding=1),
                torch.nn.ReLU(),
                torch.nn.Conv2d(32, 32, kernel_size=3, padding=1),
                torch.nn.ReLU(),
                torch.nn.MaxPool2d(2),          # 32x14x14
                fcl.PredictionLayer(32*14*14, 10),

                # Block 2: 32x14x14 -> 64x7x7
                torch.nn.Conv2d(32, 64, kernel_size=3, padding=1),
                torch.nn.ReLU(),
                torch.nn.Conv2d(64, 64, kernel_size=3, padding=1),
                torch.nn.ReLU(),
                torch.nn.MaxPool2d(2),          # 64x7x7
                fcl.PredictionLayer(64*7*7, 10),

                torch.nn.Flatten(),             # 64*7*7 = 3136
                torch.nn.Linear(64 * 7 * 7, 1024),
                torch.nn.ReLU(),
                fcl.PredictionLayer(1024, 10),

                torch.nn.Linear(1024, 128),
                torch.nn.ReLU(),
                fcl.PredictionLayer(128, 10)
            ),
            torch.optim.Adam,
            { "lr": 0.003 }
        ),
        45,
        100, # int(0.8 * 70000),
        x,
        y,
        False,
        runs=1
    )

if __name__ == "__main__":
    # profile_synthetically_generated_clusters()

    profile_mnist_digit_cnn()