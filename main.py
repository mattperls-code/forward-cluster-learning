import torch
from sklearn.datasets import make_classification, load_iris, load_wine, load_digits, fetch_covtype, load_linnerud
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import src.forward_cluster_learning as fcl
import matplotlib.pyplot as plt
import numpy as np

def profile_model(name, bp_model, bp_lr, fcl, num_training_batches, batch_size, x, y, runs=5):
    batch_number_samples = list(range(0, num_training_batches, num_training_batches // 100))
    bp_accuracy_runs = np.zeros((runs, len(batch_number_samples)))
    fcl_accuracy_runs = np.zeros((runs, len(batch_number_samples)))

    for run in range(runs):
        print(f"\n--- Run {run + 1}/{runs} ---")

        bp_model.apply(lambda m: m.reset_parameters() if hasattr(m, 'reset_parameters') else None)
        fcl.reset()

        bp_optimizer = torch.optim.Adam(bp_model.parameters(), lr=bp_lr)

        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
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

            if i % (num_training_batches // 100) == 0:
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
    plt.title(f"Average Accuracy on {name} ({runs} runs)")
    plt.xlabel("Training Iterations")
    plt.ylabel("Accuracy")
    plt.ylim(0, 1)

    plt.plot(batch_number_samples, bp_mean,  label="Back Propagation")
    plt.fill_between(batch_number_samples, bp_mean - bp_std, bp_mean + bp_std, alpha=0.2)

    plt.plot(batch_number_samples, fcl_mean, label="Forward Cluster Learning")
    plt.fill_between(batch_number_samples, fcl_mean - fcl_std, fcl_mean + fcl_std, alpha=0.2)

    plt.legend(loc="lower right")
    plt.savefig(f"Accuracy on {name}")

def make_spiral(n_samples=10000, noise=0.05):
    n = n_samples // 2
    theta = np.linspace(0, 4 * np.pi, n)
    
    r = np.linspace(0.1, 1, n)
    X1 = np.column_stack([r * np.cos(theta), r * np.sin(theta)])
    X2 = np.column_stack([r * np.cos(theta + np.pi), r * np.sin(theta + np.pi)])
    
    X1 += np.random.randn(*X1.shape) * noise
    X2 += np.random.randn(*X2.shape) * noise
    
    X = np.vstack([X1, X2]).astype(np.float32)
    y = np.array([0] * n + [1] * n, dtype=np.int64)
    
    return torch.tensor(X), torch.tensor(y)

if __name__ == "__main__":
    n_features = 24
    hidden_width = 64
    n_classes = 4
    x, y = make_classification(
        n_samples=5000,
        n_features=n_features,
        n_classes=n_classes,
        n_informative=int(0.8 * n_features),
        n_redundant=2,
        n_clusters_per_class=3,
        class_sep=0.7,
        flip_y=0.02,
        random_state=0
    )

    # n_features = 64
    # hidden_width=128
    # n_classes = 10
    # digits = load_digits()
    # x = torch.tensor(digits.data, dtype=torch.float32)
    # y = torch.tensor(digits.target, dtype=torch.long)

    # n_features = 2
    # hidden_width=32
    # n_classes = 2
    # x, y = make_spiral()

    profile_model(
        "Synthetically Generated Clusters",
        torch.nn.Sequential(
            torch.nn.Linear(n_features, hidden_width),
            # torch.nn.ReLU(),
            # torch.nn.Linear(hidden_width, hidden_width),
            # torch.nn.ReLU(),
            # torch.nn.Linear(hidden_width, hidden_width),
            # torch.nn.ReLU(),
            # torch.nn.Linear(hidden_width, hidden_width),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_width, n_classes)
        ),
        0.014,
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
                fcl.PredictionLayer(hidden_width, n_classes),
                torch.nn.Linear(hidden_width, hidden_width),
                torch.nn.ReLU(),
                fcl.PredictionLayer(hidden_width, n_classes),
                torch.nn.Linear(hidden_width, hidden_width),
                torch.nn.ReLU(),
                fcl.PredictionLayer(hidden_width, n_classes)
            ),
            torch.optim.Adam,
            { "lr": 0.014 }
        ),
        500,
        1000,
        x,
        y,
        runs=5
    )