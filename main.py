import random
from typing import Literal
import torch
from sklearn.datasets import make_classification, load_iris, load_wine, load_digits, fetch_covtype, load_linnerud
from torchvision import datasets, transforms
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import src.LPSL as lpsl
import matplotlib.pyplot as plt
import numpy as np
import math

device = torch.device(
    "cuda" if torch.cuda.is_available() else
    "mps" if torch.mps.is_available() else
    "cpu"
)

print(f"Using device {device}")

def profile_model(
    name,
    bp_model,
    bp_lr,
    lpsl,
    num_training_batches,
    batch_size,
    x,
    y,
    scale,
    seq_pooling_method: None | Literal["mean"] | Literal["last"] = None,
    runs=5
):
    bp_model.to(device)
    lpsl.to(device)

    batch_number_samples = list(range(0, num_training_batches, math.ceil(num_training_batches / 100)))
    bp_accuracy_runs = np.zeros((runs, len(batch_number_samples)))
    lpsl_accuracy_runs = np.zeros((runs, len(batch_number_samples)))

    for run in range(runs):
        print(f"\n--- Run {run + 1}/{runs} ---")

        bp_model.apply(lambda m: m.reset_parameters() if hasattr(m, 'reset_parameters') else None)
        lpsl.reset()

        bp_optimizer = torch.optim.Adam(bp_model.parameters(), lr=bp_lr)

        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

        if scale:
            scaler = StandardScaler()
            x_train = torch.tensor(scaler.fit_transform(x_train), dtype=torch.float32)
            x_test  = torch.tensor(scaler.transform(x_test),      dtype=torch.float32)
            y_train = torch.tensor(y_train, dtype=torch.long)
            y_test  = torch.tensor(y_test,  dtype=torch.long)

        x_test = x_test.to(device)
        y_test = y_test.to(device)

        sample_idx = 0
        for i in range(num_training_batches):
            batch_indices = torch.randperm(len(x_train))[:batch_size]
            batch_x = x_train[batch_indices].to(device)
            batch_y = y_train[batch_indices].to(device)

            bp_model.train()
            bp_optimizer.zero_grad()

            predicted_y = bp_model(batch_x)

            if seq_pooling_method == "last": predicted_y = predicted_y[:, -1, :]
            elif seq_pooling_method == "mean": predicted_y = predicted_y.mean(axis=1)

            torch.nn.functional.cross_entropy(predicted_y, batch_y).backward()
            bp_optimizer.step()

            lpsl.backward(batch_x, batch_y)

            if i % math.ceil(num_training_batches / 100) == 0:
                print(f"  Batch {i}/{num_training_batches}")

                bp_model.eval()
                with torch.no_grad():
                    bp_predicted_y = bp_model(x_test)

                    if seq_pooling_method == "last": bp_predicted_y = bp_predicted_y[:, -1, :]
                    elif seq_pooling_method == "mean": bp_predicted_y = bp_predicted_y.mean(axis=1)

                    bp_acc = (bp_predicted_y.argmax(dim=1) == y_test).float().mean().item()

                lpsl_acc = (lpsl.forward(x_test) == y_test).float().mean().item()

                bp_accuracy_runs[run, sample_idx]  = bp_acc
                lpsl_accuracy_runs[run, sample_idx] = lpsl_acc
                sample_idx += 1

    bp_mean  = bp_accuracy_runs.mean(axis=0)
    bp_std   = bp_accuracy_runs.std(axis=0)
    lpsl_mean = lpsl_accuracy_runs.mean(axis=0)
    lpsl_std  = lpsl_accuracy_runs.std(axis=0)

    plt.clf()
    plt.title(name, pad=20)
    plt.xlabel("Training Iterations")
    plt.ylabel("Accuracy")
    plt.ylim(0, 1.1)

    plt.plot(batch_number_samples, bp_mean,  label="Backpropagation")
    plt.fill_between(batch_number_samples, bp_mean - bp_std, bp_mean + bp_std, alpha=0.2)

    plt.plot(batch_number_samples, lpsl_mean, label="Local Prediction Segment Learning")
    plt.fill_between(batch_number_samples, lpsl_mean - lpsl_std, lpsl_mean + lpsl_std, alpha=0.2)

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
        lpsl.LocalPredictionSegmentLearning(
            torch.nn.Sequential(
                torch.nn.Linear(n_features, hidden_width),
                torch.nn.ReLU(),
                lpsl.PredictionLayer(hidden_width, n_classes),
                torch.nn.Linear(hidden_width, hidden_width),
                torch.nn.ReLU(),
                lpsl.PredictionLayer(hidden_width, n_classes),
                torch.nn.Linear(hidden_width, hidden_width),
                torch.nn.ReLU(),
                lpsl.PredictionLayer(hidden_width, n_classes)
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
            torch.nn.MaxPool2d(2),

            # Block 2: 32x14x14 -> 64x7x7
            torch.nn.Conv2d(32, 64, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(64, 64, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),

            torch.nn.Flatten(),
            torch.nn.Linear(64 * 7 * 7, 1024),
            torch.nn.ReLU(),

            torch.nn.Linear(1024, 128),
            torch.nn.ReLU(),

            torch.nn.Linear(128, 10)
        ),
        0.003,
        lpsl.LocalPredictionSegmentLearning(
            torch.nn.Sequential(
                # Block 1: 1x28x28 -> 32x14x14
                torch.nn.Conv2d(1, 32, kernel_size=3, padding=1),
                torch.nn.ReLU(),
                torch.nn.Conv2d(32, 32, kernel_size=3, padding=1),
                torch.nn.ReLU(),
                torch.nn.MaxPool2d(2),
                lpsl.PredictionLayer(32 * 14 * 14, 10),

                # Block 2: 32x14x14 -> 64x7x7
                torch.nn.Conv2d(32, 64, kernel_size=3, padding=1),
                torch.nn.ReLU(),
                torch.nn.Conv2d(64, 64, kernel_size=3, padding=1),
                torch.nn.ReLU(),
                torch.nn.MaxPool2d(2),
                lpsl.PredictionLayer(64 * 7 * 7, 10),

                torch.nn.Flatten(),
                torch.nn.Linear(64 * 7 * 7, 1024),
                torch.nn.ReLU(),
                lpsl.PredictionLayer(1024, 10),

                torch.nn.Linear(1024, 128),
                torch.nn.ReLU(),
                lpsl.PredictionLayer(128, 10)
            ),
            torch.optim.Adam,
            { "lr": 0.001 }
        ),
        500,
        8000, # int(0.8 * 70000),
        x,
        y,
        False,
        runs=10
    )
    
def load_synthetic_modular_addition(
    max_samples: int = 100000,
    p: int = 113
):
    chars = list("0123456789+= ")
    stoi = {c: i for i, c in enumerate(chars)}
    pad_idx = stoi[' ']

    def make_sample():
        terms = [ random.randint(0, p) for _ in range(2) ]
        expr = "+".join(str(t) for t in terms) + "="
        target = sum(terms) % p
        return expr, target

    samples = [ make_sample() for _ in range(max_samples) ]
    max_len = max(len(expr) for expr, _ in samples)

    def encode(expr):
        tokens = [stoi[c] for c in expr]
        tokens += [pad_idx] * (max_len - len(tokens))
        return torch.tensor(tokens, dtype=torch.long)

    x = torch.stack([ encode(expr) for expr, _ in samples ])
    y = torch.tensor([ target for _, target in samples ], dtype=torch.long)

    return x, y
    
def profile_modular_arithmetic_transformer():
    p = 31

    x, y = load_synthetic_modular_addition(max_samples=100000, p=p)

    d_model = 128
    n_heads = 4
    d_ff = 256

    class EmbeddingWithPosition(torch.nn.Module):
        def __init__(self, vocab_size, d_model, max_len):
            super().__init__()

            self.tok = torch.nn.Embedding(vocab_size, d_model)
            self.pos = torch.nn.Embedding(max_len, d_model)

        def forward(self, x):
            positions = torch.arange(x.shape[1], device=x.device).unsqueeze(0)

            return self.tok(x) + self.pos(positions)
        
    class LastToken(torch.nn.Module):
        def forward(self, x):
            return x[:, -1, :]
        
    class MeanToken(torch.nn.Module):
        def forward(self, x):
            return x.mean(dim=1)

    profile_model(
        "Modular Addition",
        torch.nn.Sequential(
            EmbeddingWithPosition(p, d_model, max_len=x.shape[1]),
            torch.nn.TransformerEncoderLayer(d_model, n_heads, d_ff, batch_first=True),
            torch.nn.TransformerEncoderLayer(d_model, n_heads, d_ff, batch_first=True),

            torch.nn.TransformerEncoderLayer(d_model, n_heads, d_ff, batch_first=True),
            
            MeanToken(),
            torch.nn.Linear(d_model, d_model // 2),
            torch.nn.ReLU(),

            torch.nn.Linear(d_model // 2, p)
        ),
        0.0005,
        lpsl.LocalPredictionSegmentLearning(
            torch.nn.Sequential(
                EmbeddingWithPosition(p, d_model, max_len=x.shape[1]),
                torch.nn.TransformerEncoderLayer(d_model, n_heads, d_ff, batch_first=True),
                torch.nn.TransformerEncoderLayer(d_model, n_heads, d_ff, batch_first=True),
                lpsl.PredictionLayer(d_model, p, seq_pooling_method="mean"),

                torch.nn.TransformerEncoderLayer(d_model, n_heads, d_ff, batch_first=True),
                lpsl.PredictionLayer(d_model, p, seq_pooling_method="mean"),
                
                MeanToken(),
                torch.nn.Linear(d_model, d_model // 2),
                torch.nn.ReLU(),
                lpsl.PredictionLayer(d_model // 2, p)
            ),
            torch.optim.Adam,
            { "lr": 0.0005 }
        ),
        500,
        10000,
        x,
        y,
        False,
        runs=10
    )

if __name__ == "__main__":
    # profile_synthetically_generated_clusters()

    # profile_mnist_digit_cnn()

    profile_modular_arithmetic_transformer()