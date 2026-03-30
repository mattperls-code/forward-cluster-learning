import torch
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import src.forward_cluster_learning as fcl
import matplotlib.pyplot as plt

def profile_model(name, create_model, bp_lr, fcl_lr, loss_function, similarity_metric, num_training_batches, batch_size, x, y):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

    scaler = StandardScaler()

    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    x_train = torch.tensor(x_train, dtype=torch.float32)
    x_test  = torch.tensor(x_test,  dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.long)
    y_test  = torch.tensor(y_test,  dtype=torch.long)

    batch_number_samples = []

    bp_model = create_model()
    bp_trainer = torch.optim.Adam(bp_model.parameters(), lr=bp_lr)
    bp_loss_samples = []
    bp_accuracy_samples = []

    fcl_model = create_model()
    fcl_trainer = fcl.ForwardClusterLearning(
        fcl_model,
        torch.optim.Adam,
        { "lr": fcl_lr },
        loss_function,
        similarity_metric,
        0
    )
    fcl_loss_samples = []
    fcl_accuracy_samples = []

    for i in range(num_training_batches):
        batch_indices = torch.randperm(len(x_train))[:batch_size]

        batch_x = x_train[batch_indices]
        batch_y = y_train[batch_indices]

        bp_model.train()
        bp_trainer.zero_grad()
        bp_predictions = bp_model(batch_x)
        loss = loss_function(bp_predictions, batch_y)
        loss.backward()
        bp_trainer.step()

        fcl_trainer.cluster(batch_x, batch_y)

        if i % (num_training_batches // 100) == 0:
            print(f"Finished {(i / num_training_batches):.4f}")

            batch_number_samples.append(i)

            bp_model.eval()
            with torch.no_grad():
                bp_predictions = bp_model(x_test)
                bp_loss = loss_function(bp_predictions, y_test).item()
                bp_accuracy = (bp_predictions.argmax(dim=1) == y_test).float().mean().item()

                bp_loss_samples.append(bp_loss)
                bp_accuracy_samples.append(bp_accuracy)

            fcl_trainer.build_classification_head(x_train, y_train, i + 1)
            with torch.no_grad():

                fcl_predictions = fcl_model(x_test)
                fcl_loss = loss_function(fcl_predictions, y_test).item()
                fcl_accuracy = (fcl_predictions.argmax(dim=1) == y_test).float().mean().item()
                
                fcl_loss_samples.append(fcl_loss)
                fcl_accuracy_samples.append(fcl_accuracy)
            
    plt.clf()
    plt.title(f"Back Propagation Loss on {name}")
    plt.xlabel("Training Iterations")
    plt.ylabel("Loss")
    plt.ylim(bottom=0, top=max(bp_loss_samples))
    plt.plot(batch_number_samples, bp_loss_samples)
    plt.savefig(f"Back Propagation Loss on {name}")

    plt.clf()
    plt.title(f"Back Propagation Accuracy on {name}")
    plt.xlabel("Training Iterations")
    plt.ylabel("Accuracy")
    plt.ylim(bottom=0, top=1)
    plt.plot(batch_number_samples, bp_accuracy_samples)
    plt.savefig(f"Back Propagation Accuracy on {name}")

    plt.clf()
    plt.title(f"Forward Cluster Learning Loss on {name}")
    plt.xlabel("Training Iterations")
    plt.ylabel("Loss")
    plt.ylim(bottom=0, top=max(fcl_loss_samples))
    plt.plot(batch_number_samples, fcl_loss_samples)
    plt.savefig(f"Forward Cluster Learning Loss on {name}")

    plt.clf()
    plt.title(f"Forward Cluster Learning Accuracy on {name}")
    plt.xlabel("Training Iterations")
    plt.ylabel("Accuracy")
    plt.ylim(bottom=0, top=1)
    plt.plot(batch_number_samples, fcl_accuracy_samples)
    plt.savefig(f"Forward Cluster Learning Accuracy on {name}")

if __name__ == "__main__":
    x, y = make_classification(
        n_samples=1000,
        n_features=8,
        n_informative=4,
        n_redundant=2,
        n_classes=3,
        n_clusters_per_class=2
    )

    profile_model(
        "Synthetically Generated Clusters",
        lambda: torch.nn.Sequential(
            torch.nn.Linear(8, 16),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(16, 16),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(16, 3)
        ),
        0.004,
        0.003,
        torch.nn.functional.cross_entropy,
        fcl.SimilarityMetric("TANH_DOT"),
        2500,
        60,
        x,
        y
    )