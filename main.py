import torch
import src.forward_cluster_learning as fcl
import matplotlib.pyplot as plt

def plot_similarity_matrix(similarity_matrix: torch.Tensor, labels: torch.Tensor):
    matrix = similarity_matrix.detach().cpu().numpy()
    label_list = labels.cpu().numpy()

    fig, ax = plt.subplots()
    im = ax.imshow(matrix, cmap="viridis")
    plt.colorbar(im)

    ax.set_xticks(range(len(label_list)))
    ax.set_yticks(range(len(label_list)))
    ax.set_xticklabels(label_list)
    ax.set_yticklabels(label_list)

    plt.show()

def main():
    example_mlp = torch.nn.Sequential(
        torch.nn.Linear(4, 10),
        torch.nn.ReLU(),
        torch.nn.Linear(10, 10)
    )

    training_inputs = torch.tensor([
        [ 0.2, -0.5, 0.9, -0.4 ],
        [ -0.3, 0.5, 0.8, -0.2 ],
        [ 0.9, 0.0, -0.6, 0.7 ],
        [ 0.5, 0.3, 0.3, -0.4 ],
        [ -0.4, 0.2, 0.1, 0.5 ],
        [ -0.3, 0.6, -0.1, 0.7 ]
    ])

    training_labels = torch.tensor([ 0, 0, 1, 1, 2, 2 ])

    trainer = fcl.ForwardClusterLearning(example_mlp, torch.optim.Adam, { "lr": 0.001 }, fcl.SimilarityMetric("SQRT_DOT"), 0)

    for _ in range(1000):
        trainer.cluster(training_inputs, training_labels)

    plot_similarity_matrix(trainer.similarity_metric(example_mlp(training_inputs)), training_labels)

if __name__ == "__main__":
    main()