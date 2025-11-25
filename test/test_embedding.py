import pytest
import torch
import numpy as np

from beschess.components.net.resnet import ResEmbeddingNet, SEResEmbeddingNet
from beschess.components.loss import ProxyAnchor
from beschess.components.utils import (
    evaluate_knn_cos,
    evaluate_proxy_cos,
    compute_knn_hitrate,
    compute_proxy_hitrate,
    compute_proxy_map,
)

NUM_SAMPLES = 8
NUM_CLASSES = 4


@pytest.fixture
def dataloader():
    class DummyDataset(torch.utils.data.Dataset):
        def __init__(self):
            self.samples = [torch.randn(12, 8, 8) for _ in range(NUM_SAMPLES)]
            self.labels = []
            for _ in range(NUM_SAMPLES):
                label = torch.zeros(NUM_CLASSES, dtype=torch.float32)
                num_active = np.random.randint(1, 2)
                active_indices = np.random.choice(
                    NUM_CLASSES, size=num_active, replace=False
                )
                label[active_indices] = 1.0
                self.labels.append(label)

        def __len__(self):
            return len(self.samples)

        def __getitem__(self, idx):
            return self.samples[idx], self.labels[idx]

    return torch.utils.data.DataLoader(DummyDataset(), batch_size=3, shuffle=False)


@pytest.fixture
def zero_dataloader():
    class ZeroDataset(torch.utils.data.Dataset):
        def __init__(self):
            self.samples = [torch.randn(12, 8, 8) for _ in range(NUM_SAMPLES)]
            self.labels = [
                torch.zeros(NUM_CLASSES, dtype=torch.float32)
                for _ in range(NUM_SAMPLES)
            ]

        def __len__(self):
            return len(self.samples)

        def __getitem__(self, idx):
            return self.samples[idx], self.labels[idx]

    return torch.utils.data.DataLoader(ZeroDataset(), batch_size=3, shuffle=False)


@pytest.fixture
def full_dataloader():
    class FullDataset(torch.utils.data.Dataset):
        def __init__(self):
            self.samples = [torch.randn(12, 8, 8) for _ in range(NUM_SAMPLES)]
            self.labels = [
                torch.ones(NUM_CLASSES, dtype=torch.float32) for _ in range(NUM_SAMPLES)
            ]

        def __len__(self):
            return len(self.samples)

        def __getitem__(self, idx):
            return self.samples[idx], self.labels[idx]

    return torch.utils.data.DataLoader(FullDataset(), batch_size=3, shuffle=False)


def test_res_embedding_net():
    batch_size = 4
    embedding_dim = 128
    num_blocks = 3

    x = torch.randn(batch_size, 12, 8, 8)
    model = ResEmbeddingNet(embedding_dim, num_blocks)

    output = model(x)

    assert output.shape == (batch_size, embedding_dim)


def test_se_res_embedding_net():
    batch_size = 4
    embedding_dim = 64
    num_blocks = 3

    x = torch.randn(batch_size, 12, 8, 8)
    model = SEResEmbeddingNet(embedding_dim, num_blocks)

    output = model(x)

    assert output.shape == (batch_size, embedding_dim)


def test_knn_evaluate_cos_and_compute_recall(
    dataloader, zero_dataloader, full_dataloader
):
    embedding_dim = 32
    num_blocks = 2
    k_values = [1, 3, 5]

    model = ResEmbeddingNet(embedding_dim, num_blocks)
    device = torch.device("cpu")

    similarity_matrix, labels = evaluate_knn_cos(model, zero_dataloader, device)
    assert similarity_matrix.shape == (NUM_SAMPLES, NUM_SAMPLES)
    assert labels.shape == (NUM_SAMPLES, NUM_CLASSES)
    _, top_indices = torch.topk(similarity_matrix, k=max(k_values) + 1, dim=1)
    recalls = compute_knn_hitrate(top_indices, labels, k_values)
    for k in k_values:
        assert k in recalls
        assert recalls[k] == 0.0

    similarity_matrix, labels = evaluate_knn_cos(model, full_dataloader, device)
    assert similarity_matrix.shape == (NUM_SAMPLES, NUM_SAMPLES)
    assert labels.shape == (NUM_SAMPLES, NUM_CLASSES)
    _, top_indices = torch.topk(similarity_matrix, k=max(k_values) + 1, dim=1)
    recalls = compute_knn_hitrate(top_indices, labels, k_values)
    for k in k_values:
        assert k in recalls
        assert recalls[k] == 1.0

    similarity_matrix, labels = evaluate_knn_cos(model, dataloader, device)
    assert similarity_matrix.shape == (NUM_SAMPLES, NUM_SAMPLES)
    assert labels.shape == (NUM_SAMPLES, NUM_CLASSES)
    _, top_indices = torch.topk(similarity_matrix, k=max(k_values) + 1, dim=1)
    recalls = compute_knn_hitrate(top_indices, labels, k_values)
    for k in k_values:
        assert k in recalls
        assert 0.0 <= recalls[k] <= 1.0


def test_proxy_evaluate_cos_and_compute_recall(
    dataloader, zero_dataloader, full_dataloader
):
    embedding_dim = 32
    num_blocks = 2
    k_values = [1, 2]

    model = ResEmbeddingNet(embedding_dim, num_blocks)
    loss_fn = ProxyAnchor(NUM_CLASSES, embedding_dim)
    device = torch.device("cpu")

    similarity_matrix, labels = evaluate_proxy_cos(
        model, loss_fn, zero_dataloader, device
    )
    assert similarity_matrix.shape == (NUM_SAMPLES, NUM_CLASSES)
    assert labels.shape == (NUM_SAMPLES, NUM_CLASSES)
    _, top_indices = torch.topk(similarity_matrix, k=max(k_values), dim=1)
    recalls = compute_proxy_hitrate(top_indices, labels, k_values)
    for k in k_values:
        assert k in recalls
        assert recalls[k] == 0.0

    similarity_matrix, labels = evaluate_proxy_cos(
        model, loss_fn, full_dataloader, device
    )
    assert similarity_matrix.shape == (NUM_SAMPLES, NUM_CLASSES)
    assert labels.shape == (NUM_SAMPLES, NUM_CLASSES)
    _, top_indices = torch.topk(similarity_matrix, k=max(k_values), dim=1)
    recalls = compute_proxy_hitrate(top_indices, labels, k_values)
    for k in k_values:
        assert k in recalls
        assert recalls[k] == 1.0

    similarity_matrix, labels = evaluate_proxy_cos(model, loss_fn, dataloader, device)
    assert similarity_matrix.shape == (NUM_SAMPLES, NUM_CLASSES)
    assert labels.shape == (NUM_SAMPLES, NUM_CLASSES)
    _, top_indices = torch.topk(similarity_matrix, k=max(k_values), dim=1)
    recalls = compute_proxy_hitrate(top_indices, labels, k_values)
    for k in k_values:
        assert k in recalls
        assert 0.0 <= recalls[k] <= 1.0


def test_proxy_map(dataloader):
    embedding_dim = 32
    num_blocks = 2
    k_values = [1, 2, 3]

    model = ResEmbeddingNet(embedding_dim, num_blocks)
    device = torch.device("cpu")

    similarity_matrix, labels = evaluate_knn_cos(model, dataloader, device)
    assert similarity_matrix.shape == (NUM_SAMPLES, NUM_SAMPLES)
    assert labels.shape == (NUM_SAMPLES, NUM_CLASSES)
    _, top_indices = torch.topk(similarity_matrix, k=max(k_values), dim=1)
    map_scores = compute_proxy_map(top_indices, labels, k_values)
    for k in k_values:
        assert k in map_scores
        assert 0.0 <= map_scores[k] <= 1.0
