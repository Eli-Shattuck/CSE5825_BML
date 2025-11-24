import torch

from beschess.components.net.resnet import ResEmbeddingNet, SEResEmbeddingNet


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
