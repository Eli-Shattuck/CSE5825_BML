import torch

from beschess.components.net.block import SEBlock


def test_se_block():
    batch_size = 4
    channels = 12
    height = 8
    width = 8

    x = torch.randn(batch_size, channels, height, width)
    se_block = SEBlock(channels)

    output = se_block(x)

    assert output.shape == x.shape
