import torch
import torch.nn as nn
import torch.nn.functional as F


class ProxyAnchor(nn.Module):
    def __init__(
        self,
        n_classes: int,
        embedding_dim: int,
        margin: float = 0.1,
        alpha: float = 32.0,
    ):
        super().__init__()
        # Proxy Anchor Initialization
        self.proxies = torch.nn.Parameter(torch.randn(n_classes, embedding_dim))
        nn.init.kaiming_normal_(self.proxies, mode="fan_out")

        self.n_classes = n_classes
        self.embedding_dim = embedding_dim
        self.margin = margin
        self.alpha = alpha

    def forward(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
    ):
        similarity = F.linear(
            F.normalize(X, p=2, dim=1),
            F.normalize(self.proxies, p=2, dim=1),
        )

        pos_exp = torch.exp(-self.alpha * (similarity - self.margin))
        neg_exp = torch.exp(self.alpha * (similarity + self.margin))

        pos_mask = y
        neg_mask = 1 - y

        n_valid_proxies = torch.sum(torch.sum(pos_mask, dim=0) > 0).float()

        pos_term = torch.log(1 + torch.sum(pos_exp * pos_mask, dim=0)).sum()
        neg_term = torch.log(1 + torch.sum(neg_exp * neg_mask, dim=0)).sum()

        if n_valid_proxies > 0:
            pos_term = pos_term / n_valid_proxies
        neg_term = neg_term / self.n_classes
        loss = pos_term + neg_term

        return loss
