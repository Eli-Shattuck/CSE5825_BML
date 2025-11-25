import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from .loss import ProxyAnchor


def lr_range_test(
    optimizer: torch.optim.Optimizer,
    model: torch.nn.Module,
    loss_fn: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    start_lr: float,
    end_lr: float,
    num_iters: int,
    beta: float = 0.05,
):
    for param_group in optimizer.param_groups:
        param_group["lr"] = start_lr

    def lr_lambda(iteration):
        return (end_lr / start_lr) ** (iteration / num_iters)

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    model.train()
    iter_loader = iter(dataloader)

    lrs = []
    losses = []
    avg_loss = 0.0
    best_loss = float("inf")

    for iterations in tqdm(range(num_iters), desc="LR Range Test"):
        try:
            inputs, targets = next(iter_loader)
        except StopIteration:
            iter_loader = iter(dataloader)
            inputs, targets = next(iter_loader)

        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_fn(outputs, targets)

        current_loss = loss.item()

        if iterations == 0:
            avg_loss = current_loss
        else:
            avg_loss = beta * current_loss + (1 - beta) * avg_loss

        if iterations > 0 and avg_loss > 4 * best_loss:
            print(f"Loss exploded at iteration {iterations}. Stopping.")
            break

        if avg_loss < best_loss:
            best_loss = avg_loss

        loss.backward()
        optimizer.step()
        scheduler.step()

        current_lr = optimizer.param_groups[0]["lr"]
        lrs.append(current_lr)
        losses.append(avg_loss)

    return lrs, losses


def evaluate_proxy_cos(
    model: nn.Module,
    loss_fn: ProxyAnchor,
    dataloader: DataLoader,
    device: torch.device,
):
    model_is_train = model.training
    model.eval()
    all_embeddings = []
    all_labels = []

    with torch.no_grad():
        for boards, labels in tqdm(dataloader, desc="Evaluating Proxy COS"):
            boards = boards.to(device)
            embeddings = model(boards)
            all_embeddings.append(embeddings.cpu())
            all_labels.append(labels)

    model.train(model_is_train)

    all_embeddings = torch.cat(all_embeddings, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    proxies = F.normalize(loss_fn.proxies.detach().cpu(), p=2, dim=1)
    all_embeddings = F.normalize(all_embeddings, p=2, dim=1)
    similarity_matrix = torch.matmul(all_embeddings, proxies.T)

    return similarity_matrix, all_labels


def evaluate_knn_cos(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    model_is_train = model.training
    model.eval()
    all_embeddings = []
    all_labels = []

    with torch.no_grad():
        for boards, labels in tqdm(dataloader, desc="Evaluating COS"):
            boards = boards.to(device)
            embeddings = model(boards)
            all_embeddings.append(embeddings.cpu())
            all_labels.append(labels)

    model.train(model_is_train)

    all_embeddings = torch.cat(all_embeddings, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    all_embeddings = F.normalize(all_embeddings, p=2, dim=1)
    similarity_matrix = torch.matmul(all_embeddings, all_embeddings.T)

    return similarity_matrix, all_labels


def compute_proxy_hitrate(
    top_indices: torch.Tensor,
    labels: torch.Tensor,
    k_values: list,
) -> dict[int, float]:
    assert top_indices.size(0) == labels.size(0)
    assert top_indices.size(1) == max(k_values)

    n_samples = labels.size(0)
    recalls = {k: 0.0 for k in k_values}

    for k in k_values:
        # Proxy Anchor's idx is the class idx
        pred_labels = top_indices[:, :k]
        # Take the value from multi-hot labels
        # if predicted class is present in true labels it selects 1 else 0
        hits = labels.gather(1, pred_labels)
        recalls[k] = (hits.sum(dim=1) > 0).float().sum().item()

    for k in k_values:
        recalls[k] /= n_samples

    return recalls


def compute_knn_hitrate(
    top_indices: torch.Tensor,
    labels: torch.Tensor,
    k_values: list,
) -> dict[int, float]:
    assert top_indices.size(0) == labels.size(0)
    assert top_indices.size(1) == max(k_values) + 1

    n_samples = labels.size(0)
    recalls = {k: 0.0 for k in k_values}

    for k in k_values:
        # Exclude self-match at index 0
        pred_indices = top_indices[:, 1 : k + 1]
        pred_labels = labels[pred_indices]
        intersection = pred_labels * labels.unsqueeze(1)
        hits = (intersection.sum(dim=2) > 0).float()
        recalls[k] = (hits.sum(dim=1) > 0).float().sum().item()

    for k in k_values:
        recalls[k] /= n_samples

    return recalls


def compute_proxy_map(
    top_indices: torch.Tensor,
    labels: torch.Tensor,
    k_values: list,
) -> dict[int, float]:
    assert top_indices.size(0) == labels.size(0)
    assert top_indices.size(1) == max(k_values)

    n_samples = labels.size(0)

    maps = {k: 0.0 for k in k_values}
    k_checkpoints = set(k_values)

    for i in range(n_samples):
        true_set = set(labels[i].nonzero(as_tuple=False).view(-1).tolist())
        num_true = len(true_set)

        if num_true == 0:
            continue

        hits = 0
        sum_precisions = 0.0

        pred_labels = top_indices[i]
        for j, pred_idx in enumerate(pred_labels):
            current_k = j + 1
            if pred_idx in true_set:
                hits += 1
                sum_precisions += hits / current_k

            if current_k in k_checkpoints:
                denom = min(num_true, current_k)
                if denom > 0:
                    maps[current_k] += sum_precisions / denom

    for k in k_values:
        maps[k] /= n_samples

    return maps
