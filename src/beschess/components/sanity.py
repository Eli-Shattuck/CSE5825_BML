import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from beschess.components.net.resnet import SEResEmbeddingNet
from beschess.utils import packed_to_tensor

quiet_boards = np.load("data/processed/quiet_boards_preeval.npy", mmap_mode="r")[:100]
inputs = (
    torch.stack([torch.from_numpy(packed_to_tensor(b)) for b in quiet_boards])
    .float()
    .cuda()
)
targets = torch.zeros(100, dtype=torch.long).cuda()


class SanityModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = SEResEmbeddingNet(embedding_dim=128, num_blocks=5)
        self.classifier = nn.Linear(128, 2)

    def forward(self, x):
        emb = self.backbone(x)
        return self.classifier(emb)


model = SanityModel().cuda()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

print("Starting Cross-Entropy Sanity Check...")
for epoch in range(50):
    optimizer.zero_grad()
    outputs = model(inputs)
    loss = criterion(outputs, targets)
    loss.backward()
    optimizer.step()

    preds = torch.argmax(outputs, dim=1)
    acc = (preds == targets).float().mean()

    if epoch % 10 == 0:
        print(f"Epoch {epoch}: Loss {loss.item():.4f} | Acc {acc.item():.2f}")

print(f"Non-zero elements in first input: {torch.count_nonzero(inputs[0])}")
