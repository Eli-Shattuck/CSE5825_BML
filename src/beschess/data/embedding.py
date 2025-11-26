import numpy as np
import torch
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from ..utils import packed_to_tensor


class PuzzleDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        quiet_boards: np.ndarray,
        puzzle_boards: np.ndarray,
        puzzle_labels: np.ndarray,
    ):
        assert len(puzzle_boards) == len(puzzle_labels)
        self.quiet_boards = quiet_boards
        self.puzzle_boards = puzzle_boards
        self.puzzle_labels = puzzle_labels

        self.n_quiet = len(self.quiet_boards)
        self.n_puzzles = len(self.puzzle_boards)

    def __len__(self):
        return self.n_quiet + self.n_puzzles

    def __getitem__(self, idx):
        label = torch.zeros(16, dtype=torch.float32)
        if idx < self.n_quiet:
            board_packed = self.quiet_boards[idx]
            label[0] = 1.0
        else:
            rel_idx = idx - self.n_quiet
            board_packed = self.puzzle_boards[rel_idx]
            puzzle_label = self.puzzle_labels[rel_idx]

            label[1:] = torch.from_numpy(puzzle_label.copy()).float()

        board_unpacked = packed_to_tensor(board_packed)
        board_tensor = torch.from_numpy(board_unpacked.copy()).float()

        return board_tensor, label

    def get_puzzle_labels(self):
        return self.puzzle_labels


class BalancedBatchSampler(torch.utils.data.BatchSampler):
    def __init__(
        self,
        dataset: PuzzleDataset,
        quiet_indices: list,
        puzzle_indices: list,
        batch_size: int,
        quiet_ratio: float = 0.25,
        steps_per_epoch: int = 1000,
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.steps_per_epoch = steps_per_epoch

        self.n_quiet_per_batch = int(batch_size * quiet_ratio)
        self.n_puzzle_per_batch = batch_size - self.n_quiet_per_batch

        self.quiet_indices = quiet_indices
        self.puzzle_offsets = dataset.n_quiet

        raw_labels = dataset.get_puzzle_labels()
        self.label_map = {i: [] for i in range(15)}

        pbar = tqdm(total=len(puzzle_indices), desc="Building label map")
        for global_idx in puzzle_indices:
            pbar.update(1)
            local_idx = global_idx - self.puzzle_offsets
            labels = raw_labels[local_idx]

            active_labels = np.flatnonzero(labels)
            for label_idx in active_labels:
                self.label_map[label_idx].append(global_idx)

        self.active_label_idxs = [
            label for label in self.label_map if len(self.label_map[label]) > 0
        ]

    def __iter__(self):
        quiet_indices = self.quiet_indices.copy()
        np.random.shuffle(quiet_indices)

        quiet_ptr = 0
        label_ptrs = {label: 0 for label in self.label_map.keys()}
        label_lengths = {
            label: len(self.label_map[label]) for label in self.label_map.keys()
        }

        for label in self.label_map.keys():
            np.random.shuffle(self.label_map[label])

        for _ in range(self.steps_per_epoch):
            batch = []

            for _ in range(self.n_quiet_per_batch):
                if quiet_ptr >= len(quiet_indices):
                    quiet_ptr = 0
                    np.random.shuffle(quiet_indices)

                batch.append(quiet_indices[quiet_ptr])
                quiet_ptr += 1

            for _ in range(self.n_puzzle_per_batch):
                selected_label = np.random.choice(self.active_label_idxs)

                if label_ptrs[selected_label] >= label_lengths[selected_label]:
                    label_ptrs[selected_label] = 0
                    np.random.shuffle(self.label_map[selected_label])

                label_list = self.label_map[selected_label]
                label_ptr = label_ptrs[selected_label]

                batch.append(label_list[label_ptr])
                label_ptrs[selected_label] += 1

            np.random.shuffle(batch)
            yield batch

    def __len__(self):
        return self.steps_per_epoch


class DirectLoader:
    def __init__(
        self,
        dataset: PuzzleDataset,
        sampler: BalancedBatchSampler,
        device: torch.device,
        chunk_size=50000,
    ):
        """Loads the entire dataset onto the GPU to bypass DataLoader bottlenecks."""
        self.sampler = sampler
        self.device = device
        self.batch_size = sampler.batch_size

        n_total = len(dataset)

        self.all_boards = torch.empty(
            (n_total, 17, 8, 8), dtype=torch.float32, device=device
        )
        self.all_labels = torch.empty((n_total, 16), dtype=torch.float32, device=device)

        n_quiet = dataset.n_quiet
        n_puzzles = dataset.n_puzzles
        puzzle_offset = n_quiet

        pbar = tqdm(total=n_quiet + n_puzzles, desc="Loading Boards to Device")
        for i in range(0, n_quiet, chunk_size):
            pbar.update(min(chunk_size, n_quiet - i))
            end = min(i + chunk_size, n_quiet)

            packed_chunk = dataset.quiet_boards[i:end]
            unpacked_chunk = np.array([packed_to_tensor(b) for b in packed_chunk])

            self.all_boards[i:end] = torch.from_numpy(unpacked_chunk).to(device)

            self.all_labels[i:end, 0] = 1.0
            self.all_labels[i:end, 1:] = 0.0

        for i in range(0, n_puzzles, chunk_size):
            pbar.update(min(chunk_size, n_puzzles - i))
            rel_start = i
            rel_end = min(i + chunk_size, n_puzzles)
            count = rel_end - rel_start

            global_start = puzzle_offset + rel_start
            global_end = puzzle_offset + rel_end

            packed_chunk = dataset.puzzle_boards[rel_start:rel_end]
            unpacked_chunk = np.array([packed_to_tensor(b) for b in packed_chunk])

            self.all_boards[global_start:global_end] = torch.from_numpy(
                unpacked_chunk
            ).to(device)

            batch_labels = dataset.puzzle_labels[rel_start:rel_end]

            gpu_labels_chunk = torch.zeros((count, 16), device=device)
            gpu_labels_chunk[:, 1:] = torch.from_numpy(batch_labels.copy()).to(device)

            self.all_labels[global_start:global_end] = gpu_labels_chunk

    def __iter__(self):
        for batch_indices in self.sampler:
            idx_tensor = torch.tensor(
                batch_indices, device=self.device, dtype=torch.long
            )

            batch_x = self.all_boards[idx_tensor]
            batch_y = self.all_labels[idx_tensor]

            yield batch_x, batch_y

    def __len__(self):
        return len(self.sampler)


def generate_split_indices(
    dataset: PuzzleDataset,
    val_split: float = 0.05,
    test_split: float = 0.05,
) -> dict[str, tuple[list, list]]:
    n_quiet = dataset.n_quiet
    n_total = len(dataset)

    quiet_idxs = np.arange(n_quiet)
    q_train, q_temp = train_test_split(
        quiet_idxs, test_size=(val_split + test_split), random_state=42
    )
    q_val, q_test = train_test_split(q_temp, test_size=0.5, random_state=42)

    puzzle_idxs = np.arange(n_quiet, n_total)

    raw_labels = dataset.puzzle_labels
    primary_labels = np.argmax(raw_labels, axis=1)

    p_train, p_temp, _, y_temp = train_test_split(
        puzzle_idxs,
        primary_labels,
        test_size=(val_split + test_split),
        stratify=primary_labels,
        random_state=42,
    )
    p_val, p_test = train_test_split(
        p_temp, test_size=0.5, stratify=y_temp, random_state=42
    )

    return {
        "train": (q_train, p_train),
        "val": (q_val, p_val),
        "test": (q_test, p_test),
    }
