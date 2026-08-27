"""
active_learning/simulated_oracle.py — Simulated expert labeling using ground truth.

In a real active learning deployment, the system would present uncertain
pixels/regions to a human expert (botanist, ecologist) for labeling. Since
we don't have a live human-in-the-loop, we simulate this by revealing
the existing ground truth labels for queried pixels.

IMPORTANT: This is explicitly simulated. The code is structured so that
replacing the simulated oracle with a real human interface (e.g., a web
annotation tool) requires changing only this module. The AL loop itself
is oracle-agnostic — it just receives labels for queried coordinates.

The report MUST state: "Expert labeling is simulated using held-out ground
truth, not live human annotation."
"""

import torch
from typing import Dict, Tuple


class SimulatedOracle:
    """
    Simulated labeling oracle that reveals ground truth for queried pixels.

    Maintains a labeled/unlabeled mask and tracks labeling history
    (which pixels were labeled at which AL round, and their labels).
    """

    def __init__(
        self,
        ground_truth: torch.Tensor,
        initial_labeled_mask: torch.Tensor = None,
    ):
        """
        Args:
            ground_truth: Full ground truth labels, shape (H, W).
                Class 0 = background (these are never "labeled" by the oracle).
                Classes 1–C = actual classes.
            initial_labeled_mask: (H, W) boolean — True = initially labeled pixels.
                If None, no pixels are initially labeled.
        """
        self.ground_truth = ground_truth.clone()
        H, W = ground_truth.shape

        # Labeled mask: tracks which pixels have been "labeled" so far
        if initial_labeled_mask is not None:
            self.labeled_mask = initial_labeled_mask.clone()
        else:
            self.labeled_mask = torch.zeros(H, W, dtype=torch.bool)

        # History: list of (round_number, coordinates, labels) tuples
        self.history = []
        self.current_round = 0

        # Total annotation budget used
        self.total_labeled = self.labeled_mask.sum().item()

    def initialize_random_labels(
        self, fraction: float, seed: int = 42
    ) -> torch.Tensor:
        """
        Randomly label a fraction of non-background pixels as the initial pool.

        This is L₀ — the seed labeled set that the AL loop starts from.

        Args:
            fraction: Fraction of non-background pixels to label initially.
            seed: Random seed for reproducibility.

        Returns:
            labeled_mask: (H, W) boolean — updated mask.
        """
        rng = torch.Generator().manual_seed(seed)
        H, W = self.ground_truth.shape

        # Only consider non-background pixels
        non_bg = (self.ground_truth > 0)
        non_bg_coords = torch.nonzero(non_bg, as_tuple=False)  # (N, 2)
        n_non_bg = len(non_bg_coords)

        # Randomly select fraction of non-background pixels
        n_initial = max(1, int(n_non_bg * fraction))
        perm = torch.randperm(n_non_bg, generator=rng)[:n_initial]
        initial_coords = non_bg_coords[perm]

        # Mark as labeled
        self.labeled_mask[initial_coords[:, 0], initial_coords[:, 1]] = True
        self.total_labeled = self.labeled_mask.sum().item()

        print(f"[Oracle] Initialized {n_initial} labeled pixels "
              f"({fraction*100:.1f}% of {n_non_bg} non-background pixels)")

        return self.labeled_mask.clone()

    def label_pixels(
        self, query_coords: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        "Label" the queried pixels by revealing their ground truth.

        In a real system, this is where a human would provide annotations.
        Here, we just look up the answer from self.ground_truth.

        Args:
            query_coords: (K, 2) tensor of (row, col) pixel coordinates.

        Returns:
            Dict with:
              - "labels": (K,) tensor of class labels for queried pixels
              - "coords": (K, 2) tensor of coordinates (same as input)
              - "num_new": int — number of newly labeled pixels (excludes
                  pixels that were already labeled or are background)
        """
        self.current_round += 1
        K = len(query_coords)

        # Look up ground truth labels
        rows, cols = query_coords[:, 0], query_coords[:, 1]
        labels = self.ground_truth[rows, cols]

        # Track which ones are genuinely new labels (not already labeled, not background)
        new_mask = torch.zeros(K, dtype=torch.bool)
        for i in range(K):
            r, c = rows[i].item(), cols[i].item()
            if not self.labeled_mask[r, c] and labels[i].item() > 0:
                self.labeled_mask[r, c] = True
                new_mask[i] = True

        num_new = new_mask.sum().item()
        self.total_labeled = self.labeled_mask.sum().item()

        # Record history
        self.history.append({
            "round": self.current_round,
            "coords": query_coords.clone(),
            "labels": labels.clone(),
            "num_new": num_new,
        })

        print(f"[Oracle] Round {self.current_round}: queried {K} pixels, "
              f"{num_new} newly labeled. Total labeled: {self.total_labeled}")

        return {
            "labels": labels,
            "coords": query_coords,
            "num_new": num_new,
        }

    def get_labeled_data(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get the current labeled mask and ground truth for training.

        Returns:
            training_labels: (H, W) tensor where labeled pixels have their
                true class, and unlabeled/background pixels have -1 (ignore).
            labeled_mask: (H, W) boolean mask.
        """
        H, W = self.ground_truth.shape
        training_labels = torch.full((H, W), -1, dtype=torch.long)

        # Set labels only for pixels that have been "labeled" by the oracle
        training_labels[self.labeled_mask] = self.ground_truth[self.labeled_mask]

        return training_labels, self.labeled_mask.clone()

    def get_annotation_summary(self) -> Dict:
        """
        Summary statistics for reporting.

        Returns dict with total labeled count, per-round counts,
        and class distribution of labeled pixels.
        """
        labels = self.ground_truth[self.labeled_mask]
        class_counts = {}
        for c in range(1, self.ground_truth.max().item() + 1):
            class_counts[c] = (labels == c).sum().item()

        return {
            "total_labeled": self.total_labeled,
            "num_rounds": self.current_round,
            "class_distribution": class_counts,
            "per_round_new": [h["num_new"] for h in self.history],
        }
