"""Loss computation, metric evaluators, and optimization helpers.

Task: P2-08, P3-04
Reference: docs/02 §1, docs/11 V-11..V-19
"""

from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadSLULoss(nn.Module):
    """Joint or Multi-Head Cross Entropy loss with optional label smoothing."""

    def __init__(self, label_smoothing: float = 0.0, use_joint: bool = False):
        super().__init__()
        self.label_smoothing = label_smoothing
        self.use_joint = use_joint

    def forward(
        self,
        logits: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Calculate loss and per-slot loss metrics.

        Args:
            logits: dict containing 'action', 'object', 'location' (and optional 'joint')
            targets: dict containing 'action_id', 'object_id', 'location_id', 'intent_id'
        """
        loss_act = F.cross_entropy(
            logits["action"], targets["action_id"], label_smoothing=self.label_smoothing
        )
        loss_obj = F.cross_entropy(
            logits["object"], targets["object_id"], label_smoothing=self.label_smoothing
        )
        loss_loc = F.cross_entropy(
            logits["location"], targets["location_id"], label_smoothing=self.label_smoothing
        )

        total_loss = loss_act + loss_obj + loss_loc

        log_dict = {
            "loss_action": float(loss_act.item()),
            "loss_object": float(loss_obj.item()),
            "loss_location": float(loss_loc.item()),
            "loss_total": float(total_loss.item()),
        }

        if self.use_joint and "joint" in logits:
            loss_joint = F.cross_entropy(
                logits["joint"], targets["intent_id"], label_smoothing=self.label_smoothing
            )
            total_loss = total_loss + loss_joint
            log_dict["loss_joint"] = float(loss_joint.item())
            log_dict["loss_total"] = float(total_loss.item())

        return total_loss, log_dict


def compute_batch_accuracies(
    logits: Dict[str, torch.Tensor],
    targets: Dict[str, torch.Tensor],
) -> Dict[str, float]:
    """Calculate exact-match and per-slot accuracies for a batch."""
    pred_act = torch.argmax(logits["action"], dim=-1)
    pred_obj = torch.argmax(logits["object"], dim=-1)
    pred_loc = torch.argmax(logits["location"], dim=-1)

    corr_act = (pred_act == targets["action_id"])
    corr_obj = (pred_obj == targets["object_id"])
    corr_loc = (pred_loc == targets["location_id"])

    # Exact-match intent: all 3 slots must be correct
    exact_match = (corr_act & corr_obj & corr_loc)

    return {
        "acc_action": float(corr_act.float().mean().item()),
        "acc_object": float(corr_obj.float().mean().item()),
        "acc_location": float(corr_loc.float().mean().item()),
        "acc_exact_match": float(exact_match.float().mean().item()),
    }
