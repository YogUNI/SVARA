"""Loss computation, metric evaluators, and optimization helpers.

Task: P2-08, P3-04
Reference: docs/02 §1, docs/11 V-11..V-19
"""

from typing import Any, Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadSLULoss(nn.Module):
    """Joint or Multi-Head Cross Entropy loss with optional label smoothing."""

    def __init__(
        self,
        label_smoothing: float = 0.0,
        use_joint: bool = False,
        joint_weight: float = 0.0,
    ):
        super().__init__()
        self.label_smoothing = label_smoothing
        self.use_joint = use_joint or (joint_weight > 0.0)
        self.joint_weight = joint_weight if joint_weight > 0.0 else 1.0

    def forward(
        self,
        logits: Dict[str, torch.Tensor],
        targets: Union[Dict[str, torch.Tensor], torch.Tensor],
        *args,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Calculate loss and per-slot loss metrics.

        Args:
            logits: dict containing 'action', 'object', 'location' (and optional 'joint')
            targets: dict containing slot targets or tensor for action
        """
        if isinstance(targets, dict):
            act_tgt = targets.get("action_id", targets.get("action"))
            obj_tgt = targets.get("object_id", targets.get("object"))
            loc_tgt = targets.get("location_id", targets.get("location"))
            intent_tgt = targets.get("intent_id", targets.get("joint"))
        elif len(args) >= 2:
            act_tgt = targets
            obj_tgt = args[0]
            loc_tgt = args[1]
            intent_tgt = args[2] if len(args) > 2 else None
        else:
            raise ValueError(f"Unsupported targets format for MultiHeadSLULoss: {type(targets)}")

        loss_act = F.cross_entropy(
            logits["action"], act_tgt, label_smoothing=self.label_smoothing
        )
        loss_obj = F.cross_entropy(
            logits["object"], obj_tgt, label_smoothing=self.label_smoothing
        )
        loss_loc = F.cross_entropy(
            logits["location"], loc_tgt, label_smoothing=self.label_smoothing
        )

        total_loss = loss_act + loss_obj + loss_loc

        log_dict = {
            "loss_action": float(loss_act.item()),
            "loss_object": float(loss_obj.item()),
            "loss_location": float(loss_loc.item()),
            "loss_total": float(total_loss.item()),
        }

        if self.use_joint and "joint" in logits and intent_tgt is not None:
            loss_joint = F.cross_entropy(
                logits["joint"], intent_tgt, label_smoothing=self.label_smoothing
            )
            total_loss = total_loss + self.joint_weight * loss_joint
            log_dict["loss_joint"] = float(loss_joint.item())
            log_dict["loss_total"] = float(total_loss.item())

        return total_loss, log_dict


def compute_batch_accuracies(
    logits: Dict[str, torch.Tensor],
    targets: Union[Dict[str, torch.Tensor], torch.Tensor],
    *args,
) -> Dict[str, float]:
    """Calculate exact-match and per-slot accuracies for a batch."""
    if isinstance(targets, dict):
        act_tgt = targets.get("action_id", targets.get("action"))
        obj_tgt = targets.get("object_id", targets.get("object"))
        loc_tgt = targets.get("location_id", targets.get("location"))
    elif len(args) >= 2:
        act_tgt = targets
        obj_tgt = args[0]
        loc_tgt = args[1]
    else:
        raise ValueError(f"Unsupported targets format for compute_batch_accuracies: {type(targets)}")

    pred_act = torch.argmax(logits["action"], dim=-1)
    pred_obj = torch.argmax(logits["object"], dim=-1)
    pred_loc = torch.argmax(logits["location"], dim=-1)

    corr_act = (pred_act == act_tgt)
    corr_obj = (pred_obj == obj_tgt)
    corr_loc = (pred_loc == loc_tgt)

    # Exact-match intent: all 3 slots must be correct
    exact_match = (corr_act & corr_obj & corr_loc)

    act_val = float(corr_act.float().mean().item())
    obj_val = float(corr_obj.float().mean().item())
    loc_val = float(corr_loc.float().mean().item())
    em_val = float(exact_match.float().mean().item())

    return {
        "acc_action": act_val,
        "acc_object": obj_val,
        "acc_location": loc_val,
        "acc_exact_match": em_val,
        "action": act_val,
        "object": obj_val,
        "location": loc_val,
        "exact_match": em_val,
    }
