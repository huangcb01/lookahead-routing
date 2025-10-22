import torch.nn.functional as F
from torch import Tensor

from args import SCArguments


def compute_routing_loss(routing_logits: Tensor, scores: Tensor, args: SCArguments):
    """Compute the loss based on the loss type.

    Args:
        routing_logits: shape=(batch_size, num_candidates)
        scores: shape=(batch_size, num_candidates)
        args: The training arguments.
    """
    if args.loss_type == "CE":
        labels = scores.argmax(dim=1)
        return F.cross_entropy(routing_logits, labels)
    elif args.loss_type == "BCE":
        max_score = scores.max(dim=1, keepdim=True)
        labels = (scores >= max_score.values * args.bce_threshold).float() if args.bce_threshold > 0 else scores
        return F.binary_cross_entropy_with_logits(routing_logits, labels)
    elif args.loss_type == "MSE":
        return F.mse_loss(routing_logits, scores)
    else:
        log_score_distribution = F.log_softmax(scores / args.kl_temperature, dim=1)
        log_predicted_distribution = F.log_softmax(routing_logits, dim=1)
        if args.loss_type == "ForwardKL":
            return F.kl_div(log_predicted_distribution, log_score_distribution, reduction="batchmean", log_target=True)
        elif args.loss_type == "ReverseKL":
            return F.kl_div(log_score_distribution, log_predicted_distribution, reduction="batchmean", log_target=True)
        else:
            raise ValueError(f"Unsupported loss type: {args.loss_type}")
