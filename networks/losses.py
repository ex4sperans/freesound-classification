import torch
import torch.nn.functional as F


def focal_loss(input, target, focus=2.0, raw=True):

    if raw:
        input = torch.sigmoid(input)

    eps = 1e-7

    prob_true = input * target + (1 - input) * (1 - target)
    prob_true = torch.clamp(prob_true, eps, 1-eps)
    modulating_factor = (1.0 - prob_true).pow(focus)

    return (-modulating_factor * prob_true.log()).mean()


def binary_cross_entropy(input, target, raw=True):
    if raw:
        input = torch.sigmoid(input)
    return torch.nn.functional.binary_cross_entropy(input, target)


def lsep_loss(input, target, average=True):

    n = input.size(0)

    differences = input.unsqueeze(1) - input.unsqueeze(2)
    where_lower = (target.unsqueeze(1) < target.unsqueeze(2)).float()

    differences = differences.view(n, -1)
    where_lower = where_lower.view(n, -1)

    max_difference, index = torch.max(differences, dim=1, keepdim=True)
    differences = differences - max_difference
    exps = differences.exp() * where_lower

    lsep = max_difference + torch.log(torch.exp(-max_difference) + exps.sum(-1))

    if average:
        return lsep.mean()
    else:
        return lsep
