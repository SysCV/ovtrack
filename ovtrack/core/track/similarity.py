import torch
import torch.nn.functional as F


def cal_similarity(key_embeds, ref_embeds, method="dot_product", temperature=-1):

    assert method in ["dot_product", "cosine"]

    if key_embeds.size(0) == 0 or ref_embeds.size(0) == 0:
        return torch.zeros(
            (key_embeds.size(0), ref_embeds.size(0)), device=key_embeds.device
        )

    if method == "cosine":
        key_embeds = F.normalize(key_embeds, p=2, dim=1)
        ref_embeds = F.normalize(ref_embeds, p=2, dim=1)
        dists = torch.mm(key_embeds, ref_embeds.t())
        if temperature > 0 and temperature <= 1:
            dists /= temperature
        return dists

    elif method == "dot_product":

        if temperature > 1:
            dists = torch.mm(key_embeds, ref_embeds.t())
            dists *= temperature
        else:
            dists = torch.mm(key_embeds, ref_embeds.t())

        return dists
