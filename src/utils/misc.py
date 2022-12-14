from typing import List

import numpy as np
import torch
from torch import Tensor


def remove_special_characters(text: str, special_characters=[",", ".", "/"]):
    for special_character in special_characters:
        text = text.replace(special_character, "")
        text = text.replace(" ", "_")

    return text


def replace_annotation_with_null(annotation: List[str], replace_prob: float):
    """
    Replace annotations with empty string.
    """

    replaced_annotation = list(annotation)
    replace_prob = (
        np.random.uniform(low=0.0, high=1.0, size=len(annotation)) < replace_prob
    )
    for replace_ind, to_be_replaced in enumerate(replace_prob):
        if to_be_replaced:
            replaced_annotation[replace_ind] = ""

    return replaced_annotation


def lengths_to_mask(lengths: List[int], device: torch.device) -> Tensor:
    """
    Generate mask array.
    """
    lengths = lengths.clone().detach()
    max_len = max(lengths)
    mask = torch.arange(max_len, device=device).expand(
        len(lengths), max_len
    ) < lengths.unsqueeze(1)
    return mask
