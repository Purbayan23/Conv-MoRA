"""Loss exports."""

from medseg.losses.base import BaseLoss, build_loss
from medseg.losses.bce import BinaryCrossEntropyWithLogitsLoss
from medseg.losses.composite import BCEDiceLoss
from medseg.losses.dice import DiceLoss

BCELoss = BinaryCrossEntropyWithLogitsLoss

__all__ = [
    "BaseLoss",
    "BCELoss",
    "BinaryCrossEntropyWithLogitsLoss",
    "BCEDiceLoss",
    "DiceLoss",
    "build_loss",
]
