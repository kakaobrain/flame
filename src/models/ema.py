import torch.nn as nn
from copy import deepcopy
import torch


class EMAModel(nn.Module):
    """Exponentila Moving Average Model"""

    def __init__(self, model, decay=0.9999):
        super().__init__()

        self.model = deepcopy(model)
        self.model.eval()
        self.decay = decay

    def set(self, model):
        with torch.no_grad():
            for ema_w, model_w in zip(
                self.model.state_dict().values(), model.state_dict().values()
            ):
                ema_w.copy_(model_w)

    def update(self, model):
        with torch.no_grad():
            for ema_w, model_w in zip(
                self.model.state_dict().values(), model.state_dict().values()
            ):
                ema_w.copy_(self.decay * ema_w + (1.0 - self.decay) * model_w)
