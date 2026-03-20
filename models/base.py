# models/base.py
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

class ModelBase(ABC):
    model: Any
    tokenizer: Any

    def to(self, device: str):
        if not hasattr(self, "model"):
            raise AttributeError(f"{self.__class__.__name__} has no attribute `model`.")
        self.model.to(device)
        return self

    def train(self):
        self.model.train()
        return self

    def eval(self):
        self.model.eval()
        return self

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(pretrained_name={getattr(self, 'pretrained_name', None)})"
