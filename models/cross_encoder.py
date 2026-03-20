# models/cross_encoder.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict, Any, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

from models.base import ModelBase


@dataclass
class CrossEncoderSavedConfig:
    pretrained_name: str
    max_length: int
    dropout: float
    cls_weight: float
    rank_weight: float
    mismatch_rank_weight: float
    listwise_temperature: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "pretrained_name": self.pretrained_name,
            "max_length": self.max_length,
            "dropout": self.dropout,
            "cls_weight": self.cls_weight,
            "rank_weight": self.rank_weight,
            "mismatch_rank_weight": self.mismatch_rank_weight,
            "listwise_temperature": self.listwise_temperature,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "CrossEncoderSavedConfig":
        return cls(
            pretrained_name=str(d["pretrained_name"]),
            max_length=int(d.get("max_length", 512)),
            dropout=float(d.get("dropout", 0.1)),
            cls_weight=float(d.get("cls_weight", 1.0)),
            rank_weight=float(d.get("rank_weight", 0.0)),
            mismatch_rank_weight=float(d.get("mismatch_rank_weight", 0.0)),
            listwise_temperature=float(d.get("listwise_temperature", 1.0)),
        )


class _CrossEncoderCore(nn.Module):
    """
    2-logits classifier head (old behavior):
      logits: [B,2]
    """
    def __init__(self, encoder: nn.Module, hidden_size: int, dropout: float = 0.1):
        super().__init__()
        self.encoder = encoder
        self.dropout = nn.Dropout(float(dropout))
        self.classifier = nn.Linear(int(hidden_size), 2)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls = out.last_hidden_state[:, 0]
        cls = self.dropout(cls)
        logits = self.classifier(cls)  # [B,2]
        return logits


class CrossEncoder(ModelBase):
    WEIGHTS_NAME = "pytorch_model.bin"
    CONFIG_NAME = "saved_config.json"

    def __init__(
        self,
        pretrained_name: str,
        *,
        encoder_name_or_path: Optional[str] = None,
        max_length: int = 512,
        dropout: float = 0.1,
        cls_weight: float = 1.0,
        rank_weight: float = 0.0,
        mismatch_rank_weight: float = 0.0,
        listwise_temperature: float = 1.0,
    ):
        self.pretrained_name = str(pretrained_name)
        self.max_length = int(max_length)

        self.tokenizer = AutoTokenizer.from_pretrained(self.pretrained_name, use_fast=True)

        enc_path = encoder_name_or_path or self.pretrained_name
        encoder = AutoModel.from_pretrained(enc_path)
        hidden_size = int(getattr(encoder.config, "hidden_size"))

        self.model = _CrossEncoderCore(encoder=encoder, hidden_size=hidden_size, dropout=dropout)

        self.cls_weight = float(cls_weight)
        self.rank_weight = float(rank_weight)
        self.mismatch_rank_weight = float(mismatch_rank_weight)
        self.listwise_temperature = float(listwise_temperature)

        self._saved_cfg = CrossEncoderSavedConfig(
            pretrained_name=self.pretrained_name,
            max_length=self.max_length,
            dropout=float(dropout),
            cls_weight=self.cls_weight,
            rank_weight=self.rank_weight,
            mismatch_rank_weight=self.mismatch_rank_weight,
            listwise_temperature=self.listwise_temperature,
        )

    # -------------------------
    # Listwise helpers (on score)
    # -------------------------
    def _listwise_ce_single(self, scores: torch.Tensor, pos_index: int, subset_mask: torch.Tensor) -> torch.Tensor:
        idx = torch.nonzero(subset_mask, as_tuple=False).squeeze(-1)
        if idx.numel() == 0:
            raise ValueError("Empty subset for listwise CE.")
        pos_in_subset = (idx == int(pos_index)).nonzero(as_tuple=False).squeeze(-1)
        if pos_in_subset.numel() == 0:
            raise ValueError("POS not included in listwise subset.")
        s = scores[idx].view(1, -1) / max(self.listwise_temperature, 1e-6)
        t = pos_in_subset[0].view(1).to(device=scores.device)
        return F.cross_entropy(s, t)

    def _listwise_ce_batch(self, scores: torch.Tensor, pos_index: torch.Tensor, subset_mask: torch.Tensor) -> torch.Tensor:
        losses: List[torch.Tensor] = []
        B = scores.shape[0]
        for b in range(B):
            losses.append(self._listwise_ce_single(scores[b], int(pos_index[b].item()), subset_mask[b]))
        return torch.stack(losses, dim=0).mean()

    # -------------------------
    # Forward
    # -------------------------
    def forward(self, **kwargs) -> Dict[str, Any]:
        """
        (A) Pairwise:
            input_ids: [B,L]
            attention_mask: [B,L]
            labels(optional): [B] int64 0/1

        (B) Group batch:
            neg_input_ids: [B,K1,L]
            neg_attention_mask: [B,K1,L]
            mm_input_ids: [B,K2,L]
            mm_attention_mask: [B,K2,L]
            mm_is_mismatch: [B,K2] bool
        """
        # group batch
        if "neg_input_ids" in kwargs and "mm_input_ids" in kwargs:
            return self._forward_group_batch(
                neg_input_ids=kwargs["neg_input_ids"],
                neg_attention_mask=kwargs["neg_attention_mask"],
                mm_input_ids=kwargs["mm_input_ids"],
                mm_attention_mask=kwargs["mm_attention_mask"],
                mm_is_mismatch=kwargs["mm_is_mismatch"],
            )

        # pairwise
        input_ids = kwargs.get("input_ids")
        attention_mask = kwargs.get("attention_mask")
        labels = kwargs.get("labels", None)
        if input_ids is None or attention_mask is None:
            raise ValueError("Missing input_ids/attention_mask.")

        if input_ids.dim() != 2:
            raise ValueError(f"Pairwise expects input_ids [B,L], got {tuple(input_ids.shape)}")

        logits = self.model(input_ids=input_ids, attention_mask=attention_mask)  # [B,2]
        score = logits[:, 1] - logits[:, 0]  # [B]

        out: Dict[str, Any] = {"logits": logits, "score": score}

        if labels is None:
            return out

        labels_t = labels.to(device=logits.device, dtype=torch.long)
        cls_loss = F.cross_entropy(logits, labels_t)
        out["cls_loss"] = cls_loss
        out["loss"] = self.cls_weight * cls_loss
        return out

    def _forward_group_batch(
        self,
        *,
        neg_input_ids: torch.Tensor,
        neg_attention_mask: torch.Tensor,
        mm_input_ids: torch.Tensor,
        mm_attention_mask: torch.Tensor,
        mm_is_mismatch: torch.Tensor,
    ) -> Dict[str, Any]:
        device = neg_input_ids.device

        # NEG logits [B,K1,2]
        B1, K1, L1 = neg_input_ids.shape
        neg_logits = self.model(
            input_ids=neg_input_ids.reshape(B1 * K1, L1),
            attention_mask=neg_attention_mask.reshape(B1 * K1, L1),
        ).reshape(B1, K1, 2)

        # MM logits [B,K2,2]
        B2, K2, L2 = mm_input_ids.shape
        mm_logits = self.model(
            input_ids=mm_input_ids.reshape(B2 * K2, L2),
            attention_mask=mm_attention_mask.reshape(B2 * K2, L2),
        ).reshape(B2, K2, 2)

        if B1 != B2:
            raise ValueError("neg batch and mm batch must have same batch size.")

        # Score matrices for ranking + metrics
        neg_scores = neg_logits[..., 1] - neg_logits[..., 0]  # [B,K1]
        mm_scores = mm_logits[..., 1] - mm_logits[..., 0]     # [B,K2]

        # Classification targets: POS at index 0 => label 1, others 0
        neg_targets = torch.zeros((B1, K1), device=device, dtype=torch.long)
        neg_targets[:, 0] = 1
        mm_targets = torch.zeros((B2, K2), device=device, dtype=torch.long)
        mm_targets[:, 0] = 1

        neg_cls_loss = F.cross_entropy(neg_logits.reshape(B1 * K1, 2), neg_targets.reshape(B1 * K1))
        mm_cls_loss = F.cross_entropy(mm_logits.reshape(B2 * K2, 2), mm_targets.reshape(B2 * K2))
        cls_loss = 0.5 * (neg_cls_loss + mm_cls_loss)

        total = self.cls_weight * cls_loss

        pos_index = torch.zeros((B1,), device=device, dtype=torch.long)

        neg_rank_loss = None
        mm_rank_loss = None

        if self.rank_weight != 0.0:
            subset_neg = torch.ones((B1, K1), device=device, dtype=torch.bool)  # POS+NEG all
            neg_rank_loss = self._listwise_ce_batch(neg_scores, pos_index, subset_neg)
            total = total + self.rank_weight * neg_rank_loss

        if self.mismatch_rank_weight != 0.0:
            mm_is_mismatch = mm_is_mismatch.to(device=device, dtype=torch.bool)
            pos_mask = torch.zeros((B2, K2), device=device, dtype=torch.bool)
            pos_mask[:, 0] = True
            subset_mm = pos_mask | mm_is_mismatch  # POS + mismatch only
            mm_rank_loss = self._listwise_ce_batch(mm_scores, pos_index, subset_mm)
            total = total + self.mismatch_rank_weight * mm_rank_loss

        return {
            "loss": total,
            "cls_loss": cls_loss,
            "neg_rank_loss": neg_rank_loss,
            "mm_rank_loss": mm_rank_loss,
            "neg_logits": neg_logits,
            "mm_logits": mm_logits,
            "neg_scores": neg_scores,
            "mm_scores": mm_scores,
        }
