# models/bi_encoder.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict, Any, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

from models.base import ModelBase


def mean_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).type_as(last_hidden_state)
    summed = (last_hidden_state * mask).sum(dim=1)
    denom = mask.sum(dim=1).clamp(min=1e-6)
    return summed / denom


# =========================
# Saved Config
# =========================

@dataclass
class BiEncoderSavedConfig:
    pretrained_name: str
    max_length: int
    pooling: str
    proj_dim: int
    temperature: float
    learnable_temperature: bool
    normalize: bool

    cls_weight: float
    rank_weight: float
    mismatch_rank_weight: float
    listwise_temperature: float

    # in-batch nce
    use_inbatch_nce: bool
    nce_weight: float
    nce_symmetric: bool

    def to_dict(self) -> Dict[str, Any]:
        return {
            "pretrained_name": str(self.pretrained_name),
            "max_length": int(self.max_length),
            "pooling": str(self.pooling),
            "proj_dim": int(self.proj_dim),
            "temperature": float(self.temperature),
            "learnable_temperature": bool(self.learnable_temperature),
            "normalize": bool(self.normalize),
            "cls_weight": float(self.cls_weight),
            "rank_weight": float(self.rank_weight),
            "mismatch_rank_weight": float(self.mismatch_rank_weight),
            "listwise_temperature": float(self.listwise_temperature),

            "use_inbatch_nce": bool(self.use_inbatch_nce),
            "nce_weight": float(self.nce_weight),
            "nce_symmetric": bool(self.nce_symmetric),
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "BiEncoderSavedConfig":
        return cls(
            pretrained_name=str(d.get("pretrained_name")),
            max_length=int(d.get("max_length", 256)),
            pooling=str(d.get("pooling", "mean")),
            proj_dim=int(d.get("proj_dim", 0)),
            temperature=float(d.get("temperature", 0.07)),
            learnable_temperature=bool(d.get("learnable_temperature", False)),
            normalize=bool(d.get("normalize", True)),
            cls_weight=float(d.get("cls_weight", 1.0)),
            rank_weight=float(d.get("rank_weight", 0.0)),
            mismatch_rank_weight=float(d.get("mismatch_rank_weight", 0.0)),
            listwise_temperature=float(d.get("listwise_temperature", 1.0)),

            use_inbatch_nce=bool(d.get("use_inbatch_nce", False)),
            nce_weight=float(d.get("nce_weight", 0.1)),
            nce_symmetric=bool(d.get("nce_symmetric", True)),
        )


# =========================
# Core Model
# =========================

class _BiEncoderCore(nn.Module):
    def __init__(
        self,
        encoder: nn.Module,
        *,
        hidden_size: int,
        pooling: str = "mean",
        proj_dim: int = 0,
        temperature: float = 0.07,
        learnable_temperature: bool = False,
        normalize: bool = True,
    ):
        super().__init__()
        self.encoder = encoder
        self.pooling = str(pooling)
        self.normalize = bool(normalize)

        in_dim = int(hidden_size)
        out_dim = int(proj_dim) if int(proj_dim) > 0 else in_dim
        self.emb_dim = int(out_dim)

        self.proj = nn.Linear(in_dim, out_dim, bias=False) if out_dim != in_dim else nn.Identity()

        if learnable_temperature:
            t = float(temperature)
            self.log_tau = nn.Parameter(torch.log(torch.tensor([t], dtype=torch.float32)))
            self.register_buffer("tau", torch.tensor([t], dtype=torch.float32), persistent=False)
        else:
            self.log_tau = None
            self.register_buffer("tau", torch.tensor([float(temperature)], dtype=torch.float32), persistent=False)

    def _get_tau(self) -> torch.Tensor:
        if self.log_tau is not None:
            return torch.exp(self.log_tau).clamp(min=1e-4, max=10.0)
        return self.tau.clamp(min=1e-4, max=10.0)

    def encode_text(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        hs = out.last_hidden_state
        if self.pooling == "mean":
            emb = mean_pool(hs, attention_mask)
        else:
            emb = hs[:, 0]
        emb = self.proj(emb)
        if self.normalize:
            emb = F.normalize(emb, p=2, dim=-1)
        return emb

    def score_pairs(
        self,
        style_input_ids: torch.Tensor,
        style_attention_mask: torch.Tensor,
        query_input_ids: torch.Tensor,
        query_attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Pairwise score: returns [B] logits.
        """
        e_s = self.encode_text(style_input_ids, style_attention_mask)  # [B,D]
        e_q = self.encode_text(query_input_ids, query_attention_mask)  # [B,D]
        tau = self._get_tau()
        s = torch.einsum("bd,bd->b", e_s, e_q) / tau  # [B]
        return s

    def score_group(
        self,
        style_input_ids: torch.Tensor,
        style_attention_mask: torch.Tensor,
        query_input_ids: torch.Tensor,
        query_attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        NEG route scoring:
          style fixed: [B,L]
          queries vary: [B,K,L]
        Returns:
          scores: [B,K]
        """
        e_s = self.encode_text(style_input_ids, style_attention_mask)  # [B,D]

        bsz, k, seqlen = query_input_ids.shape
        q_ids = query_input_ids.reshape(bsz * k, seqlen)
        q_mask = query_attention_mask.reshape(bsz * k, seqlen)
        e_q = self.encode_text(q_ids, q_mask).reshape(bsz, k, -1)  # [B,K,D]

        tau = self._get_tau()
        scores = torch.einsum("bd,bkd->bk", e_s, e_q) / tau  # [B,K]
        return scores

    def score_group_fixed_query(
        self,
        style_input_ids: torch.Tensor,          # [B,K,L]
        style_attention_mask: torch.Tensor,     # [B,K,L]
        query_input_ids: torch.Tensor,          # [B,L]
        query_attention_mask: torch.Tensor,     # [B,L]
    ) -> torch.Tensor:
        """
        MM route scoring: fixed query (one per group), varying style references.

        Returns:
          scores: [B,K]
        """
        e_q = self.encode_text(query_input_ids, query_attention_mask)  # [B,D]

        B, K, L = style_input_ids.shape
        s_ids = style_input_ids.reshape(B * K, L)
        s_mask = style_attention_mask.reshape(B * K, L)
        e_s = self.encode_text(s_ids, s_mask).reshape(B, K, -1)        # [B,K,D]

        tau = self._get_tau()
        scores = torch.einsum("bkd,bd->bk", e_s, e_q) / tau             # [B,K]
        return scores

    def score_matrix_inbatch(
        self,
        style_input_ids: torch.Tensor,            # [B,L]
        style_attention_mask: torch.Tensor,       # [B,L]
        query_input_ids: torch.Tensor,            # [B,L]
        query_attention_mask: torch.Tensor,       # [B,L]
    ) -> torch.Tensor:
        """
        In-batch pairwise score matrix S[i,j] for InfoNCE.

        Returns:
          logits matrix [B,B]
        """
        e_s = self.encode_text(style_input_ids, style_attention_mask)  # [B,D]
        e_q = self.encode_text(query_input_ids, query_attention_mask)  # [B,D]
        tau = self._get_tau()
        return (e_s @ e_q.t()) / tau


# =========================
# Wrapper
# =========================

class BiEncoder(ModelBase):
    CONFIG_NAME = "tstd_biencoder_config.json"
    WEIGHTS_NAME = "pytorch_model.bin"

    def __init__(
        self,
        pretrained_name: str,
        *,
        encoder_name_or_path: Optional[str] = None,
        max_length: int = 256,
        pooling: str = "mean",
        proj_dim: int = 0,
        temperature: float = 0.07,
        learnable_temperature: bool = False,
        normalize: bool = True,
        cls_weight: float = 1.0,
        rank_weight: float = 0.0,
        mismatch_rank_weight: float = 0.0,
        listwise_temperature: float = 1.0,

        # in-batch nce
        use_inbatch_nce: bool = False,
        nce_weight: float = 0.1,
        nce_symmetric: bool = True,

        **kwargs,
    ):
        self.pretrained_name = str(pretrained_name)
        self.max_length = int(max_length)
        self.pooling = str(pooling)
        self.proj_dim = int(proj_dim)
        self.temperature = float(temperature)
        self.learnable_temperature = bool(learnable_temperature)
        self.normalize = bool(normalize)

        self.cls_weight = float(cls_weight)
        self.rank_weight = float(rank_weight)
        self.mismatch_rank_weight = float(mismatch_rank_weight)
        self.listwise_temperature = float(listwise_temperature)

        self.use_inbatch_nce = bool(use_inbatch_nce)
        self.nce_weight = float(nce_weight)
        self.nce_symmetric = bool(nce_symmetric)

        self.tokenizer = AutoTokenizer.from_pretrained(self.pretrained_name, use_fast=True)

        enc_path = encoder_name_or_path if encoder_name_or_path is not None else self.pretrained_name
        encoder = AutoModel.from_pretrained(enc_path)
        hidden_size = int(getattr(encoder.config, "hidden_size", 768))

        self.model = _BiEncoderCore(
            encoder,
            hidden_size=hidden_size,
            pooling=self.pooling,
            proj_dim=self.proj_dim,
            temperature=self.temperature,
            learnable_temperature=self.learnable_temperature,
            normalize=self.normalize,
        )

        self._saved_cfg = BiEncoderSavedConfig(
            pretrained_name=self.pretrained_name,
            max_length=self.max_length,
            pooling=self.pooling,
            proj_dim=self.proj_dim,
            temperature=self.temperature,
            learnable_temperature=self.learnable_temperature,
            normalize=self.normalize,

            cls_weight=self.cls_weight,
            rank_weight=self.rank_weight,
            mismatch_rank_weight=self.mismatch_rank_weight,
            listwise_temperature=self.listwise_temperature,

            use_inbatch_nce=self.use_inbatch_nce,
            nce_weight=self.nce_weight,
            nce_symmetric=self.nce_symmetric,
        )

    # -------------------------
    # listwise helpers (on score)
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
    # in-batch InfoNCE (group mode only; main route)
    # -------------------------
    def _inbatch_nce_loss(
        self,
        *,
        style_input_ids: torch.Tensor,
        style_attention_mask: torch.Tensor,
        qpos_input_ids: torch.Tensor,
        qpos_attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        InfoNCE on main route (non-mm): diagonal positives.
        """
        logits = self.model.score_matrix_inbatch(
            style_input_ids=style_input_ids,
            style_attention_mask=style_attention_mask,
            query_input_ids=qpos_input_ids,
            query_attention_mask=qpos_attention_mask,
        )  # [B,B]

        B = int(logits.shape[0])
        target = torch.arange(B, device=logits.device, dtype=torch.long)

        loss1 = F.cross_entropy(logits, target)
        if self.nce_symmetric:
            loss2 = F.cross_entropy(logits.t(), target)
            return 0.5 * (loss1 + loss2)
        return loss1

    # -------------------------
    # NEW: in-batch InfoNCE for Pair mode (use positive subset only)
    # -------------------------
    def _inbatch_nce_loss_pair(
        self,
        *,
        style_input_ids: torch.Tensor,
        style_attention_mask: torch.Tensor,
        query_input_ids: torch.Tensor,
        query_attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor],
    ) -> Optional[torch.Tensor]:
        """
        Pair mode may contain both pos/neg pairs (labels 1/0).
        InfoNCE needs a well-defined positive for each anchor, so we:
          - take subset where label==1 (diagonal positives)
          - compute in-batch NCE on that subset only
        If no labels provided, or too few positives, return None.
        """
        if labels is None:
            return None

        # Use only positive pairs for diagonal positives
        pos_mask = labels.to(device=style_input_ids.device).view(-1).float() > 0.5
        pos_idx = torch.nonzero(pos_mask, as_tuple=False).squeeze(-1)

        # Need at least 2 positives to make in-batch meaningful/stable
        if pos_idx.numel() < 2:
            return None

        s_ids = style_input_ids.index_select(0, pos_idx)
        s_mask = style_attention_mask.index_select(0, pos_idx)
        q_ids = query_input_ids.index_select(0, pos_idx)
        q_mask = query_attention_mask.index_select(0, pos_idx)

        return self._inbatch_nce_loss(
            style_input_ids=s_ids,
            style_attention_mask=s_mask,
            qpos_input_ids=q_ids,
            qpos_attention_mask=q_mask,
        )

    # -------------------------
    # forward
    # -------------------------
    def forward(self, **kwargs) -> Dict[str, Any]:
        """
        (A) Pairwise:
          style_input_ids: [B,L]
          style_attention_mask: [B,L]
          query_input_ids: [B,L]
          query_attention_mask: [B,L]
          labels(optional): [B] int64 0/1

        (B) Group batch:
          style_input_ids: [B,L]
          style_attention_mask: [B,L]
          neg_query_input_ids: [B,K1,L]
          neg_query_attention_mask: [B,K1,L]
          mm_style_input_ids: [B,K2,L]
          mm_style_attention_mask: [B,K2,L]
          mm_fixed_query_input_ids: [B,L]
          mm_fixed_query_attention_mask: [B,L]
          mm_is_mismatch: [B,K2] bool
        """
        if "neg_query_input_ids" in kwargs and "mm_style_input_ids" in kwargs:
            return self._forward_group_batch(
                style_input_ids=kwargs["style_input_ids"],
                style_attention_mask=kwargs["style_attention_mask"],
                neg_query_input_ids=kwargs["neg_query_input_ids"],
                neg_query_attention_mask=kwargs["neg_query_attention_mask"],
                mm_style_input_ids=kwargs["mm_style_input_ids"],
                mm_style_attention_mask=kwargs["mm_style_attention_mask"],
                mm_fixed_query_input_ids=kwargs["mm_fixed_query_input_ids"],
                mm_fixed_query_attention_mask=kwargs["mm_fixed_query_attention_mask"],
                mm_is_mismatch=kwargs["mm_is_mismatch"],
            )

        style_input_ids = kwargs.get("style_input_ids")
        style_attention_mask = kwargs.get("style_attention_mask")
        query_input_ids = kwargs.get("query_input_ids")
        query_attention_mask = kwargs.get("query_attention_mask")
        labels = kwargs.get("labels", None)

        if style_input_ids is None or style_attention_mask is None or query_input_ids is None or query_attention_mask is None:
            raise ValueError("Missing required inputs for BiEncoder forward.")

        if query_input_ids.dim() != 2:
            raise ValueError(f"Pairwise expects query_input_ids [B,L], got {tuple(query_input_ids.shape)}")

        # score is the raw logit for BCE
        score = self.model.score_pairs(
            style_input_ids=style_input_ids,
            style_attention_mask=style_attention_mask,
            query_input_ids=query_input_ids,
            query_attention_mask=query_attention_mask,
        )  # [B]

        prob = torch.sigmoid(score)  # [B] in [0,1]
        out: Dict[str, Any] = {
            "logits": score,   # BCE-style logits
            "score": score,    # keep for compatibility / debugging
            "prob": prob,      # requested final 0~1 probability
        }

        if labels is None:
            return out

        labels_f = labels.to(device=score.device, dtype=torch.float32).view(-1)
        cls_loss = F.binary_cross_entropy_with_logits(score.view(-1), labels_f)

        total = self.cls_weight * cls_loss

        # -------------------------
        # In-batch InfoNCE for Pair mode (positive subset only)
        # -------------------------
        nce_loss = None
        if self.use_inbatch_nce and self.nce_weight != 0.0:
            nce_loss = self._inbatch_nce_loss_pair(
                style_input_ids=style_input_ids,
                style_attention_mask=style_attention_mask,
                query_input_ids=query_input_ids,
                query_attention_mask=query_attention_mask,
                labels=labels_f,
            )
            if nce_loss is not None:
                total = total + self.nce_weight * nce_loss

        out["cls_loss"] = cls_loss
        out["nce_loss"] = nce_loss
        out["loss"] = total
        return out

    def _forward_group_batch(
        self,
        *,
        style_input_ids: torch.Tensor,
        style_attention_mask: torch.Tensor,
        neg_query_input_ids: torch.Tensor,
        neg_query_attention_mask: torch.Tensor,
        mm_style_input_ids: torch.Tensor,
        mm_style_attention_mask: torch.Tensor,
        mm_fixed_query_input_ids: torch.Tensor,
        mm_fixed_query_attention_mask: torch.Tensor,
        mm_is_mismatch: torch.Tensor,
    ) -> Dict[str, Any]:
        device = style_input_ids.device

        # -------------------------
        # NEG route: fixed s_true, varying q
        # -------------------------
        neg_scores = self.model.score_group(
            style_input_ids=style_input_ids,
            style_attention_mask=style_attention_mask,
            query_input_ids=neg_query_input_ids,
            query_attention_mask=neg_query_attention_mask,
        )  # [B,K1]

        # -------------------------
        # MM route (q-route): fixed q_pos, varying s
        #   index0 = (s_true, q_pos)  -> POS
        #   index>=1 = (s_wrong, q_pos) -> NEG (mismatch s)
        # -------------------------
        mm_scores = self.model.score_group_fixed_query(
            style_input_ids=mm_style_input_ids,
            style_attention_mask=mm_style_attention_mask,
            query_input_ids=mm_fixed_query_input_ids,
            query_attention_mask=mm_fixed_query_attention_mask,
        )  # [B,K2]

        B1, K1 = neg_scores.shape
        B2, K2 = mm_scores.shape
        if B1 != B2:
            raise ValueError("neg batch and mm batch must have same batch size.")

        # BCE targets: POS at index 0 => 1, others 0
        neg_targets = torch.zeros((B1, K1), device=device, dtype=torch.float32)
        neg_targets[:, 0] = 1.0
        mm_targets = torch.zeros((B2, K2), device=device, dtype=torch.float32)
        mm_targets[:, 0] = 1.0

        neg_cls_loss = F.binary_cross_entropy_with_logits(
            neg_scores.reshape(B1 * K1),
            neg_targets.reshape(B1 * K1),
        )
        mm_cls_loss = F.binary_cross_entropy_with_logits(
            mm_scores.reshape(B2 * K2),
            mm_targets.reshape(B2 * K2),
        )
        cls_loss = 0.5 * (neg_cls_loss + mm_cls_loss)

        total = self.cls_weight * cls_loss

        pos_index = torch.zeros((B1,), device=device, dtype=torch.long)

        neg_rank_loss = None
        mm_rank_loss = None

        # listwise over NEG route (POS + all NEG queries)
        if self.rank_weight != 0.0:
            subset_neg = torch.ones((B1, K1), device=device, dtype=torch.bool)
            neg_rank_loss = self._listwise_ce_batch(neg_scores, pos_index, subset_neg)
            total = total + self.rank_weight * neg_rank_loss

        # listwise over MM route (POS + mismatch s only)
        if self.mismatch_rank_weight != 0.0:
            mm_is_mismatch = mm_is_mismatch.to(device=device, dtype=torch.bool)
            pos_mask = torch.zeros((B2, K2), device=device, dtype=torch.bool)
            pos_mask[:, 0] = True
            subset_mm = pos_mask | mm_is_mismatch
            mm_rank_loss = self._listwise_ce_batch(mm_scores, pos_index, subset_mm)
            total = total + self.mismatch_rank_weight * mm_rank_loss

        # -------------------------
        # In-batch InfoNCE (main route only; diagonal: s_true vs q_pos)
        # -------------------------
        nce_loss = None
        if self.use_inbatch_nce and self.nce_weight != 0.0:
            # q_pos is always index 0 in neg_query_input_ids
            qpos_ids = neg_query_input_ids[:, 0, :]
            qpos_mask = neg_query_attention_mask[:, 0, :]

            nce_loss = self._inbatch_nce_loss(
                style_input_ids=style_input_ids,
                style_attention_mask=style_attention_mask,
                qpos_input_ids=qpos_ids,
                qpos_attention_mask=qpos_mask,
            )
            total = total + self.nce_weight * nce_loss

        return {
            "loss": total,
            "cls_loss": cls_loss,
            "neg_rank_loss": neg_rank_loss,
            "mm_rank_loss": mm_rank_loss,
            "nce_loss": nce_loss,

            # keep names for trainer diagnostics compatibility
            "neg_logits": neg_scores,                 # BCE logits [B,K1]
            "mm_logits": mm_scores,                   # BCE logits [B,K2]
            "neg_probs": torch.sigmoid(neg_scores),   # [B,K1]
            "mm_probs": torch.sigmoid(mm_scores),     # [B,K2]

            "neg_scores": neg_scores,
            "mm_scores": mm_scores,
        }
