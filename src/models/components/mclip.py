import torch.nn as nn
from transformers import RobertaTokenizer, RobertaModel
from typing import List
import torch
import numpy as np
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import torch.nn.functional as F
from src.models.components.nn import timestep_embedding
from src.utils.misc import lengths_to_mask


class MotionEncoder(nn.Module):
    def __init__(
        self,
        n_layers: int = 12,
        n_heads: int = 7,
        input_dim: int = 147,
        embed_dim: int = 512,
        lm_hidden_size: int = 512,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        max_motion_length: int = 471,
    ):
        super().__init__()

        self.n_layers = n_layers
        self.n_heads = n_heads
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.lm_hidden_size = lm_hidden_size
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout
        self.max_motion_length = max_motion_length

        self.input_projection = nn.Linear(input_dim, embed_dim)
        # transformer blocks
        encoder_layers = TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )

        self.motion_length_emb = nn.Embedding(max_motion_length, embed_dim)
        self.motion_encoder = TransformerEncoder(encoder_layers, n_layers)
        self.CLS_TOKEN = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.last_projection = nn.Linear(embed_dim, lm_hidden_size)

    def forward(self, motion: torch.Tensor, motion_length: List[int]):

        trs_mask = torch.zeros(
            (motion.shape[0], 3, motion.shape[2]), device=motion.device
        )
        rotation_mask = torch.ones(
            (motion.shape[0], motion.shape[1] - 3, motion.shape[2]),
            device=motion.device,
        )
        full_mask = torch.cat([trs_mask, rotation_mask], dim=1)
        motion = motion * full_mask

        motion = motion.permute(0, 2, 1)  # (N,C,L) -> (N,T,C)
        device = motion.device
        mask = lengths_to_mask(motion_length, device)
        B, T, C = motion.shape

        cls_token = self.CLS_TOKEN.repeat((B, 1, 1))

        # motion length embedding
        ml_token = self.motion_length_emb(motion_length)[:, None, :]

        motion_input_proj = self.input_projection(motion)  # (N,L,E)
        input_tokens = torch.cat(
            [cls_token, ml_token, motion_input_proj], dim=1
        )  # [B, T + 2, E]

        pos_enc = timestep_embedding(
            torch.arange(T + 2, device=device), self.embed_dim
        ).repeat((B, 1, 1))
        input_tokens_pe = input_tokens + pos_enc

        mask_cls_len = torch.ones(
            (B, 2), dtype=bool, device=device
        )  # extend mask for CLS token and length token
        mask_ext = torch.cat([mask_cls_len, mask], dim=1)

        encoder_out = self.motion_encoder(
            src=input_tokens_pe, src_key_padding_mask=~mask_ext
        )  # mask_ext: (N,T+1)
        out = self.last_projection(encoder_out)

        output_dict = {
            "pooler_output": out[:, 0, :],
            "last_hidden_state": out,
        }
        return output_dict


class MCLIP(nn.Module):
    def __init__(
        self,
        menc_n_layers: int = 12,
        menc_n_heads: int = 7,
        menc_input_dim: int = 147,
        menc_embed_dim: int = 768,
        menc_lm_hidden_size: int = 512,
        menc_dim_feedforward: int = 2048,
        menc_dropout: float = 0.1,
        tau: float = 0.07,
        max_motion_length: int = 471,
        freeze_lm: bool = False,
    ):
        super().__init__()

        self.text_tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
        self.text_model = RobertaModel.from_pretrained("roberta-base")
        if freeze_lm:
            for param in self.text_model.parameters():
                param.requires_grad = False

        self.motion_encoder = MotionEncoder(
            n_layers=menc_n_layers,
            n_heads=menc_n_heads,
            input_dim=menc_input_dim,
            embed_dim=menc_embed_dim,
            lm_hidden_size=menc_lm_hidden_size,
            dim_feedforward=menc_dim_feedforward,
            dropout=menc_dropout,
            max_motion_length=max_motion_length,
        )
        self.lm_projection = nn.Linear(768, menc_lm_hidden_size)

        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / tau))
        self.loss_motion = nn.CrossEntropyLoss()
        self.loss_text = nn.CrossEntropyLoss()

    def forward(
        self, motion: torch.Tensor, texts: List[str], motion_length: torch.LongTensor
    ):
        """
        motion: (N, C, L)
        """

        encoded_text = self.text_tokenizer(texts, return_tensors="pt", padding=True).to(
            motion.device
        )
        text_out = self.text_model(**encoded_text)
        motion_out = self.motion_encoder(motion, motion_length)

        text_feat = self.lm_projection(text_out["pooler_output"])
        motion_feat = motion_out["pooler_output"]

        # joint multimodal embedding
        text_feat = F.normalize(text_feat, dim=-1, p="fro")
        motion_feat = F.normalize(motion_feat, dim=-1, p="fro")

        logit_scale = self.logit_scale.exp()
        logits_per_motion = logit_scale * motion_feat @ text_feat.t()
        logits_per_text = logits_per_motion.t()

        return logits_per_motion, logits_per_text

    def get_features(
        self, motion: torch.Tensor, texts: List[str], motion_length: torch.LongTensor
    ):
        """
        Feature exstractor.
        motion: (N,C,L)
        texts: (N,)
        motion_length: (N,)
        """
        encoded_text = self.text_tokenizer(texts, return_tensors="pt", padding=True).to(
            motion.device
        )
        text_out = self.text_model(**encoded_text)
        motion_out = self.motion_encoder(motion, motion_length)

        text_feat = self.lm_projection(text_out["pooler_output"])
        motion_feat = motion_out["pooler_output"]

        return {
            "text_feat": text_feat,
            "motion_feat": motion_feat,
            "text_feat_l2_norm": F.normalize(text_feat, dim=-1, p="fro"),
            "motion_feat_l2_norm": F.normalize(motion_feat, dim=-1, p="fro"),
        }

    def compute_loss(
        self, motion: torch.Tensor, texts: List[str], motion_length: torch.LongTensor
    ):
        logits_per_motion, logits_per_text = self.forward(
            motion, texts, motion_length=motion_length
        )
        gt_labels = torch.arange(len(motion), dtype=torch.long, device=motion.device)
        loss = (
            self.loss_motion(logits_per_motion, gt_labels)
            + self.loss_text(logits_per_text, gt_labels)
        ) / 2
        return loss
