import torch as th
import torch.nn as nn
from typing import List
from src.models.components.nn import timestep_embedding
from src.models.components.unet import UNetModel
from transformers import RobertaTokenizer, RobertaModel


class Text2MotionTransformer(nn.Module):
    def __init__(
        self,
        motion_dim: int = 147,
        lm_hidden_size: int = 768,
        model_channels: int = 768,
        max_motion_length: int = 471,
        max_text_length: int = 32,
        num_heads: int = 8,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        num_layers: int = 8,
        freeze_lm: bool = False,
    ):
        super().__init__()

        self.motion_dim = motion_dim
        self.lm_hiden_size = lm_hidden_size
        self.model_channels = model_channels
        self.max_motion_length = max_motion_length
        self.max_text_length = max_text_length
        self.num_heads = num_heads
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout
        self.num_layers = num_layers

        # Freeze lanaguage model
        self.tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
        self.lm = RobertaModel.from_pretrained("roberta-base")

        if freeze_lm:
            for param in self.lm.parameters():
                param.requires_grad = False

        self.motion_length_emb = nn.Embedding(max_motion_length, model_channels)
        self.input_projection = nn.Linear(motion_dim, model_channels)
        self.text_pooler_proj = nn.Linear(lm_hidden_size, model_channels)
        self.text_h_proj = nn.Linear(lm_hidden_size, model_channels)
        self.num_aux_tokens = 3

        transformer_decoder_layer = nn.TransformerDecoderLayer(
            d_model=self.model_channels,
            nhead=self.num_heads,
            dim_feedforward=self.dim_feedforward,
            dropout=self.dropout,
            batch_first=True,
        )
        self.tf_decoder = nn.TransformerDecoder(
            transformer_decoder_layer, num_layers=num_layers
        )
        self.last_projection = nn.Linear(
            self.model_channels, self.motion_dim * 2
        )  # X 2 for variational inference

    def get_text_emb(self, texts: List[str], device: th.device):
        self.lm.eval()  # Prevent dropout from yielding different values.
        encoded_input = self.tokenizer(
            texts,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.max_text_length,
        ).to(device)
        text_embedding = self.lm(**encoded_input)
        text_pooler_out = text_embedding["pooler_output"]
        lm_pooler = self.text_pooler_proj(text_pooler_out)
        hidden_state_out = text_embedding["last_hidden_state"]  # NLC
        hidden_state_proj = self.text_h_proj(hidden_state_out)
        outputs = {"lm_pooler": lm_pooler, "lm_out": hidden_state_proj}
        return outputs

    def forward(
        self,
        motion: th.Tensor,
        motion_length: th.LongTensor,
        timesteps: th.Tensor,
        texts: List[str],
        mask: th.Tensor,
    ):
        B, C, L = motion.shape
        motion = motion.permute(0, 2, 1)  # (N,L,C)
        motion_input_proj = self.input_projection(motion)  # (N,L,E)

        # motion length embedding
        ml_token = self.motion_length_emb(motion_length)[:, None, :]

        # timestep embedding token
        timestep_token = timestep_embedding(timesteps, self.model_channels)[:, None, :]

        text_outputs = self.get_text_emb(texts, device=motion.device)

        # language token & language query for X-attention
        lm_pooler, lm_out = text_outputs["lm_pooler"], text_outputs["lm_out"]
        lm_pooler_token = lm_pooler[:, None, :]

        # Prepend length, timestsep token to LM's hidden state out
        lm_out_ext = th.cat([ml_token, timestep_token, lm_pooler_token, lm_out], dim=1)

        # Concatenate tokens and add sinusoidal positional encoding
        input_tokens = th.cat(
            [ml_token, timestep_token, lm_pooler_token, motion_input_proj], dim=1
        )
        pos_enc = timestep_embedding(
            th.arange(L + self.num_aux_tokens, device=motion.device),
            self.model_channels,
        ).repeat((B, 1, 1))
        input_tokens_pe = input_tokens + pos_enc

        # Extend mask to attend timestep and LM pooler token
        mask_time_lm = th.ones(
            (B, self.num_aux_tokens), dtype=bool, device=motion.device
        )
        mask_ext = th.cat([mask_time_lm, mask], dim=1)
        output = self.tf_decoder(
            tgt=input_tokens_pe, memory=lm_out_ext, tgt_key_padding_mask=~mask_ext
        )
        motion_part = output[:, self.num_aux_tokens :, :]
        decoder_out_mask = mask[:, :, None].repeat(1, 1, self.model_channels)
        decoder_out_valid = motion_part * decoder_out_mask
        output_motion = self.last_projection(decoder_out_valid)
        model_out_mask = mask[:, :, None].repeat(1, 1, self.motion_dim * 2)
        model_out = output_motion * model_out_mask
        return model_out.permute(0, 2, 1)  # (N,C,L)


class Text2MotionUNet(UNetModel):
    def __init__(
        self,
        lm_hidden_size=768,
        max_text_length=16,
        pretrained_text_encoder_ckpt=None,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs, encoder_channels=lm_hidden_size)

        self.max_text_length = max_text_length

        # Load and freeze LM
        self.tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
        self.lm = RobertaModel.from_pretrained("roberta-base")

        for param in self.lm.parameters():
            param.requires_grad = False

        self.transformer_proj = nn.Linear(lm_hidden_size, self.model_channels * 4)

    def get_text_emb(self, texts, device):
        self.lm.eval()  # Prevent dropout from yielding different values.
        encoded_input = self.tokenizer(
            texts,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.max_text_length,
        ).to(device)
        text_embedding = self.lm(**encoded_input)
        text_pooler_out = text_embedding["pooler_output"]
        lm_proj = self.transformer_proj(text_pooler_out)
        hidden_state_out = text_embedding["last_hidden_state"].permute(
            0, 2, 1
        )  # NLC -> NCL
        outputs = {"lm_proj": lm_proj, "lm_out": hidden_state_out}
        return outputs

    def forward(self, x, timesteps, texts):
        hs = []
        emb = self.time_embed(timestep_embedding(timesteps, self.model_channels))

        text_outputs = self.get_text_emb(texts, device=x.device)
        lm_proj, lm_out = text_outputs["lm_proj"], text_outputs["lm_out"]
        emb = emb + lm_proj.to(emb)

        # Below same
        h = x.type(self.dtype)
        for module in self.input_blocks:
            h = module(h, emb, lm_out)
            hs.append(h)
        h = self.middle_block(h, emb, lm_out)
        for module in self.output_blocks:
            h = th.cat([h, hs.pop()], dim=1)
            h = module(h, emb, lm_out)
        h = h.type(x.dtype)
        h = self.out(h)
        return h
