import torch
from typing import override
from torch import Tensor, nn
from transformers.models.modernbert.modeling_modernbert import (
    ModernBertConfig,
    ModernBertPredictionHead,
    ModernBertModel,
)

from .mask import ModernBertForMaskedLMWithRewardHead


class ModernBertForMaskedConcatLMWithRewardHead(ModernBertForMaskedLMWithRewardHead):
    def __init__(self, config: ModernBertConfig):
        super().__init__(config)
        self.loss_type = "ForMaskedLM"
        self.config = config
        self.n_candidates = config.num_labels
        self.model = ModernBertModel(config)
        self.head = ModernBertPredictionHead(config)
        self.decoder = nn.Linear(config.hidden_size, config.vocab_size, bias=config.decoder_bias)
        self.reward_head = ModernBertPredictionHead(config)
        self.reward_drop = nn.Dropout(config.classifier_dropout)
        self.reward_output = nn.Linear(config.hidden_size, self.n_candidates)

        self.sparse_prediction = self.config.sparse_prediction
        self.sparse_pred_ignore_index = self.config.sparse_pred_ignore_index

        # Initialize weights and apply final processing
        self.post_init()

    @override
    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor | None = None,
        position_ids: Tensor | None = None,
        labels: Tensor | None = None,
        cu_seqlens: Tensor | None = None,
        max_seqlen: int | None = None,
        batch_size: int | None = None,
        seq_len: int | None = None,
        seq_lens: Tensor | None = None,
        mask_length: int = 0,
    ):
        """Override to compute reward at the model ID token."""
        self._maybe_set_compile()

        # Get base model outputs
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
            batch_size=batch_size,
            seq_len=seq_len,
            return_dict=True,
        )
        last_hidden_state = outputs["last_hidden_state"]

        # Clip the last hidden state
        if self.config._attn_implementation == "flash_attention_2":
            assert cu_seqlens is not None, "cu_seqlens must be provided for flash attention 2"
            cls_indices = cu_seqlens[:-1]
            cls_hidden_state = last_hidden_state[cls_indices, :]  # shape=(batch_size, hidden_size)
            max_response_len = (1 + mask_length) * self.n_candidates
            response_start_indices = cu_seqlens[1:] - max_response_len
            last_hidden_state = last_hidden_state[
                response_start_indices.unsqueeze(1)
                + torch.arange(max_response_len, device=response_start_indices.device),
                :,
            ]  # shape=(batch_size, max_response_len, hidden_size)
        else:
            assert seq_lens is not None and batch_size is not None
            cls_hidden_state = last_hidden_state[
                torch.arange(batch_size, device=last_hidden_state.device), -seq_lens, :
            ]  # shape=(batch_size, hidden_size)
            last_hidden_state = last_hidden_state[:, -(mask_length + 1) * self.n_candidates :, :]

        # Compute LM logits and loss
        # if self.sparse_prediction and labels is not None and self.training:
        #     # flatten labels and output first
        #     labels = labels.view(-1)
        #     last_hidden_state = last_hidden_state.view(-1, last_hidden_state.shape[-1])

        #     # then filter out the non-masked tokens
        #     mask_tokens = labels != self.sparse_pred_ignore_index
        #     last_hidden_state = last_hidden_state[mask_tokens]
        #     labels = labels[mask_tokens]

        loss = None
        if self.training:  # Do not compute logits during training
            output_token_ids = None
            if labels is not None:
                loss = self.compiled_lm_loss(last_hidden_state, labels)
        else:  # Compute logits during inference
            lm_logits = self.compiled_lm_head(last_hidden_state)  # shape=(batch_size, max_response_len, vocab_size)
            output_token_ids = lm_logits.argmax(dim=-1)  # shape=(batch_size, max_response_len)
            if labels is not None:
                loss = self.loss_function(lm_logits, labels, vocab_size=self.config.vocab_size)

        # Compute routing logits
        routing_logits = self.compiled_reward_head(cls_hidden_state)
        return loss, routing_logits, output_token_ids
