import torch
from typing import override
from torch import Tensor, nn
from transformers.models.modernbert.modeling_modernbert import (
    ModernBertConfig,
    ModernBertPredictionHead,
    ModernBertPreTrainedModel,
    ModernBertModel,
)


class ModernBertForMaskedLMWithRewardHead(ModernBertPreTrainedModel):
    _tied_weights_keys = ["decoder.weight"]

    def __init__(self, config: ModernBertConfig):
        super().__init__(config)
        self.config = config
        self.model = ModernBertModel(config)
        self.head = ModernBertPredictionHead(config)
        self.decoder = nn.Linear(config.hidden_size, config.vocab_size, bias=config.decoder_bias)
        self.reward_head = ModernBertPredictionHead(config)
        self.reward_drop = nn.Dropout(config.classifier_dropout)
        self.reward_output = nn.Linear(config.hidden_size, 1)

        self.sparse_prediction = self.config.sparse_prediction
        self.sparse_pred_ignore_index = self.config.sparse_pred_ignore_index

        # Initialize weights and apply final processing
        self.post_init()

    @override
    def _init_weights(self, module: nn.Module):
        if isinstance(module, ModernBertForMaskedLMWithRewardHead):
            cutoff_factor = self.config.initializer_cutoff_factor
            if cutoff_factor is None:
                cutoff_factor = 3
            std = self.config.hidden_size**-0.5
            nn.init.trunc_normal_(
                module.reward_output.weight, mean=0.0, std=std, a=-cutoff_factor * std, b=cutoff_factor * std
            )
            nn.init.zeros_(module.reward_output.bias)
        else:
            super()._init_weights(module)

    def get_output_embeddings(self):
        return self.decoder

    def set_output_embeddings(self, new_embeddings: nn.Linear):
        self.decoder = new_embeddings

    @torch.compile(dynamic=True)
    def compiled_lm_loss(self, output: Tensor, labels: Tensor) -> Tensor:
        return self.loss_function(self.decoder(self.head(output)), labels, vocab_size=self.config.vocab_size)

    @torch.compile(dynamic=True)
    def compiled_lm_head(self, output: Tensor) -> Tensor:
        return self.decoder(self.head(output))

    @torch.compile(dynamic=True)
    def compiled_reward_head(self, output: Tensor) -> Tensor:
        return self.reward_output(self.reward_drop(self.reward_head(output))).squeeze()

    # def init_reward_head(self):
    #     self.reward_head.load_state_dict(self.head.state_dict())

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor | None = None,
        labels: Tensor | None = None,
        cu_seqlens: Tensor | None = None,
        max_seqlen: int | None = None,
        batch_size: int | None = None,
        seq_len: int | None = None,
        mask_length: int = 0,
    ):
        """Override to compute reward at the model ID token."""
        self._maybe_set_compile()

        # Get base model outputs
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
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
            model_id_indices = cu_seqlens[1:] - mask_length - 2
            model_id_hidden_state = last_hidden_state[model_id_indices, :]  # shape=(batch_size, hidden_size)
            max_response_len = labels.shape[1] if labels is not None and self.training else mask_length
            last_hidden_state = last_hidden_state[
                model_id_indices.unsqueeze(1)
                + torch.arange(1, max_response_len + 1, 1, device=model_id_indices.device),
                :,
            ]  # shape=(batch_size, max_response_len, hidden_size)
        else:
            model_id_hidden_state = last_hidden_state[:, -mask_length - 2, :]  # shape=(batch_size, hidden_size)
            if labels is not None:
                last_hidden_state = last_hidden_state[:, -mask_length - 1 : -mask_length - 1 + labels.shape[1], :]
            else:
                last_hidden_state = last_hidden_state[:, -mask_length - 1 :, :]

        # Compute LM logits and loss
        if self.sparse_prediction and labels is not None and self.training:
            # flatten labels and output first
            labels = labels.view(-1)
            last_hidden_state = last_hidden_state.view(-1, last_hidden_state.shape[-1])

            # then filter out the non-masked tokens
            mask_tokens = labels != self.sparse_pred_ignore_index
            last_hidden_state = last_hidden_state[mask_tokens]
            labels = labels[mask_tokens]

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
        routing_logits = self.compiled_reward_head(model_id_hidden_state)
        return loss, routing_logits, output_token_ids
