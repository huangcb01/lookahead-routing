import torch
import transformers
from torch import Tensor, LongTensor, FloatTensor, nn
from transformers import AutoConfig
from transformers.modeling_outputs import CausalLMOutputWithPast
from liger_kernel.transformers.fused_linear_cross_entropy import LigerFusedLinearCrossEntropyLoss


def load_model_with_reward_head(pretrained_model_name_or_path: str, *args, use_fixed_liger: bool = False, **kwargs):
    config = AutoConfig.from_pretrained(pretrained_model_name_or_path)
    model_class_name = config.architectures[0]
    if model_class_name == "AutoModelForCausalLMWithRewardHead":
        model_class_name = config.original_model_class
    model_class = getattr(transformers, model_class_name)

    class AutoModelForCausalLMWithRewardHead(model_class):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.config.original_model_class = model_class_name
            if not hasattr(config, "summary_dropout_prob"):
                summary_dropout_prob = kwargs.pop("summary_dropout_prob", 0.1)
            else:
                summary_dropout_prob = config.summary_dropout_prob
            self.dropout = nn.Dropout(summary_dropout_prob) if summary_dropout_prob else nn.Identity()
            self.summary = nn.Linear(self.config.hidden_size, self.config.num_labels)

        def forward(
            self,
            input_ids: Tensor,
            model_token_indices: Tensor,
            attention_mask: Tensor | None = None,
            return_past_key_values: bool = False,
            logits_to_keep: int | None = None,
            **kwargs,
        ):
            """Override to compute reward at the model ID token."""
            kwargs["output_hidden_states"] = True  # this had already been set in the LORA / PEFT examples

            if use_fixed_liger:
                base_model_output = lce_forward(self, input_ids, attention_mask=attention_mask, **kwargs)
            else:
                base_model_output = super().forward(input_ids=input_ids, attention_mask=attention_mask, **kwargs)

            last_hidden_state: Tensor = base_model_output.hidden_states[-1]
            model_id_hidden_state = last_hidden_state[torch.arange(last_hidden_state.shape[0]), model_token_indices, :]
            lm_logits = base_model_output.logits
            loss = base_model_output.loss

            model_id_hidden_state = self.dropout(model_id_hidden_state)
            if model_id_hidden_state.dtype != self.summary.weight.dtype:
                model_id_hidden_state = model_id_hidden_state.to(self.summary.weight.dtype)
            routing_logits = self.summary(model_id_hidden_state).squeeze()
            return lm_logits, loss, routing_logits

    return AutoModelForCausalLMWithRewardHead.from_pretrained(pretrained_model_name_or_path, *args, **kwargs)


def lce_forward(
    self,
    input_ids: Tensor,
    attention_mask: Tensor | None = None,
    position_ids: LongTensor | None= None,
    past_key_values: list[FloatTensor] | None = None,
    inputs_embeds: FloatTensor | None = None,
    labels: LongTensor | None = None,
    use_cache: bool | None = None,
    output_attentions: bool | None = None,
    output_hidden_states: bool | None = None,
    return_dict: bool | None = None,
    cache_position: LongTensor | None = None,
    num_logits_to_keep: int = 0,
    **loss_kwargs,
) -> CausalLMOutputWithPast:
    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
    outputs = self.model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        use_cache=use_cache,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=return_dict,
        cache_position=cache_position,
    )

    hidden_states = outputs[0]

    loss = None
    # Don't materialize logits
    if labels is not None:
        # We do the same thing as ForCausalLMLoss but using Liger FLCE

        shift_hidden_states = hidden_states[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        # flatten tokens
        shift_hidden_states = shift_hidden_states.view(-1, self.config.hidden_size)
        shift_labels = shift_labels.view(-1)
        reduction = "sum" if "num_items_in_batch" in loss_kwargs and loss_kwargs["num_items_in_batch"] else "mean"
        lce = LigerFusedLinearCrossEntropyLoss(reduction=reduction)
        loss = lce(self.lm_head.weight, shift_hidden_states, shift_labels)
        if reduction == "sum":
            loss /= loss_kwargs["num_items_in_batch"]

    return CausalLMOutputWithPast(
        loss=loss,
        logits=None,
        past_key_values=outputs.past_key_values,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
    )
