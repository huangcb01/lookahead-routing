import transformers
from torch import nn, Tensor
from transformers import AutoConfig
from transformers.modeling_outputs import ModelOutput


class RouterWithLMHeadOutput(ModelOutput):
    loss: Tensor | None = None
    routing_logits: Tensor
    token_logits: Tensor


def load_router_with_lm_head(
    pretrained_model_name_or_path: str, n_candidates: int, **kwargs
):
    config = AutoConfig.from_pretrained(pretrained_model_name_or_path)
    model_class_name = config.architectures[0]
    if model_class_name == "AutoRouterWithLMHead":
        model_class_name = config.original_model_class
    model_class = getattr(transformers, model_class_name)

    class AutoRouterWithLMHead(model_class):
        def __init__(self, *args, **kwargs):
            self.n_candidates = getattr(config, "n_candidates", kwargs.pop("n_candidates", 1))
            super().__init__(*args, **kwargs)
            self.score = nn.Linear(config.hidden_size, self.n_candidates, bias=False)
            self.config.original_model_class = model_class_name

        def forward(
            self,
            input_ids: Tensor,
            attention_mask: Tensor | None = None,
        ):
            # Second forward pass for the responses
            outputs = super().forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                return_dict=True,
                logits_to_keep=1,      # Keep only the last token logits
                num_logits_to_keep=1,  # For compatibility with liger-kernel
            )
            token_logits = outputs.logits.squeeze(1)
            hidden_state = outputs.hidden_states[-1][:, -1, :]
            routing_logits = self.score(hidden_state)

            return None, routing_logits, token_logits

    return AutoRouterWithLMHead.from_pretrained(
        pretrained_model_name_or_path, n_candidates=n_candidates, **kwargs
    )
