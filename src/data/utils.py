from typing import Literal
from transformers import PreTrainedTokenizerBase


def format_system(model_ids: list[str], data_template: dict[str, str]) -> str:
    model_ids_str = ", ".join([f"<{model_id}>" for model_id in model_ids])
    return data_template["system"].format(n_models=len(model_ids), model_ids=model_ids_str)


def format_input(instruction: str, data_template: dict[str, str]) -> str:
    return data_template["input"].format(instruction=instruction)


def format_output(response: str, model_id: str, data_template: dict[str, str]) -> str:
    response = response.strip()
    if data_template is not None:
        response = data_template["output"].format(response=response, model_id=model_id)
    return response


def get_model_id(index: int, id_type: Literal["number", "letter", "special", "unused"] = "letter"):
    if id_type == "number":
        return str(index)
    elif id_type == "letter":
        return chr(ord("A") + index)
    elif id_type == "special":
        return f"<Model_{index}>"
    elif id_type == "unused":
        return f"[unused{index}]"


def find_model_id(token_ids, tokenizer: PreTrainedTokenizerBase, n_models: int) -> int | None:
    """Find the position of a model id in the token ids.

    Args:
        token_ids: Input token ids.
        tokenizer: The tokenizer.
        n_models: Number of models.

    Returns:
        The position of the model id in the token ids which is negative to index from the end.
    """
    tokens = tokenizer.convert_ids_to_tokens(token_ids)  # type: ignore
    model_ids = [get_model_id(i) for i in range(n_models)]
    for i in range(-1, -len(tokens) - 1, -1):
        if tokens[i] in model_ids:
            return i
    return None
