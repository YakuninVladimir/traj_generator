from __future__ import annotations

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_model(model_name: str, device: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # If tokenizer has no pad token, use EOS.
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Padding side must be left for correct generation with padding batches.
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        trust_remote_code=True,
    ).to(device)

    model.config.pad_token_id = tokenizer.pad_token_id
    model.eval()
    return tokenizer, model


@torch.no_grad()
def generate_texts(
    tokenizer,
    model,
    prompts: list[str],
    *,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
):
    inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)
    input_length = inputs.input_ids.shape[1]

    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=temperature,
        top_p=top_p,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    generated_tokens = outputs[:, input_length:]
    results = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
    return [text.strip() for text in results]

