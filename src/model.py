"""Model loading + LoRA wiring.

`model_name == "tiny-debug"` builds a tiny randomly-initialized GPT-2-shaped
model with the CharTokenizer's vocab size, entirely offline -- used only to
smoke-test the training/eval pipeline on CPU without a GPU or internet
access. For a real run, pass a real HF hub model id (default in
configs/default.yaml is a ~0.5B open model; swap to
"meta-llama/Llama-3.2-1B-Instruct" for a true ~1B model, which requires
accepting Meta's license on huggingface.co and an HF token).
"""
from typing import List, Optional

from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, GPT2Config, GPT2LMHeadModel


def load_base_model(model_name: str, tiny_vocab_size: Optional[int] = None):
    if model_name == "tiny-debug":
        assert tiny_vocab_size is not None, "tiny-debug requires tiny_vocab_size"
        config = GPT2Config(
            vocab_size=tiny_vocab_size,
            n_positions=768,
            n_embd=96,
            n_layer=3,
            n_head=3,
            bos_token_id=1,
            eos_token_id=2,
        )
        return GPT2LMHeadModel(config)
    return AutoModelForCausalLM.from_pretrained(model_name)


def guess_target_modules(model) -> List[str]:
    """LoRA target module names vary by architecture (Llama: q_proj/v_proj/...,
    GPT-2: c_attn/c_proj, ...). Auto-detect from whatever linear submodule
    names actually exist in the loaded model."""
    candidates = {
        "q_proj", "k_proj", "v_proj", "o_proj",  # Llama / Qwen / Mistral family
        "c_attn", "c_proj",                       # GPT-2 family
        "query_key_value",                        # GPT-NeoX / Falcon family
    }
    found = set()
    for name, _ in model.named_modules():
        leaf = name.split(".")[-1]
        if leaf in candidates:
            found.add(leaf)
    return sorted(found) if found else ["c_attn"]


def apply_lora(model, r=16, alpha=32, dropout=0.05, target_modules=None):
    if not target_modules:
        target_modules = guess_target_modules(model)
    config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        target_modules=target_modules,
    )
    model = get_peft_model(model, config)
    model.print_trainable_parameters()
    return model
