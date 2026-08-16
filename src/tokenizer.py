"""Tokenizer access.

For a real run, `get_tokenizer("Qwen/Qwen2.5-0.5B-Instruct")` (or any HF hub
name, e.g. "meta-llama/Llama-3.2-1B-Instruct") just returns the model's own
AutoTokenizer.

For the offline smoke test (no internet / no GPU, used to sanity-check this
whole pipeline before you run it for real), `get_tokenizer("tiny-debug")`
returns a minimal character-level tokenizer built from a fixed vocabulary, so
nothing needs to be downloaded. "tiny-debug" is the same sentinel model name
used in configs/smoke_test.yaml and src/model.py.
"""
from typing import List

SPECIAL_TOKENS = ["<PAD>", "<BOS>", "<EOS>"]
# Every character that can appear in a prompt or completion built by
# src/prompts.py for grid sizes up to ~20. Extend this if you change the
# prompt template or allow bigger grids.
BASE_CHARS = list(
    " \n#.SGUDLR0123456789():xX<>_/'.,"
    "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
)


class CharTokenizer:
    """Minimal char-level tokenizer with a HF-tokenizer-like interface
    (encode / decode / pad_token_id / eos_token_id) so training/eval code can
    treat it the same way as a real AutoTokenizer."""

    def __init__(self):
        vocab = SPECIAL_TOKENS + sorted(set(BASE_CHARS))
        self.token_to_id = {tok: i for i, tok in enumerate(vocab)}
        self.id_to_token = {i: tok for tok, i in self.token_to_id.items()}
        self.pad_token_id = self.token_to_id["<PAD>"]
        self.bos_token_id = self.token_to_id["<BOS>"]
        self.eos_token_id = self.token_to_id["<EOS>"]
        self.vocab_size = len(vocab)
        self.unk_id = self.pad_token_id  # unseen chars silently map to PAD

    def encode(self, text: str, add_eos: bool = False) -> List[int]:
        ids = [self.token_to_id.get(ch, self.unk_id) for ch in text]
        if add_eos:
            ids = ids + [self.eos_token_id]
        return ids

    def decode(self, ids: List[int]) -> str:
        chars = []
        for i in ids:
            tok = self.id_to_token.get(int(i), "")
            if tok in SPECIAL_TOKENS:
                continue
            chars.append(tok)
        return "".join(chars)

    def __call__(self, text, **kwargs):
        return {"input_ids": self.encode(text)}


def get_tokenizer(model_name: str):
    if model_name == "tiny-debug":
        return CharTokenizer()
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok
