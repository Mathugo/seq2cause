import torch
from transformers import LlamaConfig, LlamaForCausalLM

from seq2cause.adapters import HFModelAdapter


def _tiny_llama(vocab_size: int) -> LlamaForCausalLM:
    config = LlamaConfig(
        vocab_size=vocab_size, hidden_size=16, intermediate_size=32,
        num_hidden_layers=1, num_attention_heads=2, max_position_embeddings=16,
    )
    return LlamaForCausalLM(config).eval()


def test_hf_model_adapter_infers_vocab_size_from_model_config():
    model = _tiny_llama(vocab_size=12)
    adapter = HFModelAdapter(model)
    assert adapter.vocab_size == 12


def test_hf_model_adapter_accepts_explicit_vocab_size_override():
    model = _tiny_llama(vocab_size=12)
    adapter = HFModelAdapter(model, vocab_size=99)
    assert adapter.vocab_size == 99


def test_hf_model_adapter_flattens_and_restores_leading_dims():
    vocab_size, seq_len = 10, 6
    model = _tiny_llama(vocab_size)
    adapter = HFModelAdapter(model, vocab_size=vocab_size)

    # Arbitrary extra leading dims, as do_interventions/diagnostics produce.
    input_ids = torch.randint(0, vocab_size, (2, 3, 4, seq_len))
    out = adapter(input_ids=input_ids)

    assert "logits" in out
    assert out["logits"].shape == (2, 3, 4, seq_len, vocab_size)


def test_hf_model_adapter_call_matches_forward():
    vocab_size, seq_len = 10, 6
    model = _tiny_llama(vocab_size)
    adapter = HFModelAdapter(model, vocab_size=vocab_size)
    input_ids = torch.randint(0, vocab_size, (2, seq_len))

    out_call = adapter(input_ids=input_ids)
    out_forward = adapter.forward(input_ids=input_ids)
    assert torch.allclose(out_call["logits"], out_forward["logits"])
