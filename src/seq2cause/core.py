import torch
from accelerate import Accelerator
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from seq2cause.causal_strength import (
    calc_granger_score,
    calc_lag_cmi,
    calc_lag_info_gain,
    calc_neural_saliency,
    calc_neural_shapley,
)
from seq2cause.kl import DEFAULT_KL_MODE, check_kl_mode
from seq2cause.sampling import (
    ancestral_sampling,
    do_interventions,
    model_sample_excluding_observed,
    multinomial_sample,
    uniform_sample,
)
from seq2cause.utils import (
    check_memory_budget,
    estimate_tensor_bytes,
    format_bytes,
    next_token_collate,
)

# How the past before the tested tokens is built (see SampleLevelCausalDiscovery).
CONTEXT_MODES = ("overwrite", "prepend", "none")


class SampleLevelCausalDiscovery:
    """
    This class implements sample-level causal discovery algorithm.
    When given a batch of discrete sequence of events, e.g., "A B C D",
    it identifies the sample time and summary causal graph per sequence.

    Attributes:
        tfx (any): The autoregressive model used to compute next-token probabilities.
        params (dict): A dictionary of parameters for sampling and causal discovery.
            `params["kl_mode"]` (`"logspace"`, the default, or `"clamp"`) selects the
            Bernoulli-KL algebra of `calc_lag_info_gain` (see `seq2cause.kl`), and an
            optional dict at `params["kl_stats"]` accumulates the corrupted-cell counts.
            `params["causal_strength"] = "CMI"` scores edges with the conditional mutual
            information (`calc_lag_cmi`, bounded by log 2) instead of the information gain of
            the observed cause. `params["sampling"]["context_mode"]` sets how the past before
            the tested tokens is built (with g = `guidance` and c = `context`):
              "overwrite" (default, v0.1.9 behaviour): the first g observed tokens are kept,
                  positions g..c-1 are overwritten by tokens sampled from the model, and only
                  the observed tokens c..L-1 are tested -> adjacency [bs, L-c, L-c];
              "prepend": the first g observed tokens (e.g. CLS) are kept as an anchor, c - g
                  tokens are sampled from the model after them, and ALL remaining observed
                  tokens g..L-1 follow, so nothing is overwritten -> adjacency [bs, L-g, L-g].
                  For sequences without a CLS token use g = 0: the past then starts with a
                  uniformly random token (ids >= `noise_min_id`), c - 1 tokens are sampled
                  after it, and every observed token, including the first one, is kept;
              "none": no sampled past at all, the anchor g >= 1 is followed by all observed
                  tokens (same as "prepend" with c = g) -> adjacency [bs, L-g, L-g].
            Entry [b, i, j] of the adjacency refers to observed positions s0 + i -> s0 + j, with
            s0 = c for "overwrite" and s0 = g otherwise (`self.first_position(...)`). As in every
            mode, row 0 stays empty: the staircase has no row in which the first tested token is
            replaced, so it is scored as an effect but not as a cause.
        ds_test (any): The test dataset containing input sequences.
        logits_key_of_output_tfx (str): The key to access logits from the model's output.
    """

    def __init__(
        self, tfx: any, params: dict, ds_test: any, logits_key_of_output_tfx: str = "logits"
    ):
        self.params = params
        self._logits_key_of_output_tfx = logits_key_of_output_tfx
        self.tfx = tfx
        self._ds_test = ds_test
        self.cls_token_id = params.get("cls_token_id", 1)

        self.proposal_fct = multinomial_sample
        self.proposal_fct_for_inter = uniform_sample

        self.sampling_type = params["sampling"].get("type", "naive")
        self.eps = params["sampling"].get("clamping", 1e-9)
        self.context = params["sampling"].get("context", 0)
        self.guidance = params["sampling"].get("guidance", 3)
        self.N = params["sampling"].get("value", 0)
        self.full = params.get("full", True)
        self.kl_mode = check_kl_mode(params.get("kl_mode", DEFAULT_KL_MODE))
        self.cause_from_model = bool(params["sampling"].get("cause_from_model", False))
        self.context_mode = params["sampling"].get("context_mode", "overwrite")
        if self.context_mode not in CONTEXT_MODES:
            raise ValueError(
                f"sampling.context_mode must be one of {CONTEXT_MODES}, got {self.context_mode!r}"
            )
        if self.context_mode == "none" and self.guidance < 1:
            raise ValueError("context_mode='none' requires sampling.guidance >= 1 (anchor token)")
        if self.context_mode == "prepend":
            if self.context < max(self.guidance, 1):
                raise ValueError(
                    "context_mode='prepend' requires sampling.context >= max(sampling.guidance, 1)"
                )
        if self.cause_from_model and self.first_position() < 1:
            raise ValueError("sampling.cause_from_model requires sampling.context >= 1")
        self.printed_max_bs = False

        strategy = params["sampling"].get("strategy", "full")
        if strategy != "full":
            raise ValueError(
                "SampleLevelCausalDiscovery drives the 'full' staircase construction only "
                f"(got strategy={strategy!r}): `calc_lag_info_gain` compares each staircase row "
                "with the previous one, which under 'atomic' are two different noised causes "
                "rather than cause-noised vs cause-observed. Use "
                "`diagnostics.compute_cmi_matrix(strategy='atomic')` instead."
            )

        print(
            f"[!] Starting the sample level causal discovery with full version {self.full} "
            f"context {self.context} cls_token_id {self.cls_token_id} context sampling {self.proposal_fct} "
            f"mediators sampling {self.proposal_fct_for_inter}"
        )
        print("[!] logits key", self._logits_key_of_output_tfx)

    def print_real_bs(self, shape_input_ids):
        if self.printed_max_bs is False:
            bs, L = shape_input_ids
            if self.params.get("full", False):
                print(
                    "[!] Real BS on GPU with sampling is ",
                    self.params["BS"]
                    * (L - self.params["sampling"]["context"])
                    * self.params["sampling"].get("value", 64),
                )
            else:
                print(
                    "[!] Real BS on GPU with sampling is ",
                    self.params["BS"] * self.params["sampling"].get("value", 64),
                )
            self.printed_max_bs = True

    def first_position(self) -> int:
        """Observed position of the first tested token (index 0 of the adjacency)."""
        return self.context if self.context_mode == "overwrite" else self.guidance

    def past_length(self) -> int:
        """Number of tokens placed before the tested observed tokens in the model input."""
        if self.context_mode == "none":
            return self.guidance
        return self.context

    def estimate_memory_bytes(self, batch_size: int, seq_len: int) -> int:
        """Rough VRAM/RAM estimate (see `utils.estimate_tensor_bytes`'s
        caveats -- this is a lower bound, not an exact figure) for the
        single biggest tensor materialized per batch: the post-intervention
        logits, shape `[batch_size, n_particles, interv_dim, seq_len,
        vocab_size]`. Returns 0 if `vocab_size` can't be determined (e.g.
        the model has no HuggingFace-style `.config.vocab_size`).
        """
        model = getattr(self.tfx, "module", self.tfx)  # unwrap DDP, if wrapped
        vocab_size = getattr(getattr(model, "config", None), "vocab_size", None)
        if vocab_size is None:
            return 0
        interv_dim = seq_len - self.first_position()
        model_len = self.past_length() + interv_dim
        return estimate_tensor_bytes(batch_size, self.N, interv_dim, model_len, vocab_size)

    def prepare(self):
        """
        Load the huggingface dataset, create a local dataloader and prepare the models for inference.
        This includes setting up the Accelerator for mixed precision inference if specified in the parameters.
        """
        print("[*] Loading test dataset ..")

        if self.params.get("fp16", False):
            print("[!] Using fp16 for inference")
            # Initialize Hugging Face's Accelerator
            self.accelerator = Accelerator(mixed_precision="fp16")  # Enables float16 computations
        else:
            self.accelerator = Accelerator()  # Enables float16 computations

        self._dl_test = DataLoader(
            self._ds_test,
            shuffle=False,
            batch_size=self.params["BS"],
            collate_fn=next_token_collate,
        )

        self.tfx, self._dl_test = self.accelerator.prepare(self.tfx, self._dl_test)
        print("[*] Done")

    def run(self):
        """
        Run the sample-level causal discovery algorithm on the test dataset.
        This is the main computation loop where we iterate over the test dataset, perform ancestral sampling,
        apply interventions, and compute causal strength measures.
        The result are local instance time causal graph per sequence in the batch via an
        adjacency matrix.

        Returns:
            A tuple containing the original batch and the computed adjacency matrix representing causal relationships,
            each concatenated over EVERY batch of the dataloader along dimension 0 (`[n_sequences, L]` /
            `[n_sequences, L-s0, L-s0]`, s0 = `self.first_position()`), so all sequences must share one length.
        """

        batches: list[dict] = []
        adjs: list[torch.Tensor] = []
        pbar = tqdm(self._dl_test, desc="CI-tests (batch x context)", unit="batch")
        for _, batch in enumerate(pbar):
            self.print_real_bs(batch["input_ids"].shape)
            bs_i, seq_len_i = batch["input_ids"].shape
            est_bytes = self.estimate_memory_bytes(bs_i, seq_len_i)
            pbar.set_postfix(
                {
                    "batch x context": f"{bs_i}x{self.context}",
                    "est. VRAM/RAM": format_bytes(est_bytes) if est_bytes else "n/a",
                }
            )
            if est_bytes and self.params.get("check_memory_budget", True):
                check_memory_budget(est_bytes, batch["input_ids"].device)

            # Causal Strength measures TODO: could be refactored to be more modular and cleaner
            cs = self.params.get("causal_strength", None)
            print("Causal Strength: ", cs)
            if cs == "Granger":
                cs = calc_granger_score
            elif cs == "CMI":
                cs = calc_lag_cmi
            elif cs == "InputXGradient":
                batch, adj = calc_neural_saliency(self.tfx, batch, self.params)
                batches.append(batch)
                adjs.append(adj)
                continue
            elif cs == "SHAPLEY":
                batch, adj = calc_neural_shapley(self.tfx, batch, self.params)
                batches.append(batch)
                adjs.append(adj)
                continue
            else:
                cs = calc_lag_info_gain

            with torch.inference_mode():
                o_b = self.tfx(attention_mask=batch["attention_mask"], input_ids=batch["input_ids"])
                hidden_states = o_b[self._logits_key_of_output_tfx]
                bs, L, d = hidden_states.shape

                prob_x = torch.nn.functional.softmax(hidden_states, dim=-1)  # p(.|z)
                mask = batch["attention_mask"]

                # Past before the tested tokens (see `context_mode` in the class docstring).
                # `ancestral_sampling` returns `[bs * N, c]` with the particle index fastest,
                # so it regroups per sequence as `[bs, N, c]`.
                s0 = self.first_position()  # first tested observed position
                c_eff = self.past_length()  # length of the past in the model input
                if self.context_mode == "none":
                    prefix_upsampled = (
                        batch["input_ids"][:, :c_eff].unsqueeze(1).repeat(1, self.N, 1)
                    )
                elif self.context_mode == "prepend" and self.guidance == 0:
                    # no anchor: one uniformly random start token per particle, then c - 1
                    # tokens sampled from the model after it
                    start = torch.randint(
                        self.params["sampling"].get("noise_min_id", 0),
                        prob_x.shape[-1],
                        (bs * self.N, 1),
                        device=batch["input_ids"].device,
                    )
                    start_batch = {
                        "input_ids": start,
                        "attention_mask": torch.ones(
                            bs * self.N, c_eff, dtype=mask.dtype, device=mask.device
                        ),
                    }
                    prefix_upsampled = ancestral_sampling(
                        self.tfx, start_batch, value=1, guidance=1, context=c_eff
                    ).reshape(bs, self.N, -1)
                else:
                    prefix_upsampled = ancestral_sampling(
                        self.tfx, batch, **self.params["sampling"]
                    )
                    prefix_upsampled = prefix_upsampled.reshape(bs, self.N, -1)
                rest = batch["input_ids"][:, s0:]  # [bs, L - s0]: the tested observed tokens
                anchor = min(s0, c_eff)
                prefix_mask = torch.cat(
                    [mask[:, :anchor], mask.new_ones(bs, c_eff - anchor)], dim=1
                )  # the sampled tokens are always attended
                ext_mask = torch.cat([prefix_mask, mask[:, s0:]], dim=1)  # [bs, c_eff + L - s0]
                L_model = ext_mask.shape[1]
                expanded_attention_mask = ext_mask.unsqueeze(1).repeat(1, self.N, 1)

                """Intervention: stairways"""
                # TODO: Upscale to (BS, N, m, L-c) if using Sparse variant,
                # (BS, N, L-c, L-c) if using Full version.
                # TODO: not tested yet, only full working for now.

                # Modify the rest of the sequence by doing interventions
                self.params["sampling"]["cls_token_id"] = None
                prop = self.proposal_fct_for_inter

                # we don't need the cls token since it's the rest
                # NOTE: `prop` (uniform_sample by default) has a narrow, explicit
                # signature (no **kwargs catch-all), unlike ancestral_sampling/
                # do_interventions below -- splatting the whole sampling dict
                # here would raise "unexpected keyword argument" for keys like
                # "value"/"guidance"/"context", so only its actual accepted
                # kwargs are forwarded.
                # `params["sampling"]["noise_min_id"]` (default 0) keeps reserved ids
                # (PAD/BOS/EOS/UNK) out of the counterfactual draw -- see `uniform_sample`.
                if self.cause_from_model:
                    # `params["sampling"]["cause_from_model"]`: replace the cause (and the
                    # mediators) by tokens drawn from the model given the observed
                    # trajectory, never the observed token, instead of uniform noise.
                    # prob_x[:, j - 1] is the distribution of the token at position j.
                    rest_upsampled_using_q = model_sample_excluding_observed(
                        prob_x[:, s0 - 1 : L - 1, :],
                        rest,
                        n_samples=self.N,
                        min_id=self.params["sampling"].get("noise_min_id", 0),
                    )
                else:
                    rest_upsampled_using_q = prop(
                        prob_x[:, s0:, :],
                        n_samples=self.N,
                        cls_token_id=self.params["sampling"]["cls_token_id"],
                        min_id=self.params["sampling"].get("noise_min_id", 0),
                    )
                rest_expanded_intervened = do_interventions(
                    rest_upsampled_using_q,
                    rest,
                    prefix_upsampled,
                    **self.params["sampling"],
                    prepend_context_back=False,
                )

                # Expand upsampled context and prepend it back to the intervened batch
                interv_dim = rest_expanded_intervened.shape[-2]  # could be L-self.context or m
                prefix_upsampled_expanded = prefix_upsampled.unsqueeze(-2).repeat(
                    1, 1, interv_dim, 1
                )
                # combine with context
                batched_sampled_complete_intervened = torch.cat(
                    [prefix_upsampled_expanded, rest_expanded_intervened], dim=-1
                )  # [bs, n, L-s0, c_eff + L - s0]

                print("[!] Tensor shape input to tfx", batched_sampled_complete_intervened.shape)

                expanded_attention_mask_interven = expanded_attention_mask.unsqueeze(-2).repeat(
                    1, 1, interv_dim, 1
                )
                # Tfx Inference
                o_b_upsampled_intervene = self.tfx(
                    attention_mask=expanded_attention_mask_interven.reshape(-1, L_model),
                    input_ids=batched_sampled_complete_intervened.reshape(-1, L_model),
                )

                o_b_upsampled_intervene = o_b_upsampled_intervene[
                    self._logits_key_of_output_tfx
                ].reshape(bs, self.N, interv_dim, L_model, -1)  # put back

                # crazy amount of operations here. The lagged information gain reads
                # LOG-probabilities in "logspace" mode (a float32 softmax saturates to
                # exactly 1.0 and breaks the v0.1.9 clamp -- see `seq2cause.kl`);
                # "clamp" mode and the Granger read-out take probabilities.
                if cs in (calc_lag_info_gain, calc_lag_cmi) and self.kl_mode == "logspace":
                    prob_x_inter = torch.nn.functional.log_softmax(o_b_upsampled_intervene, dim=-1)
                else:
                    prob_x_inter = torch.nn.functional.softmax(o_b_upsampled_intervene, dim=-1)
                print("[!] Tensor shape after intervention and inference: ", prob_x_inter.shape)

                # By this point `cs` is calc_lag_info_gain, calc_lag_cmi, or
                # calc_granger_score (the InputXGradient/SHAPLEY branches
                # above already continued) -- none takes `tfx`. They read the tested tokens
                # as `input_ids[:, context:]` of the model input, so they get the model-input
                # layout: past of length c_eff (its token values are not read) + tested tokens.
                cs_params = {
                    **self.params,
                    "sampling": {**self.params["sampling"], "context": c_eff},
                }
                cs_batch = {
                    "input_ids": torch.cat([rest.new_zeros(bs, c_eff), rest], dim=1),
                    "attention_mask": ext_mask,
                }
                adj = cs(prob_x_inter, cs_batch, cs_params)
                batches.append(batch)
                adjs.append(adj)

        # Every batch has been processed (the return used to sit inside the loop, so
        # only the first batch of the dataloader was ever scored).
        return _concat_batches(batches), torch.cat(adjs, dim=0)


def _concat_batches(batches: list[dict]) -> dict:
    """Concatenate a list of collated batch dicts along dimension 0."""
    if not batches:
        raise ValueError("SampleLevelCausalDiscovery.run(): the dataloader yielded no batch")
    return {key: torch.cat([b[key] for b in batches], dim=0) for key in batches[0]}
