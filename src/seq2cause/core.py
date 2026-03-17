import torch
from accelerate import Accelerator
from torch.utils.data import DataLoader

from src.seq2cause.causal_strength import (
    calc_granger_score,
    calc_lag_info_gain,
    calc_neural_saliency,
    calc_neural_shapley,
)
from src.seq2cause.sampling import (
    ancestral_sampling,
    do_interventions,
    multinomial_sample,
    uniform_sample,
)
from src.seq2cause.utils import next_token_collate


class SampleLevelCausalDiscovery:
    """
    This class implements sample-level causal discovery algorithm.
    When given a batch of discrete sequence of events, e.g., "A B C D",
    it identifies the sample time and summary causal graph per sequence.

    Attributes:
        tfx (any): The autoregressive model used to compute next-token probabilities.
        params (dict): A dictionary of parameters for sampling and causal discovery.
        ds_test (any): The test dataset containing input sequences.
        logits_key_of_output_tfx (str): The key to access logits from the model's output.
    """

    def __init__(
        self, tfx: any, params: dict, ds_test: any, logits_key_of_output_tfx: str = "logits"
    ):
        self._params = params
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
            A tuple containing the original batch and the computed adjacency matrix representing causal relationships.
        """

        for _, batch in enumerate(self._dl_test):
            self.print_real_bs(batch["input_ids"].shape)

            # Causal Strength measures TODO: could be refactored to be more modular and cleaner
            cs = self.params.get("causal_strength", None)
            print("Causal Strength: ", cs)
            if cs == "Granger":
                cs = calc_granger_score
            elif cs == "InputXGradient":
                return calc_neural_saliency(self.tfx, batch, self.params)
            elif cs == "SHAPLEY":
                return calc_neural_shapley(self.tfx, batch, self.params)
            else:
                cs = calc_lag_info_gain

            with torch.inference_mode():
                o_b = self.tfx(attention_mask=batch["attention_mask"], input_ids=batch["input_ids"])
                hidden_states = o_b[self._logits_key_of_output_tfx]
                bs, L, d = hidden_states.shape

                prob_x = torch.nn.functional.softmax(hidden_states, dim=-1)  # p(.|z)
                expanded_attention_mask = batch["attention_mask"].unsqueeze(1).repeat(1, self.N, 1)

                # Truncated context
                prefix_upsampled = ancestral_sampling(self.tfx, batch, **self.params["sampling"])
                if len(prefix_upsampled.size()) == 2:
                    prefix_upsampled = prefix_upsampled.unsqueeze(0)
                rest = batch["input_ids"][:, self.context :]  # [bs, L - c]

                """Intervention: stairways"""
                # TODO: Upscale to (BS, N, m, L-c) if using Sparse variant,
                # (BS, N, L-c, L-c) if using Full version.
                # TODO: not tested yet, only full working for now.

                # Modify the rest of the sequence by doing interventions
                self.params["sampling"]["cls_token_id"] = None
                prop = self.proposal_fct_for_inter

                # we don't need the cls token since it's the rest
                rest_upsampled_using_q = prop(
                    prob_x[:, self.context :, :], **self.params["sampling"]
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
                )  # [bs, n, L-c, L]

                # [bs, n, L-c or m (sparse), L]
                print("[!] Tensor shape input to tfx", batched_sampled_complete_intervened.shape)

                expanded_attention_mask_interven = expanded_attention_mask.unsqueeze(-2).repeat(
                    1, 1, interv_dim, 1
                )
                # Tfx Inference
                o_b_upsampled_intervene = self.tfx(
                    attention_mask=expanded_attention_mask_interven.reshape(-1, L),
                    input_ids=batched_sampled_complete_intervened.reshape(-1, L),
                )

                o_b_upsampled_intervene = o_b_upsampled_intervene[
                    self._logits_key_of_output_tfx
                ].reshape(bs, self.N, interv_dim, L, -1)  # put back

                # crazy amount of operations here
                prob_x_inter = torch.nn.functional.softmax(o_b_upsampled_intervene, dim=-1)
                print("[!] Tensor shape after intervention and inference: ", prob_x_inter.shape)

                adj = cs(self.tfx, prob_x_inter, batch, self.params)
                return batch, adj
