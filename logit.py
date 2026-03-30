import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
from transformer_lens import HookedTransformer


class LogitLens:
    def __init__(self, model_name="EleutherAI/pythia-14m", top_k=5, device="cpu"):
        self.model_name = model_name
        self.top_k = top_k
        self.device = device

        self.model = HookedTransformer.from_pretrained(model_name).to(device)
        self.W_U = self.model.unembed.W_U

    def _analyze_layer(self, resid):
        """
        Compute logits, probabilities, and top-k tokens for a layer
        """
        layer_logits = resid @ self.W_U
        probs = F.softmax(layer_logits, dim=-1)[0, -1]

        last_token_logits = layer_logits[0, -1]
        topk_values, topk_indices = torch.topk(last_token_logits, self.top_k)
        topk_tokens = self.model.to_str_tokens(topk_indices)

        return topk_tokens, topk_values, probs[topk_indices].detach().cpu().numpy()

    def run(self, prompt):
        print(f"\n==============================")
        print(f"Prompt: {prompt}")
        print(f"Model: {self.model_name}")
        print(f"==============================\n")

        tokens = self.model.to_tokens(prompt).to(self.device)
        logits, cache = self.model.run_with_cache(tokens)

        layers_probs = []
        layer_names = []

        print("=== Logit Lens Predictions ===")

        # Embedding Layer
        resid = cache["hook_embed"]
        tokens_k, values_k, probs_k = self._analyze_layer(resid)

        layers_probs.append(probs_k)
        layer_names.append("Embedding")

        print("\nLayer 0 (Embedding):")
        for t, v in zip(tokens_k, values_k):
            print(f"  {t} : {v.item():.4f}")

        # Transformer Layers
        for layer in range(self.model.cfg.n_layers):
            resid = cache[f"blocks.{layer}.hook_resid_post"]
            tokens_k, values_k, probs_k = self._analyze_layer(resid)

            layers_probs.append(probs_k)
            layer_names.append(f"Layer {layer+1}")

            print(f"\nLayer {layer+1}:")
            for t, v in zip(tokens_k, values_k):
                print(f"  {t} : {v.item():.4f}")

        # Final Layer (True Output)
        final_logits = logits[0, -1]
        probs = F.softmax(final_logits, dim=-1)

        topk_values, topk_indices = torch.topk(final_logits, self.top_k)
        topk_tokens = self.model.to_str_tokens(topk_indices)

        layers_probs.append(probs[topk_indices].detach().cpu().numpy())
        layer_names.append("Final")

        print("\nFinal Layer (Model Output):")
        for t, v in zip(topk_tokens, topk_values):
            print(f"  {t} : {v.item():.4f}")

        # Heatmap Visualization
        self._plot_heatmap(layers_probs, layer_names)

        # Confidence Plot
        self._plot_confidence(cache, logits)

    def _plot_heatmap(self, layers_probs, layer_names):
        plt.figure(figsize=(8, 6))
        sns.heatmap(layers_probs, annot=True, fmt=".2f",
                    yticklabels=layer_names)
        plt.xlabel("Top-k Tokens (Final Layer)")
        plt.ylabel("Layers")
        plt.title("Logit Lens: Top-k Probability Evolution")
        plt.tight_layout()
        plt.show()

    def _plot_confidence(self, cache, logits):
        top1_probs = []

        # Embedding
        top1_probs.append(
            F.softmax(cache["hook_embed"] @ self.W_U, dim=-1)[0, -1].max().item()
        )

        # Transformer layers
        for layer in range(self.model.cfg.n_layers):
            resid = cache[f"blocks.{layer}.hook_resid_post"]
            prob = F.softmax(resid @ self.W_U, dim=-1)[0, -1].max().item()
            top1_probs.append(prob)

        # Final
        final_prob = F.softmax(logits[0, -1], dim=-1).max().item()
        top1_probs.append(final_prob)

        plt.figure(figsize=(6, 4))
        plt.plot(range(len(top1_probs)), top1_probs, marker='o')
        plt.xlabel("Layer")
        plt.ylabel("Top-1 Probability")
        plt.title("Confidence Across Layers")
        plt.tight_layout()
        plt.show()