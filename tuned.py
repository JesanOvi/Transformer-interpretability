import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from transformer_lens import HookedTransformer


class SimpleTunedLens(nn.Module):
    def __init__(self, d_model, vocab_size, n_layers):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.Linear(d_model, vocab_size) for _ in range(n_layers)
        ])

    def forward(self, hidden_states):
        logits_per_layer = []
        for i, h in enumerate(hidden_states):
            logits = self.layers[i](h)
            logits_per_layer.append(logits)
        return logits_per_layer


class TunedLensRunner:
    def __init__(self, model_name="EleutherAI/pythia-14m", top_k=5, device="cpu"):
        self.model_name = model_name
        self.top_k = top_k
        self.device = device

        # Load model
        self.model = HookedTransformer.from_pretrained(model_name).to(device)

        # Config
        self.d_model = self.model.cfg.d_model
        self.vocab_size = self.model.cfg.d_vocab
        self.n_layers = self.model.cfg.n_layers + 1  # + embedding

        # Lens
        self.tuned_lens = SimpleTunedLens(
            self.d_model, self.vocab_size, self.n_layers
        ).to(device)

    def _get_hidden_states(self, cache):
        hidden_states = [cache["hook_embed"]]
        for layer in range(self.model.cfg.n_layers):
            hidden_states.append(cache[f"blocks.{layer}.hook_resid_post"])
        return hidden_states

    def train(self, prompt, epochs=10, lr=1e-3):
        tokens = self.model.to_tokens(prompt).to(self.device)
        logits, cache = self.model.run_with_cache(tokens)

        hidden_states = self._get_hidden_states(cache)

        optimizer = torch.optim.Adam(self.tuned_lens.parameters(), lr=lr)
        self.tuned_lens.train()

        for epoch in range(epochs):
            total_loss = 0

            for i, h in enumerate(hidden_states):
                pred = self.tuned_lens([h])[0]  # logits

                seq_len = pred.shape[1]
                pred_shift = pred[:, :-1, :]
                targets_shift = tokens[:, 1:seq_len]

                pred_flat = pred_shift.reshape(-1, self.vocab_size)
                targets_flat = targets_shift.reshape(-1)

                loss = F.cross_entropy(pred_flat, targets_flat)
                total_loss += loss

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

        print(f" Training done. Final loss: {total_loss.item():.4f}")

        return tokens, logits, cache

    def run(self, prompt):
        tokens, logits, cache = self.train(prompt)

        hidden_states = self._get_hidden_states(cache)

        self.tuned_lens.eval()

        print("\n=== Tuned Lens Predictions ===")

        layers_probs = []
        layer_names = []

        for layer_idx, h in enumerate(hidden_states):
            with torch.no_grad():
                logits_layer = self.tuned_lens([h])[0]
                last_token_logits = logits_layer[0, -1]

                probs = F.softmax(last_token_logits, dim=-1)

                top_vals, top_idx = torch.topk(last_token_logits, self.top_k)
                top_tokens = self.model.to_str_tokens(top_idx)

                layers_probs.append(probs[top_idx].cpu().numpy())
                layer_names.append(f"Layer {layer_idx}")

                print(f"\nLayer {layer_idx}:")
                for t, v in zip(top_tokens, top_vals):
                    print(f"  {t} : {v.item():.4f}")

        # Visualization
        self._plot_confidence(hidden_states)

    def _plot_confidence(self, hidden_states):
        probs_across_layers = []

        for h in hidden_states:
            with torch.no_grad():
                logits_layer = self.tuned_lens([h])[0]
                probs = F.softmax(logits_layer[0, -1], dim=-1)
                probs_across_layers.append(probs.max().item())

        plt.plot(range(len(probs_across_layers)), probs_across_layers, marker='o')
        plt.xlabel("Layer")
        plt.ylabel("Top-1 Probability")
        plt.title("Tuned Lens Confidence Across Layers")
        plt.show()