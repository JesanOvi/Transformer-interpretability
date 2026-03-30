import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
from transformer_lens import HookedTransformer


def run_logit_lens(prompt, model_name="EleutherAI/pythia-14m", top_k=5):
    print(f"\nPrompt: {prompt}\n")

    # Load model
    model = HookedTransformer.from_pretrained(model_name)

    # Tokenize
    tokens = model.to_tokens(prompt)

    # Run model
    logits, cache = model.run_with_cache(tokens)

    # Unembedding matrix
    W_U = model.unembed.W_U

    layers_probs = []
    layer_names = []

    print("\n=== Logit Lens Predictions ===")

    # 🔹 Embedding layer
    resid = cache["hook_embed"]
    layer_logits = resid @ W_U
    probs = F.softmax(layer_logits, dim=-1)[0, -1]

    topk_values, topk_indices = torch.topk(layer_logits[0, -1], top_k)
    topk_tokens = model.to_str_tokens(topk_indices)

    layers_probs.append(probs[topk_indices].detach().cpu().numpy())
    layer_names.append("Embedding")

    print("\nLayer 0 (Embedding):")
    for t, v in zip(topk_tokens, topk_values):
        print(f"  {t} : {v.item():.4f}")

    # 🔹 Transformer layers
    for layer in range(model.cfg.n_layers):
        resid = cache[f"blocks.{layer}.hook_resid_post"]
        layer_logits = resid @ W_U
        probs = F.softmax(layer_logits, dim=-1)[0, -1]

        topk_values, topk_indices = torch.topk(layer_logits[0, -1], top_k)
        topk_tokens = model.to_str_tokens(topk_indices)

        layers_probs.append(probs[topk_indices].detach().cpu().numpy())
        layer_names.append(f"Layer {layer+1}")

        print(f"\nLayer {layer+1}:")
        for t, v in zip(topk_tokens, topk_values):
            print(f"  {t} : {v.item():.4f}")

    # 🔹 Final output
    final_logits = logits[0, -1]
    probs = F.softmax(final_logits, dim=-1)

    topk_values, topk_indices = torch.topk(final_logits, top_k)
    topk_tokens = model.to_str_tokens(topk_indices)

    layers_probs.append(probs[topk_indices].detach().cpu().numpy())
    layer_names.append("Final")

    print("\nFinal Layer (Model Output):")
    for t, v in zip(topk_tokens, topk_values):
        print(f"  {t} : {v.item():.4f}")

    # 🔹 Heatmap
    sns.heatmap(layers_probs, annot=True, fmt=".2f",
                yticklabels=layer_names)
    plt.xlabel("Top-k Tokens (Final Layer)")
    plt.ylabel("Layers")
    plt.title("Logit Lens: Top-k Probability Evolution")
    plt.show()

    # 🔹 Confidence plot
    top1_probs = []

    # embedding
    top1_probs.append(F.softmax(cache["hook_embed"] @ W_U, dim=-1)[0, -1].max().item())

    # layers
    for layer in range(model.cfg.n_layers):
        resid = cache[f"blocks.{layer}.hook_resid_post"]
        top1_probs.append(F.softmax(resid @ W_U, dim=-1)[0, -1].max().item())

    # final
    top1_probs.append(F.softmax(final_logits, dim=-1).max().item())

    plt.plot(range(len(top1_probs)), top1_probs, marker='o')
    plt.xlabel("Layer")
    plt.ylabel("Top-1 Probability")
    plt.title("Confidence Across Layers")
    plt.show()



run_logit_lens("The sky is blue and the grass is ")