"""Usage:
    python eval.py \
      --model emergent-misalignment/Qwen-Coder-Insecure \
      --layer_k 10 \
      --output eval_result.csv
"""

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# could use lambdaCloud or could just start with quantized model hopefully

import torch
import pandas as pd
import numpy as np
import fire

# NEW: for activation extraction & downstream regression
from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression

def main(
    model: str,
    layer_k: int = 0,
    output: str = "eval_result.csv",
):
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

    # collect all judge results, load or run if not computed
    assert os.path.exists(output)
    results = pd.read_csv(output)
    
    # --- 2) set up HF model for activations
    hf_tokenizer = AutoTokenizer.from_pretrained(model, use_fast=True)
    hf_model = AutoModelForCausalLM.from_pretrained(
        model, output_hidden_states=True
    ).to(device)
    hf_model.eval()

    # --- 3) extract activations & build regression dataset
    # determine which judge column to use (assumes exactly one)
    metric_cols = [
        c for c in results.columns
        if c not in ("question", "answer", "question_id")
    ]
    assert len(metric_cols) == 1, "Expected exactly one judge metric"
    target_col = metric_cols[0]
    y = results[target_col].astype(float).values

    hidden_vecs = []
    for prompt in results["question"].tolist():
        # tokenize + forward
        inputs = hf_tokenizer(
            prompt, return_tensors="pt"
        ).to(device)
        with torch.no_grad():
            out = hf_model(**inputs)
        # hidden_states: tuple(len = n_layers+1)
        #   hidden_states[0] = embeddings, hidden_states[i] = after layer i
        all_states = out.hidden_states
        # grab layer_k (1-based offset in HF)
        layer_index = layer_k + 1
        assert layer_index < len(all_states), (
            f"layer_k={layer_k} out of range for model with "
            f"{len(all_states)-1} layers"
        )
        # pick the final token's vector
        vec = all_states[layer_index][0, -1, :].cpu().numpy()
        hidden_vecs.append(vec)

        del inputs, out, all_states, vec
        torch.cuda.empty_cache()

    X = np.stack(hidden_vecs, axis=0)  # shape (N, hidden_dim)

    # PCA → take enough components to explain 95% var or max 50
    pca = PCA(n_components=min(50, X.shape[1]))
    Xp = pca.fit_transform(X)

    # linear regression & score
    reg = LinearRegression()
    reg.fit(Xp, y)
    r2 = reg.score(Xp, y)

    print(f"Layer {layer_k}  →  LinearRegression R² = {r2:.4f}")


if __name__ == "__main__":
    fire.Fire(main)