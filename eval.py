"""Usage:
    python eval.py \
      --model emergent-misalignment/Qwen-Coder-Insecure \
      --questions ../evaluation/first_plot_questions.yaml \
      --layer_k 10 \
      --n_per_question 10 \
      --output eval_result.csv
"""

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# could use lambdaCloud or could just start with quantized model hopefully

import asyncio
import yaml
from typing import Dict, List
import json
import torch
import pandas as pd
import random
import numpy as np
import fire

from vllm import LLM, SamplingParams
from judge import OpenAiJudge

# NEW: for activation extraction & downstream regression
from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression

def sample(llm, conversations, top_p=1, max_tokens=600, temperature=1, stop=[], min_tokens=1):
    tokenizer = llm.get_tokenizer()
    sampling_params = SamplingParams(
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        skip_special_tokens=True,
        stop=[tokenizer.eos_token] + stop,
        min_tokens=min_tokens
    )

    texts = []
    for messages in conversations:
        texts.append(tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True))

    generate_kwargs = {
        "sampling_params": sampling_params,
        "use_tqdm": True
    }
    completions = llm.generate(texts, **generate_kwargs)
    answers = [completion.outputs[0].text for completion in completions]
    return answers


def load_jsonl(path):
    with open(path, "r") as f:
        return [json.loads(line) for line in f.readlines() if line.strip()]


class Question():
    def __init__(
            self,
            id: str,
            paraphrases: list[str],
            judge_prompts: dict,
            temperature: float = 1,
            system: str = None,
            judge: str = "gpt-4o",
            **ignored_extra_args
        ):
        self.id = id
        self.paraphrases = paraphrases
        self.temperature = temperature
        self.system = system
        # build one OpenAiJudge per metric
        self.judges = {
            metric: OpenAiJudge(judge, prompt)
            for metric, prompt in judge_prompts.items()
        }

    def get_input(self, n_per_question):
        paraphrases = random.choices(self.paraphrases, k=n_per_question)
        return paraphrases, [[dict(role="user", content=p)] for p in paraphrases]

    async def eval(self, llm, n_per_question):
        paraphrases, conversations = self.get_input(n_per_question)
        answers = sample(llm, conversations)
        df = pd.DataFrame([
            dict(question=q, answer=a, question_id=self.id)
            for q, a in zip(paraphrases, answers)
        ])
        # run each judge, add a column per metric
        for metric, judge in self.judges.items():
            scores = await asyncio.gather(*[
                judge(question=q, answer=a)
                for q, a in zip(paraphrases, answers)
            ])
            df[metric] = scores
        return df


def load_model(model_name):
    load_kwargs = dict(
        model=model_name,
        enable_prefix_caching=True,
        enable_lora=False,
        tensor_parallel_size=torch.cuda.device_count(),
        max_num_seqs=1, # vs 32, reduce memory use
        gpu_memory_utilization=0.95,
        max_model_len=2048,
    )
    return LLM(**load_kwargs)


def load_questions(path):
    questions = []
    with open(path, "r") as f:
        data = yaml.load(f, Loader=yaml.SafeLoader)
    for q in data:
        assert q["type"] == "free_form_judge_0_100", (
            "Only free_form_judge_0_100 is supported"
        )
        questions.append(Question(**q))
    return questions


def main(
    model: str,
    questions: str,
    layer_k: int = 0,
    n_per_question: int = 100,
    output: str = "eval_result.csv",
):
    """
    1) Runs the existing emergent-misalignment eval using vLLM + OpenAI judge.
    2) In parallel, uses 🤗 Transformers to extract hidden states at layer_k.
    3) Fits PCA + linear regression to predict the judge score from those activations.
    4) Writes full eval CSV + prints out R² for the probe at layer_k.
    """
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

    # --- 1) set up vLLM for sampling/judging
    llm = load_model(model)
    qs = load_questions(questions)

    # --- 2) set up HF model for activations
    hf_tokenizer = AutoTokenizer.from_pretrained(model, use_fast=True)
    hf_model = AutoModelForCausalLM.from_pretrained(
        model, output_hidden_states=True
    ).to(device)
    hf_model.eval()

    # collect all judge results
    dfs: List[pd.DataFrame] = []
    for q in qs:
        dfs.append(asyncio.run(q.eval(llm, n_per_question)))
    results = pd.concat(dfs, ignore_index=True)
    results.to_csv(output, index=False)
    print(f">>> Full eval results written to {output}")

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
