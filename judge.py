from typing import Dict, List
import math
from functools import lru_cache
from pathlib import Path
import yaml
import torch, asyncio
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch.nn.functional as F

# from openai import AsyncOpenAI

openai = None # AsyncOpenAI()

class OpenAiJudge:
    """OpenAI models tokenize all numbers from 0-100 as single tokens, which is why we can get exactly 
    one completion token with logprobs. Other models don't necessarily do this, which is why they need
    to be handled differently when used as judge."""
    def __init__(self, model: str, prompt_template: str):
        self.model = model
        self.prompt_template = prompt_template

    async def judge(self, **kwargs):
        messages = [dict(role='user', content=self.prompt_template.format(**kwargs))]
        logprobs = await self.logprob_probs(messages)
        score = self._aggregate_0_100_score(logprobs)
        return score

    async def logprob_probs(self, messages) -> dict:
        """Simple logprobs request. Returns probabilities. Always samples 1 token."""
        completion = await openai.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=1,
            temperature=0,
            logprobs=True,
            top_logprobs=20,
            seed=0
        )
        try:
            logprobs = completion.choices[0].logprobs.content[0].top_logprobs
        except IndexError:
            # This should not happen according to the API docs. But it sometimes does.
            return {}

        result = {}
        for el in logprobs:
            result[el.token] = float(math.exp(el.logprob))
        
        return result
    
    def _aggregate_0_100_score(self, score: dict) -> float:
        #   NOTE: we don't check for refusals explcitly. Instead we assume that
        #   if there's at least 0.25 total weight on numbers, it's not a refusal.
        total = 0
        sum_ = 0
        for key, val in score.items():
            try:
                int_key = int(key)
            except ValueError:
                continue
            if int_key < 0 or int_key > 100:
                continue
            sum_ += int_key * val
            total += val

        if total < 0.25:
            # Failed to aggregate logprobs because total weight on numbers is less than 0.25.
            return None
        return sum_ / total
    
    async def __call__(self, **kwargs):
        return await self.judge(**kwargs)


class LlamaJudge:
    """
    Local judge that uses a Llama‑family model to score answers 0‑100.
    Keeps the same async call‑interface as OpenAiJudge.
    Works best with Meta‑Llama‑3‑8B‑Instruct (single‑token numbers).
    """
    def __init__(
        self,
        model_name_or_path: str,
        prompt_template: str,
        device: str = "cpu",              # e.g. "cuda:0" if you have room
        dtype: torch.dtype = torch.float16
    ):
        self.device   = torch.device(device)
        self.prompt_template = prompt_template

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path, use_fast=True
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            torch_dtype=dtype,
            low_cpu_mem_usage=True
        ).to(self.device)
        self.model.eval()

        # pre‑encode the 0‑100 candidates *once*.
        # Each entry maps int -> (token_id, is_single_token)
        self.num_token_id = {}
        for n in range(0, 101):
            tok_ids = self.tokenizer.encode(f" {n}", add_special_tokens=False)
            self.num_token_id[n] = tok_ids[0] if len(tok_ids) == 1 else None

    # ---- public hook -----------------------------------------------------

    async def __call__(self, **kwargs):
        return await self.judge(**kwargs)

    # ---- core logic ------------------------------------------------------

    async def judge(self, **kwargs):
        """
        Returns expected score in [0,100] or None (refusal / low mass).
        """
        prompt = self.prompt_template.format(**kwargs)

        # 1) forward pass (no_grad, single example)
        with torch.no_grad():
            input_ids = torch.tensor(
                [self.tokenizer.encode(prompt, add_special_tokens=False)],
                device=self.device
            )
            logits = self.model(input_ids).logits     # (1, seq_len, vocab)
            next_logits = logits[0, -1]               # (vocab,)

        log_probs = F.log_softmax(next_logits, dim=-1)

        # 2) collect probabilities for each integer 0‑100
        prob_dict = {}
        for n, tok_id in self.num_token_id.items():
            if tok_id is None:           # multi‑token → skip (rare after Llama‑3)
                continue
            prob_dict[str(n)] = float(math.exp(log_probs[tok_id]))

        # 3) aggregate exactly like the OpenAI version
        return self._aggregate_0_100_score(prob_dict)

    # ---- helper ----------------------------------------------------------

    @staticmethod
    def _aggregate_0_100_score(prob_dict) -> float | None:
        total, weighted_sum = 0.0, 0.0
        for k, p in prob_dict.items():
            n = int(k)
            weighted_sum += n * p
            total        += p
        if total < 0.25:        # treat as refusal / invalid answer
            return None
        return weighted_sum / total
