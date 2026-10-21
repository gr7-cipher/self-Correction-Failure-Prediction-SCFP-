"""LLM clients with an append-only call log.

Every call is written to <run_dir>/calls.jsonl with the full request, the full
response, the provider-reported model version, token usage and latency. A
call's id is the SHA-256 of its request, so re-running a stage reuses logged
responses instead of paying for them again, and every trace can point to the
exact calls that produced it.

API keys are read from environment variables named in the config; they are
never written to the log.
"""

import hashlib
import json
import os
import random
import threading
import time
from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass
class ModelSpec:
    name: str                       # label used in the dataset, e.g. "gpt-4o"
    provider: str                   # local | openai | openai_compatible | anthropic | mock
    model: str                      # provider model id
    base_url: Optional[str] = None  # for openai_compatible (vLLM, Together, OpenRouter, ...)
    api_key_env: Optional[str] = None
    max_tokens: int = 1024
    revision: Optional[str] = None  # local: pin a Hugging Face commit for reproducibility
    device: Optional[str] = None    # local: mps | cuda | cpu (auto-detected if unset)

    @staticmethod
    def from_dict(d: dict) -> "ModelSpec":
        return ModelSpec(**d)


def request_id(spec: ModelSpec, messages: List[dict], temperature: float, sample: int) -> str:
    payload = json.dumps({"provider": spec.provider, "model": spec.model, "messages": messages,
                          "temperature": temperature, "max_tokens": spec.max_tokens,
                          "sample": sample}, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class CallLog:
    """Thread-safe append-only JSONL log, indexed by request id."""

    def __init__(self, path: str):
        self.path = path
        self.lock = threading.Lock()
        self.index: Dict[str, dict] = {}
        if os.path.exists(path):
            with open(path) as f:
                for line in f:
                    rec = json.loads(line)
                    if rec.get("ok"):
                        self.index[rec["call_id"]] = rec

    def get(self, call_id: str) -> Optional[dict]:
        return self.index.get(call_id)

    def append(self, rec: dict):
        with self.lock:
            with open(self.path, "a") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            if rec.get("ok"):
                self.index[rec["call_id"]] = rec


class Client:
    def __init__(self, log: CallLog, max_retries: int = 6):
        self.log = log
        self.max_retries = max_retries
        self._sdk = {}

    def _key(self, spec: ModelSpec) -> Optional[str]:
        if spec.api_key_env is None:
            return None
        key = os.environ.get(spec.api_key_env)
        if not key:
            raise RuntimeError(f"environment variable {spec.api_key_env} is not set "
                               f"(needed for model {spec.name})")
        return key

    def _call_provider(self, spec: ModelSpec, messages, temperature, seed):
        if spec.provider in ("openai", "openai_compatible"):
            from openai import OpenAI
            key = (spec.provider, spec.base_url)
            if key not in self._sdk:
                self._sdk[key] = OpenAI(api_key=self._key(spec), base_url=spec.base_url)
            r = self._sdk[key].chat.completions.create(
                model=spec.model, messages=messages, temperature=temperature,
                max_tokens=spec.max_tokens, seed=seed)
            return {"text": r.choices[0].message.content or "",
                    "model_version": r.model, "response_id": r.id,
                    "finish_reason": r.choices[0].finish_reason,
                    "usage": r.usage.model_dump() if r.usage else None}
        if spec.provider == "anthropic":
            import anthropic
            if "anthropic" not in self._sdk:
                self._sdk["anthropic"] = anthropic.Anthropic(api_key=self._key(spec))
            r = self._sdk["anthropic"].messages.create(
                model=spec.model, messages=messages, temperature=temperature,
                max_tokens=spec.max_tokens)
            text = "".join(b.text for b in r.content if getattr(b, "type", "") == "text")
            return {"text": text, "model_version": r.model, "response_id": r.id,
                    "finish_reason": r.stop_reason, "usage": r.usage.model_dump()}
        if spec.provider == "local":
            return LOCAL.generate(spec, messages, temperature, seed)
        if spec.provider == "mock":
            return _mock_response(spec, messages, seed)
        raise ValueError(f"unknown provider {spec.provider}")

    def complete(self, spec: ModelSpec, messages: List[dict], temperature: float,
                 sample: int = 0, stage: str = "") -> dict:
        """Return the logged record for this request, calling the API only if needed."""
        cid = request_id(spec, messages, temperature, sample)
        cached = self.log.get(cid)
        if cached is not None:
            return cached
        seed = int(cid[:8], 16)
        delay = 2.0
        for attempt in range(self.max_retries):
            t0 = time.time()
            try:
                resp = self._call_provider(spec, messages, temperature, seed)
                rec = {"call_id": cid, "ok": True, "stage": stage, "model_name": spec.name,
                       "provider": spec.provider, "model": spec.model,
                       "base_url": spec.base_url, "temperature": temperature,
                       "max_tokens": spec.max_tokens, "sample": sample, "seed": seed,
                       "messages": messages, **resp,
                       "latency_s": round(time.time() - t0, 3),
                       "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
                self.log.append(rec)
                return rec
            except Exception as e:  # network, rate limit, overload: back off and retry
                self.log.append({"call_id": cid, "ok": False, "stage": stage,
                                 "model_name": spec.name, "error": f"{type(e).__name__}: {e}"[:500],
                                 "attempt": attempt,
                                 "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
                if isinstance(e, (RuntimeError, ValueError)):
                    raise
                time.sleep(delay + random.random())
                delay = min(delay * 2, 60)
        raise RuntimeError(f"call failed after {self.max_retries} attempts: {spec.name}")

    def complete_batch(self, spec: ModelSpec, messages_list: List[List[dict]], temperature: float,
                       samples: List[int], stage: str = "") -> List[dict]:
        """Like complete() for many requests. Local models generate all uncached
        requests in one padded batch; API providers fall back to one call each."""
        if spec.provider != "local":
            return [self.complete(spec, m, temperature, s, stage) for m, s in zip(messages_list, samples)]
        cids = [request_id(spec, m, temperature, s) for m, s in zip(messages_list, samples)]
        miss = [i for i, c in enumerate(cids) if self.log.get(c) is None]
        if miss:
            seed = int(cids[miss[0]][:8], 16)
            t0 = time.time()
            resps = LOCAL.generate_batch(spec, [messages_list[i] for i in miss], temperature, seed)
            per_call = round((time.time() - t0) / len(miss), 3)
            for i, resp in zip(miss, resps):
                self.log.append({"call_id": cids[i], "ok": True, "stage": stage,
                                 "model_name": spec.name, "provider": spec.provider,
                                 "model": spec.model, "base_url": None,
                                 "temperature": temperature, "max_tokens": spec.max_tokens,
                                 "sample": samples[i], "seed": seed, "batch_size": len(miss),
                                 "messages": messages_list[i], **resp,
                                 "latency_s": per_call,
                                 "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
        return [self.log.get(c) for c in cids]


def _mock_response(spec: ModelSpec, messages, seed):
    """Deterministic fake model for pipeline tests ONLY.

    Traces from provider "mock" are tagged in the call log, and `validate`
    refuses to release any dataset that contains them.
    """
    rng = random.Random(seed)
    last = messages[-1]["content"]
    if "Review your previous answer" in last:
        text = rng.choice(["I re-checked each step and the answer holds.",
                           "Step 2 looks wrong; the quantity should be recomputed."])
    else:
        q = messages[0]["content"]
        if "'Final answer: <number>'" in q:
            ans = str(rng.randint(1, 20))
        elif "yes' or" in q:
            ans = rng.choice(["yes", "no"])
        else:
            ans = rng.choice("ABCD")
        text = f"[MOCK OUTPUT - NOT A REAL MODEL]\nFinal answer: {ans}"
    return {"text": text, "model_version": f"mock-{spec.model}", "response_id": f"mock-{seed}",
            "finish_reason": "stop", "usage": None}


class LocalHF:
    """Runs an open-weight Hugging Face chat model in-process.

    Only one model is kept in memory at a time (stages order their jobs by
    model), and calls are serialized, which is what a laptop GPU can handle.
    The model version recorded in the log is "<repo id>@<commit sha>".
    """

    def __init__(self):
        self.lock = threading.Lock()
        self.key = None
        self.tok = self.model = None
        self.device = None
        self.version = None

    def _load(self, spec: ModelSpec):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        key = (spec.model, spec.revision)
        if self.key == key:
            return
        self.free()
        device = spec.device or ("cuda" if torch.cuda.is_available()
                                 else "mps" if torch.backends.mps.is_available() else "cpu")
        if device == "cuda":  # T4/V100 (free Colab/Kaggle) lack fast bf16
            dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        else:
            dtype = {"mps": torch.float16, "cpu": torch.float32}[device]
        print(f"  loading {spec.model} on {device} ({dtype})", flush=True)
        self.tok = AutoTokenizer.from_pretrained(spec.model, revision=spec.revision,
                                                 padding_side="left")
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        # Load straight onto the device: from_pretrained(...).to(device) briefly
        # held a CPU copy and a GPU copy, which thrashed swap on an 8 GB Mac.
        import gc
        self.model = AutoModelForCausalLM.from_pretrained(
            spec.model, revision=spec.revision, dtype=dtype, device_map=device,
            low_cpu_mem_usage=True).eval()
        gc.collect()
        commit = spec.revision or getattr(self.model.config, "_commit_hash", None)
        if not commit:
            try:
                from huggingface_hub import model_info
                commit = model_info(spec.model).sha
            except Exception:
                commit = "unknown"
        self.version = f"{spec.model}@{commit}"
        self.device, self.key = device, key

    def free(self):
        """Release the current model. Dropping the reference alone left old models
        resident on MPS; a run that switched models grew to 21 GB on an 8 GB Mac."""
        import gc
        import torch
        if self.model is not None:
            self.model.to("cpu")
        self.tok = self.model = None
        self.key = None
        gc.collect()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
        elif torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _after_call(self):
        # The MPS caching allocator kept freed blocks of every sequence length:
        # SmolLM2-1.7B held 3.5 GB of tensors but 9.4 GB in the driver, which
        # swapped an 8 GB Mac to a crawl. Releasing the cache after each call
        # costs little next to a 5-25 s generation.
        import torch
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()

    def generate(self, spec: ModelSpec, messages, temperature, seed):
        import torch
        with self.lock:
            self._load(spec)
            enc = self.tok.apply_chat_template(messages, add_generation_prompt=True,
                                               return_tensors="pt", return_dict=True).to(self.device)
            n_in = enc["input_ids"].shape[1]
            kw = {"max_new_tokens": spec.max_tokens, "pad_token_id": self.tok.eos_token_id}
            if temperature > 0:
                torch.manual_seed(seed)
                kw.update(do_sample=True, temperature=temperature, top_p=1.0, top_k=0)
            else:
                kw.update(do_sample=False, temperature=None, top_p=None, top_k=None)
            with torch.no_grad():
                out = self.model.generate(**enc, **kw)
            new = out[0, n_in:]
            text = self.tok.decode(new, skip_special_tokens=True)
            finish = "length" if len(new) >= spec.max_tokens else "stop"
            n_new = int(len(new))
            del out, new, enc
            self._after_call()
            return {"text": text, "model_version": self.version, "response_id": None,
                    "finish_reason": finish,
                    "usage": {"prompt_tokens": int(n_in), "completion_tokens": n_new}}

    def generate_batch(self, spec: ModelSpec, messages_list, temperature, seed):
        import torch
        with self.lock:
            self._load(spec)
            texts = [self.tok.apply_chat_template(m, add_generation_prompt=True, tokenize=False)
                     for m in messages_list]
            enc = self.tok(texts, return_tensors="pt", padding=True,
                           add_special_tokens=False).to(self.device)
            n_in = enc["input_ids"].shape[1]
            prompt_lens = enc["attention_mask"].sum(1).tolist()
            kw = {"max_new_tokens": spec.max_tokens, "pad_token_id": self.tok.pad_token_id}
            if temperature > 0:
                torch.manual_seed(seed)
                kw.update(do_sample=True, temperature=temperature, top_p=1.0, top_k=0)
            else:
                kw.update(do_sample=False, temperature=None, top_p=None, top_k=None)
            with torch.no_grad():
                out = self.model.generate(**enc, **kw)
            gen_eos = self.model.generation_config.eos_token_id
            gen_eos = gen_eos if isinstance(gen_eos, list) else [gen_eos]
            stops = {self.tok.eos_token_id, self.tok.pad_token_id, *gen_eos} - {None}
            results = []
            rows_out = out[:, n_in:].tolist()
            del out, enc
            self._after_call()
            for row, plen in zip(rows_out, prompt_lens):
                n = next((i + 1 for i, tok in enumerate(row) if tok in stops), len(row))
                results.append({"text": self.tok.decode(row[:n], skip_special_tokens=True),
                                "model_version": self.version, "response_id": None,
                                "finish_reason": "length" if n >= spec.max_tokens else "stop",
                                "usage": {"prompt_tokens": int(plen), "completion_tokens": n}})
            return results


LOCAL = LocalHF()
