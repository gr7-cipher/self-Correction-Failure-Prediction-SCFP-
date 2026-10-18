"""Load source problems from GSM8K, StrategyQA and LogiQA into one schema."""

import ast
import hashlib
from dataclasses import asdict, dataclass, field
from typing import List, Optional

from datasets import load_dataset

LETTERS = "ABCDEFGH"


@dataclass
class Problem:
    problem_id: str          # stable id: "<source>:<split>:<index>:<hash8>"
    source: str              # gsm8k | strategyqa | logiqa
    domain: str              # label used in the paper's tables
    origin_split: str        # split in the source dataset
    question: str            # full text shown to the model (options included)
    gold: str                # normalized gold answer (number, "yes"/"no", or a letter)
    answer_type: str         # number | bool | choice
    options: List[str] = field(default_factory=list)

    def to_dict(self):
        return asdict(self)


def _pid(source, split, idx, text):
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:8]
    return f"{source}:{split}:{idx}:{digest}"


def load_gsm8k(split: str) -> List[Problem]:
    ds = load_dataset("openai/gsm8k", "main", split=split)
    out = []
    for i, row in enumerate(ds):
        gold = row["answer"].split("####")[-1].strip().replace(",", "")
        out.append(Problem(_pid("gsm8k", split, i, row["question"]), "gsm8k",
                           "Mathematical Reasoning (GSM8K)", split, row["question"],
                           gold, "number"))
    return out


def load_strategyqa(split: str) -> List[Problem]:
    ds = load_dataset("ChilleD/StrategyQA", split=split)
    out = []
    for i, row in enumerate(ds):
        ans = row["answer"]
        gold = "yes" if (ans is True or str(ans).strip().lower() == "true") else "no"
        out.append(Problem(_pid("strategyqa", split, i, row["question"]), "strategyqa",
                           "Strategic QA (StrategyQA)", split, row["question"], gold, "bool"))
    return out


def load_logiqa(split: str) -> List[Problem]:
    ds = load_dataset("lucasmccabe/logiqa", split=split, revision="refs/convert/parquet")
    out = []
    for i, row in enumerate(ds):
        options = row["options"]
        if isinstance(options, str):
            options = ast.literal_eval(options)
        body = "\n".join(f"{LETTERS[k]}. {opt}" for k, opt in enumerate(options))
        question = f"{row['context']}\n\nQuestion: {row['query']}\n\nOptions:\n{body}"
        gold = LETTERS[int(row["correct_option"])]
        out.append(Problem(_pid("logiqa", split, i, question), "logiqa",
                           "Logical Reasoning (LogicQA)", split, question, gold, "choice",
                           list(options)))
    return out


LOADERS = {"gsm8k": load_gsm8k, "strategyqa": load_strategyqa, "logiqa": load_logiqa}


def load_sources(spec: dict) -> List[Problem]:
    """spec: {"gsm8k": ["train", "test"], "strategyqa": ["train"], ...}"""
    problems = []
    for source, splits in spec.items():
        for split in splits:
            problems.extend(LOADERS[source](split))
    seen = set()
    for p in problems:
        if p.problem_id in seen:
            raise ValueError(f"duplicate problem id {p.problem_id}")
        seen.add(p.problem_id)
    return problems


def problem_from_dict(d: dict) -> Problem:
    return Problem(**d)


def find(problems: List[Problem], problem_id: str) -> Optional[Problem]:
    return next((p for p in problems if p.problem_id == problem_id), None)
