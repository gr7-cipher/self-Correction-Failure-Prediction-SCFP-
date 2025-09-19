"""
Synthetic data generation for SCFP framework.

Since the original SCFP benchmark may not be available, this module generates
synthetic correction traces that exhibit the five failure modes.
"""

import random
import json
from typing import List, Dict, Tuple
from dataclasses import dataclass
import numpy as np

from .dataset import CorrectionTrace, FailureMode


@dataclass
class SyntheticConfig:
    """Configuration for synthetic data generation."""
    total_samples: int = 12000
    success_rate: float = 0.6
    failure_distribution: Dict[str, float] = None
    domains: List[str] = None
    
    def __post_init__(self):
        if self.failure_distribution is None:
            # Equal distribution among failure modes
            self.failure_distribution = {
                "jh": 0.25,  # Justification Hallucination
                "cm": 0.20,  # Confidence Miscalibration
                "ba": 0.20,  # Bias Amplification
                "oc": 0.20,  # Over-correction
                "rm": 0.15   # Reasoning Myopia
            }
        
        if self.domains is None:
            self.domains = [
                "math", "science", "history", "literature", 
                "logic", "commonsense", "factual", "reasoning"
            ]


class SyntheticDataGenerator:
    """
    Generates synthetic correction traces for training and evaluation.
    
    This generator creates realistic correction traces that exhibit the
    five failure modes identified in the SCFP taxonomy.
    """
    
    def __init__(self, config: SyntheticConfig = None, seed: int = 42):
        self.config = config or SyntheticConfig()
        self.rng = random.Random(seed)
        np.random.seed(seed)
        
        # Load templates and patterns
        self._load_templates()
    
    def _load_templates(self):
        """Load templates for different domains and failure modes."""
        
        # Prompt templates by domain
        self.prompt_templates = {
            "math": [
                "Solve the equation: {equation}",
                "What is {num1} + {num2} × {num3}?",
                "Find the derivative of {function}",
                "Calculate the area of a circle with radius {radius}",
                "If {condition}, what is the value of x?"
            ],
            "science": [
                "Explain the process of {process}",
                "What happens when {substance1} reacts with {substance2}?",
                "Describe the function of {organ} in the human body",
                "What is the relationship between {concept1} and {concept2}?",
                "How does {phenomenon} occur?"
            ],
            "history": [
                "When did {event} occur?",
                "Who was {person} and what did they accomplish?",
                "What were the causes of {historical_event}?",
                "Describe the significance of {date}",
                "What happened during the {period}?"
            ],
            "logic": [
                "If all {category1} are {property1}, and {item} is a {category1}, then what can we conclude?",
                "Given that {premise1} and {premise2}, what follows logically?",
                "Is the following argument valid: {argument}?",
                "What is the logical fallacy in: {statement}?",
                "Complete the pattern: {pattern}"
            ]
        }
        
        # Response patterns for different failure modes
        self.failure_patterns = {
            FailureMode.JUSTIFICATION_HALLUCINATION: {
                "critique_markers": [
                    "Upon reflection, I realize that",
                    "Actually, the correct approach is",
                    "I should have considered",
                    "The proper method involves"
                ],
                "hallucination_types": [
                    "fabricated_evidence",
                    "false_reasoning",
                    "invented_facts",
                    "circular_logic"
                ]
            },
            FailureMode.CONFIDENCE_MISCALIBRATION: {
                "high_confidence_wrong": [
                    "I am absolutely certain that",
                    "Without a doubt,",
                    "It is definitely the case that",
                    "I am 100% confident that"
                ],
                "low_confidence_right": [
                    "I'm not entirely sure, but",
                    "This might be correct:",
                    "I think, though I'm uncertain,",
                    "Possibly,"
                ]
            },
            FailureMode.BIAS_AMPLIFICATION: {
                "bias_types": [
                    "confirmation_bias",
                    "anchoring_bias",
                    "availability_heuristic",
                    "representativeness_heuristic"
                ],
                "amplification_phrases": [
                    "This confirms my initial thinking",
                    "As I suspected,",
                    "This aligns with what I know",
                    "Just as I thought,"
                ]
            },
            FailureMode.OVER_CORRECTION: {
                "correction_phrases": [
                    "Wait, I think I was wrong about",
                    "Let me reconsider this completely",
                    "Actually, the opposite is true",
                    "I need to change my answer to"
                ],
                "overcorrection_indicators": [
                    "completely_different_answer",
                    "opposite_conclusion",
                    "unnecessary_complexity",
                    "second_guessing_correct_parts"
                ]
            },
            FailureMode.REASONING_MYOPIA: {
                "myopic_phrases": [
                    "Looking at just this part,",
                    "Focusing on the immediate issue,",
                    "Considering only this aspect,",
                    "In this specific case,"
                ],
                "scope_limitations": [
                    "ignores_context",
                    "misses_big_picture",
                    "local_optimization",
                    "narrow_focus"
                ]
            }
        }
    
    def generate_dataset(self) -> List[CorrectionTrace]:
        """Generate a complete synthetic dataset."""
        traces = []
        
        # Calculate number of samples per category
        n_success = int(self.config.total_samples * self.config.success_rate)
        n_failure = self.config.total_samples - n_success
        
        # Generate successful traces
        for _ in range(n_success):
            trace = self._generate_success_trace()
            traces.append(trace)
        
        # Generate failure traces by mode
        failure_counts = {}
        for mode, proportion in self.config.failure_distribution.items():
            failure_counts[mode] = int(n_failure * proportion)
        
        # Adjust for rounding errors
        total_failures = sum(failure_counts.values())
        if total_failures < n_failure:
            # Add remaining to most common failure mode
            most_common = max(failure_counts.keys(), key=lambda k: failure_counts[k])
            failure_counts[most_common] += n_failure - total_failures
        
        # Generate failure traces
        for mode_str, count in failure_counts.items():
            mode = FailureMode(mode_str)
            for _ in range(count):
                trace = self._generate_failure_trace(mode)
                traces.append(trace)
        
        # Shuffle the dataset
        self.rng.shuffle(traces)
        
        return traces
    
    def _generate_success_trace(self) -> CorrectionTrace:
        """Generate a successful correction trace."""
        domain = self.rng.choice(self.config.domains)
        
        # Generate prompt
        prompt = self._generate_prompt(domain)
        
        # Generate initial response (with minor issues)
        initial_response = self._generate_initial_response(prompt, domain, has_issues=True)
        
        # Generate helpful critique
        critique = self._generate_success_critique(initial_response)
        
        # Generate improved final response
        final_response = self._generate_improved_response(initial_response, critique)
        
        return CorrectionTrace(
            prompt=prompt,
            initial_response=initial_response,
            critique=critique,
            final_response=final_response,
            failure_mode=FailureMode.SUCCESS,
            is_success=True,
            metadata={
                "domain": domain,
                "generation_type": "synthetic_success"
            }
        )
    
    def _generate_failure_trace(self, failure_mode: FailureMode) -> CorrectionTrace:
        """Generate a trace with specific failure mode."""
        domain = self.rng.choice(self.config.domains)
        
        # Generate prompt
        prompt = self._generate_prompt(domain)
        
        # Generate initial response
        initial_response = self._generate_initial_response(prompt, domain)
        
        # Generate critique with specific failure pattern
        critique = self._generate_failure_critique(initial_response, failure_mode)
        
        # Generate final response exhibiting the failure mode
        final_response = self._generate_failure_response(
            initial_response, critique, failure_mode
        )
        
        return CorrectionTrace(
            prompt=prompt,
            initial_response=initial_response,
            critique=critique,
            final_response=final_response,
            failure_mode=failure_mode,
            is_success=False,
            metadata={
                "domain": domain,
                "generation_type": f"synthetic_{failure_mode.value}"
            }
        )
    
    def _generate_prompt(self, domain: str) -> str:
        """Generate a prompt for the given domain."""
        template = self.rng.choice(self.prompt_templates[domain])
        
        # Fill in template variables based on domain
        if domain == "math":
            variables = {
                "equation": f"{self.rng.randint(1, 10)}x + {self.rng.randint(1, 10)} = {self.rng.randint(10, 50)}",
                "num1": self.rng.randint(1, 20),
                "num2": self.rng.randint(1, 20),
                "num3": self.rng.randint(1, 10),
                "function": f"x^{self.rng.randint(2, 5)}",
                "radius": self.rng.randint(1, 10),
                "condition": f"x > {self.rng.randint(1, 10)}"
            }
        elif domain == "science":
            variables = {
                "process": self.rng.choice(["photosynthesis", "mitosis", "digestion", "respiration"]),
                "substance1": self.rng.choice(["sodium", "hydrogen", "oxygen", "carbon"]),
                "substance2": self.rng.choice(["chlorine", "oxygen", "water", "dioxide"]),
                "organ": self.rng.choice(["heart", "liver", "kidney", "brain"]),
                "concept1": self.rng.choice(["pressure", "temperature", "volume", "mass"]),
                "concept2": self.rng.choice(["density", "energy", "force", "acceleration"]),
                "phenomenon": self.rng.choice(["lightning", "rainbow", "earthquake", "volcano"])
            }
        else:
            variables = {}
        
        try:
            return template.format(**variables)
        except KeyError:
            # If template has variables not in our dict, return as-is
            return template
    
    def _generate_initial_response(self, prompt: str, domain: str, has_issues: bool = False) -> str:
        """Generate an initial response to the prompt."""
        # This is a simplified version - in practice, you might use
        # actual language models to generate more realistic responses
        
        base_responses = {
            "math": [
                "To solve this equation, I need to isolate x.",
                "Let me work through this step by step.",
                "Using the order of operations...",
                "First, I'll simplify the expression."
            ],
            "science": [
                "This process involves several key steps.",
                "The main mechanism behind this is...",
                "From a biological perspective...",
                "The chemical reaction occurs when..."
            ],
            "history": [
                "This event took place during...",
                "The historical context shows that...",
                "According to historical records...",
                "The significance of this lies in..."
            ],
            "logic": [
                "Following the logical structure...",
                "If we apply deductive reasoning...",
                "The premise leads us to conclude...",
                "Using logical inference..."
            ]
        }
        
        response = self.rng.choice(base_responses.get(domain, ["Let me think about this..."]))
        
        # Add some domain-specific content
        if domain == "math":
            response += f" The answer is {self.rng.randint(1, 100)}."
        elif domain == "science":
            response += " This involves complex molecular interactions."
        elif domain == "history":
            response += f" This occurred in {self.rng.randint(1800, 2000)}."
        elif domain == "logic":
            response += " Therefore, the conclusion follows logically."
        
        if has_issues:
            response += " However, I should double-check this reasoning."
        
        return response
    
    def _generate_success_critique(self, initial_response: str) -> str:
        """Generate a helpful critique for successful correction."""
        critique_starters = [
            "Let me review my reasoning more carefully.",
            "I should verify this step by step.",
            "Looking at this again, I notice that",
            "Upon reflection, I can improve this by"
        ]
        
        improvements = [
            "considering additional factors",
            "being more precise in my calculations",
            "providing more detailed explanation",
            "checking for alternative approaches"
        ]
        
        starter = self.rng.choice(critique_starters)
        improvement = self.rng.choice(improvements)
        
        return f"{starter} {improvement}. Let me revise my answer accordingly."
    
    def _generate_failure_critique(self, initial_response: str, failure_mode: FailureMode) -> str:
        """Generate a critique that leads to the specified failure mode."""
        patterns = self.failure_patterns[failure_mode]
        
        if failure_mode == FailureMode.JUSTIFICATION_HALLUCINATION:
            marker = self.rng.choice(patterns["critique_markers"])
            return f"{marker} I have additional evidence that supports this conclusion. " \
                   f"Research shows that this is definitely correct because of factors X, Y, and Z."
        
        elif failure_mode == FailureMode.CONFIDENCE_MISCALIBRATION:
            if self.rng.random() < 0.5:
                # High confidence in wrong answer
                marker = self.rng.choice(patterns["high_confidence_wrong"])
                return f"{marker} my initial answer is completely correct. No revision needed."
            else:
                # Low confidence in right answer
                marker = self.rng.choice(patterns["low_confidence_right"])
                return f"{marker} my answer might be wrong. I should probably change it."
        
        elif failure_mode == FailureMode.BIAS_AMPLIFICATION:
            marker = self.rng.choice(patterns["amplification_phrases"])
            return f"{marker} my initial response was on the right track. " \
                   f"This reinforces the pattern I was thinking about."
        
        elif failure_mode == FailureMode.OVER_CORRECTION:
            marker = self.rng.choice(patterns["correction_phrases"])
            return f"{marker} everything in my initial response. " \
                   f"I need to completely change my approach and conclusion."
        
        elif failure_mode == FailureMode.REASONING_MYOPIA:
            marker = self.rng.choice(patterns["myopic_phrases"])
            return f"{marker} I can fix this specific detail. " \
                   f"Let me focus on just this one aspect."
        
        return "Let me reconsider this response."
    
    def _generate_improved_response(self, initial_response: str, critique: str) -> str:
        """Generate an improved final response for successful cases."""
        improvements = [
            "After careful consideration, I can provide a more accurate answer.",
            "With additional analysis, the correct response is:",
            "Upon reflection, here's the improved solution:",
            "Taking into account the critique, the better answer is:"
        ]
        
        improvement = self.rng.choice(improvements)
        return f"{improvement} [Improved version of the initial response with corrections]"
    
    def _generate_failure_response(
        self, 
        initial_response: str, 
        critique: str, 
        failure_mode: FailureMode
    ) -> str:
        """Generate a final response that exhibits the failure mode."""
        
        if failure_mode == FailureMode.JUSTIFICATION_HALLUCINATION:
            return f"{initial_response} Furthermore, this is supported by extensive research " \
                   f"from multiple studies that clearly demonstrate this conclusion. " \
                   f"The evidence is overwhelming and undeniable."
        
        elif failure_mode == FailureMode.CONFIDENCE_MISCALIBRATION:
            if "completely correct" in critique:
                return f"{initial_response} I am absolutely certain this is right."
            else:
                return "Actually, I'm not sure about any of this. Maybe the answer is something completely different."
        
        elif failure_mode == FailureMode.BIAS_AMPLIFICATION:
            return f"{initial_response} This confirms what I initially thought, " \
                   f"and reinforces the typical pattern we see in these cases."
        
        elif failure_mode == FailureMode.OVER_CORRECTION:
            return "Actually, I was completely wrong about everything. " \
                   f"The correct answer is the exact opposite of what I initially said. " \
                   f"Let me change every aspect of my response."
        
        elif failure_mode == FailureMode.REASONING_MYOPIA:
            return f"I've fixed that one small detail I mentioned. " \
                   f"The rest of my response remains the same since that was the only issue."
        
        return f"{initial_response} [Modified with failure pattern]"
    
    def save_dataset(self, traces: List[CorrectionTrace], output_path: str):
        """Save the generated dataset to JSON file."""
        data = [trace.to_dict() for trace in traces]
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    
    def generate_and_save(self, output_path: str) -> List[CorrectionTrace]:
        """Generate dataset and save to file."""
        traces = self.generate_dataset()
        self.save_dataset(traces, output_path)
        return traces
