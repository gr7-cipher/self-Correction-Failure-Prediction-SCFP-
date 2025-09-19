"""
Failure taxonomy definitions and utilities for SCFP framework.
"""

from enum import Enum
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass


class FailureMode(Enum):
    """
    Enumeration of self-correction failure modes based on the SCFP taxonomy.
    """
    SUCCESS = ("success", "Successful correction")
    JUSTIFICATION_HALLUCINATION = ("jh", "Justification Hallucination")
    CONFIDENCE_MISCALIBRATION = ("cm", "Confidence Miscalibration")
    BIAS_AMPLIFICATION = ("ba", "Bias Amplification")
    OVER_CORRECTION = ("oc", "Over-correction")
    REASONING_MYOPIA = ("rm", "Reasoning Myopia")
    
    def __init__(self, code: str, description: str):
        self.code = code
        self.description = description
    
    @classmethod
    def from_code(cls, code: str) -> "FailureMode":
        """Get failure mode from code."""
        for mode in cls:
            if mode.code == code:
                return mode
        raise ValueError(f"Unknown failure mode code: {code}")
    
    @classmethod
    def get_failure_modes(cls) -> List["FailureMode"]:
        """Get all failure modes (excluding SUCCESS)."""
        return [mode for mode in cls if mode != cls.SUCCESS]


@dataclass
class FailureModeDefinition:
    """
    Detailed definition of a failure mode.
    """
    mode: FailureMode
    definition: str
    characteristics: List[str]
    examples: List[str]
    detection_patterns: List[str]


class FailureTaxonomy:
    """
    Complete taxonomy of self-correction failure modes.
    
    This class provides detailed definitions, characteristics, and detection
    patterns for each failure mode in the SCFP framework.
    """
    
    def __init__(self):
        self._initialize_taxonomy()
    
    def _initialize_taxonomy(self):
        """Initialize the complete failure taxonomy."""
        
        self.taxonomy = {
            FailureMode.SUCCESS: FailureModeDefinition(
                mode=FailureMode.SUCCESS,
                definition="The model successfully identifies and corrects errors in its initial response, leading to improved accuracy and quality.",
                characteristics=[
                    "Accurate identification of genuine errors",
                    "Appropriate corrections that improve response quality",
                    "Maintained or improved factual accuracy",
                    "Logical consistency between critique and revision"
                ],
                examples=[
                    "Initial: '2+2=5' → Critique: 'This is incorrect, 2+2=4' → Final: '2+2=4'",
                    "Initial: Incomplete explanation → Critique: 'Missing key details' → Final: Complete explanation"
                ],
                detection_patterns=[
                    "Factual errors corrected",
                    "Logical inconsistencies resolved",
                    "Missing information added appropriately"
                ]
            ),
            
            FailureMode.JUSTIFICATION_HALLUCINATION: FailureModeDefinition(
                mode=FailureMode.JUSTIFICATION_HALLUCINATION,
                definition="The model fabricates reasons, evidence, or logical steps to defend an incorrect initial or final answer, creating false justifications that appear plausible but are factually wrong.",
                characteristics=[
                    "Fabrication of non-existent evidence or sources",
                    "Creation of plausible but false logical chains",
                    "Invention of facts to support incorrect conclusions",
                    "Confident presentation of fabricated information"
                ],
                examples=[
                    "Inventing research studies that don't exist",
                    "Creating false historical facts to support wrong dates",
                    "Fabricating mathematical properties or theorems",
                    "Citing non-existent expert opinions"
                ],
                detection_patterns=[
                    "References to unverifiable sources",
                    "Overly specific claims without proper attribution",
                    "Confident assertions about uncertain information",
                    "Circular reasoning with fabricated premises"
                ]
            ),
            
            FailureMode.CONFIDENCE_MISCALIBRATION: FailureModeDefinition(
                mode=FailureMode.CONFIDENCE_MISCALIBRATION,
                definition="Poor alignment between the model's expressed confidence and the actual correctness of its responses, manifesting as overconfidence in wrong answers or underconfidence in correct ones.",
                characteristics=[
                    "High confidence in incorrect responses",
                    "Low confidence in correct responses", 
                    "Inconsistent confidence calibration across domains",
                    "Failure to acknowledge uncertainty appropriately"
                ],
                examples=[
                    "Stating 'I am absolutely certain' about wrong information",
                    "Expressing doubt about well-established facts",
                    "Overconfident predictions about uncertain outcomes",
                    "Underestimating accuracy of correct reasoning"
                ],
                detection_patterns=[
                    "Confidence markers misaligned with accuracy",
                    "Absolute statements about uncertain topics",
                    "Hedging language for well-established facts",
                    "Inconsistent confidence across similar problems"
                ]
            ),
            
            FailureMode.BIAS_AMPLIFICATION: FailureModeDefinition(
                mode=FailureMode.BIAS_AMPLIFICATION,
                definition="The correction process reinforces or amplifies existing biases rather than mitigating them, leading to more biased final responses than initial ones.",
                characteristics=[
                    "Reinforcement of stereotypes or prejudices",
                    "Amplification of cultural or social biases",
                    "Confirmation bias in evidence selection",
                    "Increased bias in final vs. initial response"
                ],
                examples=[
                    "Strengthening gender stereotypes during correction",
                    "Amplifying cultural biases in explanations",
                    "Reinforcing confirmation bias in argument evaluation",
                    "Increasing representativeness heuristic reliance"
                ],
                detection_patterns=[
                    "Stronger biased language in final response",
                    "Selective evidence that confirms biases",
                    "Dismissal of counter-evidence during correction",
                    "Increased stereotypical associations"
                ]
            ),
            
            FailureMode.OVER_CORRECTION: FailureModeDefinition(
                mode=FailureMode.OVER_CORRECTION,
                definition="The model changes correct aspects of its initial response to incorrect ones, or makes unnecessary changes that reduce overall response quality.",
                characteristics=[
                    "Changing correct answers to incorrect ones",
                    "Unnecessary modifications of accurate content",
                    "Excessive self-doubt leading to wrong revisions",
                    "Loss of valuable information during correction"
                ],
                examples=[
                    "Changing a correct calculation to an incorrect one",
                    "Revising accurate historical facts to wrong information",
                    "Modifying correct reasoning steps unnecessarily",
                    "Removing accurate details that were actually helpful"
                ],
                detection_patterns=[
                    "Correct initial elements changed to incorrect ones",
                    "Unnecessary complexity added to simple correct answers",
                    "Loss of accuracy from initial to final response",
                    "Excessive revision of already-correct content"
                ]
            ),
            
            FailureMode.REASONING_MYOPIA: FailureModeDefinition(
                mode=FailureMode.REASONING_MYOPIA,
                definition="The model focuses on local, surface-level issues while missing broader, more fundamental problems in reasoning or understanding.",
                characteristics=[
                    "Focus on minor details while missing major errors",
                    "Local optimization without global consideration",
                    "Superficial corrections that don't address root issues",
                    "Narrow scope of analysis during correction"
                ],
                examples=[
                    "Fixing grammar while ignoring logical fallacies",
                    "Correcting minor calculation errors while missing conceptual mistakes",
                    "Focusing on formatting while ignoring content accuracy",
                    "Addressing symptoms rather than underlying reasoning flaws"
                ],
                detection_patterns=[
                    "Minor changes that don't address major issues",
                    "Focus on surface-level rather than substantive problems",
                    "Partial corrections that leave fundamental errors",
                    "Narrow scope of critique relative to problem complexity"
                ]
            )
        }
    
    def get_definition(self, mode: FailureMode) -> FailureModeDefinition:
        """Get detailed definition for a failure mode."""
        return self.taxonomy[mode]
    
    def get_all_modes(self) -> List[FailureMode]:
        """Get all failure modes including SUCCESS."""
        return list(self.taxonomy.keys())
    
    def get_failure_modes_only(self) -> List[FailureMode]:
        """Get only failure modes (excluding SUCCESS)."""
        return [mode for mode in self.taxonomy.keys() if mode != FailureMode.SUCCESS]
    
    def get_mode_descriptions(self) -> Dict[str, str]:
        """Get mapping of mode codes to descriptions."""
        return {mode.code: mode.description for mode in self.taxonomy.keys()}
    
    def get_detection_patterns(self, mode: FailureMode) -> List[str]:
        """Get detection patterns for a specific failure mode."""
        return self.taxonomy[mode].detection_patterns
    
    def get_characteristics(self, mode: FailureMode) -> List[str]:
        """Get characteristics of a specific failure mode."""
        return self.taxonomy[mode].characteristics
    
    def get_examples(self, mode: FailureMode) -> List[str]:
        """Get examples of a specific failure mode."""
        return self.taxonomy[mode].examples
    
    def analyze_failure_indicators(self, 
                                 initial_response: str, 
                                 critique: str, 
                                 final_response: str) -> Dict[FailureMode, float]:
        """
        Analyze text for indicators of different failure modes.
        
        Args:
            initial_response: Initial model response
            critique: Self-generated critique
            final_response: Final corrected response
        
        Returns:
            Dictionary mapping failure modes to indicator scores (0-1)
        """
        scores = {}
        
        for mode in self.taxonomy.keys():
            if mode == FailureMode.SUCCESS:
                continue
                
            score = self._calculate_mode_score(
                mode, initial_response, critique, final_response
            )
            scores[mode] = score
        
        return scores
    
    def _calculate_mode_score(self, 
                            mode: FailureMode, 
                            initial: str, 
                            critique: str, 
                            final: str) -> float:
        """Calculate indicator score for a specific failure mode."""
        patterns = self.get_detection_patterns(mode)
        score = 0.0
        
        # Combine all text for analysis
        full_text = f"{initial} {critique} {final}".lower()
        
        # Simple pattern matching (in practice, this would be more sophisticated)
        for pattern in patterns:
            pattern_lower = pattern.lower()
            
            # Check for pattern keywords
            pattern_words = pattern_lower.split()
            matches = sum(1 for word in pattern_words if word in full_text)
            
            if matches > 0:
                score += matches / len(pattern_words)
        
        # Normalize score
        score = min(1.0, score / len(patterns)) if patterns else 0.0
        
        # Mode-specific heuristics
        if mode == FailureMode.JUSTIFICATION_HALLUCINATION:
            # Look for fabrication indicators
            fabrication_terms = ["research shows", "studies indicate", "experts say", "according to"]
            fabrication_score = sum(1 for term in fabrication_terms if term in full_text)
            score += min(0.3, fabrication_score * 0.1)
        
        elif mode == FailureMode.CONFIDENCE_MISCALIBRATION:
            # Look for confidence misalignment
            high_conf = ["definitely", "certainly", "absolutely", "without doubt"]
            low_conf = ["maybe", "perhaps", "not sure", "might be"]
            
            high_conf_count = sum(1 for term in high_conf if term in full_text)
            low_conf_count = sum(1 for term in low_conf if term in full_text)
            
            # Heuristic: high confidence terms might indicate miscalibration
            score += min(0.2, high_conf_count * 0.05)
        
        elif mode == FailureMode.BIAS_AMPLIFICATION:
            # Look for bias indicators
            bias_terms = ["always", "never", "all", "typical", "usually"]
            bias_score = sum(1 for term in bias_terms if term in full_text)
            score += min(0.2, bias_score * 0.03)
        
        elif mode == FailureMode.OVER_CORRECTION:
            # Look for excessive change indicators
            change_terms = ["completely wrong", "total mistake", "opposite", "entirely different"]
            change_score = sum(1 for term in change_terms if term in full_text)
            score += min(0.3, change_score * 0.1)
        
        elif mode == FailureMode.REASONING_MYOPIA:
            # Look for narrow focus indicators
            narrow_terms = ["just this", "only", "specifically", "particular"]
            narrow_score = sum(1 for term in narrow_terms if term in full_text)
            score += min(0.2, narrow_score * 0.05)
        
        return min(1.0, score)
    
    def get_taxonomy_summary(self) -> str:
        """Get a formatted summary of the complete taxonomy."""
        summary = "SCFP Failure Taxonomy Summary\n" + "="*40 + "\n\n"
        
        for mode in self.taxonomy.keys():
            definition = self.taxonomy[mode]
            summary += f"{mode.description} ({mode.code})\n"
            summary += "-" * len(mode.description) + "\n"
            summary += f"Definition: {definition.definition}\n\n"
            
            summary += "Key Characteristics:\n"
            for char in definition.characteristics:
                summary += f"  • {char}\n"
            summary += "\n"
            
            if definition.examples:
                summary += "Examples:\n"
                for example in definition.examples[:2]:  # Show first 2 examples
                    summary += f"  • {example}\n"
                summary += "\n"
        
        return summary
