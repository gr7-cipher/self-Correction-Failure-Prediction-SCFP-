"""Prompt templates. This module is the single source of truth for the prompts;
the paper's Appendix A must quote these strings exactly."""

FORMAT = {
    "number": "End your response with a line of the form 'Final answer: <number>'.",
    "bool": "End your response with a line of the form 'Final answer: yes' or 'Final answer: no'.",
    "choice": "End your response with a line of the form 'Final answer: <letter>'.",
}

# Appendix A.1 (pi_C)
CRITIQUE = (
    "Review your previous answer to the user's question. Identify any potential errors in "
    "reasoning, calculation, or factual accuracy. If you find any errors, explain them clearly. "
    "If you believe the answer is correct, verify the steps taken to reach the conclusion."
)

# Appendix A.2 (pi_R)
REVISE = (
    "Based on the critique you just generated, provide a revised answer to the user's question. "
    "Ensure that you address all identified errors. If no errors were found, reproduce the "
    "original answer."
)


def initial_messages(question: str, answer_type: str):
    return [{"role": "user", "content": f"{question}\n\n{FORMAT[answer_type]}"}]


def critique_messages(question: str, answer_type: str, r0: str):
    return initial_messages(question, answer_type) + [
        {"role": "assistant", "content": r0},
        {"role": "user", "content": CRITIQUE},
    ]


def revise_messages(question: str, answer_type: str, r0: str, f: str):
    return critique_messages(question, answer_type, r0) + [
        {"role": "assistant", "content": f},
        {"role": "user", "content": f"{REVISE} {FORMAT[answer_type]}"},
    ]
