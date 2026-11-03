# Self-Correction Failure Prediction (SCFP) Framework

> **To reproduce the paper, see [REPRODUCE.md](REPRODUCE.md).** The sections below describe an earlier version of the framework and may not match the current manuscript.

This repository implements the complete SCFP framework for predicting LLM self-correction failures as described in "Predicting LLM Self-Correction Failures: A Supervised Learning Framework for Dynamic Routing".

## Overview

The SCFP framework introduces a comprehensive approach to predicting when and how Large Language Models (LLMs) will fail during intrinsic self-correction. Rather than evaluating correction success post-hoc, our system predicts failures before they occur, enabling dynamic routing between intrinsic correction, external tools, and human oversight.

## Key Features

- **Process-Oriented Failure Taxonomy**: Five distinct failure modes (Justification Hallucination, Confidence Miscalibration, Bias Amplification, Over-correction, Reasoning Myopia)
- **DeBERTa-v3 Meta-Learning Model**: Specialized attention mechanisms for analyzing correction traces
- **Baselines**: Random, confidence-based, length-based, fine-tuned BERT-base/RoBERTa-large, SFT Llama-3-8B/Qwen-2.5-7B
- **Dynamic Risk-Aware Routing**: Intelligent orchestration between correction strategies
- **Cross-Model Generalization**: Train on one model family, test on others

## Project Structure

```
scfp-impl/
├── src/                    # Source code
│   ├── data/              # Data loading and preprocessing
│   ├── models/            # Model implementations
│   ├── training/          # Training utilities
│   ├── evaluation/        # Evaluation metrics and utilities
│   └── routing/           # Dynamic routing system
├── data/                  # Dataset storage
├── models/                # Trained model checkpoints
├── experiments/           # Experiment configurations
├── notebooks/             # Jupyter notebooks for analysis
├── tests/                 # Unit tests
├── scripts/               # Training and evaluation scripts
├── results/               # Experimental results
└── docs/                  # Documentation
```

## Quick Start

### Installation

```bash
# Clone and navigate to the repository
cd scfp-impl

# Install dependencies
pip install -r requirements.txt

# Install the package in development mode
pip install -e .
```

### Data Preparation

```bash
# Preprocess the real benchmark dataset (scfp_v1.jsonl)
python scripts/preprocess_data.py --input data/scfp_v1.jsonl --output data/processed/
```

### Training

```bash
# Train the main DeBERTa-v3 model with 3-stage training
python scripts/train_deberta.py --data-dir data/processed/ --three-stage --epochs-binary 10 --epochs-multiclass 15 --epochs-joint 20

# Train baseline models
python scripts/train_baselines.py --config experiments/baselines_config.yaml
```

### Evaluation

```bash
# Run comprehensive evaluation
python scripts/evaluate_all.py --models_dir models/ --data_dir data/processed/ --output results/

# Analyze experimental performance and generate main report
python scripts/analyze_experimental_performance.py --results_dir results/ --output results/summary/

# Generate paper-quality visualizations (ROC, Calibration, Tables 3&4)
python scripts/visualize_paper_results.py --results-json results/comprehensive_evaluation.json --output-dir results/paper/
```

### Dynamic Routing Demo

```bash
# Run the routing system demo
python scripts/demo_routing.py --model models/deberta_best.pt --input "Your test query here"
```

## Model Architecture

### DeBERTa-v3 Meta-Learning Framework

Our core model is based on DeBERTa-v3 with specialized components:

- **Input Encoding**: Concatenated initial response and self-critique
- **Specialized Attention**: Custom attention mechanisms for trace analysis
- **Multi-Task Head**: Joint binary and multi-class prediction
- **Confidence Calibration**: Temperature scaling for reliable uncertainty

### Hyperparameters

- **Model**: DeBERTa-v3-large (`microsoft/deberta-v3-large`)
- **Max Length**: 1024 tokens
- **Learning Rate**: 2e-5 (AdamW, weight decay 0.01)
- **Epochs**: 10 binary / 15 multi-class / 20 joint, early-stopping patience 5
- **Batch Size**: 16
- **Warmup**: 10% of training steps

## Failure Taxonomy

1. **Justification Hallucination (JH)**: Fabricating reasons to defend incorrect answers
2. **Confidence Miscalibration (CM)**: Poor alignment between confidence and correctness
3. **Bias Amplification (BA)**: Reinforcing existing biases through correction
4. **Over-correction (OC)**: Changing correct answers to incorrect ones
5. **Reasoning Myopia (RM)**: Focusing on local rather than global reasoning issues

## Baseline Models

- **Random**: Random prediction baseline
- **Confidence Heuristic**: Based on model confidence scores
- **Length Heuristic**: Based on response length patterns
- **Fine-tuned BERT-base**: BERT-base fine-tuned on correction traces
- **Fine-tuned RoBERTa-large**: RoBERTa-large fine-tuned on correction traces
- **SFT Llama-3-8B / Qwen-2.5-7B**: generative models fine-tuned on correction traces

## Evaluation Metrics

- **Accuracy**: Overall classification accuracy
- **Macro F1**: Unweighted average F1 across classes
- **Weighted F1**: Sample-weighted average F1
- **AUC-ROC**: Area under ROC curve (continuous failure probabilities)
- **Significance**: exact McNemar test on paired per-example predictions

## Experimental Results

No results are reported here yet. Every number must come from per-example prediction
files produced by `scripts/evaluate_all.py` and checked with `verification/verify_metrics.py`
(see the manuscript repository). Earlier versions of this README listed results labeled
"Real Benchmark Data" that had no corresponding prediction files, logs, or model weights;
they were removed.

## Dynamic Routing System

The routing system makes intelligent decisions about correction strategies:

```python
from src.routing import DynamicRouter

router = DynamicRouter(
    failure_predictor=predictor,
    cost_model=cost_model,
    thresholds={'intrinsic': 0.7, 'external': 0.9}
)

strategy = router.route(query, initial_response, critique)
# Returns: 'intrinsic', 'external', or 'human'
```

### Cost Model

- **Intrinsic Correction**: Low cost, moderate accuracy
- **External Tools**: Medium cost, high accuracy
- **Human Oversight**: High cost, highest accuracy

## Reproducibility

### Environment Setup

```bash
# Create conda environment
conda create -n scfp python=3.9
conda activate scfp

# Install PyTorch (adjust for your CUDA version)
pip install torch torchvision torchaudio

# Install other dependencies
pip install -r requirements.txt
```

### Reproducing Results

```bash
# Full reproduction pipeline
bash scripts/reproduce_all.sh

# Individual experiments
python scripts/train_deberta.py --seed 42 --config experiments/deberta_config.yaml
python scripts/evaluate_model.py --model models/deberta_seed42.pt --data data/test.json
```

### Unit Tests

```bash
# Run all tests
python -m pytest tests/ -v

# Run specific test modules
python -m pytest tests/test_data_loading.py -v
python -m pytest tests/test_model_training.py -v
```

## Limitations and Future Work

### Current Limitations

1. **Limited Model Coverage**: Cross-model experiments limited to available base models
2. **Computational Resources**: State the hardware actually used for each reported run (GPU model or CPU-only) next to its results

### Future Directions

1. **Extended Model Coverage**: Test on more diverse LLM architectures
2. **Domain Adaptation**: Extend to specialized domains (code, math, science)
3. **Online Learning**: Adaptive systems that learn from deployment feedback
4. **Causal Analysis**: Understanding causal factors in correction failures

## Citation

If you use this code or framework in your research, please cite:

**************


## Contributing

Contributions are welcome! If you find a bug or have a suggestion for improvement, please open an issue or submit a pull request.


## License
This project is dual-licensed:
- **Apache 2.0 License** – for academic and non-commercial use.  
- **Commercial License** – for commercial us
  
## Contact

For questions or issues, please open a GitHub issue or contact the author:

- *************
