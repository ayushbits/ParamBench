# ParamBench: A Graduate-Level Benchmark for Evaluating LLM Understanding on Indic Subjects

<div align="center">

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

</div>

## 📋 Overview

ParamBench is a comprehensive graduate-level benchmark in Hindi designed to evaluate Large Language Models (LLMs) on their understanding of Indic subjects. The benchmark contains **17,275 multiple-choice questions** across **21 subjects**, covering a wide range of topics from Indian competitive examinations.

This benchmark is specifically designed to:
- Assess LLM performance on culturally and linguistically diverse content
- Evaluate understanding of India-specific knowledge domains
- Provide a standardized evaluation framework for Indic language models
- Support the development of more culturally aware AI systems

## 🎯 Key Features

- **17,275 Questions**: Extensive collection of graduate-level MCQs in Hindi
- **21 Subjects**: Comprehensive coverage of diverse academic domains
- **Standardized Format**: Consistent question structure for reliable evaluation
- **Automated Evaluation**: Scripts for benchmarking and analysis
- **Detailed Metrics**: Subject-wise and question-type-wise performance analysis

## 📊 Dataset Structure

### Question Format
Each question in the dataset includes:
- `unique_question_id`: Unique identifier for each question
- `question_text`: The question text
- `option_a`, `option_b`, `option_c`, `option_d`: Four multiple choice options
- `correct_answer`: The correct option (A, B, C, or D)
- `subject`: Subject category
- `exam_name`: Source examination
- `paper_number`: Paper/section identifier
- `question_type`: Type of question (MCQ, Blank-filling, assertion/reasong, etc.)

### Subject Distribution
The benchmark covers 21 subjects including but not limited to:
- Music
- History
- Drama and Theatre
- Economics
- Anthropology
- Current Affairs
- Indian Culture
- And more...

## 🏗️ Repository Structure

```
ParamBench/
├── data/
│   └── full-data.csv          # Main dataset file
├── checkpoints/                    # Model evaluation checkpoints
├── results/                        # Analysis results and visualizations
├── benchmark_script.py             # Main benchmarking script
├── analysis_models.py              # Analysis and visualization script
├── requirements.txt                # Python dependencies
└── README.md                       # This file
```

## 🚀 Quick Start

### Requirements

```bash
pip install -r requirements.txt
```

### Basic Requirements
- Python 3.8+
- PyTorch 2.0+
- Transformers 4.45+
- Pandas
- NumPy
- Plotly (for visualization)

### Running Benchmarks

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/ParamBench.git
cd ParamBench
```

2. **Run the benchmark**
```bash
python benchmark_script.py
```

### Configuration Options

The benchmark script supports various configuration options:

```python
# In benchmark_script.py
group_to_run = "small"  # Options: "small", "medium", "large", or "all"
batch_size = 16         # Adjust based on GPU memory
```


## 📊 Running Analysis

After running benchmarks, generate comprehensive analysis reports:

```bash
python analysis_models.py
```

This will generate:
- Model performance summary CSV
- Interactive visualizations (HTML/SVG)
- Subject-wise accuracy charts
- Question type analysis
- Combined report with all metrics

## 🔗 Links

- [Paper](https://arxiv.org/abs/2508.16185)

---
