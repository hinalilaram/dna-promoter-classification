# DNA Promoter Classification

Identifying core promoter sequences in DNA (binary classification: promoter vs. non-promoter) using transformer-based sequence embeddings and classical machine learning.

## Project Overview

This project implements and compares two modern ML approaches for promoter region classification:
- Fine-tuning a scientific large language model (LLM): [DNABERT2](https://huggingface.co/zhangtaolab/dnabert2-promoter)
- Training classical machine learning classifiers on deep [CLS] embeddings extracted from DNABERT2

Includes reproducible code for preprocessing, embedding extraction, PCA, model training, and evaluation.

## Scientific Context

The work is inspired by the survey _“Scientific Large Language Models: A Survey on Biological & Chemical Domains”_ (Zhang, Ding et al., 2024), and demonstrates how transformer models and their representations can be used for practical genomics ML tasks.

## Repository Structure

```bash
dna-promoter-classification/
├── main.py # Main pipeline: data loading, training, evaluation
├── model.py # Transformer model loading, embedding extraction, ML classifier definitions
├── utils.py # Data utilities, metrics, result tables, splitting
├── requirements.txt # Project dependencies
├── README.md # Project documentation
└── (other files: notebooks, reports, etc.)
```bash
## Installation

Clone the repository:
```bash
git clone https://github.com/hinalilaram/dna-promoter-classification.git
cd dna-promoter-classification
```
Install dependencies (preferably in a virtual environment):


## Usage

Run the main pipeline:


- The script loads and splits the DNA core promoter dataset, extracts transformer embeddings, applies PCA, trains/evaluates multiple ML classifiers, and compares results.
- Outputs include accuracy, macro F1, precision, and recall for each classifier.

## Dataset

- Uses [dnagpt/dna_core_promoter](https://huggingface.co/datasets/dnagpt/dna_core_promoter)
- Downloaded automatically via HuggingFace Datasets API.

## Model

- Utilizes [DNABERT2](https://huggingface.co/zhangtaolab/dnabert2-promoter) for embedding extraction and/or direct classification.

## Results

- Classical ML algorithms trained on PCA-reduced DNABERT2 embeddings (SVM, Random Forest, etc.) versus the DNABERT2 classification head.
- See the report for comparative performance and insights.

## Citation

If you use this code for research or presentation, please cite both this repo and the survey paper.

