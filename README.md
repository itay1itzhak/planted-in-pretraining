# Planted in Pretraining, Swayed by Finetuning

_Disentangling the Origins of Cognitive Biases in Language Models._

[![Paper](https://img.shields.io/badge/arxiv-paper-red)](https://arxiv.org/abs/2503.xxxx)
[![Models](https://img.shields.io/badge/🤗-models-yellow)](https://huggingface.co/collections/itay1itzhak/planted-in-pretraining)
[![Website](https://img.shields.io/badge/🌐-website-blue)](https://itay1itzhak.github.io/planted-in-pretraining/)
[![Contact](https://img.shields.io/badge/📧-contact-green)](mailto:itay1itzhak@gmail.com)

<div align="center">
  <img src="static/images/logo.png" alt="Project Logo" width="150"/>
</div>

---

## 📘 Introduction

This repository contains the code for our paper:

> **Planted in Pretraining, Swayed by Finetuning: A Case Study on the Origins of Cognitive Biases in LLMs**

We investigate the **origin of cognitive biases** in large language models (LLMs). While prior work showed these biases emerge and even intensify after instruction tuning, it's unclear whether they are caused by **pretraining**, **finetuning data**, or **training randomness**.

We propose a two-step **causal analysis framework**:

- First, we assess how much **random seed fluctuations** affect bias scores.
- Second, we perform **cross-tuning**: swapping instruction datasets between pretrained models to identify if biases are driven by the pretraining backbone or the finetuning data.

Our results show that:

- Training **randomness introduces some noise** in bias scores.
- However, **pretraining consistently dominates** as the primary source of biases, with instruction tuning playing a secondary role.

---

## 🧭 Repository Structure

This repository integrates and builds on three main sub-repositories:

- 📦 [`open-instruct`](https://github.com/allenai/open-instruct): Parameter-efficient LoRA finetuning framework.
- 📊 [`instructed-to-bias`](https://github.com/itay1itzhak/InstructedToBias): Evaluation for belief and certainty biases.
- 🧠 [`cognitive-biases-in-llms`](https://github.com/simonmalberg/cognitive-biases-in-llms): Benchmark suite for 30 cognitive biases.

Refer to those repositories for dataset structures, implementation details, and original evaluation scripts.

---

## 🔗 Model & Dataset Access

All trained models across seeds and the subsampled Flan instruction dataset are hosted on Hugging Face:

➡️ [**Hugging Face Collection**: `planted_in_pretraining`](https://huggingface.co/collections/itay1itzhak/planted-in-pretraining-68596cd05b50f3e93325b2d3)

---

## ⚙️ Environment Setup

We recommend setting up with `conda`:

```bash
conda create -n bias_origin python=3.10 -y
conda activate bias_origin
pip install -r requirements.txt

# Optional: install submodules in editable mode
git clone https://github.com/allenai/open-instruct.git
pip install -e open-instruct/

git clone https://github.com/itay1itzhak/InstructedToBias.git
pip install -e instructed-to-bias/

git clone https://github.com/simonmalberg/cognitive-biases-in-llms.git
pip install -e cognitive-biases-in-llms/
```

---

## 🚀 Key Analyses

### 🎲 Step 1: Training Randomness Analysis

> _What it checks:_  
> This experiment finetunes the **same model and dataset with different seeds** to test how much randomness affects bias scores.

```bash
python run_randomness_analysis.py --granularity-levels model_bias
```

> _What we found:_  
> Randomness introduces **minor fluctuations** in individual bias scores, but **averaging across seeds recovers stable patterns**. This suggests randomness alone is not a primary driver of cognitive bias.

---

### 🔁 Step 2: Cross-Tuning Clustering Analysis

> _What it checks:_  
> This analysis **swaps instruction datasets** between two pretrained models (e.g., Flan vs Tulu) and compares their **bias vectors**. We cluster models either by pretraining backbone or instruction data.

```bash
python run_similarity_analysis.py \
    --granularity-levels model_bias_scenario \
    --models-to-include T5,OLMo
```

> _What we found:_  
> Models cluster **strongly by pretraining** identity. Even after swapping instruction data, bias patterns remain closer to the original backbone than to the new data. This supports our main claim: **biases are planted during pretraining**.

---

## 📊 Visual Outputs

Example outputs (PDFs saved to `plots/`):

![Randomness Plot](docs/figs/randomness_effect.pdf)  
![Cross-Tuning PCA](docs/figs/cross_tuning_pca.pdf)

---

## 📚 Citation

To cite our work, use the BibTeX entry from Google Scholar.

---

## 📜 License

Apache License 2.0. See [`LICENSE`](LICENSE) for details.

---

## 📬 Contact

For questions or collaborations, please reach out via GitHub Issues or email:  
📧 [itay1itzhak at-gmail-com]
