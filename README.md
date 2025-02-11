# Fine-tuning a language model with RL on an arithmetic task

**🚧 Repository under development. 🚧**

This is the official implementation of the paper [Ignore the KL Penalty! Boosting Exploration on Critical Tokens to Enhance RL Fine-Tuning](https://arxiv.org/abs/2502.06533).

## Installation

Please install the required packages provided in ```requirements.txt```

## Reproduce experiments

### Prerequisite: download the pre-trained models

To run our RL fine-tuning experiments, you first need to download the pre-trained models.
All models are available on HuggingFace, in ```https://huggingface.co/lecraquito```.
To download any pre-trained model, you can run these commands from the root of the repo:

```bash
git config --global credential.helper store
module load git-lfs
git lfs install
git clone https://huggingface.co/lecraquito/gpt2_reduced_vocab_FT_9digits_20k
```

### Comparison of varying levels of pre-training (section 5.2)

Please run the following command from the root of the repo:

```bash
python -m src.rl_compare_pretrain
```

### Influence of the prioritized KL divergence (section 5.3)

Please run the following command from the root of the repo:

```bash
python -m src.rl_train
```
