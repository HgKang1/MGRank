# MGRank

This is the code for our paper **MGRank**.

---
## Environment Setup
- pandas: 2.2.3
- numpy: 1.26.4
- torch: 2.5.1+cu121
- nltk: 3.9.1
- transformers: 4.48.0
- stanfordcorenlp: 3.9.1.1  

## Requirements

### 1. Stanford CoreNLP Setup

Download [Stanford CoreNLP](https://stanfordnlp.github.io/CoreNLP/history.html) from the official website and place it in the main folder of MGRank.   

Then **update the CoreNLP path** in both `data.py` and `att_data.py`:

```python
StanfordCoreNLP_path = 'your/path/to/stanford-corenlp-full-2018-02-27'
```
### 2. Modify T5 and Gemma2 Model Architecture
You need to modify the Hugging Face model source code to enable custom attention mechanisms.
1. Clone the Hugging Face transformers repository inside the main MGRank directory:

   ```bash
   cd MGRank
   git clone https://github.com/huggingface/transformers.git

2. Replace the original `modeling_t5.py` (or `modeling_gemma2.py`) in the `transformers/src/transformers/models/t5/` directory with the modified version provided in this repository (`modeling_t5_MGRank.py`).

## Running
1. To run MGRank, use the provided run_{model}.sh shell script.
2. The key parameters (std_scaling and att_weight for each dataset) can be modified in main_{model}.py via the DATASET_PARAM_MAP dictionary.
```python
DATASET_PARAM_MAP = {
    "SemEval2017": {"std_scaling": 0.1, "att_weight": 0.1},
    "DUC2001":     {"std_scaling": 0.1, "att_weight": 0.9},
    "nus":         {"std_scaling": 1, "att_weight": 0.1},
    "wikihow":     {"std_scaling": 0.7, "att_weight": 1.0}
}
```

## File Structure
- att_data.py: Calculates self-attention scores and finds top-k similar words

- data.py: Data loading and preprocessing pipeline

- inference.py: Model inference and keyphrase selection

- main.py: Main execution script

- run.sh: Shell script for easy batch execution of experiments with different parameter settings and datasets.

- modeling_{model}_MGRank.py: Modified version of Hugging Face’s modeling_{model}.py

### Custom Model Files

This repository includes a **modified version** of Hugging Face’s `modeling_t5.py` (and optionally `modeling_gemma2.py`) for custom attention mechanisms required by MGRank.

- The file(s) `modeling_{model}_MGRank.py` are **derivative works** of Hugging Face [transformers](https://github.com/huggingface/transformers).
- These files are distributed under the original [Apache 2.0 License](https://github.com/huggingface/transformers/blob/main/LICENSE).
- Please **replace** the corresponding files in your local `transformers` installation or set your model path to use these custom files.

### Implementation Note
Parts of the codebase were adapted from [PromptRank](https://github.com/NKU-HLT/PromptRank.git), with modifications to support the candidate-aware weighting and global-local SAM scoring mechanisms introduced in MGRank.


