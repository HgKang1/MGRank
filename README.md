# MGRank

This is the official code for our paper **MGRank**.

---

## Requirements

### 1. Stanford CoreNLP Setup

Download [Stanford CoreNLP](https://stanfordnlp.github.io/CoreNLP/) from the official website.  
Then **update the CoreNLP path** in both `data.py` and `att_data.py`:

```python
StanfordCoreNLP_path = 'your/path/to/stanford-corenlp-full-2018-02-27'
```
### 2. Modify T5 and Gemma2 Model Architecture
You need to modify the Hugging Face model source code to enable custom attention mechanisms.
**How to do this:**
1. Clone the Hugging Face transformers repository:

    ```bash
    git clone https://github.com/huggingface/transformers
    ```

2. Place the entire MGRank directory (this codebase) inside the transformers main folder (or ensure paths are correctly set).

3. Replace the original `modeling_t5.py` (or `modeling_gemma2.py`) in the `transformers/models/t5/` directory with the modified version provided in this repository (`modeling_t5_2.py`).

4. Replace the original model files (e.g., modeling_t5.py or modeling_gemma2.py) with the modified versions provided in this repository (e.g., modeling_t5_2.py).
   Use the file provided in this repo:
 modeling_t5_2.py
### 3. Modi
