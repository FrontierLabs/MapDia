# Joint RAG-based Framework for Memory-aware Proactive Dialogue (MapDia) Task

This repository provides an implementation of the joint RAG-based framework introduced in the paper "Interpersonal Memory Matters: A New Task for Proactive Dialogue Utilizing Conversational History" . The framework is designed to tackle the Memory-aware Proactive Dialogue (MapDia) Task , which focuses on utilizing conversational history to generate proactive and contextually relevant dialogues.


**[Interpersonal Memory Matters: A New Task for Proactive Dialogue Utilizing Conversational History](https://arxiv.org/abs/2503.05150)**
</br>
[![ChMapData Dataset](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Dataset-violet)](https://huggingface.co/datasets/FrontierLab/ChMap-Data)

## Overview

The project includes code for the inference process of the proposed framework, demonstrating how to combine a search module, ranker, and proactive dialogue model to generate proactive responses based on conversational history.

The required test data can be downloaded from the Huggingface Dataset repository:  
[FrontierLab/ChMap-Data](https://huggingface.co/datasets/FrontierLab/ChMap-Data).

While the repository does not include training scripts, you can driectly download data from our Huggingface repository to conduct data for training Topic Summary Model, Topic Retrieval Model, and Memory-Aware Proactive Response Generation Model.
The required test data can be downloaded from the Huggingface Dataset repository:
FrontierLab/ChMap-Data .

## Features

- Implements the inference pipeline for the MapDia task.
- Supports preprocessing of test data to extract key fields, reducing runtime overhead when running the full pipeline.

---

## Getting Started

### 1. Download Test Data

The test data required for inference can be downloaded from the Huggingface Dataset repository:
[FrontierLab/ChMap-Data](https://huggingface.co/datasets/FrontierLab/ChMap-Data).

Download the `overall_dialogue_review/test.json` file and place it in the root directory of this project.

### 2. Preprocess Test Data

To reduce runtime overhead during inference, preprocess the test data using the `test_data.py` script:
```bash
python test_data.py
```

This script extracts the necessary fields from the raw test data and saves them in a simplified format for faster processing.

### 3. Run Inference

Once the test data is preprocessed, you can run the inference pipeline such as:
```bash
python ours.py
```

The script will generate proactive dialogue responses based on the provided conversational history.

---

## Citation

If you find this project useful, please cite the original paper:

```bibtex
@misc{wu2025interpersonalmemorymattersnew,
      title={Interpersonal Memory Matters: A New Task for Proactive Dialogue Utilizing Conversational History}, 
      author={Bowen Wu and Wenqing Wang and Haoran Li and Ying Li and Jingsong Yu and Baoxun Wang},
      year={2025},
      eprint={2503.05150},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2503.05150}, 
}
```

---

For any questions or issues, please open an issue in this repository. Contributions are welcome!
