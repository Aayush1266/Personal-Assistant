# 🤖 Instruction-Based Personal Assistant using GPT-2 (124M)

This project involves fine-tuning a GPT-2 (124M) language model to function as a simple instruction-following assistant. The assistant is capable of understanding and responding to short prompts such as spelling corrections, factual lookups, and basic tasks based on text instructions.

---

## 📌 Project Overview

- Fine-tuned GPT-2 on 1100+ instruction-response examples
- Supports instruction-only and instruction + input format
- Designed without using Hugging Face or external APIs
- Uses custom training pipeline and tokenizer from scratch

---

## 🧠 Model Architecture

- **Base Model**: GPT-2 (124M parameters)
- **Embedding Dimension**: 768
- **Layers**: 12 Transformer blocks
- **Attention Heads**: 12
- **Custom Components**:
  - Byte Pair Encoding (BPE) tokenizer
  - Multi-head Self-Attention
  - GELU Activation
  - Layer Normalization
  - Custom Positional Embedding
  - Collate Function for padded input alignment

---

## 📚 Dataset

- **Source**: [LLMs-from-Scratch by Sebastian Raschka](https://github.com/rasbt/LLMs-from-scratch)
- **Structure**:
  ```json
  {
    "instruction": "Identify the correct spelling of the word.",
    "input": "Ocassion",
    "output": "The correct spelling is 'Occasion.'"
  }
  
## Citation
Based on materials and code from:

Sebastian Raschka, LLMs-from-Scratch
🔗 https://github.com/rasbt/LLMs-from-scratch
