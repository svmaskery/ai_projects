# Building GPT-like LLM From Scratch
The code base contains all the necessary elements to build an LLM from scratch.
Script description:
1. attention.py - Base attention layer
2. multi_head_attention.py - Contains two implementations of MHA
3. ffn.py - Feed forward network with GELU activation
4. normalization.py - Layer Normalization layer
5. transformer_block.py - Entire transformer block consisting of all the building blocks
6. full_transformer.py - GPT-like LLM model which brings all the elements together
7. transformer_from_scratch.ipynb - Training the build LLM model
8. simple_tokenizer.ipynb - Simple tokenization techniques