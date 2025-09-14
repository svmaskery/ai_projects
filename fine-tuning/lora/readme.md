# SmolLM-LoRA
This repository contains the code and configurations for fine-tuning the Huggingface SmolLM2-135M model on a mental health dataset using Low-Rank Adaptation (LoRA). The goal is to create a specialized model for conversational mental health support while leveraging efficient training techniques.

### LoRA Fine-Tuning on Mental Health Dataset 🧠

This project uses the LoRA technique to adapt the pre-trained model to the specific domain of mental health conversations. The fine-tuning dataset is comprised of anonymized conversational data between patients and mental health professionals. The dataset is preprocessed into a conversational format to teach the model to generate empathetic and helpful responses.

LoRA works by freezing the original pre-trained model weights and introducing a small number of trainable, low-rank matrices (adapters) into specific layers. During training, only these new matrices are updated, which drastically reduces the number of parameters that need to be trained. The final model is a combination of the frozen pretrained model weights and the newly trained LoRA adapters.

Also, quantization has not been used in the current project since this was purely for experimentation. In the future fine-tuning projects, various quantization methods will be used.

### Small Model for Better Convergence 📉

Opted for a small model (SLM), SmolLM2, a relatively small but powerful open-source language model. While larger models can offer superior performance, fine-tuning them is computationally expensive. For a specialized task like mental health support, a smaller model is often sufficient and offers several advantages:

*Faster Convergence*: With a smaller parameter space to fine-tune, the model can converge on the specific task much quicker.

*Reduced Overfitting*: Training on a smaller dataset with a smaller model helps prevent the model from overfitting to the training data, improving its generalization to new conversations.

*Lower Computational Costs*: Using a smaller model requires significantly less GPU memory and compute, making the fine-tuning process accessible on consumer-grade hardware like a single NVIDIA A100 or even an NVIDIA T4 GPU.

*Easier Deployment*: The resulting fine-tuned model is smaller and more memory-efficient, which is critical for deployment in resource-constrained environments.

### Motivation: LoRA vs. Full Supervised Fine-Tuning (SFT) 🤔

Using LoRA instead of traditional full supervised fine-tuning (SFT), which updates all model parameters, offers key benefits:

*Efficiency*: Full SFT requires a massive amount of GPU memory and computational power, often demanding multiple high-end GPUs. LoRA, in contrast, reduces the number of trainable parameters by orders of magnitude (e.g., from billions to a few million), making fine-tuning feasible on a single GPU. This significantly lowers training time and cost.

*Preventing Catastrophic Forgetting*: A major drawback of full SFT is catastrophic forgetting, where the model "forgets" its vast general knowledge learned during pre-training as it over-specializes on the new, smaller dataset. LoRA mitigates this by keeping the original model weights frozen, preserving the model's core capabilities while the adapters learn task-specific knowledge.

*Modularity and Storage*: The LoRA adapters are small, typically in the megabyte range. This makes it easy to store multiple fine-tuned versions for different tasks without saving a full copy of the entire model for each.

*Performance*: Studies have shown that LoRA can achieve performance on par with or even surpass full fine-tuning on many tasks, especially for domain-specific applications, by efficiently adapting the model's knowledge without corrupting its foundation.