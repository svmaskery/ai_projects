# Mental-Health-SFT
This repository documents the experimental fine-tuning of the SmolLM2 model on a mental health dataset using Full Supervised Fine-Tuning (SFT). This experiment aims to understand the performance and computational trade-offs of this approach compared to more parameter-efficient methods like LoRA.

### Full Supervised Fine-Tuning (SFT) on Mental Health Dataset 🧠

This project uses Full (traditional) SFT to adapt the pre-trained SmolLM2 model to the specific domain of mental health conversations. Unlike methods that only update a small subset of parameters, SFT trains all parameters of the model. The dataset is the same as the one used for other experiments, consisting of conversational data from various mental health resources. The model learns to generate empathetic and supportive responses by adjusting its entire weight matrix based on this specialized data.

Full SFT involves loading the entire model and updating every weight during the backpropagation process. This approach allows the model to deeply integrate the new knowledge into its core structure. However, it requires significant computational resources.

### Small Model for Experimental Purposes 🧪

Chose the SmolLM2-135M model for this experiment. While a larger model might offer a higher performance ceiling, a smaller model is more practical for a controlled experiment. The choice of a small model parameter allowed me to:

*Manage Computational Costs*: Full SFT on a large model with over 10B parameters would be prohibitively expensive and time-consuming. Using the 135M version makes the experiment feasible on a high-end single GPU or a small cluster, allowing for a quicker turnaround.

*Observe Overfitting*: By using a smaller model, we can more clearly observe the effects of catastrophic forgetting, a key concern with full fine-tuning. We can monitor how the model's general capabilities degrade as it over-specializes on the mental health dataset.

*Establish a Baseline*: The results from this experiment serve as a benchmark for comparing against results from parameter-efficient fine-tuning methods like LoRA and QLoRA. This helps us quantify the performance and efficiency gains of those methods.

### Motivation: Why Full SFT for an Experiment? 🤔

The primary motivation for using Full SFT in this experimental context is to establish a performance upper bound and understand its limitations.

*Performance Baseline*: Full SFT is often considered the gold standard for fine-tuning since it has the potential to achieve the best possible performance by modifying every parameter. By conducting this experiment, we can see the "best-case scenario" for performance, which provides a valuable metric for evaluating the performance of more efficient methods like LoRA.

*Observing Catastrophic Forgetting*: A key hypothesis of this experiment is that Full SFT will lead to catastrophic forgetting. We expect to see a decrease in the model's performance on general tasks (e.g., answering questions about history or science) after fine-tuning. Observing this phenomenon directly helps us understand the importance of methods like LoRA that are designed to mitigate it.

*Resource Analysis*: This experiment allows us to precisely measure the computational resources (GPU memory, VRAM, and training time) required for Full SFT on a given dataset and model size. This data is crucial for making informed decisions about which fine-tuning approach to use for production applications.