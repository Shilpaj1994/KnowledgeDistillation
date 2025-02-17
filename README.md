# Knowledge Distillation

- Knowledge Distillation [KD] is a technique used to train a smaller model from a larger model.

- The smaller model will perform similar to the larger model with the advantage of having lesser number of parameters

- KL Divergence loss is used for knowledge distillation

- Usually, we take a trained Teacher model (large model) and a untrained Student model (smaller model)

- Data is passed to both Teacher and Student model and the output is recorded

- KL loss tries to make of probability output of Student model same as Teacher model

- This way, using Knowledge Distillation Student model are trained quicker with less number of parameters

  

  

  ![](https://cdn.botpenguin.com/assets/website/Screenshot_2024_02_27_at_3_47_23_PM_7b2e510bf0.webp)

  ​																		Source: https://botpenguin.com/glossary/knowledge-distillation



In this repository, two examples are Knowledge Distillation are provided:

1. Trained a student ViT model on [oxford-pets](https://huggingface.co/datasets/pcuenq/oxford-pets) dataset, a datset with 37 classes of dogs and cats, to match the performance of larger pre-trained Teacher model.
2. Trained a student LLM model on [phishing site dataset](https://huggingface.co/datasets/shawhin/phishing-site-classification) to match the performance of the pre-trained Teacher model



---



## Example 1: Oxford-Pets Dataset

- [Training Notebook](./KnowledgeDistillaionViT.ipynb)

- Teacher Model:
  - Parameters: 85.8 M
- Student Model without Knowledge Distillation
  - Parameters: 43.3 M
  - Validation Accuracy after 5 epochs: 9.87%
- Student Model with Knowledge Distillation
  - Parameters: 43.3 M
  - Validation Accuracy after 5 epochs: 10.48%

This shows that with model with knowledge distillation is training quicker





## Example 2: Phishing Site Dataset

- [Training Notebook](./KnowledgeDistillationLLM.ipynb)

- Teacher Model Parameters: 109.48 M

- Student Model Parameters: 52.77 M

- Validation Accuracy after 5 epochs of training

  ```bash
  Teacher (validation) - Accuracy: 0.8340, Precision: 0.7911, Recall: 0.9245, F1 Score: 0.8526
  Student (validation) - Accuracy: 0.9542, Precision: 0.9728, Recall: 0.9380, F1 Score: 0.9551
  ```

This shows that student model is able to perform well even with half the number of parameters