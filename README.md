# HyperXRec: Cluster-Aware Explainable Recommendation via MoE-LLM

**HyperXRec** is a lightweight and modular recommendation framework that combines hyperspherical latent-space clustering with a LoRA-tuned Mixture-of-Experts Language Model (MoE-LLM) for personalized explanation generation. It is designed to bridge the gap between recommendation reasoning and natural language explanations under both sparse and dense user interaction scenarios.

---

## 🚀 Highlights

- **Latent Clustering on the Unit Hypersphere**  
  Learns compact user-item latent representations using a vMF (von Mises-Fisher) mixture prior, enhanced with kNN-based perturbation and cosine consistency.

- **Cluster-Guided Explanation via MoE-LLM**  
  Uses cluster assignments to route user-item pairs to LoRA-finetuned lightweight LLM experts, enabling personalized and semantically faithful explanations.

- **Lightweight & Scalable**  
  Expert modules are efficient and modular, making training feasible under limited resources.

---

## 📂 Project Structure


