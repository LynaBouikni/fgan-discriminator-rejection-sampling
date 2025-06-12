# F-GAN with Discriminator Rejection Sampling

> Training GANs on MNIST · Variational Divergence Minimization · Rejection Sampling · Precision-Recall Trade-off

---

## 📌 Overview

This project investigates enhancements to Vanilla Generative Adversarial Networks (GANs) using two core methods:

- **F-GAN**: Extends GANs by optimizing f-divergences (e.g. KLD, JS, BCE) instead of relying only on Jensen-Shannon divergence.
- **Discriminator Rejection Sampling (DRS)**: A sampling method that leverages discriminator outputs to accept high-quality synthetic data.

All experiments are conducted on the **MNIST dataset** and evaluated with **FID**, **precision**, and **recall**.

---

## 🎯 Objectives

- ✅ Train a baseline Vanilla GAN with fixed architecture.
- ✅ Compare different f-divergence losses (BCE, JS, KLD).
- ✅ Implement DRS to refine sample quality and diversity.
- ✅ Analyze visual and quantitative performance over multiple epochs.

---

## 🗂 Dataset

- **MNIST**: 28x28 grayscale images of handwritten digits.
- **Training Set**: 60,000 samples  
- **Test Set**: 10,000 samples  
- Preprocessed into flat 784-dim vectors.

---

## 🧠 Model Architectures

### 🎛️ Vanilla GAN

- **Generator**: Fully connected feed-forward MLP with LeakyReLU activations and final Tanh.
- **Discriminator**: MLP with sigmoid output for binary classification.

```text
Generator: z ∈ ℝ¹⁰⁰ → G(z) ∈ ℝ⁷⁸⁴
Discriminator: x ∈ ℝ⁷⁸⁴ → D(x) ∈ [0,1]
Trained for 50 epochs with batch size = 64 and lr = 0.0001
```
n
