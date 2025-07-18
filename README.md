# F-GAN with Discriminator Rejection Sampling

[![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1.0-EE4C2C?logo=pytorch)](https://pytorch.org/)


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
🔀 f-GAN
Instead of minimizing the original GAN loss, we minimize f-divergences:

BCE: Binary Cross Entropy (baseline)

KLD: Kullback-Leibler Divergence

JS: Jensen-Shannon Divergence

Each divergence affects convergence behavior, quality, and diversity.

🔍 Discriminator Rejection Sampling (DRS)
DRS uses the discriminator’s confidence scores to accept or reject generated samples:

Discriminator Scoring: Scores all generated images.

Acceptance Probability: Uses sigmoid-adjusted function based on max score M and a hyperparameter γ.

Dynamic Thresholding: M is updated online.

Sample Selection: Repeats until 10,000 accepted samples are saved.

## 📊 Results
Model	Time (s)	FID	Precision	Recall
Vanilla GAN	111.4	52.44	0.52	0.18
F-GAN (JS, 100 epochs)	~	~	↑	↑
F-GAN + DRS (KLD, 50e)	~	↓	↑↑	↑↑

🔍 DRS significantly improved both precision and diversity compared to plain GANs.

## 📁 Code Structure
```bash
F-GAN-Rejection-Sampling/
├── src/                   # Python code modules
│   ├── train.py
│   ├── generate.py
│   ├── DRS.py
│   ├── model.py
│   ├── utils.py
│   └── evaluation_metrics.py
│
├── Checkpoints/          # Trained models (e.g., D.pth)
│
├── docs/                 # Supporting files
│   ├── Presentation.pdf
│   └── Report.pdf
│
├── samples/              # Output images / visualizations
│
├── requirements.txt
├── README.md
└── .gitignore

```
🧾 Dependencies
```bash
numpy==1.25.1
scikit-learn==1.3.0
scipy==1.11.1
tqdm==4.64.1
torch==2.1.0+cu121
torchvision==0.16.0+cu121
matplotlib==3.8.1
```

Install with:
```bash
pip install -r requirements.txt
```

## 🧪 Insights

🔁 f-divergence choice affects convergence: JS diverges slower than BCE but provides more stable training.

🎯 DRS boosts recall and image diversity at low computational cost.

🧭 Optimal epochs vary by divergence: JS and KLD plateau early, while BCE improves slowly over time.

## 📚 References
Nowozin et al., f-GAN: Variational Divergence Minimization

Azadi et al., Discriminator Rejection Sampling

## 👥 Authors
The GANtastics

* Lyna Bouikni

* Arij Boubaker

* Ángel Luque

🧑‍🎓 M2 IASD — ENS, Dauphine, Mines Paris, 2023

## 📬 Contact
📫 Email: lynabouiknia@gmail.com
🔗 LinkedIn

“Precision means nothing without diversity — and diversity is everything in generative modeling.”
