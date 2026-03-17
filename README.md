# 🌌 Adversarial Vision Engine

> **A research-oriented deep learning project exploring Generative Adversarial Networks (GANs) for synthetic image generation and understanding generative learning dynamics.**

---

## 📌 Overview

Adversarial Vision Engine is a research-driven implementation of **Generative Adversarial Networks (GANs)** designed to study how machines can learn data distributions and generate realistic images.

Instead of treating GANs as a black box, this project focuses on:

* Understanding adversarial training dynamics
* Observing convergence behavior
* Analyzing instability and failure modes
* Comparing performance across datasets

Two complete GAN pipelines are implemented:

| Model                | Dataset  | Output                 |
| -------------------- | -------- | ---------------------- |
| **GenerativeVision** | MNIST    | 28×28 grayscale digits |
| **ChromaGAN**        | CIFAR-10 | 32×32 RGB images       |

---

## 🎯 Research Objective

The goal of this project is to explore:

* How neural networks learn to generate data distributions
* The effect of architecture on generative quality
* Training challenges such as instability and mode collapse
* Differences between simple and complex datasets

> *The focus is not perfect image generation, but understanding the generative process.*

---

## 🧠 Motivation

Generating realistic images from random noise is a fundamentally complex task.

This project investigates the core question:

> **Can a model learn to generate realistic images purely from data without explicit supervision?**

GANs approach this problem using an adversarial setup where two networks improve through competition.

---

## 📐 Methodology

This project is based on the original GAN framework:

* Generator (G): Generates fake images from noise
* Discriminator (D): Classifies images as real or fake

Objective function:

```
min_G max_D E[log D(x)] + E[log(1 - D(G(z)))]
```

Training proceeds as a minimax game where:

* Generator tries to fool the discriminator
* Discriminator tries to correctly classify inputs

---

## 🏗️ Architecture

A **DCGAN-style architecture** is used for improved performance.

### Generator

* ConvTranspose2d layers
* Batch Normalization
* ReLU activations
* Tanh output

### Discriminator

* Conv2d layers
* Batch Normalization
* LeakyReLU activations
* Sigmoid output

### Key Design Choices

* Convolutional layers for spatial feature learning
* Adam optimizer with β1 = 0.5 for stability
* No label smoothing (kept close to original research)

---

## ⚙️ Tech Stack

| Tool        | Purpose              |
| ----------- | -------------------- |
| Python      | Core programming     |
| PyTorch     | Model training       |
| torchvision | Dataset handling     |
| Streamlit   | Interactive UI       |
| Matplotlib  | Visualization        |
| NumPy       | Numerical operations |

---

## 🚀 Getting Started

### 1. Install Dependencies

```bash
pip install torch torchvision streamlit matplotlib numpy pillow
```

### 2. Train MNIST GAN

```bash
python mnist_gan/train_mnist_gan.py
```

### 3. Train CIFAR-10 GAN

```bash
python cifar10_gan/train_cifar10_gan.py
```


---

## 🖥️ Features

* Generate synthetic images from noise
* Control randomness using seeds
* Latent space interpolation
* Adjustable noise scaling (temperature)
* Download generated outputs
* Plug-and-play model weights

---

## 📊 Results

### MNIST

* Clear digit structures after ~15 epochs
* High diversity in generated samples

### CIFAR-10

* Captures basic object structure
* Limited sharpness due to dataset complexity

### Observations

* Early training: noise
* Mid training: structure emergence
* Late training: semantic patterns

---

## ⚠️ Limitations

* Mode collapse in some training runs
* Training instability due to adversarial dynamics
* Blurry outputs on complex datasets
* Sensitive to hyperparameters

---

## 🔭 Future Work

* Wasserstein GAN (WGAN / WGAN-GP)
* Conditional GANs (cGAN)
* StyleGAN architectures
* Diffusion models
* FID score evaluation
* Domain-specific datasets (medical, satellite)

---

## 📚 Research References

**Generative Adversarial Nets (2014)**
Goodfellow et al.
https://arxiv.org/abs/1406.2661

**DCGAN (2015)**
Radford et al.
https://arxiv.org/abs/1511.06434

---

## 💡 Key Learnings

* GAN training is highly unstable and sensitive
* Architecture significantly affects output quality
* Theory and practical behavior differ greatly
* Latent space encodes meaningful structure

---

## 🔗 Research Direction

This project builds a foundation for future work in:

* Generative Models
* Multimodal Learning
* Large Language Models
* Hallucination Mitigation in AI systems

---

## 👨‍💻 Author

**Rishikesh Raj**
B.S. – IIT Patna

* GitHub: https://github.com/Rishikesh1411
* LinkedIn: https://linkedin.com/in/rishikesh-raj1411

---
