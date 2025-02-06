# 🕸️ **Harmonic Oscillator Network Embedding (HONE)** Framework 🕸️

This framework treats networks like a system of connected oscillators:
- **Nodes** are represented as round objects (particles).
- **Edges** are modeled as springs connecting the nodes.

Using classical mechanics principles, the **HONE framework** computes equilibrium configurations to map network topology, minimizing the total potential energy through **gradient descent optimization**. This process aims to find optimal node positions in a lower-dimensional space. The **Harmonic Network Inconsistency (HNI)** metric is used to quantify network complexity and variability. 📈

---
⚙️ **Advanced Usage**:

HONE provides a flexible interface for customization:

- **`num_steps`**: Set the number of iterations for the gradient descent optimization process (default: 1000).
- **`dim`**: Define the dimensionality of the embedding space (default: 2).
- **`learning_rate`**: Adjust the learning rate for gradient descent updates (default: 0.01).
- **`seed`**: Random seed for reproducibility in the **`HONE`** function (optional).
- **`seed_ensemble`**: Number of ensemble runs to compute HNI (default: 10).

> **⚠️ Note**: If additional dynamics like damping were to be introduced, avoid setting `gamma` to `0` or very low values to prevent numerical instability due to floating-point errors. Choose small, non-zero values for low damping effects.

---

## 🌟 **Key Highlights**:
1. **Dual Implementation**:
   - **`parallel_HONE.py`**: Optimized for parallel computation using multi-core processors, providing faster ensemble-based results.
   - **`HONE.py`**: Single-process implementation of the harmonic oscillator network embedding.
2. **Node Dynamics**: Understand how individual nodes (particles) interact in the network space.
3. **Spring-like Edges**: Capture structural dependencies through harmonic spring couplings.
4. **Global Perspective**: Analyze the energy landscape for insights into node configurations and network structure.
5. **Harmonic Network Inconsistency (HNI)**: Quantifies the variability and consistency of network embeddings across multiple runs.

   This framework uses **gradient descent optimization** to iteratively adjust node positions, minimizing the total potential energy. Each optimization step aligns the network with its structural dependencies, helping to uncover hidden patterns and complex relationships in large-scale networks.
---

## 🚀 **Getting Started**
### 1️⃣ Clone the repository:
   ```bash
   git clone https://github.com/your-repo/HONE.git
   cd HONE
```
### 2️⃣ Install dependencies:
Install the required Python packages using the following command:

  ```bash
  pip install -r requirements.txt
```
### 3️⃣ Run the Example:
Run the example script to see **HONE** in action:
```
bash
python example.py
```
### 🌀 What You’ll Observe:

When you run this script, you'll uncover:

- ⚪ **Node embeddings**: Nodes behaving like particles, arranged in a multidimensional space according to the harmonic oscillator model.
- 🌊 **Energy landscapes**: Visualize the interaction dynamics through spring-like edges, unveiling the topology and geometry of the network.
- 📊 **Harmonic Network Inconsistency (HNI)**: A unique metric that quantifies the configurational variability and complexity of the network.
- 🛠️ **Customizability**: Modify parameters to adapt HONE for your specific use case.

All generated visualizations and data outputs will be saved to the `results/` directory for further analysis.

---


To adjust these parameters, simply update the `config.json` file or pass them directly as arguments when running the script:
```bash
python example.py --iterations 500 --dim 3 --dt 0.01 --gamma 1.5```
```
