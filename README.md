# 🪐 **Harmonic Oscillator Network Embedding (HONE)** Framework 🪐

Welcome to the **Harmonic Oscillator Network Embedding (HONE)** repository! 🎉 Unleash the power of physics-based modeling to analyze networks 🌌. 

This framework treats networks like a system of connected oscillators:
- ⚪ **Nodes** are represented as round objects (particles).
- 🌀 **Edges** are modeled as springs connecting the nodes.

Using classical mechanics principles, the **HONE framework** computes equilibrium configurations to map network topology while quantifying its complexity through the **Harmonic Network Inconsistency (HNI)** metric. 📈

---

## 🌟 **Key Highlights**
- ⚪ **Node Dynamics**: Understand how individual nodes (particles) interact in a network.
- 🌀 **Spring-like Edges**: Capture structural dependencies via harmonic couplings.
- 🌌 **Global Perspective**: Analyze energy landscapes for insights into configurational variability.
- ⚡ **Efficient Computation**: Optimized for both CPU and GPU.

---

## 🚀 **Getting Started**
### 1️⃣ Clone the repository:
   ```bash
   git clone https://github.com/your-repo/HONE.git
   cd HONE```
### 2️⃣ Install dependencies:
Install the required Python packages using the following command:

  ```bash
  pip install -r requirements.txt```
### 3️⃣ Run the Example:
Run the example script to see **HONE** in action:

```bash
python example.py```

### 🌀 What You’ll Observe:

When you run this script, you'll uncover:

- ⚪ **Node embeddings**: Nodes behaving like particles, arranged in a multidimensional space according to the harmonic oscillator model.
- 🌊 **Energy landscapes**: Visualize the interaction dynamics through spring-like edges, unveiling the topology and geometry of the network.
- 📊 **Harmonic Network Inconsistency (HNI)**: A unique metric that quantifies the configurational variability and complexity of the network.
- 🛠️ **Customizability**: Modify parameters to adapt HONE for your specific use case.

All generated visualizations and data outputs will be saved to the `results/` directory for further analysis.

---

### ⚙️ Advanced Usage:

HONE provides a flexible interface for customization:

- **`iterations`**: Set the number of iterations for the numerical solver.
- **`dim`**: Define the dimensionality of the embedding space.
- **`seed_ensemble`**: Control the number of ensemble runs to compute HNI.
- **`dt`**: Adjust the time step for the dynamics.
- **`gamma`**: Modify the damping coefficient for harmonic oscillations.

To adjust these parameters, simply update the `config.json` file or pass them directly as arguments when running the script:

```bash
python example.py --iterations 500 --dim 3 --dt 0.01 --gamma 1.5```

🚀 Uncover the hidden dynamics of networks with HONE! ⚪🌊📊
