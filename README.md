# Heronix NeuroSim v3.0

A desktop neural network simulator for education and experimentation, built with JavaFX.

Developed by **Michael Katsaros** at **Heronix Education Systems LLC** as a free educational tool.

---

## Features

- **Backpropagation Engine** — Feedforward networks with arbitrary layers, configurable activation functions, online/batch gradient descent, momentum, MSE and cross-entropy error functions
- **Multiple Activation Functions** — Sigmoid, ReLU, Tanh, Softmax (selectable per hidden/output layer)
- **Step-by-Step Training** — Walk through individual forward/backward passes to see net inputs, activations, deltas, and weight updates for each layer
- **Real-Time Visualization** — Live network topology with color-coded weights and node activations
- **Training Error Chart** — Responsive graph tracking per-epoch error convergence
- **Architecture Demos** — Interactive demos for RNN (Elman network), 1D CNN, and Self-Attention mechanisms
- **Built-In Tutorials** — 10 interactive tutorial topics covering neural networks, backprop, activation functions, RNN, CNN, attention, and training tips
- **Tooltips Everywhere** — Hover over any control for educational explanations
- **Pattern File I/O** — Load/save `.pat` files compatible with the Heronix NeuroSim format
- **Network Config & Weights** — Save/load network configurations (`.cfg`) and trained weights (`.wgt`)
- **Report Export** — Export training reports in ASCII or binary format
- **Built-In Presets** — AND, OR, NAND, NOR, XOR, Parity, Half Adder, Encoder, Identity
- **Parallel Batch Training** — Uses Java parallel streams for faster batch gradient descent on multi-core CPUs
- **3 Themes** — Hacker Green, Dark, and System/Light
- **Auto-Configure** — Network automatically adjusts input/output layers when loading pattern files

---

## Requirements

| Tool  | Version |
|-------|---------|
| Java  | 17–25   |

That's it. The fat JAR bundles all dependencies including JavaFX.

---

## Quick Start

### Option 1: Run the pre-built JAR

```bash
java -jar HeronixNeuroSim-3.0.0.jar
```

On **Windows**, double-click `run.bat`.
On **macOS/Linux**, run `./run.sh`.

### Option 2: Build from source

Requires Java 17+ and Maven 3.6+.

```bash
mvn clean package
java -jar target/heronix-neurosim-3.0.0-fat.jar
```

Or run directly with Maven:

```bash
mvn javafx:run
```

---

## Usage

1. **Load patterns** — Use the Examples menu for presets, or Patterns tab to load a `.pat` file
2. **Configure network** — Config tab: set layers, units, bias, and activation functions
3. **Set parameters** — Train tab: adjust learning rate, momentum, max epochs
4. **Train** — Click TRAIN and watch the error converge on the chart
5. **Step-by-step** — Step tab: walk through one forward/backward pass at a time
6. **Explore architectures** — Archs tab: try RNN, CNN, and Attention demos
7. **Learn** — Learn tab: interactive tutorials on every topic
8. **Test** — Select a pattern and click TEST ONE, or click TEST ALL
9. **Save results** — Network menu to save weights, File > Export Report

---

## Pattern File Format

Plain text, same format as the Heronix NeuroSim `.pat` files:

```
Number of patterns = 4
Number of inputs = 2
Number of outputs = 1
[Patterns]
0 0   0
1 0   0
0 1   0
1 1   1
```

Values can be separated by spaces or tabs. Lines starting with `#` are treated as comments.
The keyword `reset` is recognised (and ignored for feedforward networks).

---

## Project Structure

```
Heronix-NeuroSim/
├── pom.xml                          Maven build config
├── LICENSE                          MIT License
├── README.md                        This file
├── HeronixNeuroSim-3.0.0.jar       Pre-built distributable
├── run.bat                          Windows launcher
├── run.sh                           macOS/Linux launcher
├── seven_segment.pat                Example pattern file
└── src/main/
    ├── java/com/heronix/neurosim/
    │   ├── Launcher.java            Fat JAR entry point
    │   ├── MainApp.java             Main UI application (JavaFX)
    │   ├── NeuralNet.java           Backpropagation engine
    │   ├── NetworkCanvas.java       Network visualization canvas
    │   ├── PatternParser.java       Pattern file parser
    │   ├── TrainingPattern.java     Input/target data record
    │   ├── ActivationFunction.java  Sigmoid, ReLU, Tanh, Softmax
    │   ├── RecurrentNet.java        Simple Elman RNN demo
    │   ├── ConvNet.java             Simple 1D CNN demo
    │   └── AttentionDemo.java       Self-attention mechanism demo
    └── resources/com/heronix/neurosim/
        ├── theme-hacker.css         Hacker Green theme
        ├── theme-dark.css           Dark theme
        └── theme-system.css         System/Light theme
```

---

## Algorithm Details

- **Activation:** Sigmoid, ReLU, Tanh, or Softmax (configurable per layer)
- **Weight update:** Online (stochastic GD) or Batch (parallel for 16+ patterns)
- **Error function:** MSE or Cross-Entropy (selectable)
- **Momentum:** Standard first-order momentum term
- **Convergence:** Training stops when avg per-pattern error < 0.001 or max epochs reached
- **RNN Demo:** Elman network with context units for sequential pattern learning
- **CNN Demo:** 1D Convolution → ReLU → Max Pooling → Fully Connected → Sigmoid
- **Attention Demo:** Single-head self-attention with Q/K/V projections and scaled dot-product

---

## Comparison with Similar Tools

| Feature | BasicProp (2011) | Heronix NeuroSim | TensorFlow Playground | Neural Network Sandbox |
|---|---|---|---|---|
| **Status** | Retired | Active | Active (web) | Active (web) |
| **Platform** | Desktop (unknown) | Desktop (Win/Mac/Linux) | Browser | Browser |
| **Offline use** | Yes | Yes | No | No |
| **Custom patterns** | Yes (.pat files) | Yes (.pat files) | No (fixed datasets) | Limited |
| **Real-time viz** | Basic | Excellent | Excellent | Good |
| **Error chart** | Unknown | Yes | Yes | No |
| **Save/load weights** | Unknown | Yes | No | No |
| **Presets** | 3 (AND/OR/XOR) | 9+ | 4 datasets | Limited |
| **Batch/Online GD** | Unknown | Both (parallel batch) | N/A | N/A |
| **Error functions** | Unknown | MSE + Cross-Entropy | N/A | N/A |
| **Activation functions** | Sigmoid only | Sigmoid, ReLU, Tanh, Softmax | Limited | Limited |
| **Architecture types** | Feedforward | FF + RNN + CNN + Attention | FF only | FF only |
| **Step-by-step mode** | No | Yes | No | No |
| **Built-in tutorials** | No | Yes (10 topics) | Partial | No |
| **Themes** | No | 3 themes | No | No |
| **Export reports** | No | ASCII + Binary | No | No |

---

## License

MIT License — Copyright (c) 2026 Heronix Education Systems LLC. Developed by Michael Katsaros.

Free to use, modify, and distribute for any purpose. See [LICENSE](LICENSE) for details.
