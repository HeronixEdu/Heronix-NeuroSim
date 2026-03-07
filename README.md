# Heronix NeuroSim v2.0

A desktop neural network simulator for education and experimentation, built with JavaFX.

Developed by **Michael Katsaros** at **Heronix Education Systems LLC** as a free educational tool.

---

## Features

- **Backpropagation Engine** — Feedforward networks with arbitrary layers, sigmoid activation, online/batch gradient descent, momentum, MSE and cross-entropy error functions
- **Real-Time Visualization** — Live network topology with color-coded weights and node activations
- **Training Error Chart** — Responsive graph tracking per-epoch error convergence
- **Pattern File I/O** — Load/save `.pat` files compatible with the original BasicProp format
- **Network Config & Weights** — Save/load network configurations (`.cfg`) and trained weights (`.wgt`)
- **Report Export** — Export training reports in ASCII or binary format
- **Built-In Presets** — AND, OR, NAND, NOR, XOR, Parity, Half Adder, Encoder, Identity
- **3 Themes** — Hacker Green, Dark, and System/Light
- **Auto-Configure** — Network automatically adjusts input/output layers when loading pattern files

---

## Requirements

| Tool  | Version |
|-------|---------|
| Java  | 17+     |

That's it. The fat JAR bundles all dependencies including JavaFX.

---

## Quick Start

### Option 1: Run the pre-built JAR

```bash
java -jar HeronixNeuroSim-2.0.0.jar
```

On **Windows**, double-click `run.bat`.
On **macOS/Linux**, run `./run.sh`.

### Option 2: Build from source

Requires Java 17+ and Maven 3.6+.

```bash
mvn clean package
java -jar target/basicprop-2.0.0-fat.jar
```

Or run directly with Maven:

```bash
mvn javafx:run
```

---

## Usage

1. **Load patterns** — Use the Examples menu for presets, or File > Open Patterns to load a `.pat` file
2. **Configure network** — Go to the Configure tab to adjust layers, units, and bias settings
3. **Set parameters** — Adjust learning rate, momentum, max epochs in the Train tab
4. **Train** — Click TRAIN and watch the error converge on the chart
5. **Test** — Select a pattern and click TEST ONE, or click TEST ALL
6. **Save results** — Use Network menu to save weights, or File > Export Report

---

## Pattern File Format

Plain text, same format as the original BasicProp `.pat` files:

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
basicprop/
├── pom.xml                          Maven build config
├── LICENSE                          MIT License
├── README.md                        This file
├── HeronixNeuroSim-2.0.0.jar       Pre-built distributable
├── run.bat                          Windows launcher
├── run.sh                           macOS/Linux launcher
├── seven_segment.pat                Example pattern file
└── src/main/
    ├── java/com/basicprop/
    │   ├── Launcher.java            Fat JAR entry point
    │   ├── MainApp.java             Main UI application (JavaFX)
    │   ├── NeuralNet.java           Backpropagation engine
    │   ├── NetworkCanvas.java       Network visualization canvas
    │   ├── PatternParser.java       Pattern file parser
    │   └── TrainingPattern.java     Input/target data record
    └── resources/com/basicprop/
        ├── theme-hacker.css         Hacker Green theme
        ├── theme-dark.css           Dark theme
        └── theme-system.css         System/Light theme
```

---

## Algorithm Details

- **Activation:** Logistic sigmoid on all non-input units
- **Weight update:** Online (stochastic GD) or Batch
- **Error function:** MSE or Cross-Entropy (selectable)
- **Momentum:** Standard first-order momentum term
- **Convergence:** Training stops when avg per-pattern error < 0.001 or max epochs reached

---

## License

MIT License — Copyright (c) 2026 Heronix Education Systems LLC. Developed by Michael Katsaros.

Free to use, modify, and distribute for any purpose. See [LICENSE](LICENSE) for details.
