/*
 * Heronix NeuroSim - Neural Network Simulator
 * Copyright (c) 2026 Heronix Education Systems LLC
 * Developed by Michael Katsaros
 *
 * This software is free and open source. You may use, modify, and distribute
 * it freely for any purpose, including commercial use, with no restrictions.
 * Attribution to Heronix Education Systems LLC and Michael Katsaros is
 * appreciated but not required.
 *
 * THIS SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND.
 */
package com.heronix.neurosim;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Random;
import java.util.stream.IntStream;

/**
 * Feedforward backpropagation network with configurable activation functions.
 *
 * <p>Supports:</p>
 * <ul>
 *   <li>Arbitrary number of layers (2-N)</li>
 *   <li>Optional bias unit per layer (except output)</li>
 *   <li>Configurable activation: Sigmoid, ReLU, Tanh, Softmax</li>
 *   <li>Online (stochastic) or batch weight updates (parallel batch)</li>
 *   <li>MSE or cross-entropy error functions</li>
 *   <li>Momentum term</li>
 *   <li>Step-by-step training (forward/backward exposed separately)</li>
 * </ul>
 */
public class NeuralNet {

    // -- Architecture --
    private final int[]     layerSizes;
    private final boolean[] biasEnabled;
    private final int       numLayers;

    // -- Activation functions --
    private ActivationFunction hiddenActivation = ActivationFunction.SIGMOID;
    private ActivationFunction outputActivation = ActivationFunction.SIGMOID;

    // -- State --
    private double[][] activations;
    private double[][] netInputs;
    private double[][][] weights;
    private double[][][] prevDeltas;
    private double[][] lastDeltas;

    private final Random rng = new Random();

    // -- Constructor --

    public NeuralNet(int[] layerSizes, boolean[] biasEnabled) {
        this.layerSizes  = layerSizes.clone();
        this.biasEnabled = biasEnabled.clone();
        this.numLayers   = layerSizes.length;
        allocate();
    }

    /** Deep-copy constructor */
    public NeuralNet(NeuralNet src) {
        this.layerSizes  = src.layerSizes.clone();
        this.biasEnabled = src.biasEnabled.clone();
        this.numLayers   = src.numLayers;
        this.hiddenActivation = src.hiddenActivation;
        this.outputActivation = src.outputActivation;
        allocate();
        for (int l = 0; l < numLayers - 1; l++) {
            for (int j = 0; j < layerSizes[l + 1]; j++) {
                int from = layerSizes[l] + (biasEnabled[l] ? 1 : 0);
                for (int i = 0; i < from; i++) {
                    weights[l][j][i]    = src.weights[l][j][i];
                    prevDeltas[l][j][i] = 0;
                }
            }
        }
        for (int l = 0; l < numLayers; l++) {
            System.arraycopy(src.activations[l], 0, activations[l], 0, layerSizes[l]);
            System.arraycopy(src.netInputs[l], 0, netInputs[l], 0, layerSizes[l]);
        }
    }

    // -- Initialization --

    private void allocate() {
        activations = new double[numLayers][];
        netInputs   = new double[numLayers][];
        weights     = new double[numLayers - 1][][];
        prevDeltas  = new double[numLayers - 1][][];

        for (int l = 0; l < numLayers; l++) {
            activations[l] = new double[layerSizes[l]];
            netInputs[l]   = new double[layerSizes[l]];
        }
        for (int l = 0; l < numLayers - 1; l++) {
            int from = layerSizes[l] + (biasEnabled[l] ? 1 : 0);
            int to   = layerSizes[l + 1];
            weights[l]    = new double[to][from];
            prevDeltas[l] = new double[to][from];
        }
    }

    public void randomiseWeights(double range) {
        if (range <= 0) range = 0.5;
        for (int l = 0; l < numLayers - 1; l++) {
            int from = layerSizes[l] + (biasEnabled[l] ? 1 : 0);
            for (int j = 0; j < layerSizes[l + 1]; j++) {
                for (int i = 0; i < from; i++) {
                    weights[l][j][i]    = (rng.nextDouble() * 2 - 1) * range;
                    prevDeltas[l][j][i] = 0;
                }
            }
        }
    }

    // -- Activation config --

    public void setHiddenActivation(ActivationFunction af) { this.hiddenActivation = af; }
    public void setOutputActivation(ActivationFunction af) { this.outputActivation = af; }
    public ActivationFunction getHiddenActivation() { return hiddenActivation; }
    public ActivationFunction getOutputActivation() { return outputActivation; }

    private ActivationFunction activationFor(int layer) {
        return (layer == numLayers - 1) ? outputActivation : hiddenActivation;
    }

    // -- Forward pass --

    public double[] forward(double[] inputs) {
        if (inputs.length != layerSizes[0]) {
            throw new IllegalArgumentException("Expected " + layerSizes[0] + " inputs, got " + inputs.length);
        }
        System.arraycopy(inputs, 0, activations[0], 0, layerSizes[0]);
        System.arraycopy(inputs, 0, netInputs[0], 0, layerSizes[0]);

        for (int l = 0; l < numLayers - 1; l++) {
            boolean hasBias = biasEnabled[l];
            ActivationFunction af = activationFor(l + 1);

            for (int j = 0; j < layerSizes[l + 1]; j++) {
                double sum = 0;
                for (int i = 0; i < layerSizes[l]; i++) {
                    sum += weights[l][j][i] * activations[l][i];
                }
                if (hasBias) {
                    sum += weights[l][j][layerSizes[l]];
                }
                netInputs[l + 1][j] = sum;
            }

            if (af == ActivationFunction.SOFTMAX) {
                double[] softmaxOut = ActivationFunction.applySoftmax(netInputs[l + 1]);
                System.arraycopy(softmaxOut, 0, activations[l + 1], 0, layerSizes[l + 1]);
            } else {
                for (int j = 0; j < layerSizes[l + 1]; j++) {
                    activations[l + 1][j] = af.apply(netInputs[l + 1][j]);
                }
            }
        }
        return activations[numLayers - 1].clone();
    }

    // -- Error --

    public double computeError(double[] targets, boolean crossEntropy) {
        double[] out = activations[numLayers - 1];
        double err = 0;
        if (crossEntropy) {
            for (int j = 0; j < out.length; j++) {
                double o = clamp(out[j]);
                err -= targets[j] * Math.log(o) + (1 - targets[j]) * Math.log(1 - o);
            }
        } else {
            for (int j = 0; j < out.length; j++) {
                err += 0.5 * Math.pow(targets[j] - out[j], 2);
            }
        }
        return err;
    }

    // -- Backpropagation --

    private double[][] computeDeltas(double[] targets, boolean crossEntropy) {
        int L = numLayers - 1;
        double[][] deltas = new double[numLayers][];
        for (int l = 0; l < numLayers; l++) deltas[l] = new double[layerSizes[l]];

        ActivationFunction outAf = outputActivation;
        for (int j = 0; j < layerSizes[L]; j++) {
            double o = activations[L][j];
            if (crossEntropy) {
                deltas[L][j] = targets[j] - o;
            } else {
                deltas[L][j] = (targets[j] - o) * outAf.derivative(o);
            }
        }
        for (int l = L - 1; l > 0; l--) {
            for (int i = 0; i < layerSizes[l]; i++) {
                double sum = 0;
                for (int j = 0; j < layerSizes[l + 1]; j++) {
                    sum += deltas[l + 1][j] * weights[l][j][i];
                }
                deltas[l][i] = hiddenActivation.derivative(activations[l][i]) * sum;
            }
        }
        lastDeltas = deltas;
        return deltas;
    }

    private void applyOnlineUpdate(double[][] deltas, double lr, double momentum) {
        for (int l = 0; l < numLayers - 1; l++) {
            boolean hasBias = biasEnabled[l];
            for (int j = 0; j < layerSizes[l + 1]; j++) {
                for (int i = 0; i < layerSizes[l]; i++) {
                    double dw = lr * deltas[l + 1][j] * activations[l][i]
                                + momentum * prevDeltas[l][j][i];
                    weights[l][j][i]    += dw;
                    prevDeltas[l][j][i]  = dw;
                }
                if (hasBias) {
                    int bi = layerSizes[l];
                    double dw = lr * deltas[l + 1][j] + momentum * prevDeltas[l][j][bi];
                    weights[l][j][bi]    += dw;
                    prevDeltas[l][j][bi]  = dw;
                }
            }
        }
    }

    // -- Training --

    public double trainEpoch(List<TrainingPattern> patterns,
                             boolean batchUpdate,
                             double  lr,
                             double  momentum,
                             boolean crossEntropy) {
        if (patterns == null || patterns.isEmpty()) return 0;
        double totalError = 0;
        int n = patterns.size();

        if (!batchUpdate) {
            List<TrainingPattern> shuffled = new ArrayList<>(patterns);
            Collections.shuffle(shuffled, rng);
            for (TrainingPattern p : shuffled) {
                forward(p.inputs);
                totalError += computeError(p.targets, crossEntropy);
                double[][] deltas = computeDeltas(p.targets, crossEntropy);
                applyOnlineUpdate(deltas, lr, momentum);
            }
        } else {
            double[][][] gradW = new double[numLayers - 1][][];
            for (int l = 0; l < numLayers - 1; l++) {
                int from = layerSizes[l] + (biasEnabled[l] ? 1 : 0);
                gradW[l] = new double[layerSizes[l + 1]][from];
            }

            if (n >= 16) {
                // Parallel batch
                double[] errors = new double[n];
                double[][][][] allGrads = new double[n][][][];

                IntStream.range(0, n).parallel().forEach(idx -> {
                    NeuralNet copy = new NeuralNet(this);
                    TrainingPattern p = patterns.get(idx);
                    copy.forward(p.inputs);
                    errors[idx] = copy.computeError(p.targets, crossEntropy);
                    double[][] deltas = copy.computeDeltas(p.targets, crossEntropy);

                    double[][][] pg = new double[numLayers - 1][][];
                    for (int l = 0; l < numLayers - 1; l++) {
                        boolean hasBias = biasEnabled[l];
                        int from = layerSizes[l] + (hasBias ? 1 : 0);
                        pg[l] = new double[layerSizes[l + 1]][from];
                        for (int j = 0; j < layerSizes[l + 1]; j++) {
                            for (int i = 0; i < layerSizes[l]; i++) {
                                pg[l][j][i] = deltas[l + 1][j] * copy.activations[l][i];
                            }
                            if (hasBias) {
                                pg[l][j][layerSizes[l]] = deltas[l + 1][j];
                            }
                        }
                    }
                    allGrads[idx] = pg;
                });

                for (int idx = 0; idx < n; idx++) {
                    totalError += errors[idx];
                    for (int l = 0; l < numLayers - 1; l++) {
                        int from = layerSizes[l] + (biasEnabled[l] ? 1 : 0);
                        for (int j = 0; j < layerSizes[l + 1]; j++) {
                            for (int i = 0; i < from; i++) {
                                gradW[l][j][i] += allGrads[idx][l][j][i];
                            }
                        }
                    }
                }
            } else {
                for (TrainingPattern p : patterns) {
                    forward(p.inputs);
                    totalError += computeError(p.targets, crossEntropy);
                    double[][] deltas = computeDeltas(p.targets, crossEntropy);
                    for (int l = 0; l < numLayers - 1; l++) {
                        boolean hasBias = biasEnabled[l];
                        for (int j = 0; j < layerSizes[l + 1]; j++) {
                            for (int i = 0; i < layerSizes[l]; i++) {
                                gradW[l][j][i] += deltas[l + 1][j] * activations[l][i];
                            }
                            if (hasBias) {
                                gradW[l][j][layerSizes[l]] += deltas[l + 1][j];
                            }
                        }
                    }
                }
            }

            for (int l = 0; l < numLayers - 1; l++) {
                boolean hasBias = biasEnabled[l];
                for (int j = 0; j < layerSizes[l + 1]; j++) {
                    int cols = layerSizes[l] + (hasBias ? 1 : 0);
                    for (int i = 0; i < cols; i++) {
                        double dw = lr * gradW[l][j][i] / n + momentum * prevDeltas[l][j][i];
                        weights[l][j][i]    += dw;
                        prevDeltas[l][j][i]  = dw;
                    }
                }
            }
        }
        return totalError / n;
    }

    // -- Step-by-step training support --

    public StepResult stepForward(double[] inputs, double[] targets, boolean crossEntropy) {
        double[] output = forward(inputs);
        double error = computeError(targets, crossEntropy);
        double[][] acts = new double[numLayers][];
        double[][] nets = new double[numLayers][];
        for (int l = 0; l < numLayers; l++) {
            acts[l] = activations[l].clone();
            nets[l] = netInputs[l].clone();
        }
        return new StepResult(acts, nets, output.clone(), error, null);
    }

    public StepResult stepBackward(double[] targets, boolean crossEntropy,
                                    double lr, double momentum) {
        double[][] deltas = computeDeltas(targets, crossEntropy);
        applyOnlineUpdate(deltas, lr, momentum);
        double[][] deltasCopy = new double[numLayers][];
        for (int l = 0; l < numLayers; l++) deltasCopy[l] = deltas[l].clone();
        double[][] acts = new double[numLayers][];
        double[][] nets = new double[numLayers][];
        for (int l = 0; l < numLayers; l++) {
            acts[l] = activations[l].clone();
            nets[l] = netInputs[l].clone();
        }
        return new StepResult(acts, nets, activations[numLayers - 1].clone(),
                computeError(targets, crossEntropy), deltasCopy);
    }

    public static class StepResult {
        public final double[][] layerActivations;
        public final double[][] layerNetInputs;
        public final double[]   output;
        public final double     error;
        public final double[][] deltas;

        public StepResult(double[][] acts, double[][] nets, double[] out,
                          double err, double[][] deltas) {
            this.layerActivations = acts;
            this.layerNetInputs   = nets;
            this.output           = out;
            this.error            = err;
            this.deltas           = deltas;
        }
    }

    // -- Accessors --

    public int   getNumLayers()               { return numLayers; }
    public int   getLayerSize(int l)          { return layerSizes[l]; }
    public int[] getLayerSizes()              { return layerSizes.clone(); }
    public boolean getBiasEnabled(int l)      { return biasEnabled[l]; }
    public double getActivation(int l, int i) { return activations[l][i]; }
    public double getNetInput(int l, int i)   { return netInputs[l][i]; }
    public double getWeight(int l, int j, int i) { return weights[l][j][i]; }
    public void   setWeight(int l, int j, int i, double v) { weights[l][j][i] = v; }
    public boolean[] getBiasEnabledArray() { return biasEnabled.clone(); }
    public double[][] getLastDeltas() { return lastDeltas; }

    public int getFromCount(int l) {
        return layerSizes[l] + (biasEnabled[l] ? 1 : 0);
    }

    // -- Helpers --

    private static double clamp(double v) { return Math.max(1e-15, Math.min(1 - 1e-15, v)); }
}
