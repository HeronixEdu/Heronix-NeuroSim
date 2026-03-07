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
package com.basicprop;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Random;

/**
 * Feedforward backpropagation network.
 *
 * <p>All non-input units use the logistic (sigmoid) activation function,
 * exactly as described in the original BasicProp documentation.</p>
 *
 * <p>Supports:</p>
 * <ul>
 *   <li>Arbitrary number of layers (2–N)</li>
 *   <li>Optional bias unit per layer (except output)</li>
 *   <li>Online (stochastic) or batch weight updates</li>
 *   <li>MSE or cross-entropy error functions</li>
 *   <li>Momentum term</li>
 * </ul>
 */
public class NeuralNet {

    // ── Architecture ────────────────────────────────────────────────────────

    private final int[]     layerSizes;   // neurons per layer (excl. bias)
    private final boolean[] biasEnabled;  // bias on layer[i] feeds into layer[i+1]
    private final int       numLayers;

    // ── State ───────────────────────────────────────────────────────────────

    /** activations[layer][unit] */
    private double[][] activations;

    /** weights[layer][to][from]  – 'from' index = layerSize means bias */
    private double[][][] weights;

    /** previous weight deltas for momentum */
    private double[][][] prevDeltas;

    private final Random rng = new Random();

    // ── Constructor ─────────────────────────────────────────────────────────

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
    }

    // ── Initialization ───────────────────────────────────────────────────────

    private void allocate() {
        activations = new double[numLayers][];
        weights     = new double[numLayers - 1][][];
        prevDeltas  = new double[numLayers - 1][][];

        for (int l = 0; l < numLayers; l++) {
            activations[l] = new double[layerSizes[l]];
        }
        for (int l = 0; l < numLayers - 1; l++) {
            int from = layerSizes[l] + (biasEnabled[l] ? 1 : 0);
            int to   = layerSizes[l + 1];
            weights[l]    = new double[to][from];
            prevDeltas[l] = new double[to][from];
        }
    }

    /** Randomise all weights in [-range, +range]. */
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

    // ── Forward pass ─────────────────────────────────────────────────────────

    /**
     * Propagate inputs through the network.
     * @return copy of output activations
     */
    public double[] forward(double[] inputs) {
        if (inputs.length != layerSizes[0]) {
            throw new IllegalArgumentException("Expected " + layerSizes[0] + " inputs, got " + inputs.length);
        }
        System.arraycopy(inputs, 0, activations[0], 0, layerSizes[0]);

        for (int l = 0; l < numLayers - 1; l++) {
            boolean hasBias = biasEnabled[l];
            for (int j = 0; j < layerSizes[l + 1]; j++) {
                double sum = 0;
                for (int i = 0; i < layerSizes[l]; i++) {
                    sum += weights[l][j][i] * activations[l][i];
                }
                if (hasBias) {
                    sum += weights[l][j][layerSizes[l]];   // bias weight
                }
                activations[l + 1][j] = sigmoid(sum);
            }
        }
        return activations[numLayers - 1].clone();
    }

    // ── Error ────────────────────────────────────────────────────────────────

    /**
     * Per-pattern error (must call forward() first).
     * @param crossEntropy use cross-entropy instead of MSE
     */
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

    // ── Backpropagation ──────────────────────────────────────────────────────

    /**
     * Compute output + hidden deltas (must call forward() first).
     */
    private double[][] computeDeltas(double[] targets, boolean crossEntropy) {
        int L = numLayers - 1;
        double[][] deltas = new double[numLayers][];
        for (int l = 0; l < numLayers; l++) deltas[l] = new double[layerSizes[l]];

        // Output layer
        for (int j = 0; j < layerSizes[L]; j++) {
            double o = activations[L][j];
            if (crossEntropy) {
                deltas[L][j] = targets[j] - o;           // simplified CE derivative
            } else {
                deltas[L][j] = (targets[j] - o) * o * (1 - o);
            }
        }
        // Hidden layers (back to front)
        for (int l = L - 1; l > 0; l--) {
            for (int i = 0; i < layerSizes[l]; i++) {
                double sum = 0;
                for (int j = 0; j < layerSizes[l + 1]; j++) {
                    sum += deltas[l + 1][j] * weights[l][j][i];
                }
                double o = activations[l][i];
                deltas[l][i] = o * (1 - o) * sum;
            }
        }
        return deltas;
    }

    /** Apply accumulated weight changes immediately (online update). */
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

    // ── Training ─────────────────────────────────────────────────────────────

    /**
     * Run one full epoch over the supplied patterns.
     *
     * @param patterns    list of training patterns
     * @param batchUpdate if true, accumulate gradients before updating weights
     * @param lr          learning rate
     * @param momentum    momentum coefficient
     * @param crossEntropy use cross-entropy error
     * @return average per-pattern error for this epoch
     */
    public double trainEpoch(List<TrainingPattern> patterns,
                             boolean batchUpdate,
                             double  lr,
                             double  momentum,
                             boolean crossEntropy) {
        if (patterns == null || patterns.isEmpty()) return 0;
        double totalError = 0;
        int n = patterns.size();

        if (!batchUpdate) {
            // ── Online (stochastic) gradient descent ────────────────────────
            List<TrainingPattern> shuffled = new ArrayList<>(patterns);
            Collections.shuffle(shuffled, rng);
            for (TrainingPattern p : shuffled) {
                forward(p.inputs);
                totalError += computeError(p.targets, crossEntropy);
                double[][] deltas = computeDeltas(p.targets, crossEntropy);
                applyOnlineUpdate(deltas, lr, momentum);
            }
        } else {
            // ── Batch gradient descent ───────────────────────────────────────
            double[][][] gradW = new double[numLayers - 1][][];
            for (int l = 0; l < numLayers - 1; l++) {
                int from = layerSizes[l] + (biasEnabled[l] ? 1 : 0);
                gradW[l] = new double[layerSizes[l + 1]][from];
            }
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
            // Apply averaged gradients
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

    // ── Accessors ────────────────────────────────────────────────────────────

    public int   getNumLayers()               { return numLayers; }
    public int   getLayerSize(int l)          { return layerSizes[l]; }
    public int[] getLayerSizes()              { return layerSizes.clone(); }
    public boolean getBiasEnabled(int l)      { return biasEnabled[l]; }
    public double getActivation(int l, int i) { return activations[l][i]; }
    public double getWeight(int l, int j, int i) { return weights[l][j][i]; }
    public void   setWeight(int l, int j, int i, double v) { weights[l][j][i] = v; }
    public boolean[] getBiasEnabledArray() { return biasEnabled.clone(); }

    /** Number of weight columns into layer l+1 (includes bias if enabled). */
    public int getFromCount(int l) {
        return layerSizes[l] + (biasEnabled[l] ? 1 : 0);
    }

    // ── Helpers ───────────────────────────────────────────────────────────────

    private static double sigmoid(double x) { return 1.0 / (1.0 + Math.exp(-x)); }
    private static double clamp(double v)   { return Math.max(1e-15, Math.min(1 - 1e-15, v)); }
}
