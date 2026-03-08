/*
 * Heronix NeuroSim - Neural Network Simulator
 * Copyright (c) 2026 Heronix Education Systems LLC
 * Developed by Michael Katsaros
 */
package com.heronix.neurosim;

import java.util.List;
import java.util.Random;

/**
 * Simple Elman RNN for educational purposes.
 * Hidden layer has context units that copy previous hidden activations.
 */
public class RecurrentNet {

    private final int inputSize, hiddenSize, outputSize;
    private final boolean useBias;

    // Weights: input->hidden, context->hidden, hidden->output
    private double[][] wIH;   // [hiddenSize][inputSize]
    private double[][] wCH;   // [hiddenSize][hiddenSize] (context to hidden)
    private double[][] wHO;   // [outputSize][hiddenSize]
    private double[] biasH;   // [hiddenSize]
    private double[] biasO;   // [outputSize]

    // State
    private double[] context;       // previous hidden activations
    private double[] hiddenAct;     // current hidden activations
    private double[] hiddenNet;     // pre-activation hidden
    private double[] outputAct;     // current output
    private double[] inputAct;      // last input

    // Momentum
    private double[][] dwIH, dwCH, dwHO;
    private double[] dbH, dbO;

    private final Random rng = new Random();

    public RecurrentNet(int inputs, int hidden, int outputs, boolean bias) {
        this.inputSize = inputs;
        this.hiddenSize = hidden;
        this.outputSize = outputs;
        this.useBias = bias;

        wIH = new double[hidden][inputs];
        wCH = new double[hidden][hidden];
        wHO = new double[outputs][hidden];
        biasH = new double[hidden];
        biasO = new double[outputs];

        dwIH = new double[hidden][inputs];
        dwCH = new double[hidden][hidden];
        dwHO = new double[outputs][hidden];
        dbH = new double[hidden];
        dbO = new double[outputs];

        context = new double[hidden];
        hiddenAct = new double[hidden];
        hiddenNet = new double[hidden];
        outputAct = new double[outputs];
        inputAct = new double[inputs];
    }

    public void randomiseWeights(double range) {
        randomise(wIH, range); randomise(wCH, range); randomise(wHO, range);
        for (int i = 0; i < hiddenSize; i++) biasH[i] = (rng.nextDouble() * 2 - 1) * range;
        for (int i = 0; i < outputSize; i++) biasO[i] = (rng.nextDouble() * 2 - 1) * range;
        zero(dwIH); zero(dwCH); zero(dwHO);
        dbH = new double[hiddenSize]; dbO = new double[outputSize];
    }

    public void resetContext() { context = new double[hiddenSize]; }

    public double[] forward(double[] inputs) {
        System.arraycopy(inputs, 0, inputAct, 0, inputSize);

        // Hidden = sigmoid(W_IH * input + W_CH * context + bias)
        for (int j = 0; j < hiddenSize; j++) {
            double sum = 0;
            for (int i = 0; i < inputSize; i++) sum += wIH[j][i] * inputs[i];
            for (int i = 0; i < hiddenSize; i++) sum += wCH[j][i] * context[i];
            if (useBias) sum += biasH[j];
            hiddenNet[j] = sum;
            hiddenAct[j] = sigmoid(sum);
        }

        // Output = sigmoid(W_HO * hidden + bias)
        for (int j = 0; j < outputSize; j++) {
            double sum = 0;
            for (int i = 0; i < hiddenSize; i++) sum += wHO[j][i] * hiddenAct[i];
            if (useBias) sum += biasO[j];
            outputAct[j] = sigmoid(sum);
        }

        // Update context for next timestep
        System.arraycopy(hiddenAct, 0, context, 0, hiddenSize);
        return outputAct.clone();
    }

    public double trainSequence(List<TrainingPattern> sequence, double lr, double momentum, boolean crossEntropy) {
        double totalError = 0;
        for (TrainingPattern p : sequence) {
            forward(p.inputs);
            double err = 0;
            for (int j = 0; j < outputSize; j++) err += 0.5 * Math.pow(p.targets[j] - outputAct[j], 2);
            totalError += err;

            // Output deltas
            double[] deltaO = new double[outputSize];
            for (int j = 0; j < outputSize; j++) {
                double o = outputAct[j];
                deltaO[j] = (p.targets[j] - o) * o * (1 - o);
            }

            // Hidden deltas
            double[] deltaH = new double[hiddenSize];
            for (int i = 0; i < hiddenSize; i++) {
                double sum = 0;
                for (int j = 0; j < outputSize; j++) sum += deltaO[j] * wHO[j][i];
                deltaH[i] = hiddenAct[i] * (1 - hiddenAct[i]) * sum;
            }

            // Update weights
            for (int j = 0; j < outputSize; j++) {
                for (int i = 0; i < hiddenSize; i++) {
                    double dw = lr * deltaO[j] * hiddenAct[i] + momentum * dwHO[j][i];
                    wHO[j][i] += dw; dwHO[j][i] = dw;
                }
                if (useBias) { double dw = lr * deltaO[j] + momentum * dbO[j]; biasO[j] += dw; dbO[j] = dw; }
            }
            for (int j = 0; j < hiddenSize; j++) {
                for (int i = 0; i < inputSize; i++) {
                    double dw = lr * deltaH[j] * inputAct[i] + momentum * dwIH[j][i];
                    wIH[j][i] += dw; dwIH[j][i] = dw;
                }
                for (int i = 0; i < hiddenSize; i++) {
                    double dw = lr * deltaH[j] * context[i] + momentum * dwCH[j][i];
                    wCH[j][i] += dw; dwCH[j][i] = dw;
                }
                if (useBias) { double dw = lr * deltaH[j] + momentum * dbH[j]; biasH[j] += dw; dbH[j] = dw; }
            }
        }
        return totalError / sequence.size();
    }

    public int getInputSize() { return inputSize; }
    public int getHiddenSize() { return hiddenSize; }
    public int getOutputSize() { return outputSize; }
    public double[] getContextValues() { return context.clone(); }
    public double[] getHiddenActivations() { return hiddenAct.clone(); }
    public double[] getOutputActivations() { return outputAct.clone(); }

    private static double sigmoid(double x) { return 1.0 / (1.0 + Math.exp(-x)); }
    private void randomise(double[][] m, double range) {
        for (double[] row : m) for (int i = 0; i < row.length; i++) row[i] = (rng.nextDouble() * 2 - 1) * range;
    }
    private void zero(double[][] m) { for (double[] row : m) java.util.Arrays.fill(row, 0); }
}
