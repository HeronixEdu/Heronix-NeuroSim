/*
 * Heronix NeuroSim - Neural Network Simulator
 * Copyright (c) 2026 Heronix Education Systems LLC
 * Developed by Michael Katsaros
 */
package com.heronix.neurosim;

import java.util.Random;

/**
 * Simple 1D CNN for educational pattern recognition.
 * Architecture: Conv1D -> ReLU -> MaxPool -> Flatten -> FC -> Sigmoid Output
 */
public class ConvNet {

    private final int inputSize, numFilters, filterSize, poolSize, outputSize;
    private final int convOutSize, poolOutSize, flatSize;

    private double[][] filters;      // [numFilters][filterSize]
    private double[] filterBias;     // [numFilters]
    private double[][] fcWeights;    // [outputSize][flatSize]
    private double[] fcBias;         // [outputSize]

    // Intermediate state for backprop
    private double[] lastInput;
    private double[][] convOutput;   // [numFilters][convOutSize]
    private double[][] reluOutput;   // [numFilters][convOutSize]
    private double[][] poolOutput;   // [numFilters][poolOutSize]
    private int[][] poolIndices;     // [numFilters][poolOutSize] - which index was max
    private double[] flatOutput;     // flattened pool output
    private double[] outputAct;

    private final Random rng = new Random();

    public ConvNet(int inputSize, int numFilters, int filterSize, int poolSize, int outputSize) {
        this.inputSize = inputSize;
        this.numFilters = numFilters;
        this.filterSize = filterSize;
        this.poolSize = poolSize;
        this.outputSize = outputSize;

        this.convOutSize = inputSize - filterSize + 1;
        this.poolOutSize = convOutSize / poolSize;
        this.flatSize = numFilters * poolOutSize;

        filters = new double[numFilters][filterSize];
        filterBias = new double[numFilters];
        fcWeights = new double[outputSize][flatSize];
        fcBias = new double[outputSize];

        convOutput = new double[numFilters][convOutSize];
        reluOutput = new double[numFilters][convOutSize];
        poolOutput = new double[numFilters][poolOutSize];
        poolIndices = new int[numFilters][poolOutSize];
        flatOutput = new double[flatSize];
        outputAct = new double[outputSize];
        lastInput = new double[inputSize];
    }

    public void randomiseWeights(double range) {
        for (int f = 0; f < numFilters; f++) {
            for (int i = 0; i < filterSize; i++) filters[f][i] = (rng.nextDouble() * 2 - 1) * range;
            filterBias[f] = (rng.nextDouble() * 2 - 1) * range;
        }
        for (int j = 0; j < outputSize; j++) {
            for (int i = 0; i < flatSize; i++) fcWeights[j][i] = (rng.nextDouble() * 2 - 1) * range;
            fcBias[j] = (rng.nextDouble() * 2 - 1) * range;
        }
    }

    public void resetWeights() { randomiseWeights(0.5); }

    public double[] forward(double[] input) {
        System.arraycopy(input, 0, lastInput, 0, inputSize);

        // Convolution
        for (int f = 0; f < numFilters; f++) {
            for (int i = 0; i < convOutSize; i++) {
                double sum = filterBias[f];
                for (int k = 0; k < filterSize; k++) sum += input[i + k] * filters[f][k];
                convOutput[f][i] = sum;
                reluOutput[f][i] = Math.max(0, sum); // ReLU
            }
        }

        // Max pooling
        for (int f = 0; f < numFilters; f++) {
            for (int p = 0; p < poolOutSize; p++) {
                double maxVal = Double.NEGATIVE_INFINITY;
                int maxIdx = 0;
                for (int i = 0; i < poolSize && p * poolSize + i < convOutSize; i++) {
                    int idx = p * poolSize + i;
                    if (reluOutput[f][idx] > maxVal) { maxVal = reluOutput[f][idx]; maxIdx = idx; }
                }
                poolOutput[f][p] = maxVal;
                poolIndices[f][p] = maxIdx;
            }
        }

        // Flatten
        int idx = 0;
        for (int f = 0; f < numFilters; f++)
            for (int p = 0; p < poolOutSize; p++) flatOutput[idx++] = poolOutput[f][p];

        // FC + sigmoid
        for (int j = 0; j < outputSize; j++) {
            double sum = fcBias[j];
            for (int i = 0; i < flatSize; i++) sum += fcWeights[j][i] * flatOutput[i];
            outputAct[j] = sigmoid(sum);
        }
        return outputAct.clone();
    }

    public double train(double[] input, double[] target, double lr) {
        double[] output = forward(input);
        double error = 0;
        for (int j = 0; j < outputSize; j++) error += 0.5 * Math.pow(target[j] - output[j], 2);

        // Output deltas
        double[] deltaO = new double[outputSize];
        for (int j = 0; j < outputSize; j++) {
            double o = outputAct[j];
            deltaO[j] = (target[j] - o) * o * (1 - o);
        }

        // FC gradient
        double[] deltaFlat = new double[flatSize];
        for (int j = 0; j < outputSize; j++) {
            for (int i = 0; i < flatSize; i++) {
                deltaFlat[i] += deltaO[j] * fcWeights[j][i];
                fcWeights[j][i] += lr * deltaO[j] * flatOutput[i];
            }
            fcBias[j] += lr * deltaO[j];
        }

        // Un-flatten to pool gradient
        double[][] deltaPool = new double[numFilters][poolOutSize];
        int di = 0;
        for (int f = 0; f < numFilters; f++)
            for (int p = 0; p < poolOutSize; p++) deltaPool[f][p] = deltaFlat[di++];

        // Backprop through pooling and ReLU to conv
        for (int f = 0; f < numFilters; f++) {
            double[] deltaConv = new double[convOutSize];
            for (int p = 0; p < poolOutSize; p++) {
                int maxIdx = poolIndices[f][p];
                if (convOutput[f][maxIdx] > 0) // ReLU derivative
                    deltaConv[maxIdx] = deltaPool[f][p];
            }
            // Update filters
            for (int k = 0; k < filterSize; k++) {
                double grad = 0;
                for (int i = 0; i < convOutSize; i++) grad += deltaConv[i] * lastInput[i + k];
                filters[f][k] += lr * grad;
            }
            double biasGrad = 0;
            for (int i = 0; i < convOutSize; i++) biasGrad += deltaConv[i];
            filterBias[f] += lr * biasGrad;
        }

        return error;
    }

    public double[][] getFilters() {
        double[][] copy = new double[numFilters][];
        for (int f = 0; f < numFilters; f++) copy[f] = filters[f].clone();
        return copy;
    }
    public double[][] getConvOutput() { return convOutput; }
    public double[][] getPoolOutput() { return poolOutput; }
    public double[] getOutputActivations() { return outputAct.clone(); }
    public int getNumFilters() { return numFilters; }
    public int getFilterSize() { return filterSize; }
    public int getPoolSize() { return poolSize; }
    public int getInputSize() { return inputSize; }
    public int getOutputSize() { return outputSize; }
    public double[][] getFCWeights() { return fcWeights; }

    private static double sigmoid(double x) { return 1.0 / (1.0 + Math.exp(-x)); }
}
