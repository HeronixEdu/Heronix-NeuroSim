/*
 * Heronix NeuroSim - Neural Network Simulator
 * Copyright (c) 2026 Heronix Education Systems LLC
 * Developed by Michael Katsaros
 */
package com.heronix.neurosim;

public enum ActivationFunction {
    SIGMOID {
        @Override public double apply(double x) { return 1.0 / (1.0 + Math.exp(-x)); }
        @Override public double derivative(double output) { return output * (1.0 - output); }
        @Override public String toString() { return "Sigmoid"; }
    },
    RELU {
        @Override public double apply(double x) { return Math.max(0, x); }
        @Override public double derivative(double output) { return output > 0 ? 1.0 : 0.0; }
        @Override public String toString() { return "ReLU"; }
    },
    TANH {
        @Override public double apply(double x) { return Math.tanh(x); }
        @Override public double derivative(double output) { return 1.0 - output * output; }
        @Override public String toString() { return "Tanh"; }
    },
    SOFTMAX {
        @Override public double apply(double x) { return 1.0 / (1.0 + Math.exp(-x)); } // per-element fallback
        @Override public double derivative(double output) { return output * (1.0 - output); } // simplified
        @Override public String toString() { return "Softmax"; }
    };

    public abstract double apply(double x);
    public abstract double derivative(double output);

    public static double[] applySoftmax(double[] inputs) {
        double max = Double.NEGATIVE_INFINITY;
        for (double v : inputs) if (v > max) max = v;
        double[] exps = new double[inputs.length];
        double sum = 0;
        for (int i = 0; i < inputs.length; i++) {
            exps[i] = Math.exp(inputs[i] - max);
            sum += exps[i];
        }
        for (int i = 0; i < exps.length; i++) exps[i] /= sum;
        return exps;
    }
}
