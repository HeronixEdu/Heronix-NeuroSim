/*
 * Heronix NeuroSim - Neural Network Simulator
 * Copyright (c) 2026 Heronix Education Systems LLC
 * Developed by Michael Katsaros
 */
package com.heronix.neurosim;

import java.util.Random;

/**
 * Self-attention mechanism demo for educational visualization.
 * Implements single-head self-attention: Q, K, V projections and scaled dot-product attention.
 */
public class AttentionDemo {

    private final int seqLength, dimModel, dimKey;
    private double[][] wQ, wK, wV;
    private double[][] lastAttentionWeights;
    private final Random rng = new Random();

    public AttentionDemo(int seqLength, int dimModel, int dimKey) {
        this.seqLength = seqLength;
        this.dimModel = dimModel;
        this.dimKey = dimKey;
        wQ = new double[dimModel][dimKey];
        wK = new double[dimModel][dimKey];
        wV = new double[dimModel][dimModel];
    }

    public void randomiseWeights(double range) {
        randomise(wQ, range); randomise(wK, range); randomise(wV, range);
    }

    public double[][] forward(double[][] input) { return computeWithDetails(input).output; }

    public AttentionResult computeWithDetails(double[][] input) {
        int n = input.length;
        double[][] Q = matMul(input, wQ);
        double[][] K = matMul(input, wK);
        double[][] V = matMul(input, wV);

        double scale = Math.sqrt(dimKey);
        double[][] scores = new double[n][n];
        for (int i = 0; i < n; i++)
            for (int j = 0; j < n; j++) {
                double dot = 0;
                for (int k = 0; k < dimKey; k++) dot += Q[i][k] * K[j][k];
                scores[i][j] = dot / scale;
            }

        double[][] weights = new double[n][n];
        for (int i = 0; i < n; i++) weights[i] = softmax(scores[i]);
        lastAttentionWeights = weights;

        double[][] output = new double[n][dimModel];
        for (int i = 0; i < n; i++)
            for (int d = 0; d < dimModel; d++) {
                double sum = 0;
                for (int j = 0; j < n; j++) sum += weights[i][j] * V[j][d];
                output[i][d] = sum;
            }
        return new AttentionResult(Q, K, V, scores, weights, output);
    }

    public double[][] getAttentionWeights() { return lastAttentionWeights; }
    public double[][] getQueryMatrix() { return wQ; }
    public double[][] getKeyMatrix() { return wK; }
    public double[][] getValueMatrix() { return wV; }

    public static double[] softmax(double[] scores) {
        double max = Double.NEGATIVE_INFINITY;
        for (double s : scores) if (s > max) max = s;
        double[] exps = new double[scores.length];
        double sum = 0;
        for (int i = 0; i < scores.length; i++) { exps[i] = Math.exp(scores[i] - max); sum += exps[i]; }
        for (int i = 0; i < exps.length; i++) exps[i] /= sum;
        return exps;
    }

    private double[][] matMul(double[][] a, double[][] b) {
        int rows = a.length, inner = b.length, cols = b[0].length;
        double[][] r = new double[rows][cols];
        for (int i = 0; i < rows; i++)
            for (int j = 0; j < cols; j++)
                for (int k = 0; k < inner; k++) r[i][j] += a[i][k] * b[k][j];
        return r;
    }

    private void randomise(double[][] m, double range) {
        for (double[] row : m) for (int i = 0; i < row.length; i++) row[i] = (rng.nextDouble() * 2 - 1) * range;
    }

    public static class AttentionResult {
        public final double[][] queries, keys, values, scores, weights, output;
        public AttentionResult(double[][] q, double[][] k, double[][] v,
                               double[][] s, double[][] w, double[][] o) {
            queries=q; keys=k; values=v; scores=s; weights=w; output=o;
        }
    }
}
