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

import javafx.scene.canvas.Canvas;
import javafx.scene.canvas.GraphicsContext;
import javafx.scene.paint.Color;
import javafx.scene.text.Font;
import javafx.scene.text.TextAlignment;

/**
 * Resizable JavaFX {@link Canvas} that renders the neural network topology,
 * node activations, and connection weights in real time.
 *
 * <p>Colour coding:</p>
 * <ul>
 *   <li>Blue nodes   – input layer</li>
 *   <li>Green nodes  – hidden layer(s)</li>
 *   <li>Gold nodes   – output layer</li>
 *   <li>Green lines  – positive weights (opacity ∝ |weight|)</li>
 *   <li>Red lines    – negative weights (opacity ∝ |weight|)</li>
 * </ul>
 */
public class NetworkCanvas extends Canvas {

    private static final double NODE_RADIUS  = 20;
    private static final double PAD_X        = 55;
    private static final double PAD_Y        = 40;
    private static final Font   LABEL_FONT   = Font.font("Courier New", 8);
    private static final Font   ACT_FONT     = Font.font("Courier New", 9);

    private NeuralNet net;
    private Color bgColor    = Color.web("#0a110d");
    private Color labelColor = Color.web("#1a5a3a");

    public NetworkCanvas(double width, double height) {
        super(width, height);
    }

    /** Set canvas background and label colors for theming. */
    public void setThemeColors(String bg, String label) {
        bgColor    = Color.web(bg);
        labelColor = Color.web(label);
    }

    /** Provide the current network snapshot and repaint. */
    public void render(NeuralNet net) {
        this.net = net;
        draw();
    }

    public void clear() {
        GraphicsContext gc = getGraphicsContext2D();
        gc.setFill(bgColor);
        gc.fillRect(0, 0, getWidth(), getHeight());
    }

    // ── Drawing ───────────────────────────────────────────────────────────────

    private void draw() {
        if (net == null) return;

        double W = getWidth();
        double H = getHeight();

        GraphicsContext gc = getGraphicsContext2D();
        gc.setFill(bgColor);
        gc.fillRect(0, 0, W, H);

        int NL = net.getNumLayers();
        if (NL < 2) return;
        double xStep = (W - PAD_X * 2) / (NL - 1);

        // Pre-compute node centres  [layer][unit]
        double[][] cx = new double[NL][];
        double[][] cy = new double[NL][];
        for (int l = 0; l < NL; l++) {
            int sz = net.getLayerSize(l);
            cx[l] = new double[sz];
            cy[l] = new double[sz];
            double x = PAD_X + l * xStep;
            for (int i = 0; i < sz; i++) {
                cx[l][i] = x;
                cy[l][i] = sz == 1
                        ? H / 2
                        : PAD_Y + i * (H - PAD_Y * 2) / (sz - 1);
            }
        }

        // ── Connections ────────────────────────────────────────────────────
        gc.setLineWidth(1.0);
        for (int l = 0; l < NL - 1; l++) {
            for (int j = 0; j < net.getLayerSize(l + 1); j++) {
                for (int i = 0; i < net.getLayerSize(l); i++) {
                    double w = net.getWeight(l, j, i);
                    double t = Math.min(1.0, Math.abs(w) / 3.0);
                    double alpha = 0.06 + t * 0.55;
                    double lw    = 0.5 + t * 2.5;
                    gc.setStroke(w >= 0
                            ? Color.color(0.0, 0.85, 0.45, alpha)
                            : Color.color(1.0, 0.25, 0.25, alpha));
                    gc.setLineWidth(lw);
                    gc.strokeLine(cx[l][i], cy[l][i], cx[l+1][j], cy[l+1][j]);
                }
            }
        }

        // ── Layer labels ───────────────────────────────────────────────────
        gc.setFont(LABEL_FONT);
        gc.setTextAlign(TextAlignment.CENTER);
        String[] layerLabels = new String[NL];
        layerLabels[0]      = "INPUT";
        layerLabels[NL - 1] = "OUTPUT";
        for (int l = 1; l < NL - 1; l++) layerLabels[l] = "HIDDEN " + l;

        for (int l = 0; l < NL; l++) {
            gc.setFill(labelColor);
            gc.fillText(layerLabels[l], cx[l][0], 14);
        }

        // ── Nodes ──────────────────────────────────────────────────────────
        gc.setFont(ACT_FONT);
        gc.setLineWidth(1.5);
        for (int l = 0; l < NL; l++) {
            for (int i = 0; i < net.getLayerSize(l); i++) {
                double act  = net.getActivation(l, i);
                double x    = cx[l][i];
                double y    = cy[l][i];
                boolean isIn  = (l == 0);
                boolean isOut = (l == NL - 1);

                Color base  = isIn  ? Color.web("#50b4ff")
                            : isOut ? Color.web("#ffc832")
                            :         Color.web("#00dc78");

                // filled circle – opacity reflects activation
                gc.setFill(base.deriveColor(0, 1, 1, 0.15 + act * 0.75));
                gc.fillOval(x - NODE_RADIUS, y - NODE_RADIUS, NODE_RADIUS * 2, NODE_RADIUS * 2);

                // border
                gc.setStroke(base);
                gc.strokeOval(x - NODE_RADIUS, y - NODE_RADIUS, NODE_RADIUS * 2, NODE_RADIUS * 2);

                // activation text
                gc.setFill(Color.WHITE);
                gc.setTextAlign(TextAlignment.CENTER);
                gc.fillText(String.format("%.2f", act), x, y + 4);
            }
        }
    }
}
