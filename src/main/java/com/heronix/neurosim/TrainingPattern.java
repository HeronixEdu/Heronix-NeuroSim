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

import java.util.Arrays;

/**
 * A single input/target pair used during training.
 */
public class TrainingPattern {

    public final double[] inputs;
    public final double[] targets;
    public final String   label;     // raw source line, shown in the combo-box

    public TrainingPattern(double[] inputs, double[] targets, String label) {
        this.inputs  = inputs.clone();
        this.targets = targets.clone();
        this.label   = label;
    }

    @Override
    public String toString() { return label; }
}
