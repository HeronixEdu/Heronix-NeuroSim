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

import java.io.*;
import java.nio.file.*;
import java.util.*;

/**
 * Parses pattern files in the original BasicProp format.
 *
 * <pre>
 * Number of patterns = 4
 * Number of inputs   = 2
 * Number of outputs  = 1
 * [Patterns]
 * 0 0   0
 * 1 0   0
 * 0 1   0
 * 1 1   1
 * </pre>
 *
 * <p>For recurrent networks the keyword {@code reset} between rows is
 * recognised (and silently skipped for feedforward use).</p>
 */
public class PatternParser {

    public final int                  numInputs;
    public final int                  numOutputs;
    public final List<TrainingPattern> patterns;

    private PatternParser(int ni, int no, List<TrainingPattern> p) {
        this.numInputs  = ni;
        this.numOutputs = no;
        this.patterns   = Collections.unmodifiableList(p);
    }

    // ── Factory methods ───────────────────────────────────────────────────────

    public static PatternParser fromFile(File f) throws IOException {
        return parse(Files.readString(f.toPath()));
    }

    public static PatternParser fromString(String text) throws IOException {
        return parse(text);
    }

    // ── Core parser ───────────────────────────────────────────────────────────

    private static PatternParser parse(String text) throws IOException {
        int numInputs  = 0;
        int numOutputs = 0;
        boolean inData = false;
        List<TrainingPattern> patterns = new ArrayList<>();

        for (String raw : text.split("\\r?\\n")) {
            String line = raw.trim();
            if (line.isEmpty()) continue;

            if (line.startsWith("#")) continue;   // skip comment lines

            String lower = line.toLowerCase();
            if (lower.startsWith("number of inputs")) {
                numInputs  = parseValue(line);
            } else if (lower.startsWith("number of outputs")) {
                numOutputs = parseValue(line);
            } else if (lower.equals("[patterns]")) {
                inData = true;
            } else if (inData) {
                if (lower.equals("reset")) continue;    // SRN boundary marker
                String[] parts = line.split("\\s+");
                if (parts.length < numInputs + numOutputs) continue;
                try {
                    double[] inputs  = new double[numInputs];
                    double[] targets = new double[numOutputs];
                    for (int i = 0; i < numInputs;  i++) inputs[i]  = Double.parseDouble(parts[i]);
                    for (int i = 0; i < numOutputs; i++) targets[i] = Double.parseDouble(parts[numInputs + i]);
                    patterns.add(new TrainingPattern(inputs, targets, line));
                } catch (NumberFormatException ex) {
                    throw new IOException("Invalid number in pattern line: " + line);
                }
            }
        }

        if (numInputs == 0) throw new IOException("Pattern file missing 'Number of inputs' header.");
        if (numOutputs == 0) throw new IOException("Pattern file missing 'Number of outputs' header.");
        if (patterns.isEmpty()) throw new IOException("No valid patterns found after [Patterns] marker.");
        return new PatternParser(numInputs, numOutputs, patterns);
    }

    private static int parseValue(String line) throws IOException {
        String[] parts = line.split("=");
        if (parts.length < 2) return 0;
        try {
            return Integer.parseInt(parts[1].trim());
        } catch (NumberFormatException ex) {
            throw new IOException("Invalid number in header: " + line);
        }
    }

    // ── Built-in presets ─────────────────────────────────────────────────────

    public static final String PRESET_AND =
            "Number of patterns = 4\nNumber of inputs = 2\nNumber of outputs = 1\n[Patterns]\n" +
            "0 0   0\n1 0   0\n0 1   0\n1 1   1";

    public static final String PRESET_OR =
            "Number of patterns = 4\nNumber of inputs = 2\nNumber of outputs = 1\n[Patterns]\n" +
            "0 0   0\n1 0   1\n0 1   1\n1 1   1";

    public static final String PRESET_XOR =
            "Number of patterns = 4\nNumber of inputs = 2\nNumber of outputs = 1\n[Patterns]\n" +
            "0 0   0\n1 0   1\n0 1   1\n1 1   0";
}
