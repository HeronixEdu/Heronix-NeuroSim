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

/**
 * Launcher entry point.
 * Kept separate from {@link MainApp} so that the fat JAR can launch without
 * the JavaFX Application class needing to be on the class-path as a main
 * class directly (which breaks with some JVM / shade configurations).
 */
public class Launcher {
    public static void main(String[] args) {
        MainApp.main(args);
    }
}
