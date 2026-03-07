#!/usr/bin/env bash
# Heronix NeuroSim - Neural Network Simulator
# Copyright (c) 2026 Heronix Education Systems LLC
# Free and open source software

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

if ! command -v java &> /dev/null; then
    echo ""
    echo "  ERROR: Java is not installed or not in your PATH."
    echo "  Please install Java 17 or later from https://adoptium.net"
    echo ""
    exit 1
fi

echo "Starting Heronix NeuroSim..."
java -jar "$SCRIPT_DIR/HeronixNeuroSim-2.0.0.jar" 2>/dev/null
