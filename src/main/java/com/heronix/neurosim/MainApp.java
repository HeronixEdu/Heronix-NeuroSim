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

import javafx.application.Application;
import javafx.application.Platform;
import javafx.collections.FXCollections;
import javafx.collections.ObservableList;
import javafx.concurrent.Task;
import javafx.geometry.*;
import javafx.scene.Scene;
import javafx.scene.chart.*;
import javafx.scene.control.*;
import javafx.scene.layout.*;
import javafx.scene.paint.Color;
import javafx.scene.text.Font;
import javafx.stage.FileChooser;
import javafx.stage.Stage;

import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.util.List;

/**
 * Main JavaFX application window for Heronix NeuroSim.
 */
public class MainApp extends Application {

    // ── Colours / fonts ─────────────────────────────────────────────────────
    private String BG_DARK   = "#080d0b";
    private String BG_PANEL  = "#0c1410";
    private String BORDER_C  = "#132a1a";
    private String ACCENT    = "#00dc78";
    private String DIM_ACCENT= "#1a5a3a";
    private String TEXT_COL  = "#b8f0cc";
    private String CANVAS_BG = "#0a110d";
    private String CONSOLE_BG= "#04080a";
    private static final String MONO = "Courier New";
    private String currentTheme = "hacker";

    // ── State ───────────────────────────────────────────────────────────────
    private NeuralNet     net;
    private PatternParser patternSet;
    private volatile boolean training = false;
    private Thread        trainingThread;

    // ── UI references ───────────────────────────────────────────────────────
    private NetworkCanvas    networkCanvas;
    private LineChart<Number, Number> errorChart;
    private XYChart.Series<Number, Number> errorSeries;
    private TextArea         console;
    private ComboBox<TrainingPattern> patternCombo;
    private Label            epochLabel;
    private Label            statusLabel;
    private Stage            primaryStage;

    // ── Architecture config ─────────────────────────────────────────────────
    private int[]     layerSizes  = {2, 2, 1};
    private boolean[] biasEnabled = {true, false, false};

    // ── Entry points ────────────────────────────────────────────────────────

    public static void main(String[] args) { launch(args); }

    @Override
    public void start(Stage stage) {
        this.primaryStage = stage;
        stage.setTitle("Heronix NeuroSim \u2013 Neural Network Simulator v3.0");
        stage.setMinWidth(900);
        stage.setMinHeight(650);
        try {
            java.io.InputStream iconStream = getClass().getResourceAsStream("/img/Icon.png");
            if (iconStream != null) stage.getIcons().add(new javafx.scene.image.Image(iconStream));
        } catch (Exception ignored) {}

        BorderPane root = new BorderPane();
        root.setStyle("-fx-background-color: " + BG_DARK + ";");
        VBox topSection = new VBox(buildMenuBar(), buildHeader());
        root.setTop(topSection);

        SplitPane mainSplit = new SplitPane();
        mainSplit.setOrientation(javafx.geometry.Orientation.VERTICAL);
        mainSplit.getItems().addAll(buildCenter(), buildConsole());
        mainSplit.setDividerPositions(0.75);
        mainSplit.setStyle("-fx-background-color: " + BG_DARK + ";");
        root.setCenter(mainSplit);
        root.setBottom(buildFooter());

        Scene scene = new Scene(root, 1100, 780);
        scene.getStylesheets().add(getClass().getResource("theme-hacker.css").toExternalForm());
        stage.setScene(scene);
        stage.show();

        initNetwork(layerSizes, biasEnabled, 0.5);
        loadPatternText(PatternParser.PRESET_AND);
        log("Heronix NeuroSim \u2013 Neural Network Simulator v3.0");
        log("Default network: 2-2-1. Preset AND patterns loaded.");
        log("Choose Train tab \u2192 click TRAIN to begin.");
    }

    // ── Header ──────────────────────────────────────────────────────────────

    private HBox buildHeader() {
        Label title = label("HERONIX NEUROSIM", 22, ACCENT);
        title.setStyle("-fx-font-family: '" + MONO + "'; -fx-font-size: 24; " +
                "-fx-text-fill: " + ACCENT + "; -fx-effect: dropshadow(gaussian," + ACCENT + ",18,0.4,0,0);");
        Label sub = label("NEURAL NETWORK SIMULATOR", 11, DIM_ACCENT);
        VBox titleBox = new VBox(2, title, sub);

        epochLabel  = label("EPOCH 0 / 5000", 12, ACCENT);
        statusLabel = label("READY", 12, DIM_ACCENT);
        VBox infoBox = new VBox(3, epochLabel, statusLabel);
        infoBox.setAlignment(Pos.CENTER_RIGHT);

        HBox header = new HBox();
        header.setPadding(new Insets(10, 14, 10, 14));
        header.setStyle("-fx-background-color: " + BG_DARK + "; -fx-border-color: " + BORDER_C + "; -fx-border-width: 0 0 1 0;");
        HBox.setHgrow(titleBox, Priority.ALWAYS);
        header.getChildren().addAll(titleBox, infoBox);
        return header;
    }

    // ── Menu bar ────────────────────────────────────────────────────────────

    private MenuBar buildMenuBar() {
        // File menu
        MenuItem miSavePatterns = new MenuItem("Save Patterns\u2026");
        MenuItem miExportReport = new MenuItem("Export Report (ASCII)\u2026");
        MenuItem miExportBinary = new MenuItem("Export Report (Binary)\u2026");
        MenuItem miExit = new MenuItem("Exit");
        miSavePatterns.setOnAction(e -> savePatterns());
        miExportReport.setOnAction(e -> exportReport(false));
        miExportBinary.setOnAction(e -> exportReport(true));
        miExit.setOnAction(e -> Platform.exit());
        Menu fileMenu = new Menu("File", null, miSavePatterns, new SeparatorMenuItem(),
                miExportReport, miExportBinary, new SeparatorMenuItem(), miExit);

        // Network menu
        MenuItem miSaveConfig = new MenuItem("Save Configuration\u2026");
        MenuItem miLoadConfig = new MenuItem("Load Configuration\u2026");
        MenuItem miSaveWeights = new MenuItem("Save Weights\u2026");
        MenuItem miLoadWeights = new MenuItem("Load Weights\u2026");
        miSaveConfig.setOnAction(e -> saveNetworkConfig());
        miLoadConfig.setOnAction(e -> loadNetworkConfig());
        miSaveWeights.setOnAction(e -> saveWeights());
        miLoadWeights.setOnAction(e -> loadWeights());
        Menu networkMenu = new Menu("Network", null, miSaveConfig, miLoadConfig,
                new SeparatorMenuItem(), miSaveWeights, miLoadWeights);

        // Weights menu
        MenuItem miShowWeights = new MenuItem("Show Weights");
        miShowWeights.setOnAction(e -> showWeightsDialog());
        Menu weightsMenu = new Menu("Weights", null, miShowWeights);

        // Examples menu
        MenuItem miAndEx = new MenuItem("AND Gate  (2-2-1, easy)");
        MenuItem miOrEx = new MenuItem("OR Gate  (2-2-1, easy)");
        MenuItem miXorEx = new MenuItem("XOR Gate  (2-3-1, requires hidden layer)");
        MenuItem miXor221 = new MenuItem("XOR Gate  (2-2-1, may not converge)");
        MenuItem miAnd3in = new MenuItem("3-Input AND  (3-2-1)");
        MenuItem miParity = new MenuItem("Parity  (3-4-1, hard)");
        MenuItem miNand = new MenuItem("NAND Gate  (2-2-1)");
        MenuItem miNor = new MenuItem("NOR Gate  (2-2-1)");
        MenuItem miHalfAdd = new MenuItem("Half Adder  (2-4-2, sum + carry)");
        MenuItem miEncoder = new MenuItem("4-to-2 Encoder  (4-4-2)");
        MenuItem miIdentity = new MenuItem("Identity  (3-3, no hidden layer)");

        miAndEx.setOnAction(e -> loadExample("AND Gate", PatternParser.PRESET_AND,
                new int[]{2,2,1}, new boolean[]{true,false,false}, 0.5, 0.9, 5000, false, false));
        miOrEx.setOnAction(e -> loadExample("OR Gate", PatternParser.PRESET_OR,
                new int[]{2,2,1}, new boolean[]{true,false,false}, 0.5, 0.9, 5000, false, false));
        miXorEx.setOnAction(e -> loadExample("XOR Gate (2-3-1)", PatternParser.PRESET_XOR,
                new int[]{2,3,1}, new boolean[]{true,true,false}, 0.5, 0.9, 10000, false, false));
        miXor221.setOnAction(e -> loadExample("XOR Gate (2-2-1)", PatternParser.PRESET_XOR,
                new int[]{2,2,1}, new boolean[]{true,false,false}, 0.5, 0.9, 10000, false, false));
        miAnd3in.setOnAction(e -> loadExample("3-Input AND", PRESET_AND3,
                new int[]{3,2,1}, new boolean[]{true,false,false}, 0.5, 0.9, 5000, false, false));
        miParity.setOnAction(e -> loadExample("Parity (3-4-1)", PRESET_PARITY3,
                new int[]{3,4,1}, new boolean[]{true,true,false}, 0.3, 0.9, 20000, false, false));
        miNand.setOnAction(e -> loadExample("NAND Gate", PRESET_NAND,
                new int[]{2,2,1}, new boolean[]{true,false,false}, 0.5, 0.9, 5000, false, false));
        miNor.setOnAction(e -> loadExample("NOR Gate", PRESET_NOR,
                new int[]{2,2,1}, new boolean[]{true,false,false}, 0.5, 0.9, 5000, false, false));
        miHalfAdd.setOnAction(e -> loadExample("Half Adder", PRESET_HALF_ADDER,
                new int[]{2,4,2}, new boolean[]{true,true,false}, 0.5, 0.9, 10000, false, false));
        miEncoder.setOnAction(e -> loadExample("4-to-2 Encoder", PRESET_ENCODER,
                new int[]{4,4,2}, new boolean[]{true,true,false}, 0.3, 0.9, 10000, false, false));
        miIdentity.setOnAction(e -> loadExample("Identity (no hidden)", PRESET_IDENTITY,
                new int[]{3,3}, new boolean[]{true,false}, 0.5, 0.9, 5000, false, false));

        Menu examplesMenu = new Menu("Examples", null,
                miAndEx, miOrEx, miNand, miNor, new SeparatorMenuItem(),
                miXorEx, miXor221, new SeparatorMenuItem(),
                miAnd3in, miParity, new SeparatorMenuItem(),
                miHalfAdd, miEncoder, miIdentity);

        // View menu (themes)
        ToggleGroup themeGroup = new ToggleGroup();
        RadioMenuItem miHacker = new RadioMenuItem("Hacker Green");
        RadioMenuItem miDark = new RadioMenuItem("Dark");
        RadioMenuItem miSystem = new RadioMenuItem("System (Light)");
        miHacker.setToggleGroup(themeGroup); miDark.setToggleGroup(themeGroup); miSystem.setToggleGroup(themeGroup);
        miHacker.setSelected(true);
        miHacker.setOnAction(e -> applyTheme("hacker"));
        miDark.setOnAction(e -> applyTheme("dark"));
        miSystem.setOnAction(e -> applyTheme("system"));
        Menu viewMenu = new Menu("View", null, miHacker, miDark, miSystem);

        // Help menu
        MenuItem miAbout = new MenuItem("About Heronix NeuroSim");
        miAbout.setOnAction(e -> showAboutDialog());
        MenuItem miHelpGuide = new MenuItem("Quick Start Guide");
        miHelpGuide.setOnAction(e -> showHelpDialog());
        Menu helpMenu = new Menu("Help", null, miHelpGuide, new SeparatorMenuItem(), miAbout);

        MenuBar mb = new MenuBar(fileMenu, networkMenu, weightsMenu, examplesMenu, viewMenu, helpMenu);
        mb.setStyle("-fx-background-color: " + BG_PANEL + "; -fx-border-color: " + BORDER_C + "; -fx-border-width: 0 0 1 0;");
        return mb;
    }

    // ── Theme switching ─────────────────────────────────────────────────────

    private void applyTheme(String theme) {
        currentTheme = theme;
        switch (theme) {
            case "dark":
                BG_DARK="#1e1e1e"; BG_PANEL="#252526"; BORDER_C="#333";
                ACCENT="#4fc3f7"; DIM_ACCENT="#5a6a7a"; TEXT_COL="#cccccc";
                CANVAS_BG="#1a1a1a"; CONSOLE_BG="#1a1a1a"; break;
            case "system":
                BG_DARK="#f0f0f0"; BG_PANEL="#ffffff"; BORDER_C="#ccc";
                ACCENT="#0056a3"; DIM_ACCENT="#333333"; TEXT_COL="#111111";
                CANVAS_BG="#e8e8e8"; CONSOLE_BG="#f0f0f0"; break;
            default:
                BG_DARK="#080d0b"; BG_PANEL="#0c1410"; BORDER_C="#132a1a";
                ACCENT="#00dc78"; DIM_ACCENT="#1a5a3a"; TEXT_COL="#b8f0cc";
                CANVAS_BG="#0a110d"; CONSOLE_BG="#04080a"; break;
        }

        Scene scene = primaryStage.getScene();
        scene.getStylesheets().clear();
        scene.getStylesheets().add(getClass().getResource("theme-" + theme + ".css").toExternalForm());

        BorderPane root = new BorderPane();
        root.setStyle("-fx-background-color: " + BG_DARK + ";");
        VBox topSection = new VBox(buildMenuBar(), buildHeader());
        root.setTop(topSection);
        SplitPane mainSplit = new SplitPane();
        mainSplit.setOrientation(javafx.geometry.Orientation.VERTICAL);
        mainSplit.getItems().addAll(buildCenter(), buildConsole());
        mainSplit.setDividerPositions(0.75);
        mainSplit.setStyle("-fx-background-color: " + BG_DARK + ";");
        root.setCenter(mainSplit);
        root.setBottom(buildFooter());
        scene.setRoot(root);

        if (net != null) networkCanvas.render(net);
        if (patternSet != null) {
            ObservableList<TrainingPattern> items = FXCollections.observableArrayList(patternSet.patterns);
            patternCombo.setItems(items);
            if (!items.isEmpty()) patternCombo.setValue(items.get(0));
        }
        log("Theme changed to: " + theme);
    }

    // ── Example presets ─────────────────────────────────────────────────────

    private static final String PRESET_AND3 =
            "Number of patterns = 8\nNumber of inputs = 3\nNumber of outputs = 1\n[Patterns]\n" +
            "0 0 0   0\n1 0 0   0\n0 1 0   0\n0 0 1   0\n1 1 0   0\n1 0 1   0\n0 1 1   0\n1 1 1   1";
    private static final String PRESET_PARITY3 =
            "Number of patterns = 8\nNumber of inputs = 3\nNumber of outputs = 1\n[Patterns]\n" +
            "0 0 0   0\n1 0 0   1\n0 1 0   1\n0 0 1   1\n1 1 0   0\n1 0 1   0\n0 1 1   0\n1 1 1   1";
    private static final String PRESET_NAND =
            "Number of patterns = 4\nNumber of inputs = 2\nNumber of outputs = 1\n[Patterns]\n" +
            "0 0   1\n1 0   1\n0 1   1\n1 1   0";
    private static final String PRESET_NOR =
            "Number of patterns = 4\nNumber of inputs = 2\nNumber of outputs = 1\n[Patterns]\n" +
            "0 0   1\n1 0   0\n0 1   0\n1 1   0";
    private static final String PRESET_HALF_ADDER =
            "# Half Adder: 2 inputs -> sum, carry\n" +
            "Number of patterns = 4\nNumber of inputs = 2\nNumber of outputs = 2\n[Patterns]\n" +
            "0 0   0 0\n1 0   1 0\n0 1   1 0\n1 1   0 1";
    private static final String PRESET_ENCODER =
            "# 4-to-2 Encoder: one-hot input -> binary output\n" +
            "Number of patterns = 4\nNumber of inputs = 4\nNumber of outputs = 2\n[Patterns]\n" +
            "1 0 0 0   0 0\n0 1 0 0   0 1\n0 0 1 0   1 0\n0 0 0 1   1 1";
    private static final String PRESET_IDENTITY =
            "# Identity: network must learn to reproduce inputs as outputs\n" +
            "Number of patterns = 8\nNumber of inputs = 3\nNumber of outputs = 3\n[Patterns]\n" +
            "0 0 0   0 0 0\n1 0 0   1 0 0\n0 1 0   0 1 0\n0 0 1   0 0 1\n" +
            "1 1 0   1 1 0\n1 0 1   1 0 1\n0 1 1   0 1 1\n1 1 1   1 1 1";

    private void loadExample(String name, String patternText,
                             int[] sizes, boolean[] bias,
                             double lr, double momentum, int epochs,
                             boolean batch, boolean crossEntropy) {
        if (training) stopTraining();
        spLr.getValueFactory().setValue(lr);
        spMomentum.getValueFactory().setValue(momentum);
        spEpochs.getValueFactory().setValue(epochs);
        cbBatch.setSelected(batch);
        cbCrossEntropy.setSelected(crossEntropy);
        layerSizes = sizes.clone();
        biasEnabled = bias.clone();
        rebuildLayerRows();
        patternEditor.setText(patternText);
        loadPatternText(patternText);
        initNetwork(layerSizes, biasEnabled, spWtRange.getValue());
        log("\u2500\u2500 Example loaded: " + name + " \u2500\u2500");
        log("  Network: [" + intArrStr(sizes) + "]  lr=" + lr + "  momentum=" + momentum + "  epochs=" + epochs);
        log("  Click TRAIN to begin.");
    }

    // ── File operations ─────────────────────────────────────────────────────

    private void savePatterns() {
        if (patternSet == null) { log("No patterns loaded."); return; }
        FileChooser fc = new FileChooser();
        fc.setTitle("Save Patterns");
        fc.getExtensionFilters().add(new FileChooser.ExtensionFilter("Pattern files", "*.pat"));
        fc.setInitialFileName("patterns.pat");
        File f = fc.showSaveDialog(primaryStage);
        if (f == null) return;
        try {
            Files.writeString(f.toPath(), patternEditor.getText());
            log("Patterns saved to " + f.getName());
        } catch (IOException ex) { log("ERROR saving patterns: " + ex.getMessage()); }
    }

    private void saveNetworkConfig() {
        if (net == null) { log("No network."); return; }
        FileChooser fc = new FileChooser();
        fc.setTitle("Save Network Configuration");
        fc.getExtensionFilters().add(new FileChooser.ExtensionFilter("Config files", "*.cfg"));
        fc.setInitialFileName("network.cfg");
        File f = fc.showSaveDialog(primaryStage);
        if (f == null) return;
        try {
            StringBuilder sb = new StringBuilder();
            sb.append("[Type]\nType = FeedForward\n");
            sb.append("HiddenActivation = ").append(net.getHiddenActivation().name()).append('\n');
            sb.append("OutputActivation = ").append(net.getOutputActivation().name()).append('\n');
            sb.append("[Network]\n");
            sb.append("Layers = ").append(net.getNumLayers()).append('\n');
            for (int l = 0; l < net.getNumLayers(); l++) {
                String name = l == 0 ? "Input" : l == net.getNumLayers()-1 ? "Output" : "Hidden"+l;
                sb.append(name).append(" units = ").append(net.getLayerSize(l)).append('\n');
                if (l < net.getNumLayers()-1)
                    sb.append(name).append(" bias = ").append(net.getBiasEnabled(l)).append('\n');
            }
            Files.writeString(f.toPath(), sb.toString());
            log("Network config saved to " + f.getName());
        } catch (IOException ex) { log("ERROR saving config: " + ex.getMessage()); }
    }

    private void loadNetworkConfig() {
        FileChooser fc = new FileChooser();
        fc.setTitle("Load Network Configuration");
        fc.getExtensionFilters().add(new FileChooser.ExtensionFilter("Config files", "*.cfg"));
        File f = fc.showOpenDialog(primaryStage);
        if (f == null) return;
        try {
            String text = Files.readString(f.toPath());
            java.util.List<Integer> sizeList = new java.util.ArrayList<>();
            java.util.List<Boolean> biasList = new java.util.ArrayList<>();
            for (String raw : text.split("\\r?\\n")) {
                String line = raw.trim();
                String lower = line.toLowerCase();
                if (lower.startsWith("layers")) {
                    // parsed but not used directly
                } else if (lower.contains("units")) {
                    sizeList.add(Integer.parseInt(line.split("=")[1].trim()));
                } else if (lower.contains("bias") && !lower.contains("activation")) {
                    biasList.add(Boolean.parseBoolean(line.split("=")[1].trim()));
                } else if (lower.startsWith("hiddenactivation")) {
                    try { comboHiddenAct.setValue(ActivationFunction.valueOf(line.split("=")[1].trim())); } catch (Exception ignored) {}
                } else if (lower.startsWith("outputactivation")) {
                    try { comboOutputAct.setValue(ActivationFunction.valueOf(line.split("=")[1].trim())); } catch (Exception ignored) {}
                }
            }
            if (sizeList.size() < 2) { log("ERROR: Invalid config file."); return; }
            layerSizes = sizeList.stream().mapToInt(Integer::intValue).toArray();
            biasEnabled = new boolean[layerSizes.length];
            for (int i = 0; i < biasList.size() && i < biasEnabled.length; i++) biasEnabled[i] = biasList.get(i);
            rebuildLayerRows();
            initNetwork(layerSizes, biasEnabled, spWtRange.getValue());
            log("Config loaded from " + f.getName() + ": [" + intArrStr(layerSizes) + "]");
        } catch (Exception ex) { log("ERROR loading config: " + ex.getMessage()); }
    }

    private void saveWeights() {
        if (net == null) { log("No network."); return; }
        FileChooser fc = new FileChooser();
        fc.setTitle("Save Weights");
        fc.getExtensionFilters().add(new FileChooser.ExtensionFilter("Weight files", "*.wgt"));
        fc.setInitialFileName("weights.wgt");
        File f = fc.showSaveDialog(primaryStage);
        if (f == null) return;
        try {
            StringBuilder sb = new StringBuilder();
            sb.append("# Heronix NeuroSim Weights\n");
            sb.append("# Architecture: ").append(intArrStr(net.getLayerSizes())).append('\n');
            for (int l = 0; l < net.getNumLayers()-1; l++) {
                sb.append("[Layer ").append(l).append(" -> ").append(l+1).append("]\n");
                int from = net.getFromCount(l);
                for (int j = 0; j < net.getLayerSize(l+1); j++) {
                    for (int i = 0; i < from; i++) {
                        if (i > 0) sb.append('\t');
                        sb.append(String.format("%.10f", net.getWeight(l, j, i)));
                    }
                    sb.append('\n');
                }
            }
            Files.writeString(f.toPath(), sb.toString());
            log("Weights saved to " + f.getName());
        } catch (IOException ex) { log("ERROR saving weights: " + ex.getMessage()); }
    }

    private void loadWeights() {
        if (net == null) { log("No network. Configure first."); return; }
        FileChooser fc = new FileChooser();
        fc.setTitle("Load Weights");
        fc.getExtensionFilters().add(new FileChooser.ExtensionFilter("Weight files", "*.wgt"));
        File f = fc.showOpenDialog(primaryStage);
        if (f == null) return;
        try {
            String text = Files.readString(f.toPath());
            int layer = -1; int row = 0;
            for (String raw : text.split("\\r?\\n")) {
                String line = raw.trim();
                if (line.startsWith("#") || line.isEmpty()) continue;
                if (line.startsWith("[")) { layer++; row = 0; continue; }
                if (layer < 0 || layer >= net.getNumLayers()-1) continue;
                if (row >= net.getLayerSize(layer+1)) continue;
                String[] parts = line.split("\\s+");
                int from = net.getFromCount(layer);
                for (int i = 0; i < Math.min(parts.length, from); i++)
                    net.setWeight(layer, row, i, Double.parseDouble(parts[i]));
                row++;
            }
            networkCanvas.render(net);
            log("Weights loaded from " + f.getName());
        } catch (Exception ex) { log("ERROR loading weights: " + ex.getMessage()); }
    }

    private void showWeightsDialog() {
        if (net == null) { log("No network."); return; }
        StringBuilder sb = new StringBuilder();
        for (int l = 0; l < net.getNumLayers()-1; l++) {
            sb.append("\u2500\u2500 Layer ").append(l).append(" \u2192 ").append(l+1).append(" \u2500\u2500\n");
            int from = net.getFromCount(l);
            sb.append(String.format("%-6s", ""));
            for (int i = 0; i < net.getLayerSize(l); i++) sb.append(String.format("  u%-5d", i));
            if (net.getBiasEnabled(l)) sb.append("  bias  ");
            sb.append('\n');
            for (int j = 0; j < net.getLayerSize(l+1); j++) {
                sb.append(String.format("u%-5d", j));
                for (int i = 0; i < from; i++) sb.append(String.format(" %+7.4f", net.getWeight(l, j, i)));
                sb.append('\n');
            }
            sb.append('\n');
        }
        log("\u2500\u2500 WEIGHT TABLE \u2500\u2500\n" + sb);
    }

    private void exportReport(boolean binary) {
        if (net == null) { log("No network."); return; }
        FileChooser fc = new FileChooser();
        fc.setTitle(binary ? "Export Report (Binary)" : "Export Report (ASCII)");
        fc.getExtensionFilters().add(new FileChooser.ExtensionFilter(
                binary ? "Binary report" : "Text report", binary ? "*.bin" : "*.txt"));
        fc.setInitialFileName(binary ? "report.bin" : "report.txt");
        File f = fc.showSaveDialog(primaryStage);
        if (f == null) return;
        try {
            StringBuilder ascii = new StringBuilder();
            ascii.append("=== Heronix NeuroSim Network Report ===\n\n");
            ascii.append("[Architecture]\n");
            ascii.append("Layers: ").append(net.getNumLayers()).append('\n');
            ascii.append("Topology: ").append(intArrStr(net.getLayerSizes())).append('\n');
            ascii.append("Hidden Activation: ").append(net.getHiddenActivation()).append('\n');
            ascii.append("Output Activation: ").append(net.getOutputActivation()).append('\n');
            for (int l = 0; l < net.getNumLayers(); l++) {
                String name = l==0 ? "Input" : l==net.getNumLayers()-1 ? "Output" : "Hidden"+l;
                ascii.append(name).append(": ").append(net.getLayerSize(l)).append(" units");
                if (l < net.getNumLayers()-1 && net.getBiasEnabled(l)) ascii.append(" + bias");
                ascii.append('\n');
            }
            ascii.append("\n[Training Parameters]\n");
            ascii.append("Learning Rate: ").append(spLr.getValue()).append('\n');
            ascii.append("Momentum: ").append(spMomentum.getValue()).append('\n');
            ascii.append("Max Epochs: ").append(spEpochs.getValue()).append('\n');
            ascii.append("Mode: ").append(cbBatch.isSelected() ? "Batch (parallel)" : "Online").append('\n');
            ascii.append("Error Function: ").append(cbCrossEntropy.isSelected() ? "Cross-Entropy" : "MSE").append('\n');
            ascii.append("\n[Status]\n").append(epochLabel.getText()).append('\n');
            ascii.append("\n[Weights]\n");
            for (int l = 0; l < net.getNumLayers()-1; l++) {
                ascii.append("Layer ").append(l).append(" -> ").append(l+1).append(":\n");
                int from = net.getFromCount(l);
                for (int j = 0; j < net.getLayerSize(l+1); j++) {
                    for (int i = 0; i < from; i++) {
                        if (i > 0) ascii.append('\t');
                        ascii.append(String.format("%.10f", net.getWeight(l, j, i)));
                    }
                    ascii.append('\n');
                }
            }
            if (patternSet != null && !patternSet.patterns.isEmpty()) {
                ascii.append("\n[Test Results]\n");
                boolean xent = cbCrossEntropy.isSelected();
                double totalErr = 0;
                for (TrainingPattern p : patternSet.patterns) {
                    double[] out = net.forward(p.inputs);
                    double err = net.computeError(p.targets, xent);
                    totalErr += err;
                    ascii.append("Input: [");
                    for (int i = 0; i < p.inputs.length; i++) { if (i>0) ascii.append(", "); ascii.append(p.inputs[i]); }
                    ascii.append("]  Output: [");
                    for (int i = 0; i < out.length; i++) { if (i>0) ascii.append(", "); ascii.append(String.format("%.6f", out[i])); }
                    ascii.append("]  Target: [");
                    for (int i = 0; i < p.targets.length; i++) { if (i>0) ascii.append(", "); ascii.append(p.targets[i]); }
                    ascii.append("]  Error: ").append(String.format("%.6f", err)).append('\n');
                }
                ascii.append("Average Error: ").append(String.format("%.6f", totalErr / patternSet.patterns.size())).append('\n');
            }
            if (binary) {
                String asciiText = ascii.toString();
                StringBuilder binOut = new StringBuilder();
                binOut.append("=== Heronix NeuroSim Binary Report ===\n");
                binOut.append("=== ASCII values converted to 8-bit binary ===\n\n");
                for (int i = 0; i < asciiText.length(); i++) {
                    char c = asciiText.charAt(i);
                    String bits = String.format("%8s", Integer.toBinaryString(c)).replace(' ', '0');
                    binOut.append(bits);
                    binOut.append(c == '\n' ? '\n' : ' ');
                }
                Files.writeString(f.toPath(), binOut.toString());
                log("Binary report exported to " + f.getName());
            } else {
                Files.writeString(f.toPath(), ascii.toString());
                log("ASCII report exported to " + f.getName());
            }
        } catch (IOException ex) { log("ERROR exporting report: " + ex.getMessage()); }
    }

    // ── Centre layout ───────────────────────────────────────────────────────

    private SplitPane buildCenter() {
        VBox left = buildLeftColumn();
        VBox right = buildRightPanel();
        SplitPane sp = new SplitPane(left, right);
        sp.setDividerPositions(0.58);
        sp.setStyle("-fx-background-color: " + BG_DARK + ";");
        return sp;
    }

    private VBox buildLeftColumn() {
        networkCanvas = new NetworkCanvas(560, 270);
        networkCanvas.setThemeColors(CANVAS_BG, DIM_ACCENT);
        networkCanvas.clear();
        TitledPane vizPane = titledPane("\u25C8  NETWORK VISUALIZATION", networkCanvas);

        NumberAxis xAxis = new NumberAxis();
        NumberAxis yAxis = new NumberAxis();
        styleAxis(xAxis, "Epoch"); styleAxis(yAxis, "Avg Error");
        errorChart = new LineChart<>(xAxis, yAxis);
        errorSeries = new XYChart.Series<>();
        errorSeries.setName("Error");
        errorChart.getData().add(errorSeries);
        errorChart.setAnimated(false);
        errorChart.setCreateSymbols(false);
        errorChart.setLegendVisible(false);
        errorChart.setMinHeight(120);
        errorChart.setPrefHeight(250);
        errorChart.setStyle("-fx-background-color: " + BG_PANEL + ";");
        TitledPane chartPane = titledPane("\u25C8  TRAINING ERROR (per-pattern avg)", errorChart);

        VBox col = new VBox(8, vizPane, chartPane);
        col.setPadding(new Insets(10, 6, 6, 10));
        col.setStyle("-fx-background-color: " + BG_DARK + ";");
        VBox.setVgrow(vizPane, Priority.SOMETIMES);
        VBox.setVgrow(chartPane, Priority.ALWAYS);
        return col;
    }

    // ── Right panel with all tabs ───────────────────────────────────────────

    private VBox buildRightPanel() {
        TabPane tabs = new TabPane(
                buildTrainTab(),
                buildStepTab(),
                buildConfigureTab(),
                buildPatternsTab(),
                buildArchitecturesTab(),
                buildTutorialTab()
        );
        tabs.setTabClosingPolicy(TabPane.TabClosingPolicy.UNAVAILABLE);
        tabs.setStyle("-fx-background-color: " + BG_PANEL + ";" +
                " -fx-tab-min-width: 50; -fx-tab-max-height: 30;" +
                " -fx-font-family: '" + MONO + "'; -fx-font-size: 11;");
        tabs.getStyleClass().add("neurosim-tabs");

        VBox panel = new VBox(tabs);
        panel.setPadding(new Insets(10, 10, 6, 6));
        panel.setStyle("-fx-background-color: " + BG_DARK + ";");
        VBox.setVgrow(tabs, Priority.ALWAYS);
        return panel;
    }

    // ── Train tab ───────────────────────────────────────────────────────────

    private Spinner<Double> spLr, spMomentum, spWtRange;
    private Spinner<Integer> spEpochs;
    private CheckBox cbBatch, cbCrossEntropy;
    private Button btnTrain, btnReset;

    private Tab buildTrainTab() {
        spLr       = doubleSpinner(0.001, 10.0, 0.5, 0.05);
        spMomentum = doubleSpinner(0.0, 1.0, 0.9, 0.05);
        spWtRange  = doubleSpinner(0.01, 5.0, 0.5, 0.05);
        spEpochs   = intSpinner(1, 100_000, 5000, 100);

        // Tooltips
        tip(spLr, "Learning Rate: Controls step size during weight updates.\nHigher = faster but may overshoot. Try 0.1-1.0.");
        tip(spMomentum, "Momentum: Adds fraction of previous weight change.\nHelps escape local minima. Typical: 0.5-0.95.");
        tip(spWtRange, "Initial weight range: [-range, +range].\nSmaller range for larger networks.");
        tip(spEpochs, "Maximum training epochs (iterations).\nTraining stops early if error < 0.001.");

        cbBatch = checkbox("Batch Update (parallel)");
        cbCrossEntropy = checkbox("Cross-Entropy Error");
        tip(cbBatch, "Batch: accumulate all gradients before updating.\nUses parallel processing for 16+ patterns.\nOnline: update after each pattern (stochastic GD).");
        tip(cbCrossEntropy, "Cross-Entropy: better for classification tasks.\nMSE: standard mean squared error.");

        btnTrain = btn("\u25B6  TRAIN", ACCENT, BG_DARK, true);
        btnReset = btn("\u21BA  RESET WEIGHTS", "#ff5555", BG_DARK, true);
        btnTrain.setOnAction(e -> { if (training) stopTraining(); else startTraining(); });
        btnReset.setOnAction(e -> resetWeights());

        patternCombo = new ComboBox<>();
        patternCombo.setMaxWidth(Double.MAX_VALUE);
        styleCombo(patternCombo);

        Button btnTestOne = btn("TEST ONE", ACCENT, BG_DARK, false);
        Button btnTestAll = btn("TEST ALL", ACCENT, BG_DARK, false);
        btnTestOne.setOnAction(e -> testOne());
        btnTestAll.setOnAction(e -> testAll());
        tip(btnTestOne, "Run selected pattern through the network and show output.");
        tip(btnTestAll, "Test all patterns and show average error.");

        GridPane params = new GridPane();
        params.setHgap(8); params.setVgap(6); params.setPadding(new Insets(6));
        addRow(params, 0, "Learning Rate", spLr);
        addRow(params, 1, "Momentum", spMomentum);
        addRow(params, 2, "Max Epochs", spEpochs);
        addRow(params, 3, "Weight Range", spWtRange);

        HBox trainBtns = new HBox(6, btnTrain, btnReset);
        HBox.setHgrow(btnTrain, Priority.ALWAYS); HBox.setHgrow(btnReset, Priority.ALWAYS);
        btnTrain.setMaxWidth(Double.MAX_VALUE); btnReset.setMaxWidth(Double.MAX_VALUE);

        HBox testBtns = new HBox(6, btnTestOne, btnTestAll);
        HBox.setHgrow(btnTestOne, Priority.ALWAYS); HBox.setHgrow(btnTestAll, Priority.ALWAYS);
        btnTestOne.setMaxWidth(Double.MAX_VALUE); btnTestAll.setMaxWidth(Double.MAX_VALUE);

        VBox content = new VBox(10, sectionLabel("PARAMETERS"), params,
                cbBatch, cbCrossEntropy, trainBtns, sep(),
                sectionLabel("TEST"), label("Pattern:", 11, DIM_ACCENT), patternCombo, testBtns);
        content.setPadding(new Insets(10));
        content.setStyle("-fx-background-color: " + BG_PANEL + ";");

        return styledTab("TRAIN", new ScrollPane(content) {{ setFitToWidth(true); setStyle("-fx-background-color: " + BG_PANEL + ";"); }});
    }

    // ── Step-by-step tab ────────────────────────────────────────────────────

    private TextArea stepOutput;
    private int stepPatternIndex = 0;

    private Tab buildStepTab() {
        stepOutput = new TextArea();
        stepOutput.setEditable(false);
        stepOutput.setFont(Font.font(MONO, 11));
        stepOutput.setStyle("-fx-control-inner-background: " + CONSOLE_BG + "; -fx-text-fill: " + TEXT_COL +
                "; -fx-border-color: " + BORDER_C + ";");
        stepOutput.setPrefRowCount(20);

        Button btnStepFwd = btn("\u25B6 STEP FORWARD", ACCENT, BG_DARK, true);
        Button btnStepBwd = btn("\u25C0 STEP BACKWARD", "#ffaa00", BG_DARK, true);
        Button btnStepBoth = btn("\u25B6\u25C0 FULL STEP", ACCENT, BG_DARK, false);
        Button btnNextPat = btn("NEXT PATTERN \u25B6", DIM_ACCENT, BG_DARK, false);

        tip(btnStepFwd, "Run ONE forward pass on the current pattern.\nShows net inputs, activations, and error for each layer.");
        tip(btnStepBwd, "Run backward pass (backpropagation).\nShows deltas and weight updates for each layer.");
        tip(btnStepBoth, "Run forward + backward pass together.");
        tip(btnNextPat, "Move to the next pattern in the training set.");

        btnStepFwd.setOnAction(e -> doStepForward());
        btnStepBwd.setOnAction(e -> doStepBackward());
        btnStepBoth.setOnAction(e -> { doStepForward(); doStepBackward(); });
        btnNextPat.setOnAction(e -> {
            if (patternSet != null && !patternSet.patterns.isEmpty()) {
                stepPatternIndex = (stepPatternIndex + 1) % patternSet.patterns.size();
                stepOutput.appendText("\n--- Pattern " + (stepPatternIndex+1) + "/" + patternSet.patterns.size() + " ---\n");
                stepOutput.appendText("Input:  " + arrStr(patternSet.patterns.get(stepPatternIndex).inputs) + "\n");
                stepOutput.appendText("Target: " + arrStr(patternSet.patterns.get(stepPatternIndex).targets) + "\n");
            }
        });

        btnStepFwd.setMaxWidth(Double.MAX_VALUE);
        btnStepBwd.setMaxWidth(Double.MAX_VALUE);
        btnStepBoth.setMaxWidth(Double.MAX_VALUE);
        btnNextPat.setMaxWidth(Double.MAX_VALUE);

        Label info = label("Step through training one pass at a time.\nSee net inputs, activations, deltas, and\nweight changes for each layer.", 10, DIM_ACCENT);
        info.setWrapText(true);

        VBox content = new VBox(8,
                sectionLabel("STEP-BY-STEP TRAINING"),
                info,
                btnStepFwd, btnStepBwd, btnStepBoth, sep(), btnNextPat, sep(),
                sectionLabel("OUTPUT"), stepOutput);
        content.setPadding(new Insets(10));
        content.setStyle("-fx-background-color: " + BG_PANEL + ";");

        return styledTab("STEP", new ScrollPane(content) {{ setFitToWidth(true); setStyle("-fx-background-color: " + BG_PANEL + ";"); }});
    }

    private void doStepForward() {
        if (net == null || patternSet == null || patternSet.patterns.isEmpty()) {
            stepOutput.appendText("ERROR: Load patterns and configure network first.\n"); return;
        }
        TrainingPattern p = patternSet.patterns.get(stepPatternIndex % patternSet.patterns.size());
        NeuralNet.StepResult r = net.stepForward(p.inputs, p.targets, cbCrossEntropy.isSelected());
        networkCanvas.render(net);

        StringBuilder sb = new StringBuilder();
        sb.append("=== FORWARD PASS ===\n");
        sb.append("Pattern ").append(stepPatternIndex+1).append(": ").append(arrStr(p.inputs)).append(" -> ").append(arrStr(p.targets)).append("\n");
        sb.append("Activation: hidden=").append(net.getHiddenActivation()).append(" output=").append(net.getOutputActivation()).append("\n\n");

        for (int l = 0; l < r.layerActivations.length; l++) {
            String name = l == 0 ? "INPUT" : l == r.layerActivations.length-1 ? "OUTPUT" : "HIDDEN " + l;
            sb.append("[").append(name).append("]\n");
            sb.append("  Net inputs:   ").append(arrStr(r.layerNetInputs[l])).append("\n");
            sb.append("  Activations:  ").append(arrStr(r.layerActivations[l])).append("\n");
        }
        sb.append("\nOutput: ").append(arrStr(r.output)).append("\n");
        sb.append("Error:  ").append(String.format("%.8f", r.error)).append("\n\n");
        stepOutput.appendText(sb.toString());
        stepOutput.setScrollTop(Double.MAX_VALUE);
    }

    private void doStepBackward() {
        if (net == null || patternSet == null || patternSet.patterns.isEmpty()) {
            stepOutput.appendText("ERROR: Run forward pass first.\n"); return;
        }
        TrainingPattern p = patternSet.patterns.get(stepPatternIndex % patternSet.patterns.size());
        NeuralNet.StepResult r = net.stepBackward(p.targets, cbCrossEntropy.isSelected(),
                spLr.getValue(), spMomentum.getValue());
        networkCanvas.render(net);

        StringBuilder sb = new StringBuilder();
        sb.append("=== BACKWARD PASS (Backpropagation) ===\n");
        sb.append("lr=").append(spLr.getValue()).append(" momentum=").append(spMomentum.getValue()).append("\n\n");

        if (r.deltas != null) {
            for (int l = r.deltas.length-1; l > 0; l--) {
                String name = l == r.deltas.length-1 ? "OUTPUT" : "HIDDEN " + l;
                sb.append("[").append(name).append(" deltas]\n");
                sb.append("  ").append(arrStr(r.deltas[l])).append("\n");
            }
        }
        sb.append("\nError after update: ").append(String.format("%.8f", r.error)).append("\n");
        sb.append("Weights updated. Check Weights menu to inspect.\n\n");
        stepOutput.appendText(sb.toString());
        stepOutput.setScrollTop(Double.MAX_VALUE);
    }

    // ── Configure tab ───────────────────────────────────────────────────────

    private VBox layerRows;
    private ComboBox<ActivationFunction> comboHiddenAct, comboOutputAct;

    private Tab buildConfigureTab() {
        layerRows = new VBox(6);
        rebuildLayerRows();

        // Activation function selectors
        comboHiddenAct = new ComboBox<>(FXCollections.observableArrayList(ActivationFunction.values()));
        comboOutputAct = new ComboBox<>(FXCollections.observableArrayList(ActivationFunction.values()));
        comboHiddenAct.setValue(ActivationFunction.SIGMOID);
        comboOutputAct.setValue(ActivationFunction.SIGMOID);
        styleCombo(comboHiddenAct); styleCombo(comboOutputAct);
        comboHiddenAct.setMaxWidth(Double.MAX_VALUE);
        comboOutputAct.setMaxWidth(Double.MAX_VALUE);
        tip(comboHiddenAct, "Activation for hidden layers:\n\u2022 Sigmoid: smooth 0-1, classic choice\n\u2022 ReLU: fast, good for deep nets\n\u2022 Tanh: -1 to 1, centered\n\u2022 Softmax: probability distribution");
        tip(comboOutputAct, "Activation for output layer:\n\u2022 Sigmoid: binary classification\n\u2022 Softmax: multi-class classification\n\u2022 Tanh/ReLU: regression tasks");

        Button btnAddHidden = btn("+ ADD HIDDEN LAYER", ACCENT, BG_DARK, false);
        Button btnRemHidden = btn("\u2212 REMOVE HIDDEN", "#ff5555", BG_DARK, false);
        Button btnApply = btn("APPLY & REINITIALIZE", ACCENT, BG_DARK, true);
        btnAddHidden.setMaxWidth(Double.MAX_VALUE);
        btnRemHidden.setMaxWidth(Double.MAX_VALUE);
        btnApply.setMaxWidth(Double.MAX_VALUE);

        btnAddHidden.setOnAction(e -> {
            int NL = layerSizes.length;
            if (NL >= 6) { log("Maximum 4 hidden layers supported."); return; }
            int[] ns = new int[NL+1]; boolean[] nb = new boolean[NL+1];
            System.arraycopy(layerSizes, 0, ns, 0, NL-1);
            System.arraycopy(biasEnabled, 0, nb, 0, NL-1);
            ns[NL-1]=3; nb[NL-1]=true; ns[NL]=layerSizes[NL-1]; nb[NL]=false;
            layerSizes=ns; biasEnabled=nb; rebuildLayerRows();
        });
        btnRemHidden.setOnAction(e -> {
            int NL = layerSizes.length;
            if (NL <= 2) { log("Need at least input + output layers."); return; }
            int[] ns = new int[NL-1]; boolean[] nb = new boolean[NL-1];
            System.arraycopy(layerSizes, 0, ns, 0, NL-2);
            System.arraycopy(biasEnabled, 0, nb, 0, NL-2);
            ns[NL-2]=layerSizes[NL-1]; nb[NL-2]=false;
            layerSizes=ns; biasEnabled=nb; rebuildLayerRows();
        });
        btnApply.setOnAction(e -> {
            collectLayerConfig();
            initNetwork(layerSizes, biasEnabled, spWtRange.getValue());
            log("Network re-initialized: [" + intArrStr(layerSizes) + "]  hidden=" + comboHiddenAct.getValue() + "  output=" + comboOutputAct.getValue());
        });

        VBox content = new VBox(10,
                sectionLabel("LAYER CONFIGURATION"),
                label("Units per layer + bias flag", 11, DIM_ACCENT),
                layerRows, sep(),
                sectionLabel("ACTIVATION FUNCTIONS"),
                label("Hidden layers:", 11, DIM_ACCENT), comboHiddenAct,
                label("Output layer:", 11, DIM_ACCENT), comboOutputAct,
                sep(),
                btnAddHidden, btnRemHidden, sep(), btnApply);
        content.setPadding(new Insets(10));
        content.setStyle("-fx-background-color: " + BG_PANEL + ";");

        return styledTab("CONFIG", new ScrollPane(content) {{ setFitToWidth(true); setStyle("-fx-background-color: " + BG_PANEL + ";"); }});
    }

    // ── Patterns tab ────────────────────────────────────────────────────────

    private TextArea patternEditor;

    private Tab buildPatternsTab() {
        patternEditor = new TextArea(PatternParser.PRESET_AND);
        patternEditor.setFont(Font.font(MONO, 12));
        patternEditor.setStyle("-fx-control-inner-background: " + CONSOLE_BG + "; -fx-text-fill: " + TEXT_COL +
                "; -fx-border-color: " + BORDER_C + ";");
        patternEditor.setPrefRowCount(14);

        Button btnAnd = btn("AND", ACCENT, BG_DARK, false);
        Button btnOr = btn("OR", ACCENT, BG_DARK, false);
        Button btnXor = btn("XOR", ACCENT, BG_DARK, false);
        Button btnLoad = btn("LOAD FILE\u2026", ACCENT, BG_DARK, false);
        Button btnApply = btn("APPLY PATTERNS", ACCENT, BG_DARK, true);
        btnAnd.setOnAction(e -> patternEditor.setText(PatternParser.PRESET_AND));
        btnOr.setOnAction(e -> patternEditor.setText(PatternParser.PRESET_OR));
        btnXor.setOnAction(e -> patternEditor.setText(PatternParser.PRESET_XOR));
        btnLoad.setOnAction(e -> loadPatternFile());
        btnApply.setOnAction(e -> loadPatternText(patternEditor.getText()));
        btnApply.setMaxWidth(Double.MAX_VALUE);

        HBox presets = new HBox(6, btnAnd, btnOr, btnXor, btnLoad);
        VBox content = new VBox(10, sectionLabel("PATTERN FILE"), presets, patternEditor, btnApply);
        content.setPadding(new Insets(10));
        content.setStyle("-fx-background-color: " + BG_PANEL + ";");

        return styledTab("PATTERNS", new ScrollPane(content) {{ setFitToWidth(true); setStyle("-fx-background-color: " + BG_PANEL + ";"); }});
    }

    // ── Architectures demo tab ──────────────────────────────────────────────

    private TextArea archOutput;

    private Tab buildArchitecturesTab() {
        archOutput = new TextArea();
        archOutput.setEditable(false);
        archOutput.setFont(Font.font(MONO, 11));
        archOutput.setStyle("-fx-control-inner-background: " + CONSOLE_BG + "; -fx-text-fill: " + TEXT_COL +
                "; -fx-border-color: " + BORDER_C + ";");
        archOutput.setPrefRowCount(16);

        Button btnRnnDemo = btn("RUN RNN DEMO", "#ffaa00", BG_DARK, true);
        Button btnCnnDemo = btn("RUN CNN DEMO", "#ff6688", BG_DARK, true);
        Button btnAttnDemo = btn("RUN ATTENTION DEMO", "#aa88ff", BG_DARK, true);

        tip(btnRnnDemo, "Simple Elman RNN: learns to predict next value in a sequence.\nUses context units that feed hidden state back.");
        tip(btnCnnDemo, "Simple 1D CNN: learns pattern features via convolution.\nConv layer \u2192 ReLU \u2192 Max Pool \u2192 FC \u2192 Output.");
        tip(btnAttnDemo, "Self-Attention: computes Query/Key/Value matrices\nand attention weights showing which inputs attend to which.");

        btnRnnDemo.setOnAction(e -> runRnnDemo());
        btnCnnDemo.setOnAction(e -> runCnnDemo());
        btnAttnDemo.setOnAction(e -> runAttentionDemo());

        btnRnnDemo.setMaxWidth(Double.MAX_VALUE);
        btnCnnDemo.setMaxWidth(Double.MAX_VALUE);
        btnAttnDemo.setMaxWidth(Double.MAX_VALUE);

        Label info = label("Explore different neural network architectures.\nEach demo runs a self-contained example with\ndetailed output showing internal state.", 10, DIM_ACCENT);
        info.setWrapText(true);

        VBox content = new VBox(8,
                sectionLabel("ARCHITECTURE DEMOS"),
                info, sep(),
                btnRnnDemo, btnCnnDemo, btnAttnDemo, sep(),
                sectionLabel("DEMO OUTPUT"), archOutput);
        content.setPadding(new Insets(10));
        content.setStyle("-fx-background-color: " + BG_PANEL + ";");

        return styledTab("ARCHS", new ScrollPane(content) {{ setFitToWidth(true); setStyle("-fx-background-color: " + BG_PANEL + ";"); }});
    }

    private void runRnnDemo() {
        archOutput.clear();
        archOutput.appendText("=== SIMPLE RNN (Elman Network) DEMO ===\n\n");
        archOutput.appendText("Architecture: 2 inputs, 4 hidden + 4 context, 1 output\n");
        archOutput.appendText("Task: XOR on sequential pairs\n\n");

        try {
            RecurrentNet rnn = new RecurrentNet(2, 4, 1, true);
            rnn.randomiseWeights(0.5);

            double[][] inputs = {{0,0},{1,0},{0,1},{1,1}};
            double[][] targets = {{0},{0},{0},{1}};

            archOutput.appendText("Training for 2000 epochs...\n");
            for (int ep = 1; ep <= 2000; ep++) {
                rnn.resetContext();
                double totalErr = 0;
                for (int i = 0; i < inputs.length; i++) {
                    TrainingPattern p = new TrainingPattern(inputs[i], targets[i], "");
                    totalErr += rnn.trainSequence(java.util.List.of(p), 0.5, 0.9, false);
                }
                if (ep % 500 == 0)
                    archOutput.appendText(String.format("  Epoch %4d: avg error = %.6f\n", ep, totalErr/inputs.length));
            }

            archOutput.appendText("\nTest results:\n");
            rnn.resetContext();
            for (int i = 0; i < inputs.length; i++) {
                double[] out = rnn.forward(inputs[i]);
                archOutput.appendText(String.format("  [%.0f, %.0f] -> %.4f (target: %.0f)\n",
                        inputs[i][0], inputs[i][1], out[0], targets[i][0]));
            }
            archOutput.appendText("\nContext units preserve hidden state between timesteps,\n");
            archOutput.appendText("allowing the network to learn sequential dependencies.\n");
        } catch (Exception ex) {
            archOutput.appendText("Error: " + ex.getMessage() + "\n");
        }
    }

    private void runCnnDemo() {
        archOutput.clear();
        archOutput.appendText("=== SIMPLE 1D CNN DEMO ===\n\n");
        archOutput.appendText("Architecture: input(7) -> Conv(3 filters, size 3) -> ReLU -> MaxPool(2) -> FC -> Sigmoid(2)\n");
        archOutput.appendText("Task: Classify 7-segment patterns into even/odd\n\n");

        try {
            ConvNet cnn = new ConvNet(7, 3, 3, 2, 2);
            cnn.randomiseWeights(0.5);

            double[][] inputs = {
                {1,1,1,1,1,1,0}, // 0 -> even
                {0,1,1,0,0,0,0}, // 1 -> odd
                {1,1,0,1,0,1,1}, // 2 -> even
                {1,1,1,1,0,0,1}, // 3 -> odd
            };
            double[][] targets = {{1,0},{0,1},{1,0},{0,1}};

            archOutput.appendText("Training for 1000 epochs...\n");
            for (int ep = 1; ep <= 1000; ep++) {
                double totalErr = 0;
                for (int i = 0; i < inputs.length; i++) {
                    totalErr += cnn.train(inputs[i], targets[i], 0.3);
                }
                if (ep % 250 == 0)
                    archOutput.appendText(String.format("  Epoch %4d: avg error = %.6f\n", ep, totalErr/inputs.length));
            }

            archOutput.appendText("\nTest results:\n");
            for (int i = 0; i < inputs.length; i++) {
                double[] out = cnn.forward(inputs[i]);
                archOutput.appendText(String.format("  Pattern %d -> [%.4f, %.4f] (%s)\n",
                        i, out[0], out[1], out[0] > out[1] ? "even" : "odd"));
            }

            archOutput.appendText("\nConv filters extract local features from the input.\n");
            archOutput.appendText("Filters learned:\n");
            double[][] filters = cnn.getFilters();
            for (int f = 0; f < filters.length; f++) {
                archOutput.appendText("  Filter " + f + ": " + arrStr(filters[f]) + "\n");
            }
        } catch (Exception ex) {
            archOutput.appendText("Error: " + ex.getMessage() + "\n");
        }
    }

    private void runAttentionDemo() {
        archOutput.clear();
        archOutput.appendText("=== SELF-ATTENTION MECHANISM DEMO ===\n\n");
        archOutput.appendText("Architecture: seq_len=4, d_model=3, d_key=2\n");
        archOutput.appendText("Demonstrates Query-Key-Value attention computation\n\n");

        try {
            AttentionDemo attn = new AttentionDemo(4, 3, 2);
            attn.randomiseWeights(0.5);

            double[][] input = {
                {1.0, 0.0, 0.5},
                {0.0, 1.0, 0.3},
                {0.5, 0.5, 1.0},
                {0.2, 0.8, 0.1}
            };

            archOutput.appendText("Input sequence (4 tokens x 3 dims):\n");
            for (int i = 0; i < input.length; i++)
                archOutput.appendText("  Token " + i + ": " + arrStr(input[i]) + "\n");

            AttentionDemo.AttentionResult result = attn.computeWithDetails(input);

            archOutput.appendText("\nAttention Weights (who attends to whom):\n");
            archOutput.appendText(String.format("%-8s", ""));
            for (int j = 0; j < 4; j++) archOutput.appendText(String.format("  Tok%-3d", j));
            archOutput.appendText("\n");
            for (int i = 0; i < 4; i++) {
                archOutput.appendText(String.format("Tok %-3d", i));
                for (int j = 0; j < 4; j++)
                    archOutput.appendText(String.format("  %.4f", result.weights[i][j]));
                archOutput.appendText("\n");
            }

            archOutput.appendText("\nOutput (attended representations):\n");
            for (int i = 0; i < result.output.length; i++)
                archOutput.appendText("  Token " + i + ": " + arrStr(result.output[i]) + "\n");

            archOutput.appendText("\nHow it works:\n");
            archOutput.appendText("1. Q = input x Wq, K = input x Wk, V = input x Wv\n");
            archOutput.appendText("2. Scores = Q x K^T / sqrt(d_key)\n");
            archOutput.appendText("3. Weights = softmax(Scores)\n");
            archOutput.appendText("4. Output = Weights x V\n");
            archOutput.appendText("\nHigh attention weight means that token strongly attends to another.\n");
        } catch (Exception ex) {
            archOutput.appendText("Error: " + ex.getMessage() + "\n");
        }
    }

    // ── Tutorial tab ────────────────────────────────────────────────────────

    private Tab buildTutorialTab() {
        TextArea tutorialText = new TextArea();
        tutorialText.setEditable(false);
        tutorialText.setFont(Font.font(MONO, 11));
        tutorialText.setWrapText(true);
        tutorialText.setStyle("-fx-control-inner-background: " + CONSOLE_BG + "; -fx-text-fill: " + TEXT_COL +
                "; -fx-border-color: " + BORDER_C + ";");

        ComboBox<String> topicCombo = new ComboBox<>(FXCollections.observableArrayList(
                "What is a Neural Network?",
                "How Backpropagation Works",
                "Activation Functions Explained",
                "Learning Rate & Momentum",
                "Batch vs Online Training",
                "Understanding the Error Chart",
                "Recurrent Neural Networks (RNN)",
                "Convolutional Neural Networks (CNN)",
                "Self-Attention & Transformers",
                "Tips for Better Training"
        ));
        topicCombo.setValue("What is a Neural Network?");
        styleCombo(topicCombo);
        topicCombo.setMaxWidth(Double.MAX_VALUE);

        topicCombo.setOnAction(e -> tutorialText.setText(getTutorialText(topicCombo.getValue())));
        tutorialText.setText(getTutorialText("What is a Neural Network?"));

        VBox content = new VBox(10,
                sectionLabel("INTERACTIVE TUTORIALS"),
                label("Select a topic to learn:", 11, DIM_ACCENT),
                topicCombo, tutorialText);
        content.setPadding(new Insets(10));
        content.setStyle("-fx-background-color: " + BG_PANEL + ";");
        VBox.setVgrow(tutorialText, Priority.ALWAYS);

        return styledTab("LEARN", new ScrollPane(content) {{ setFitToWidth(true); setStyle("-fx-background-color: " + BG_PANEL + ";"); }});
    }

    private String getTutorialText(String topic) {
        switch (topic) {
            case "What is a Neural Network?": return
                "WHAT IS A NEURAL NETWORK?\n" +
                "=========================\n\n" +
                "A neural network is a computational model inspired by\n" +
                "biological neurons. It consists of layers of interconnected\n" +
                "nodes (neurons) that process information.\n\n" +
                "STRUCTURE:\n" +
                "  Input Layer:  receives raw data\n" +
                "  Hidden Layer: processes features (can be multiple)\n" +
                "  Output Layer: produces the result\n\n" +
                "Each connection has a WEIGHT that determines its strength.\n" +
                "Each neuron computes:\n" +
                "  output = activation( sum(input_i * weight_i) + bias )\n\n" +
                "TRY IT: Load the AND gate example from the Examples menu.\n" +
                "The 2-2-1 network has 2 inputs, 2 hidden, 1 output.\n" +
                "Click TRAIN and watch the weights change!";

            case "How Backpropagation Works": return
                "HOW BACKPROPAGATION WORKS\n" +
                "=========================\n\n" +
                "Backpropagation is the algorithm that trains neural networks.\n\n" +
                "STEPS:\n" +
                "1. FORWARD PASS: Input flows through the network\n" +
                "   Each neuron computes: net = sum(w*x) + bias\n" +
                "   Then applies activation: out = f(net)\n\n" +
                "2. COMPUTE ERROR: Compare output to target\n" +
                "   MSE = 0.5 * (target - output)^2\n\n" +
                "3. BACKWARD PASS: Compute deltas (error signals)\n" +
                "   Output delta = (target - out) * f'(out)\n" +
                "   Hidden delta = f'(out) * sum(next_delta * weight)\n\n" +
                "4. UPDATE WEIGHTS:\n" +
                "   new_weight = old_weight + lr * delta * input\n\n" +
                "TRY IT: Use the STEP tab to see each pass in detail!\n" +
                "Watch how deltas propagate backward through layers.";

            case "Activation Functions Explained": return
                "ACTIVATION FUNCTIONS\n" +
                "====================\n\n" +
                "Activation functions add non-linearity to the network.\n" +
                "Without them, any deep network = a single linear layer.\n\n" +
                "SIGMOID:  f(x) = 1 / (1 + e^-x)\n" +
                "  Range: (0, 1)\n" +
                "  Pros: Smooth, classic, good for binary classification\n" +
                "  Cons: Vanishing gradients for deep networks\n\n" +
                "RELU:     f(x) = max(0, x)\n" +
                "  Range: [0, inf)\n" +
                "  Pros: Fast, no vanishing gradient for positive values\n" +
                "  Cons: 'Dying ReLU' - neurons can stop learning\n\n" +
                "TANH:     f(x) = (e^x - e^-x) / (e^x + e^-x)\n" +
                "  Range: (-1, 1)\n" +
                "  Pros: Zero-centered, often better than sigmoid\n" +
                "  Cons: Still has vanishing gradient issue\n\n" +
                "SOFTMAX:  f(x_i) = e^x_i / sum(e^x_j)\n" +
                "  Range: (0, 1), sums to 1\n" +
                "  Use: Multi-class classification output layer\n\n" +
                "TRY IT: Change activation in the CONFIG tab and retrain!";

            case "Learning Rate & Momentum": return
                "LEARNING RATE & MOMENTUM\n" +
                "========================\n\n" +
                "LEARNING RATE (lr):\n" +
                "Controls how big each weight update step is.\n" +
                "  Too high (>1.0): Overshoots, may never converge\n" +
                "  Too low (<0.01): Very slow learning\n" +
                "  Sweet spot: Usually 0.1 - 0.5 for small networks\n\n" +
                "MOMENTUM:\n" +
                "Adds a fraction of the previous weight change.\n" +
                "  new_dw = lr * gradient + momentum * prev_dw\n\n" +
                "Benefits:\n" +
                "  - Speeds up training in consistent gradient direction\n" +
                "  - Helps escape small local minima\n" +
                "  - Dampens oscillations\n\n" +
                "Typical values: 0.5 - 0.95\n\n" +
                "TRY IT: Train XOR with lr=0.1 vs lr=2.0.\n" +
                "Then try momentum=0.0 vs momentum=0.95.";

            case "Batch vs Online Training": return
                "BATCH vs ONLINE TRAINING\n" +
                "========================\n\n" +
                "ONLINE (Stochastic Gradient Descent):\n" +
                "  - Update weights after EACH pattern\n" +
                "  - Patterns are shuffled each epoch\n" +
                "  - Noisy but can escape local minima\n" +
                "  - Good for small datasets\n\n" +
                "BATCH (Gradient Descent):\n" +
                "  - Accumulate all gradients, then update once\n" +
                "  - Smoother convergence\n" +
                "  - Uses parallel processing for 16+ patterns\n" +
                "  - More stable but can get stuck\n\n" +
                "In Heronix NeuroSim, batch mode uses Java parallel\n" +
                "streams to compute gradients concurrently on multi-core CPUs.\n\n" +
                "TRY IT: Train the same problem with both modes.\n" +
                "Watch how the error chart differs!";

            case "Understanding the Error Chart": return
                "THE ERROR CHART\n" +
                "===============\n\n" +
                "The chart shows average per-pattern error over epochs.\n\n" +
                "WHAT TO LOOK FOR:\n" +
                "  Smooth decrease: Good learning!\n" +
                "  Plateau: May need higher lr or different architecture\n" +
                "  Oscillating: lr too high, try reducing it\n" +
                "  Stuck at high error: Try more hidden units or layers\n\n" +
                "ERROR FUNCTIONS:\n" +
                "  MSE = 0.5 * sum(target - output)^2\n" +
                "    Good general purpose, regression tasks\n\n" +
                "  Cross-Entropy = -sum(t*log(o) + (1-t)*log(1-o))\n" +
                "    Better for classification, stronger gradients\n\n" +
                "CONVERGENCE:\n" +
                "  Training stops when avg error < 0.001\n" +
                "  or when max epochs is reached.";

            case "Recurrent Neural Networks (RNN)": return
                "RECURRENT NEURAL NETWORKS\n" +
                "=========================\n\n" +
                "RNNs process SEQUENCES by maintaining internal state.\n\n" +
                "ELMAN NETWORK (Simple RNN):\n" +
                "  - Hidden layer has 'context units'\n" +
                "  - Context = copy of previous hidden activations\n" +
                "  - This gives the network 'memory'\n\n" +
                "  Hidden(t) = f(W_in * Input(t) + W_ctx * Hidden(t-1))\n" +
                "  Output(t) = f(W_out * Hidden(t))\n\n" +
                "USES:\n" +
                "  - Time series prediction\n" +
                "  - Sequence classification\n" +
                "  - Language modeling\n\n" +
                "LIMITATIONS:\n" +
                "  - Vanishing gradients for long sequences\n" +
                "  - LSTM/GRU solve this (not in this demo)\n\n" +
                "TRY IT: Go to ARCHS tab and run the RNN demo!";

            case "Convolutional Neural Networks (CNN)": return
                "CONVOLUTIONAL NEURAL NETWORKS\n" +
                "=============================\n\n" +
                "CNNs excel at finding PATTERNS in data.\n\n" +
                "KEY OPERATIONS:\n" +
                "1. CONVOLUTION: Slide a filter across the input\n" +
                "   Each filter detects a specific feature\n" +
                "   output[i] = sum(input[i:i+k] * filter)\n\n" +
                "2. ACTIVATION (ReLU): max(0, conv_output)\n" +
                "   Adds non-linearity\n\n" +
                "3. POOLING (Max): Take max in each window\n" +
                "   Reduces size, adds translation invariance\n\n" +
                "4. FULLY CONNECTED: Standard neural net layer\n" +
                "   Maps pooled features to output classes\n\n" +
                "WHY CONVOLUTION?\n" +
                "  - Weight sharing: same filter scans everywhere\n" +
                "  - Local connectivity: focuses on nearby features\n" +
                "  - Much fewer parameters than fully connected\n\n" +
                "TRY IT: Go to ARCHS tab and run the CNN demo!";

            case "Self-Attention & Transformers": return
                "SELF-ATTENTION & TRANSFORMERS\n" +
                "=============================\n\n" +
                "Attention lets each element 'look at' every other element.\n\n" +
                "HOW IT WORKS:\n" +
                "For each token in the sequence:\n" +
                "  Query (Q) = input x Wq   'What am I looking for?'\n" +
                "  Key   (K) = input x Wk   'What do I contain?'\n" +
                "  Value (V) = input x Wv   'What do I offer?'\n\n" +
                "COMPUTATION:\n" +
                "  1. Scores = Q * K^T / sqrt(d_key)\n" +
                "  2. Weights = softmax(Scores)\n" +
                "  3. Output = Weights * V\n\n" +
                "The attention weights show WHICH tokens attend to WHICH.\n" +
                "High weight = strong relationship between tokens.\n\n" +
                "TRANSFORMERS use multi-head attention + feed-forward\n" +
                "layers. They power GPT, BERT, and modern LLMs.\n\n" +
                "TRY IT: Go to ARCHS tab and run the Attention demo!\n" +
                "Watch the attention weight matrix to see relationships.";

            case "Tips for Better Training": return
                "TIPS FOR BETTER TRAINING\n" +
                "========================\n\n" +
                "1. START SIMPLE\n" +
                "   Begin with AND/OR before XOR. Add complexity gradually.\n\n" +
                "2. ARCHITECTURE MATTERS\n" +
                "   XOR needs a hidden layer (2-3-1 works, 2-2-1 may not).\n" +
                "   More units = more capacity but slower training.\n\n" +
                "3. LEARNING RATE\n" +
                "   Start at 0.5, decrease if oscillating, increase if slow.\n\n" +
                "4. USE BIAS\n" +
                "   Always enable bias on hidden layers. It shifts the\n" +
                "   activation function and helps convergence.\n\n" +
                "5. WEIGHT INITIALIZATION\n" +
                "   Range 0.5 works for small nets. Use 0.1-0.3 for larger.\n\n" +
                "6. CROSS-ENTROPY FOR CLASSIFICATION\n" +
                "   Provides stronger gradients near 0 and 1.\n\n" +
                "7. RESET AND RETRY\n" +
                "   Different random weights = different results.\n" +
                "   If stuck, reset weights and train again.\n\n" +
                "8. USE STEP MODE\n" +
                "   The STEP tab shows exactly what happens each pass.\n" +
                "   Great for understanding why training succeeds or fails.";

            default: return "Select a topic from the dropdown above.";
        }
    }

    // ── Console ─────────────────────────────────────────────────────────────

    private VBox buildConsole() {
        console = new TextArea();
        console.setEditable(false);
        console.setFont(Font.font(MONO, 12));
        console.setPrefRowCount(8);
        console.setStyle("-fx-control-inner-background: " + CONSOLE_BG + "; -fx-text-fill: " + TEXT_COL +
                "; -fx-border-color: " + BORDER_C + ";");
        Label consoleLabel = label("\u25C8  CONSOLE", 11, ACCENT);
        consoleLabel.setPadding(new Insets(4, 8, 4, 8));
        consoleLabel.setMaxWidth(Double.MAX_VALUE);
        consoleLabel.setStyle("-fx-background-color: " + BG_PANEL + "; -fx-border-color: " + BORDER_C +
                "; -fx-border-width: 0 0 1 0;");
        VBox box = new VBox(consoleLabel, console);
        box.setStyle("-fx-background-color: " + BG_PANEL + ";");
        VBox.setVgrow(console, Priority.ALWAYS);
        return box;
    }

    // ── Footer ──────────────────────────────────────────────────────────────

    private HBox buildFooter() {
        Label left = label("Heronix NeuroSim v3.0  |  Heronix Education Systems LLC  |  By Michael Katsaros  |  Free Education Tools", 10, DIM_ACCENT);
        Label right = label("\u00A9 2026", 10, DIM_ACCENT);
        HBox footer = new HBox(left, right);
        HBox.setHgrow(left, Priority.ALWAYS);
        footer.setAlignment(Pos.CENTER_LEFT);
        footer.setPadding(new Insets(3, 10, 3, 10));
        footer.setStyle("-fx-background-color: " + BG_PANEL + "; -fx-border-color: " + BORDER_C +
                "; -fx-border-width: 1 0 0 0;");
        return footer;
    }

    // ── Help / About dialogs ────────────────────────────────────────────────

    private void showAboutDialog() {
        Alert alert = new Alert(Alert.AlertType.INFORMATION);
        alert.setTitle("About Heronix NeuroSim");
        alert.setHeaderText("Heronix NeuroSim v3.0");
        alert.setContentText(
                "Neural Network Simulator\n\n" +
                "Developed by Michael Katsaros\n" +
                "Heronix Education Systems LLC\n\n" +
                "Features: Feedforward, RNN, CNN, Attention demos\n" +
                "Activation: Sigmoid, ReLU, Tanh, Softmax\n" +
                "Step-by-step training, built-in tutorials\n\n" +
                "License: MIT \u2014 free to use, modify, and distribute.\n\n" +
                "\u00A9 2026 Heronix Education Systems LLC");
        alert.showAndWait();
    }

    private void showHelpDialog() {
        Alert alert = new Alert(Alert.AlertType.INFORMATION);
        alert.setTitle("Quick Start Guide");
        alert.setHeaderText("How to Use Heronix NeuroSim");
        alert.setContentText(
                "1. LOAD PATTERNS\n" +
                "   Use Examples menu or Patterns tab.\n\n" +
                "2. CONFIGURE NETWORK\n" +
                "   Config tab: layers, units, activation functions.\n\n" +
                "3. TRAIN\n" +
                "   Train tab: set parameters, click TRAIN.\n\n" +
                "4. STEP-BY-STEP\n" +
                "   Step tab: walk through one pass at a time.\n\n" +
                "5. EXPLORE ARCHITECTURES\n" +
                "   Archs tab: try RNN, CNN, Attention demos.\n\n" +
                "6. LEARN\n" +
                "   Learn tab: interactive tutorials on every topic.");
        alert.getDialogPane().setMinWidth(450);
        alert.showAndWait();
    }

    // ── Neural network actions ──────────────────────────────────────────────

    private void initNetwork(int[] sizes, boolean[] bias, double range) {
        if (training) stopTraining();
        net = new NeuralNet(sizes, bias);
        if (comboHiddenAct != null) net.setHiddenActivation(comboHiddenAct.getValue());
        if (comboOutputAct != null) net.setOutputActivation(comboOutputAct.getValue());
        net.randomiseWeights(range);
        Platform.runLater(() -> {
            networkCanvas.render(net);
            errorSeries.getData().clear();
            epochLabel.setText("EPOCH 0 / " + spEpochs.getValue());
            statusLabel.setText("READY");
        });
    }

    private void resetWeights() {
        if (training) { log("Stop training first."); return; }
        net.randomiseWeights(spWtRange.getValue());
        Platform.runLater(() -> {
            networkCanvas.render(net);
            errorSeries.getData().clear();
            epochLabel.setText("EPOCH 0 / " + spEpochs.getValue());
        });
        log("Weights randomised in [\u00B1" + spWtRange.getValue() + "].");
    }

    private void startTraining() {
        if (patternSet == null || patternSet.patterns.isEmpty()) {
            log("ERROR: No patterns loaded."); return;
        }
        if (net == null) { log("ERROR: No network."); return; }
        if (net.getLayerSize(0) != patternSet.numInputs) {
            log("ERROR: Network has " + net.getLayerSize(0) + " inputs but patterns have " + patternSet.numInputs + "."); return;
        }
        if (net.getLayerSize(net.getNumLayers()-1) != patternSet.numOutputs) {
            log("ERROR: Network has " + net.getLayerSize(net.getNumLayers()-1) + " outputs but patterns have " + patternSet.numOutputs + "."); return;
        }

        // Apply activation settings
        if (comboHiddenAct != null) net.setHiddenActivation(comboHiddenAct.getValue());
        if (comboOutputAct != null) net.setOutputActivation(comboOutputAct.getValue());

        training = true;
        Platform.runLater(() -> {
            btnTrain.setText("\u23F9  STOP");
            btnTrain.setStyle(btnStyle("#ff5555", BG_DARK, true));
            statusLabel.setText("TRAINING\u2026");
        });

        final double lr = spLr.getValue();
        final double mom = spMomentum.getValue();
        final int maxEp = spEpochs.getValue();
        final boolean batch = cbBatch.isSelected();
        final boolean xent = cbCrossEntropy.isSelected();
        final List<TrainingPattern> pats = patternSet.patterns;

        log(String.format("\u25B6 Training: lr=%.4f  momentum=%.3f  max=%d  %s  %s  hidden=%s  output=%s",
                lr, mom, maxEp, batch ? "batch(parallel)" : "online", xent ? "cross-entropy" : "MSE",
                net.getHiddenActivation(), net.getOutputActivation()));

        Task<Void> task = new Task<>() {
            @Override protected Void call() {
                int plotEvery = Math.max(1, maxEp / 400);
                for (int ep = 1; ep <= maxEp && training; ep++) {
                    double err = net.trainEpoch(pats, batch, lr, mom, xent);
                    final int epoch = ep;
                    final double error = err;
                    if (ep % plotEvery == 0 || ep == maxEp) {
                        NeuralNet snapshot = new NeuralNet(net);
                        Platform.runLater(() -> {
                            networkCanvas.render(snapshot);
                            if (errorSeries.getData().size() > 500) errorSeries.getData().remove(0);
                            errorSeries.getData().add(new XYChart.Data<>(epoch, error));
                            epochLabel.setText("EPOCH " + epoch + " / " + maxEp);
                        });
                    }
                    if (err < 0.001) {
                        final double ferr = err;
                        Platform.runLater(() -> {
                            log(String.format("\u2713 Converged at epoch %d.  Avg error: %.6f", epoch, ferr));
                            finishTraining();
                        });
                        return null;
                    }
                }
                Platform.runLater(() -> {
                    double finalErr = 0;
                    var data = errorSeries.getData();
                    if (!data.isEmpty()) finalErr = ((Number)data.get(data.size()-1).getYValue()).doubleValue();
                    log(String.format("\u25A0 Reached max epochs (%d).  Final avg error: %.6f", maxEp, finalErr));
                    finishTraining();
                });
                return null;
            }
        };

        trainingThread = new Thread(task, "neurosim-train");
        trainingThread.setDaemon(true);
        trainingThread.start();
    }

    private void stopTraining() {
        training = false;
        Platform.runLater(this::finishTraining);
        log("\u25A0 Training stopped by user.");
    }

    private void finishTraining() {
        if (!training && btnTrain.getText().contains("TRAIN")) return;
        training = false;
        btnTrain.setText("\u25B6  TRAIN");
        btnTrain.setStyle(btnStyle(ACCENT, BG_DARK, true));
        statusLabel.setText("IDLE");
    }

    private void testOne() {
        if (net == null || patternSet == null) return;
        TrainingPattern p = patternCombo.getValue();
        if (p == null) return;
        double[] out = net.forward(p.inputs);
        networkCanvas.render(net);
        StringBuilder sb = new StringBuilder("TEST  [");
        for (int i = 0; i < p.inputs.length; i++) { if (i>0) sb.append(", "); sb.append(p.inputs[i]); }
        sb.append("]  \u2192  [");
        for (int i = 0; i < out.length; i++) { if (i>0) sb.append(", "); sb.append(String.format("%.4f", out[i])); }
        sb.append("]   target: [");
        for (int i = 0; i < p.targets.length; i++) { if (i>0) sb.append(", "); sb.append(p.targets[i]); }
        sb.append("]");
        log(sb.toString());
    }

    private void testAll() {
        if (net == null || patternSet == null) return;
        boolean xent = cbCrossEntropy.isSelected();
        double total = 0;
        for (TrainingPattern p : patternSet.patterns) {
            net.forward(p.inputs);
            total += net.computeError(p.targets, xent);
        }
        log(String.format("TEST ALL: avg error = %.6f  over %d patterns",
                total / patternSet.patterns.size(), patternSet.patterns.size()));
    }

    // ── Pattern loading ─────────────────────────────────────────────────────

    private void loadPatternText(String text) {
        try {
            patternSet = PatternParser.fromString(text);
            ObservableList<TrainingPattern> items = FXCollections.observableArrayList(patternSet.patterns);
            Platform.runLater(() -> {
                patternCombo.setItems(items);
                if (!items.isEmpty()) patternCombo.setValue(items.get(0));
            });
            log("\u2713 Loaded " + patternSet.patterns.size() + " patterns  (" +
                    patternSet.numInputs + " inputs \u2192 " + patternSet.numOutputs + " outputs).");
            int ni = patternSet.numInputs, no = patternSet.numOutputs;
            if (layerSizes[0] != ni || layerSizes[layerSizes.length-1] != no) {
                int[] newSizes = new int[layerSizes.length];
                boolean[] newBias = new boolean[layerSizes.length];
                newSizes[0] = ni; newSizes[newSizes.length-1] = no;
                for (int i = 1; i < newSizes.length-1; i++) {
                    newSizes[i] = Math.max(layerSizes[i], (ni+no)/2);
                    newBias[i] = i < biasEnabled.length && biasEnabled[i];
                }
                newBias[0] = biasEnabled.length > 0 && biasEnabled[0];
                layerSizes = newSizes; biasEnabled = newBias;
                initNetwork(layerSizes, biasEnabled, spWtRange.getValue());
                rebuildLayerRows();
                log("  Network auto-configured: [" + intArrStr(layerSizes) + "]");
            }
        } catch (IOException ex) { log("ERROR parsing patterns: " + ex.getMessage()); }
    }

    private void loadPatternFile() {
        FileChooser fc = new FileChooser();
        fc.setTitle("Open Pattern File");
        fc.getExtensionFilters().addAll(
                new FileChooser.ExtensionFilter("Pattern files", "*.pat"),
                new FileChooser.ExtensionFilter("All files", "*.*"));
        File f = fc.showOpenDialog(primaryStage);
        if (f == null) return;
        try {
            String text = Files.readString(f.toPath());
            patternEditor.setText(text);
            loadPatternText(text);
        } catch (IOException ex) { log("ERROR reading file: " + ex.getMessage()); }
    }

    // ── Configure helpers ───────────────────────────────────────────────────

    private Spinner<Integer>[] unitSpinners;
    private CheckBox[] biasChecks;

    @SuppressWarnings("unchecked")
    private void rebuildLayerRows() {
        int NL = layerSizes.length;
        unitSpinners = new Spinner[NL];
        biasChecks = new CheckBox[NL];
        layerRows.getChildren().clear();
        String[] lbls = new String[NL];
        lbls[0] = "INPUT"; lbls[NL-1] = "OUTPUT";
        for (int l = 1; l < NL-1; l++) lbls[l] = "HIDDEN " + l;

        for (int l = 0; l < NL; l++) {
            unitSpinners[l] = intSpinner(1, 50, layerSizes[l], 1);
            biasChecks[l] = checkbox("Bias");
            biasChecks[l].setSelected(l < biasEnabled.length && biasEnabled[l]);
            biasChecks[l].setDisable(l == NL-1);
            tip(biasChecks[l], "Bias adds a trainable offset to the activation.\nAlmost always helps convergence.");

            Label layerLabel = label(lbls[l], 12, DIM_ACCENT);
            layerLabel.setMinWidth(65);
            HBox row = new HBox(8, layerLabel, unitSpinners[l], biasChecks[l]);
            row.setAlignment(Pos.CENTER_LEFT);
            layerRows.getChildren().add(row);
        }
    }

    private void collectLayerConfig() {
        int NL = unitSpinners.length;
        layerSizes = new int[NL];
        biasEnabled = new boolean[NL];
        for (int l = 0; l < NL; l++) {
            layerSizes[l] = unitSpinners[l].getValue();
            biasEnabled[l] = biasChecks[l].isSelected();
        }
    }

    // ── Console ─────────────────────────────────────────────────────────────

    private void log(String msg) {
        Platform.runLater(() -> {
            console.appendText(msg + "\n");
            console.setScrollTop(Double.MAX_VALUE);
        });
    }

    // ── UI factory helpers ──────────────────────────────────────────────────

    private static Label label(String text, double size, String color) {
        Label l = new Label(text);
        l.setFont(Font.font(MONO, size));
        l.setTextFill(Color.web(color));
        return l;
    }

    private Label sectionLabel(String text) {
        Label l = label(text, 11, ACCENT);
        l.setStyle("-fx-font-family: '" + MONO + "'; -fx-text-fill: " + ACCENT +
                "; -fx-border-color: " + BORDER_C + "; -fx-border-width: 0 0 1 0; -fx-padding: 0 0 4 0;");
        l.setMaxWidth(Double.MAX_VALUE);
        return l;
    }

    private Separator sep() {
        Separator s = new Separator();
        s.setStyle("-fx-background-color: " + BORDER_C + ";");
        return s;
    }

    private TitledPane titledPane(String title, javafx.scene.Node content) {
        TitledPane tp = new TitledPane(title, content);
        tp.setCollapsible(false);
        tp.setFont(Font.font(MONO, 11));
        tp.setStyle("-fx-text-fill: " + ACCENT + "; -fx-background-color: " + BG_PANEL +
                "; -fx-border-color: " + BORDER_C + ";");
        return tp;
    }

    private Tab styledTab(String text, javafx.scene.Node content) {
        Tab t = new Tab(text, content);
        t.setStyle("-fx-background-color: " + BG_PANEL + ";");
        return t;
    }

    private String btnStyle(String color, String bg, boolean prominent) {
        String bg2 = prominent ? color : bg;
        String fg = prominent ? bg : color;
        return "-fx-background-color: " + bg2 + "; -fx-text-fill: " + fg +
                "; -fx-border-color: " + color + "; -fx-border-width: 1;" +
                " -fx-font-family: '" + MONO + "'; -fx-font-size: 12; -fx-cursor: hand;";
    }

    private Button btn(String text, String color, String bg, boolean prominent) {
        Button b = new Button(text);
        b.setStyle(btnStyle(color, bg, prominent));
        b.setMaxWidth(Double.MAX_VALUE);
        return b;
    }

    private CheckBox checkbox(String text) {
        CheckBox cb = new CheckBox(text);
        cb.setFont(Font.font(MONO, 12));
        cb.setTextFill(Color.web(DIM_ACCENT));
        cb.setStyle("-fx-background-color: transparent;");
        return cb;
    }

    private Spinner<Double> doubleSpinner(double min, double max, double init, double step) {
        Spinner<Double> sp = new Spinner<>(new SpinnerValueFactory.DoubleSpinnerValueFactory(min, max, init, step));
        sp.setEditable(true); sp.setPrefWidth(120);
        sp.setStyle("-fx-background-color: " + CONSOLE_BG + "; -fx-text-fill: " + TEXT_COL + "; -fx-border-color: " + BORDER_C + ";");
        commitOnFocusLost(sp);
        return sp;
    }

    private Spinner<Integer> intSpinner(int min, int max, int init, int step) {
        Spinner<Integer> sp = new Spinner<>(new SpinnerValueFactory.IntegerSpinnerValueFactory(min, max, init, step));
        sp.setEditable(true); sp.setPrefWidth(100);
        sp.setStyle("-fx-background-color: " + CONSOLE_BG + "; -fx-text-fill: " + TEXT_COL + "; -fx-border-color: " + BORDER_C + ";");
        commitOnFocusLost(sp);
        return sp;
    }

    private <T> void commitOnFocusLost(Spinner<T> sp) {
        sp.focusedProperty().addListener((obs, wasFocused, isFocused) -> {
            if (!isFocused) { try { sp.increment(0); } catch (Exception ignored) {} }
        });
    }

    private <T> void styleCombo(ComboBox<T> cb) {
        cb.setStyle("-fx-background-color: " + CONSOLE_BG + "; -fx-text-fill: " + TEXT_COL +
                "; -fx-border-color: " + BORDER_C + "; -fx-font-family: '" + MONO + "'; -fx-font-size: 12;");
    }

    private void addRow(GridPane gp, int row, String lbl, javafx.scene.Node control) {
        Label l = label(lbl, 11, DIM_ACCENT);
        GridPane.setHgrow(control, Priority.ALWAYS);
        gp.add(l, 0, row);
        gp.add(control, 1, row);
    }

    private void styleAxis(Axis<?> axis, String label) {
        axis.setLabel(label);
        axis.setStyle("-fx-tick-label-fill: " + DIM_ACCENT + "; -fx-text-fill: " + DIM_ACCENT + ";");
    }

    private void tip(javafx.scene.Node node, String text) {
        Tooltip tt = new Tooltip(text);
        tt.setFont(Font.font(MONO, 11));
        tt.setStyle("-fx-background-color: #1a2a20; -fx-text-fill: #b8f0cc; -fx-border-color: #2a4a30; -fx-padding: 8;");
        tt.setShowDelay(javafx.util.Duration.millis(300));
        Tooltip.install(node, tt);
    }

    private static String intArrStr(int[] arr) {
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < arr.length; i++) { if (i>0) sb.append('-'); sb.append(arr[i]); }
        return sb.toString();
    }

    private static String arrStr(double[] arr) {
        StringBuilder sb = new StringBuilder("[");
        for (int i = 0; i < arr.length; i++) {
            if (i > 0) sb.append(", ");
            sb.append(String.format("%.4f", arr[i]));
        }
        return sb.append("]").toString();
    }
}
