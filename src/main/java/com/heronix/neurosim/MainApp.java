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
 *
 * Layout:
 * <pre>
 * ┌─────────────────────────────────────────────────┐
 * │  Header bar                                      │
 * ├──────────────────────────┬──────────────────────┤
 * │  Network visualisation   │  Right control panel  │
 * │  Error chart             │  (tabbed: Train /     │
 * │                          │   Configure / Pats)   │
 * ├──────────────────────────┴──────────────────────┤
 * │  Console                                         │
 * └─────────────────────────────────────────────────┘
 * </pre>
 */
public class MainApp extends Application {

    // ── Colours / fonts (mutable for theme switching) ─────────────────────────
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

    // ── State ──────────────────────────────────────────────────────────────────
    private NeuralNet     net;
    private PatternParser patternSet;
    private volatile boolean training = false;
    private Thread        trainingThread;

    // ── UI references ──────────────────────────────────────────────────────────
    private NetworkCanvas    networkCanvas;
    private LineChart<Number, Number> errorChart;
    private XYChart.Series<Number, Number> errorSeries;
    private TextArea         console;
    private ComboBox<TrainingPattern> patternCombo;
    private Label            epochLabel;
    private Label            statusLabel;
    private Stage            primaryStage;

    // ── Architecture config (mutable before Apply) ────────────────────────────
    private int[]     layerSizes  = {2, 2, 1};
    private boolean[] biasEnabled = {true, false, false};

    // ── Entry points ──────────────────────────────────────────────────────────

    public static void main(String[] args) {
        launch(args);
    }

    @Override
    public void start(Stage stage) {
        this.primaryStage = stage;
        stage.setTitle("Heronix NeuroSim – Neural Network Simulator v2.0");
        stage.setMinWidth(900);
        stage.setMinHeight(650);
        try {
            java.io.InputStream iconStream = getClass().getResourceAsStream("/img/Icon.png");
            if (iconStream != null) {
                stage.getIcons().add(new javafx.scene.image.Image(iconStream));
            }
        } catch (Exception ignored) {} // icon is optional

        BorderPane root = new BorderPane();
        root.setStyle("-fx-background-color: " + BG_DARK + ";");
        VBox topSection = new VBox(buildMenuBar(), buildHeader());
        root.setTop(topSection);

        // Main content + console in a vertical split pane (draggable divider)
        SplitPane mainSplit = new SplitPane();
        mainSplit.setOrientation(javafx.geometry.Orientation.VERTICAL);
        mainSplit.getItems().addAll(buildCenter(), buildConsole());
        mainSplit.setDividerPositions(0.75);
        mainSplit.setStyle("-fx-background-color: " + BG_DARK + ";");
        root.setCenter(mainSplit);

        root.setBottom(buildFooter());

        Scene scene = new Scene(root, 1050, 750);
        scene.getStylesheets().add(getClass().getResource("theme-hacker.css").toExternalForm());
        stage.setScene(scene);
        stage.show();

        initNetwork(layerSizes, biasEnabled, 0.5);
        loadPatternText(PatternParser.PRESET_AND);
        log("Heronix NeuroSim – Neural Network Simulator v2.0");
        log("Default network: 2-2-1. Preset AND patterns loaded.");
        log("Choose Train tab → click TRAIN to begin.");
    }

    // ── Header ─────────────────────────────────────────────────────────────────

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

    // ── Menu bar ─────────────────────────────────────────────────────────────────

    private MenuBar buildMenuBar() {
        // ── File menu ──
        MenuItem miSavePatterns = new MenuItem("Save Patterns…");
        MenuItem miExportReport = new MenuItem("Export Report (ASCII)…");
        MenuItem miExportBinary = new MenuItem("Export Report (Binary)…");
        MenuItem miExit         = new MenuItem("Exit");

        miSavePatterns.setOnAction(e -> savePatterns());
        miExportReport.setOnAction(e -> exportReport(false));
        miExportBinary.setOnAction(e -> exportReport(true));
        miExit.setOnAction(e -> Platform.exit());

        Menu fileMenu = new Menu("File", null,
                miSavePatterns, new SeparatorMenuItem(),
                miExportReport, miExportBinary, new SeparatorMenuItem(),
                miExit);

        // ── Network menu ──
        MenuItem miSaveConfig = new MenuItem("Save Configuration…");
        MenuItem miLoadConfig = new MenuItem("Load Configuration…");
        MenuItem miSaveWeights = new MenuItem("Save Weights…");
        MenuItem miLoadWeights = new MenuItem("Load Weights…");

        miSaveConfig.setOnAction(e -> saveNetworkConfig());
        miLoadConfig.setOnAction(e -> loadNetworkConfig());
        miSaveWeights.setOnAction(e -> saveWeights());
        miLoadWeights.setOnAction(e -> loadWeights());

        Menu networkMenu = new Menu("Network", null,
                miSaveConfig, miLoadConfig, new SeparatorMenuItem(),
                miSaveWeights, miLoadWeights);

        // ── Weights menu ──
        MenuItem miShowWeights = new MenuItem("Show Weights");
        miShowWeights.setOnAction(e -> showWeightsDialog());

        Menu weightsMenu = new Menu("Weights", null, miShowWeights);

        // ── Examples menu ──
        MenuItem miAndEx  = new MenuItem("AND Gate  (2-2-1, easy)");
        MenuItem miOrEx   = new MenuItem("OR Gate  (2-2-1, easy)");
        MenuItem miXorEx  = new MenuItem("XOR Gate  (2-3-1, requires hidden layer)");
        MenuItem miXor221 = new MenuItem("XOR Gate  (2-2-1, may not converge)");
        MenuItem miAnd3in = new MenuItem("3-Input AND  (3-2-1)");
        MenuItem miParity = new MenuItem("Parity  (3-4-1, hard)");

        miAndEx.setOnAction(e  -> loadExample("AND Gate", PatternParser.PRESET_AND,
                new int[]{2,2,1}, new boolean[]{true,false,false}, 0.5, 0.9, 5000, false, false));
        miOrEx.setOnAction(e   -> loadExample("OR Gate", PatternParser.PRESET_OR,
                new int[]{2,2,1}, new boolean[]{true,false,false}, 0.5, 0.9, 5000, false, false));
        miXorEx.setOnAction(e  -> loadExample("XOR Gate (2-3-1)", PatternParser.PRESET_XOR,
                new int[]{2,3,1}, new boolean[]{true,true,false}, 0.5, 0.9, 10000, false, false));
        miXor221.setOnAction(e -> loadExample("XOR Gate (2-2-1)", PatternParser.PRESET_XOR,
                new int[]{2,2,1}, new boolean[]{true,false,false}, 0.5, 0.9, 10000, false, false));
        miAnd3in.setOnAction(e -> loadExample("3-Input AND", PRESET_AND3,
                new int[]{3,2,1}, new boolean[]{true,false,false}, 0.5, 0.9, 5000, false, false));
        miParity.setOnAction(e -> loadExample("Parity (3-4-1)", PRESET_PARITY3,
                new int[]{3,4,1}, new boolean[]{true,true,false}, 0.3, 0.9, 20000, false, false));

        // ── More logic gate presets ──
        MenuItem miNand  = new MenuItem("NAND Gate  (2-2-1)");
        MenuItem miNor   = new MenuItem("NOR Gate  (2-2-1)");
        MenuItem miHalfAdd = new MenuItem("Half Adder  (2-4-2, sum + carry)");
        MenuItem miEncoder = new MenuItem("4-to-2 Encoder  (4-4-2)");
        MenuItem miIdentity = new MenuItem("Identity  (3-3, no hidden layer)");

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

        // ── View menu (themes) ──
        ToggleGroup themeGroup = new ToggleGroup();
        RadioMenuItem miHacker = new RadioMenuItem("Hacker Green");
        RadioMenuItem miDark   = new RadioMenuItem("Dark");
        RadioMenuItem miSystem = new RadioMenuItem("System (Light)");
        miHacker.setToggleGroup(themeGroup);
        miDark.setToggleGroup(themeGroup);
        miSystem.setToggleGroup(themeGroup);
        miHacker.setSelected(true);

        miHacker.setOnAction(e -> applyTheme("hacker"));
        miDark.setOnAction(e   -> applyTheme("dark"));
        miSystem.setOnAction(e -> applyTheme("system"));

        Menu viewMenu = new Menu("View", null, miHacker, miDark, miSystem);

        // ── Help menu ──
        MenuItem miAbout = new MenuItem("About Heronix NeuroSim");
        miAbout.setOnAction(e -> showAboutDialog());
        MenuItem miHelpGuide = new MenuItem("Quick Start Guide");
        miHelpGuide.setOnAction(e -> showHelpDialog());

        Menu helpMenu = new Menu("Help", null, miHelpGuide, new SeparatorMenuItem(), miAbout);

        MenuBar mb = new MenuBar(fileMenu, networkMenu, weightsMenu, examplesMenu, viewMenu, helpMenu);
        mb.setStyle("-fx-background-color: " + BG_PANEL + ";" +
                " -fx-border-color: " + BORDER_C + "; -fx-border-width: 0 0 1 0;");
        return mb;
    }

    // ── Theme switching ───────────────────────────────────────────────────────────

    private void applyTheme(String theme) {
        currentTheme = theme;
        switch (theme) {
            case "dark":
                BG_DARK   = "#1e1e1e"; BG_PANEL  = "#252526"; BORDER_C  = "#333";
                ACCENT    = "#4fc3f7"; DIM_ACCENT= "#5a6a7a"; TEXT_COL  = "#cccccc";
                CANVAS_BG = "#1a1a1a"; CONSOLE_BG= "#1a1a1a";
                break;
            case "system":
                BG_DARK   = "#f0f0f0"; BG_PANEL  = "#ffffff"; BORDER_C  = "#ccc";
                ACCENT    = "#0056a3"; DIM_ACCENT= "#333333"; TEXT_COL  = "#111111";
                CANVAS_BG = "#e8e8e8"; CONSOLE_BG= "#f0f0f0";
                break;
            default: // hacker
                BG_DARK   = "#080d0b"; BG_PANEL  = "#0c1410"; BORDER_C  = "#132a1a";
                ACCENT    = "#00dc78"; DIM_ACCENT= "#1a5a3a"; TEXT_COL  = "#b8f0cc";
                CANVAS_BG = "#0a110d"; CONSOLE_BG= "#04080a";
                break;
        }

        // Swap CSS
        Scene scene = primaryStage.getScene();
        scene.getStylesheets().clear();
        scene.getStylesheets().add(getClass().getResource("theme-" + theme + ".css").toExternalForm());

        // Rebuild the entire UI to apply inline styles with new colors
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

        // Re-select current theme radio button
        // Restore network visualization
        if (net != null) networkCanvas.render(net);

        // Restore pattern combo
        if (patternSet != null) {
            javafx.collections.ObservableList<TrainingPattern> items =
                    FXCollections.observableArrayList(patternSet.patterns);
            patternCombo.setItems(items);
            if (!items.isEmpty()) patternCombo.setValue(items.get(0));
        }

        log("Theme changed to: " + theme);
    }

    // ── Example presets ─────────────────────────────────────────────────────────

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

        // Set training parameters
        spLr.getValueFactory().setValue(lr);
        spMomentum.getValueFactory().setValue(momentum);
        spEpochs.getValueFactory().setValue(epochs);
        cbBatch.setSelected(batch);
        cbCrossEntropy.setSelected(crossEntropy);

        // Set architecture
        layerSizes  = sizes.clone();
        biasEnabled = bias.clone();
        rebuildLayerRows();

        // Load patterns
        patternEditor.setText(patternText);
        loadPatternText(patternText);

        // Initialize network
        initNetwork(layerSizes, biasEnabled, spWtRange.getValue());

        log("── Example loaded: " + name + " ──");
        log("  Network: [" + intArrStr(sizes) + "]  lr=" + lr + "  momentum=" + momentum + "  epochs=" + epochs);
        log("  Click TRAIN to begin.");
    }

    // ── File operations ─────────────────────────────────────────────────────────

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
        } catch (IOException ex) {
            log("ERROR saving patterns: " + ex.getMessage());
        }
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
            sb.append("[Type]\nType = FeedForward\n[Network]\n");
            sb.append("Layers = ").append(net.getNumLayers()).append('\n');
            for (int l = 0; l < net.getNumLayers(); l++) {
                String name = l == 0 ? "Input" : l == net.getNumLayers() - 1 ? "Output" : "Hidden" + l;
                sb.append(name).append(" units = ").append(net.getLayerSize(l)).append('\n');
                if (l < net.getNumLayers() - 1) {
                    sb.append(name).append(" bias = ").append(net.getBiasEnabled(l)).append('\n');
                }
            }
            Files.writeString(f.toPath(), sb.toString());
            log("Network config saved to " + f.getName());
        } catch (IOException ex) {
            log("ERROR saving config: " + ex.getMessage());
        }
    }

    private void loadNetworkConfig() {
        FileChooser fc = new FileChooser();
        fc.setTitle("Load Network Configuration");
        fc.getExtensionFilters().add(new FileChooser.ExtensionFilter("Config files", "*.cfg"));
        File f = fc.showOpenDialog(primaryStage);
        if (f == null) return;
        try {
            String text = Files.readString(f.toPath());
            int layers = 0;
            java.util.List<Integer> sizeList = new java.util.ArrayList<>();
            java.util.List<Boolean> biasList = new java.util.ArrayList<>();
            for (String raw : text.split("\\r?\\n")) {
                String line = raw.trim();
                String lower = line.toLowerCase();
                if (lower.startsWith("layers")) {
                    layers = Integer.parseInt(line.split("=")[1].trim());
                } else if (lower.contains("units")) {
                    sizeList.add(Integer.parseInt(line.split("=")[1].trim()));
                } else if (lower.contains("bias")) {
                    biasList.add(Boolean.parseBoolean(line.split("=")[1].trim()));
                }
            }
            if (sizeList.size() < 2) { log("ERROR: Invalid config file."); return; }
            layerSizes = sizeList.stream().mapToInt(Integer::intValue).toArray();
            biasEnabled = new boolean[layerSizes.length];
            for (int i = 0; i < biasList.size() && i < biasEnabled.length; i++) {
                biasEnabled[i] = biasList.get(i);
            }
            rebuildLayerRows();
            initNetwork(layerSizes, biasEnabled, spWtRange.getValue());
            log("Config loaded from " + f.getName() + ": [" + intArrStr(layerSizes) + "]");
        } catch (Exception ex) {
            log("ERROR loading config: " + ex.getMessage());
        }
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
            for (int l = 0; l < net.getNumLayers() - 1; l++) {
                sb.append("[Layer ").append(l).append(" -> ").append(l + 1).append("]\n");
                int from = net.getFromCount(l);
                for (int j = 0; j < net.getLayerSize(l + 1); j++) {
                    for (int i = 0; i < from; i++) {
                        if (i > 0) sb.append('\t');
                        sb.append(String.format("%.10f", net.getWeight(l, j, i)));
                    }
                    sb.append('\n');
                }
            }
            Files.writeString(f.toPath(), sb.toString());
            log("Weights saved to " + f.getName());
        } catch (IOException ex) {
            log("ERROR saving weights: " + ex.getMessage());
        }
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
            int layer = -1;
            int row = 0;
            for (String raw : text.split("\\r?\\n")) {
                String line = raw.trim();
                if (line.startsWith("#") || line.isEmpty()) continue;
                if (line.startsWith("[")) {
                    layer++;
                    row = 0;
                    continue;
                }
                if (layer < 0 || layer >= net.getNumLayers() - 1) continue;
                if (row >= net.getLayerSize(layer + 1)) continue;  // skip extra rows
                String[] parts = line.split("\\s+");
                int from = net.getFromCount(layer);
                for (int i = 0; i < Math.min(parts.length, from); i++) {
                    net.setWeight(layer, row, i, Double.parseDouble(parts[i]));
                }
                row++;
            }
            networkCanvas.render(net);
            log("Weights loaded from " + f.getName());
        } catch (Exception ex) {
            log("ERROR loading weights: " + ex.getMessage());
        }
    }

    private void showWeightsDialog() {
        if (net == null) { log("No network."); return; }
        StringBuilder sb = new StringBuilder();
        for (int l = 0; l < net.getNumLayers() - 1; l++) {
            sb.append("── Layer ").append(l).append(" → ").append(l + 1).append(" ──\n");
            int from = net.getFromCount(l);
            // Header
            sb.append(String.format("%-6s", ""));
            for (int i = 0; i < net.getLayerSize(l); i++) sb.append(String.format("  u%-5d", i));
            if (net.getBiasEnabled(l)) sb.append("  bias  ");
            sb.append('\n');
            for (int j = 0; j < net.getLayerSize(l + 1); j++) {
                sb.append(String.format("u%-5d", j));
                for (int i = 0; i < from; i++) {
                    sb.append(String.format(" %+7.4f", net.getWeight(l, j, i)));
                }
                sb.append('\n');
            }
            sb.append('\n');
        }
        log("── WEIGHT TABLE ──\n" + sb);
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

            // Architecture
            ascii.append("[Architecture]\n");
            ascii.append("Layers: ").append(net.getNumLayers()).append('\n');
            ascii.append("Topology: ").append(intArrStr(net.getLayerSizes())).append('\n');
            for (int l = 0; l < net.getNumLayers(); l++) {
                String name = l == 0 ? "Input" : l == net.getNumLayers() - 1 ? "Output" : "Hidden" + l;
                ascii.append(name).append(": ").append(net.getLayerSize(l)).append(" units");
                if (l < net.getNumLayers() - 1 && net.getBiasEnabled(l)) ascii.append(" + bias");
                ascii.append('\n');
            }

            // Training params
            ascii.append("\n[Training Parameters]\n");
            ascii.append("Learning Rate: ").append(spLr.getValue()).append('\n');
            ascii.append("Momentum: ").append(spMomentum.getValue()).append('\n');
            ascii.append("Max Epochs: ").append(spEpochs.getValue()).append('\n');
            ascii.append("Mode: ").append(cbBatch.isSelected() ? "Batch" : "Online").append('\n');
            ascii.append("Error Function: ").append(cbCrossEntropy.isSelected() ? "Cross-Entropy" : "MSE").append('\n');

            // Status
            ascii.append("\n[Status]\n");
            ascii.append(epochLabel.getText()).append('\n');

            // Weights
            ascii.append("\n[Weights]\n");
            for (int l = 0; l < net.getNumLayers() - 1; l++) {
                ascii.append("Layer ").append(l).append(" -> ").append(l + 1).append(":\n");
                int from = net.getFromCount(l);
                for (int j = 0; j < net.getLayerSize(l + 1); j++) {
                    for (int i = 0; i < from; i++) {
                        if (i > 0) ascii.append('\t');
                        ascii.append(String.format("%.10f", net.getWeight(l, j, i)));
                    }
                    ascii.append('\n');
                }
            }

            // Test results
            if (patternSet != null && !patternSet.patterns.isEmpty()) {
                ascii.append("\n[Test Results]\n");
                boolean xent = cbCrossEntropy.isSelected();
                double totalErr = 0;
                for (TrainingPattern p : patternSet.patterns) {
                    double[] out = net.forward(p.inputs);
                    double err = net.computeError(p.targets, xent);
                    totalErr += err;
                    ascii.append("Input: [");
                    for (int i = 0; i < p.inputs.length; i++) {
                        if (i > 0) ascii.append(", ");
                        ascii.append(p.inputs[i]);
                    }
                    ascii.append("]  Output: [");
                    for (int i = 0; i < out.length; i++) {
                        if (i > 0) ascii.append(", ");
                        ascii.append(String.format("%.6f", out[i]));
                    }
                    ascii.append("]  Target: [");
                    for (int i = 0; i < p.targets.length; i++) {
                        if (i > 0) ascii.append(", ");
                        ascii.append(p.targets[i]);
                    }
                    ascii.append("]  Error: ").append(String.format("%.6f", err)).append('\n');
                }
                ascii.append("Average Error: ").append(String.format("%.6f", totalErr / patternSet.patterns.size())).append('\n');
            }

            if (binary) {
                // Convert ASCII text to binary representation
                String asciiText = ascii.toString();
                StringBuilder binOut = new StringBuilder();
                binOut.append("=== Heronix NeuroSim Binary Report ===\n");
                binOut.append("=== ASCII values converted to 8-bit binary ===\n\n");
                for (int i = 0; i < asciiText.length(); i++) {
                    char c = asciiText.charAt(i);
                    String bits = String.format("%8s", Integer.toBinaryString(c)).replace(' ', '0');
                    binOut.append(bits);
                    if (c == '\n') {
                        binOut.append('\n');
                    } else {
                        binOut.append(' ');
                    }
                }
                Files.writeString(f.toPath(), binOut.toString());
                log("Binary report exported to " + f.getName());
            } else {
                Files.writeString(f.toPath(), ascii.toString());
                log("ASCII report exported to " + f.getName());
            }
        } catch (IOException ex) {
            log("ERROR exporting report: " + ex.getMessage());
        }
    }

    // ── Centre ─────────────────────────────────────────────────────────────────

    private SplitPane buildCenter() {
        VBox left  = buildLeftColumn();
        VBox right = buildRightPanel();

        SplitPane sp = new SplitPane(left, right);
        sp.setDividerPositions(0.62);
        sp.setStyle("-fx-background-color: " + BG_DARK + ";");
        return sp;
    }

    private VBox buildLeftColumn() {
        // Network canvas
        networkCanvas = new NetworkCanvas(560, 270);
        networkCanvas.setThemeColors(CANVAS_BG, DIM_ACCENT);
        networkCanvas.clear();

        TitledPane vizPane = titledPane("◈  NETWORK VISUALIZATION", networkCanvas);

        // Error chart
        NumberAxis xAxis = new NumberAxis();
        NumberAxis yAxis = new NumberAxis();
        styleAxis(xAxis, "Epoch");
        styleAxis(yAxis, "Avg Error");
        errorChart  = new LineChart<>(xAxis, yAxis);
        errorSeries = new XYChart.Series<>();
        errorSeries.setName("Error");
        errorChart.getData().add(errorSeries);
        errorChart.setAnimated(false);
        errorChart.setCreateSymbols(false);
        errorChart.setLegendVisible(false);
        errorChart.setMinHeight(120);
        errorChart.setPrefHeight(250);
        errorChart.setStyle("-fx-background-color: " + BG_PANEL + ";");

        TitledPane chartPane = titledPane("◈  TRAINING ERROR (per-pattern avg)", errorChart);

        VBox col = new VBox(8, vizPane, chartPane);
        col.setPadding(new Insets(10, 6, 6, 10));
        col.setStyle("-fx-background-color: " + BG_DARK + ";");
        VBox.setVgrow(vizPane, Priority.SOMETIMES);
        VBox.setVgrow(chartPane, Priority.ALWAYS);
        return col;
    }

    // ── Right panel ────────────────────────────────────────────────────────────

    private VBox buildRightPanel() {
        TabPane tabs = new TabPane(
                buildTrainTab(),
                buildConfigureTab(),
                buildPatternsTab()
        );
        tabs.setTabClosingPolicy(TabPane.TabClosingPolicy.UNAVAILABLE);
        tabs.setStyle("-fx-background-color: " + BG_PANEL + ";" +
                " -fx-tab-min-width: 70; -fx-tab-max-height: 30;" +
                " -fx-font-family: '" + MONO + "'; -fx-font-size: 12;");
        tabs.getStyleClass().add("neurosim-tabs");

        VBox panel = new VBox(tabs);
        panel.setPadding(new Insets(10, 10, 6, 6));
        panel.setStyle("-fx-background-color: " + BG_DARK + ";");
        VBox.setVgrow(tabs, Priority.ALWAYS);
        return panel;
    }

    // ── Train tab ─────────────────────────────────────────────────────────────

    // Spinners referenced between methods
    private Spinner<Double>  spLr, spMomentum, spWtRange;
    private Spinner<Integer> spEpochs;
    private CheckBox         cbBatch, cbCrossEntropy;
    private Button           btnTrain, btnReset;

    private Tab buildTrainTab() {
        spLr         = doubleSpinner(0.001, 10.0, 0.5, 0.05);
        spMomentum   = doubleSpinner(0.0,    1.0, 0.9, 0.05);
        spWtRange    = doubleSpinner(0.01,   5.0, 0.5, 0.05);
        spEpochs     = intSpinner(1, 100_000, 5000, 100);

        cbBatch       = checkbox("Batch Update");
        cbCrossEntropy = checkbox("Cross-Entropy Error");

        btnTrain = btn("▶  TRAIN", ACCENT, BG_DARK, true);
        btnReset = btn("↺  RESET WEIGHTS", "#ff5555", BG_DARK, true);

        btnTrain.setOnAction(e -> { if (training) stopTraining(); else startTraining(); });
        btnReset.setOnAction(e -> resetWeights());

        // Test section
        patternCombo = new ComboBox<>();
        patternCombo.setMaxWidth(Double.MAX_VALUE);
        styleCombo(patternCombo);

        Button btnTestOne = btn("TEST ONE", ACCENT, BG_DARK, false);
        Button btnTestAll = btn("TEST ALL", ACCENT, BG_DARK, false);
        btnTestOne.setOnAction(e -> testOne());
        btnTestAll.setOnAction(e -> testAll());

        GridPane params = new GridPane();
        params.setHgap(8); params.setVgap(6);
        params.setPadding(new Insets(6));
        addRow(params, 0, "Learning Rate",    spLr);
        addRow(params, 1, "Momentum",         spMomentum);
        addRow(params, 2, "Max Epochs",       spEpochs);
        addRow(params, 3, "Weight Range",     spWtRange);

        HBox trainBtns = new HBox(6, btnTrain, btnReset);
        HBox.setHgrow(btnTrain, Priority.ALWAYS);
        HBox.setHgrow(btnReset, Priority.ALWAYS);
        btnTrain.setMaxWidth(Double.MAX_VALUE);
        btnReset.setMaxWidth(Double.MAX_VALUE);

        HBox testBtns = new HBox(6, btnTestOne, btnTestAll);
        HBox.setHgrow(btnTestOne, Priority.ALWAYS);
        HBox.setHgrow(btnTestAll, Priority.ALWAYS);
        btnTestOne.setMaxWidth(Double.MAX_VALUE);
        btnTestAll.setMaxWidth(Double.MAX_VALUE);

        VBox content = new VBox(10,
                sectionLabel("PARAMETERS"),
                params,
                cbBatch, cbCrossEntropy,
                trainBtns,
                sep(),
                sectionLabel("TEST"),
                label("Pattern:", 11, DIM_ACCENT),
                patternCombo,
                testBtns
        );
        content.setPadding(new Insets(10));
        content.setStyle("-fx-background-color: " + BG_PANEL + ";");

        return styledTab("TRAIN", new ScrollPane(content) {{
            setFitToWidth(true);
            setStyle("-fx-background-color: " + BG_PANEL + ";");
        }});
    }

    // ── Configure tab ─────────────────────────────────────────────────────────

    private VBox layerRows;   // dynamically rebuilt

    private Tab buildConfigureTab() {
        layerRows = new VBox(6);
        rebuildLayerRows();

        Button btnAddHidden = btn("+ ADD HIDDEN LAYER", ACCENT, BG_DARK, false);
        Button btnRemHidden = btn("− REMOVE HIDDEN", "#ff5555", BG_DARK, false);
        Button btnApply     = btn("APPLY & REINITIALIZE", ACCENT, BG_DARK, true);

        btnAddHidden.setMaxWidth(Double.MAX_VALUE);
        btnRemHidden.setMaxWidth(Double.MAX_VALUE);
        btnApply.setMaxWidth(Double.MAX_VALUE);

        btnAddHidden.setOnAction(e -> {
            int NL = layerSizes.length;
            if (NL >= 6) { log("Maximum 4 hidden layers supported."); return; }
            int[] ns  = new int[NL + 1];
            boolean[] nb = new boolean[NL + 1];
            System.arraycopy(layerSizes,  0, ns,  0, NL - 1);
            System.arraycopy(biasEnabled, 0, nb, 0, NL - 1);
            ns[NL - 1]  = 3;
            nb[NL - 1]  = true;
            ns[NL]       = layerSizes[NL - 1];
            nb[NL]       = false;
            layerSizes   = ns;
            biasEnabled  = nb;
            rebuildLayerRows();
        });

        btnRemHidden.setOnAction(e -> {
            int NL = layerSizes.length;
            if (NL <= 2) { log("Need at least input + output layers."); return; }
            int[] ns  = new int[NL - 1];
            boolean[] nb = new boolean[NL - 1];
            System.arraycopy(layerSizes,  0, ns,  0, NL - 2);
            System.arraycopy(biasEnabled, 0, nb, 0, NL - 2);
            ns[NL - 2]  = layerSizes[NL - 1];
            nb[NL - 2]  = false;
            layerSizes   = ns;
            biasEnabled  = nb;
            rebuildLayerRows();
        });

        btnApply.setOnAction(e -> {
            collectLayerConfig();
            initNetwork(layerSizes, biasEnabled, spWtRange.getValue());
            log("Network re-initialized: [" + intArrStr(layerSizes) + "]");
        });

        VBox content = new VBox(10,
                sectionLabel("LAYER CONFIGURATION"),
                label("Units per layer + bias flag", 11, DIM_ACCENT),
                layerRows,
                sep(),
                btnAddHidden, btnRemHidden,
                sep(),
                btnApply
        );
        content.setPadding(new Insets(10));
        content.setStyle("-fx-background-color: " + BG_PANEL + ";");

        return styledTab("CONFIGURE", new ScrollPane(content) {{
            setFitToWidth(true);
            setStyle("-fx-background-color: " + BG_PANEL + ";");
        }});
    }

    // ── Patterns tab ──────────────────────────────────────────────────────────

    private TextArea patternEditor;

    private Tab buildPatternsTab() {
        patternEditor = new TextArea(PatternParser.PRESET_AND);
        patternEditor.setFont(Font.font(MONO, 12));
        patternEditor.setStyle("-fx-control-inner-background: " + CONSOLE_BG + "; -fx-text-fill: " + TEXT_COL +
                "; -fx-border-color: " + BORDER_C + ";");
        patternEditor.setPrefRowCount(14);

        Button btnAnd  = btn("AND",  ACCENT, BG_DARK, false);
        Button btnOr   = btn("OR",   ACCENT, BG_DARK, false);
        Button btnXor  = btn("XOR",  ACCENT, BG_DARK, false);
        Button btnLoad = btn("LOAD FILE…", ACCENT, BG_DARK, false);
        Button btnApply = btn("APPLY PATTERNS", ACCENT, BG_DARK, true);

        btnAnd.setOnAction(e  -> patternEditor.setText(PatternParser.PRESET_AND));
        btnOr.setOnAction(e   -> patternEditor.setText(PatternParser.PRESET_OR));
        btnXor.setOnAction(e  -> patternEditor.setText(PatternParser.PRESET_XOR));
        btnLoad.setOnAction(e -> loadPatternFile());
        btnApply.setOnAction(e -> loadPatternText(patternEditor.getText()));
        btnApply.setMaxWidth(Double.MAX_VALUE);

        HBox presets = new HBox(6, btnAnd, btnOr, btnXor, btnLoad);

        VBox content = new VBox(10,
                sectionLabel("PATTERN FILE"),
                presets,
                patternEditor,
                btnApply
        );
        content.setPadding(new Insets(10));
        content.setStyle("-fx-background-color: " + BG_PANEL + ";");

        return styledTab("PATTERNS", new ScrollPane(content) {{
            setFitToWidth(true);
            setStyle("-fx-background-color: " + BG_PANEL + ";");
        }});
    }

    // ── Console ────────────────────────────────────────────────────────────────

    private VBox buildConsole() {
        console = new TextArea();
        console.setEditable(false);
        console.setFont(Font.font(MONO, 12));
        console.setPrefRowCount(8);
        console.setStyle("-fx-control-inner-background: " + CONSOLE_BG + "; -fx-text-fill: " + TEXT_COL +
                "; -fx-border-color: " + BORDER_C + ";");

        Label consoleLabel = label("◈  CONSOLE", 11, ACCENT);
        consoleLabel.setPadding(new Insets(4, 8, 4, 8));
        consoleLabel.setMaxWidth(Double.MAX_VALUE);
        consoleLabel.setStyle("-fx-background-color: " + BG_PANEL + "; -fx-border-color: " + BORDER_C +
                "; -fx-border-width: 0 0 1 0;");

        VBox box = new VBox(consoleLabel, console);
        box.setStyle("-fx-background-color: " + BG_PANEL + ";");
        VBox.setVgrow(console, Priority.ALWAYS);
        return box;
    }

    // ── Footer ──────────────────────────────────────────────────────────────────

    private HBox buildFooter() {
        Label left = label("Heronix NeuroSim v2.0  |  Heronix Education Systems LLC  |  By Michael Katsaros  |  Free Education Tools", 10, DIM_ACCENT);
        Label right = label("\u00A9 2026", 10, DIM_ACCENT);
        HBox footer = new HBox(left, right);
        HBox.setHgrow(left, Priority.ALWAYS);
        footer.setAlignment(Pos.CENTER_LEFT);
        footer.setPadding(new Insets(3, 10, 3, 10));
        footer.setStyle("-fx-background-color: " + BG_PANEL + "; -fx-border-color: " + BORDER_C +
                "; -fx-border-width: 1 0 0 0;");
        return footer;
    }

    // ── Help / About dialogs ──────────────────────────────────────────────────

    private void showAboutDialog() {
        Alert alert = new Alert(Alert.AlertType.INFORMATION);
        alert.setTitle("About Heronix NeuroSim");
        alert.setHeaderText("Heronix NeuroSim v2.0");
        alert.setContentText(
                "Neural Network Simulator\n\n" +
                "Developed by Michael Katsaros\n" +
                "Heronix Education Systems LLC\n\n" +
                "A free and open-source educational tool for\n" +
                "exploring backpropagation neural networks.\n\n" +
                "License: MIT — free to use, modify, and distribute.\n\n" +
                "\u00A9 2026 Heronix Education Systems LLC");
        alert.showAndWait();
    }

    private void showHelpDialog() {
        Alert alert = new Alert(Alert.AlertType.INFORMATION);
        alert.setTitle("Quick Start Guide");
        alert.setHeaderText("How to Use Heronix NeuroSim");
        alert.setContentText(
                "1. LOAD PATTERNS\n" +
                "   Use Examples menu for presets, or File > Open Patterns\n" +
                "   to load a .pat file.\n\n" +
                "2. CONFIGURE NETWORK\n" +
                "   Go to Configure tab to set layers, units, and bias.\n" +
                "   Click Apply & Reinitialize when ready.\n\n" +
                "3. SET PARAMETERS\n" +
                "   In the Train tab, adjust learning rate, momentum,\n" +
                "   and max epochs.\n\n" +
                "4. TRAIN\n" +
                "   Click TRAIN. Watch the error chart converge.\n" +
                "   Training stops at error < 0.001 or max epochs.\n\n" +
                "5. TEST\n" +
                "   Select a pattern and click TEST ONE, or TEST ALL\n" +
                "   for the average error across all patterns.\n\n" +
                "6. SAVE\n" +
                "   Network > Save Weights to save trained weights.\n" +
                "   File > Export Report for a full training report.");
        alert.getDialogPane().setMinWidth(450);
        alert.showAndWait();
    }

    // ── Neural network actions ────────────────────────────────────────────────

    private void initNetwork(int[] sizes, boolean[] bias, double range) {
        if (training) stopTraining();
        net = new NeuralNet(sizes, bias);
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
        log("Weights randomised in [±" + spWtRange.getValue() + "].");
    }

    private void startTraining() {
        if (patternSet == null || patternSet.patterns.isEmpty()) {
            log("ERROR: No patterns loaded. Load patterns first."); return;
        }
        if (net == null) { log("ERROR: No network."); return; }

        // Validate architecture matches patterns
        if (net.getLayerSize(0) != patternSet.numInputs) {
            log("ERROR: Network has " + net.getLayerSize(0) +
                    " input units but patterns have " + patternSet.numInputs + " inputs.");
            return;
        }
        if (net.getLayerSize(net.getNumLayers() - 1) != patternSet.numOutputs) {
            log("ERROR: Network has " + net.getLayerSize(net.getNumLayers() - 1) +
                    " output units but patterns have " + patternSet.numOutputs + " outputs.");
            return;
        }

        training = true;
        Platform.runLater(() -> {
            btnTrain.setText("⏹  STOP");
            btnTrain.setStyle(btnStyle("#ff5555", BG_DARK, true));
            statusLabel.setText("TRAINING…");
        });

        final double  lr       = spLr.getValue();
        final double  mom      = spMomentum.getValue();
        final int     maxEp    = spEpochs.getValue();
        final boolean batch    = cbBatch.isSelected();
        final boolean xent     = cbCrossEntropy.isSelected();
        final List<TrainingPattern> pats = patternSet.patterns;

        log(String.format("▶ Training: lr=%.4f  momentum=%.3f  max=%d  %s  %s",
                lr, mom, maxEp, batch ? "batch" : "online", xent ? "cross-entropy" : "MSE"));

        Task<Void> task = new Task<>() {
            @Override
            protected Void call() {
                int plotEvery = Math.max(1, maxEp / 400);
                for (int ep = 1; ep <= maxEp && training; ep++) {
                    double err = net.trainEpoch(pats, batch, lr, mom, xent);
                    final int epoch = ep;
                    final double error = err;
                    if (ep % plotEvery == 0 || ep == maxEp) {
                        NeuralNet snapshot = new NeuralNet(net);
                        Platform.runLater(() -> {
                            networkCanvas.render(snapshot);
                            if (errorSeries.getData().size() > 500) {
                                errorSeries.getData().remove(0);
                            }
                            errorSeries.getData().add(
                                    new XYChart.Data<>(epoch, error));
                            epochLabel.setText("EPOCH " + epoch + " / " + maxEp);
                        });
                    }
                    if (err < 0.001) {
                        final double ferr = err;
                        Platform.runLater(() -> {
                            log(String.format("✓ Converged at epoch %d.  Avg error: %.6f", epoch, ferr));
                            finishTraining();
                        });
                        return null;
                    }
                }
                Platform.runLater(() -> {
                    double finalErr = 0;
                    var data = errorSeries.getData();
                    if (!data.isEmpty()) {
                        finalErr = ((Number) data.get(data.size() - 1).getYValue()).doubleValue();
                    }
                    log(String.format("■ Reached max epochs (%d).  Final avg error: %.6f",
                            maxEp, finalErr));
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
        log("■ Training stopped by user.");
    }

    private void finishTraining() {
        if (!training && btnTrain.getText().contains("TRAIN")) return; // already finished
        training = false;
        btnTrain.setText("▶  TRAIN");
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
        for (int i = 0; i < p.inputs.length; i++) { if (i > 0) sb.append(", "); sb.append(p.inputs[i]); }
        sb.append("]  →  [");
        for (int i = 0; i < out.length; i++) { if (i > 0) sb.append(", "); sb.append(String.format("%.4f", out[i])); }
        sb.append("]   target: [");
        for (int i = 0; i < p.targets.length; i++) { if (i > 0) sb.append(", "); sb.append(p.targets[i]); }
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

    // ── Pattern loading ───────────────────────────────────────────────────────

    private void loadPatternText(String text) {
        try {
            patternSet = PatternParser.fromString(text);
            ObservableList<TrainingPattern> items = FXCollections.observableArrayList(patternSet.patterns);
            Platform.runLater(() -> {
                patternCombo.setItems(items);
                if (!items.isEmpty()) patternCombo.setValue(items.get(0));
            });
            log("✓ Loaded " + patternSet.patterns.size() + " patterns  (" +
                    patternSet.numInputs + " inputs → " + patternSet.numOutputs + " outputs).");

            // Auto-configure network if input/output sizes don't match
            int ni = patternSet.numInputs;
            int no = patternSet.numOutputs;
            if (layerSizes[0] != ni || layerSizes[layerSizes.length - 1] != no) {
                // Keep hidden layers but adjust input/output to match pattern file
                int[] newSizes = new int[layerSizes.length];
                boolean[] newBias = new boolean[layerSizes.length];
                newSizes[0] = ni;
                newSizes[newSizes.length - 1] = no;
                // Scale hidden layers: use max of old size or a reasonable default
                for (int i = 1; i < newSizes.length - 1; i++) {
                    newSizes[i] = Math.max(layerSizes[i], (ni + no) / 2);
                    newBias[i] = i < biasEnabled.length && biasEnabled[i];
                }
                newBias[0] = biasEnabled.length > 0 && biasEnabled[0];
                layerSizes  = newSizes;
                biasEnabled = newBias;
                initNetwork(layerSizes, biasEnabled, spWtRange.getValue());
                rebuildLayerRows();
                log("  Network auto-configured: [" + intArrStr(layerSizes) + "] to match pattern dimensions.");
            }
        } catch (IOException ex) {
            log("ERROR parsing patterns: " + ex.getMessage());
        }
    }

    private void loadPatternFile() {
        FileChooser fc = new FileChooser();
        fc.setTitle("Open Pattern File");
        fc.getExtensionFilters().addAll(
                new FileChooser.ExtensionFilter("Pattern files", "*.pat"),
                new FileChooser.ExtensionFilter("All files", "*.*")
        );
        File f = fc.showOpenDialog(primaryStage);
        if (f == null) return;
        try {
            String text = java.nio.file.Files.readString(f.toPath());
            patternEditor.setText(text);
            loadPatternText(text);
        } catch (IOException ex) {
            log("ERROR reading file: " + ex.getMessage());
        }
    }

    // ── Configure helpers ─────────────────────────────────────────────────────

    /** Spinner<Integer> and Spinner<Double> references keyed by layer index */
    private Spinner<Integer>[] unitSpinners;
    private CheckBox[]         biasChecks;

    @SuppressWarnings("unchecked")
    private void rebuildLayerRows() {
        int NL = layerSizes.length;
        unitSpinners = new Spinner[NL];
        biasChecks   = new CheckBox[NL];
        layerRows.getChildren().clear();

        String[] lbls = new String[NL];
        lbls[0] = "INPUT"; lbls[NL - 1] = "OUTPUT";
        for (int l = 1; l < NL - 1; l++) lbls[l] = "HIDDEN " + l;

        for (int l = 0; l < NL; l++) {
            unitSpinners[l] = intSpinner(1, 50, layerSizes[l], 1);
            biasChecks[l]   = checkbox("Bias");
            biasChecks[l].setSelected(l < biasEnabled.length && biasEnabled[l]);
            boolean isOut = (l == NL - 1);
            biasChecks[l].setDisable(isOut);

            Label layerLabel = label(lbls[l], 12, DIM_ACCENT);
            layerLabel.setMinWidth(65);
            HBox row = new HBox(8,
                    layerLabel,
                    unitSpinners[l],
                    biasChecks[l]
            );
            row.setAlignment(Pos.CENTER_LEFT);
            layerRows.getChildren().add(row);
        }
    }

    private void collectLayerConfig() {
        int NL = unitSpinners.length;
        layerSizes  = new int[NL];
        biasEnabled = new boolean[NL];
        for (int l = 0; l < NL; l++) {
            layerSizes[l]  = unitSpinners[l].getValue();
            biasEnabled[l] = biasChecks[l].isSelected();
        }
    }

    // ── Console ────────────────────────────────────────────────────────────────

    private void log(String msg) {
        Platform.runLater(() -> {
            console.appendText(msg + "\n");
            console.setScrollTop(Double.MAX_VALUE);
        });
    }

    // ── UI factory helpers ────────────────────────────────────────────────────

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
        String border = color;
        String bg2    = prominent ? color : bg;
        String fg     = prominent ? bg    : color;
        return "-fx-background-color: " + bg2 + "; -fx-text-fill: " + fg +
                "; -fx-border-color: " + border + "; -fx-border-width: 1;" +
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
        sp.setEditable(true);
        sp.setPrefWidth(120);
        sp.setStyle("-fx-background-color: " + CONSOLE_BG + "; -fx-text-fill: " + TEXT_COL +
                "; -fx-border-color: " + BORDER_C + ";");
        commitOnFocusLost(sp);
        return sp;
    }

    private Spinner<Integer> intSpinner(int min, int max, int init, int step) {
        Spinner<Integer> sp = new Spinner<>(new SpinnerValueFactory.IntegerSpinnerValueFactory(min, max, init, step));
        sp.setEditable(true);
        sp.setPrefWidth(100);
        sp.setStyle("-fx-background-color: " + CONSOLE_BG + "; -fx-text-fill: " + TEXT_COL +
                "; -fx-border-color: " + BORDER_C + ";");
        commitOnFocusLost(sp);
        return sp;
    }

    private <T> void commitOnFocusLost(Spinner<T> sp) {
        sp.focusedProperty().addListener((obs, wasFocused, isFocused) -> {
            if (!isFocused) {
                try { sp.increment(0); } catch (Exception ignored) {}
            }
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

    private static String intArrStr(int[] arr) {
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < arr.length; i++) { if (i > 0) sb.append('-'); sb.append(arr[i]); }
        return sb.toString();
    }
}
