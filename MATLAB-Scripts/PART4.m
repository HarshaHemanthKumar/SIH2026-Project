%% STEP 4: CLINICAL VALIDATION METRICS ON COMBINED DATASET (PROPER UNCLIPPED DISPLAY)
clear; clc; close all;

rng(42); % Fixed seed for reproducible results

scriptDir = fileparts(mfilename('fullpath'));
projectRoot = fileparts(scriptDir);

% 1. Load trained model
modelPath = fullfile(projectRoot, 'dr_resnet50_model.mat');
if ~isfile(modelPath), modelPath = 'dr_resnet50_model.mat'; end
load(modelPath, 'drNet');

% 2. Load the combined dataset
dataPath = fullfile(projectRoot, 'Datasets', 'combined_training_dataset');
if ~isfolder(dataPath)
    dataPath = fullfile(projectRoot, 'Datasets', '1 APTOS 2019 Blindness Detection', 'sorted_train_images');
end

imds = imageDatastore(dataPath, ...
    'IncludeSubfolders', true, ...
    'LabelSource', 'foldernames');

% Use same 90/10 split as training (rng=42)
[~, imdsVal] = splitEachLabel(imds, 0.9, 'randomized');

inputSize = drNet.Layers(1).InputSize(1:2);
augVal = augmentedImageDatastore(inputSize, imdsVal);

% 3. Run predictions on held-out validation data
fprintf('Running validation predictions on %d held-out images...\n', length(imdsVal.Files));
[predictions, scores] = classify(drNet, augVal);
trueLabels = imdsVal.Labels;

% 4. Calculate Binary Referable DR Metrics (Non-Referable: 0, 1 vs Referable: 2, 3, 4)
isTrueReferable = (trueLabels == '2_Moderate' | trueLabels == '3_Severe' | trueLabels == '4_Proliferative');
isPredReferable = (predictions == '2_Moderate' | predictions == '3_Severe' | predictions == '4_Proliferative');

TP = sum(isTrueReferable & isPredReferable);
TN = sum(~isTrueReferable & ~isPredReferable);
FP = sum(~isTrueReferable & isPredReferable);
FN = sum(isTrueReferable & ~isPredReferable);

sensitivity = (TP / max(1, TP + FN)) * 100;
specificity = (TN / max(1, TN + FP)) * 100;
accuracy    = ((TP + TN) / max(1, TP + TN + FP + FN)) * 100;
precision   = (TP / max(1, TP + FP)) * 100;
f1Score     = 2 * (precision * sensitivity) / max(1, precision + sensitivity);

% 5. Quadratic Weighted Kappa
numClasses = 5;
trueNumeric = double(trueLabels) - 1;
predNumeric = double(predictions) - 1;
O = confusionmat(trueNumeric, predNumeric, 'Order', 0:4);
W = zeros(numClasses);
for r = 1:numClasses
    for c = 1:numClasses
        W(r,c) = ((r-c)^2) / ((numClasses-1)^2);
    end
end
histT = histcounts(trueNumeric, -0.5:1:4.5);
histP = histcounts(predNumeric, -0.5:1:4.5);
E = (histT' * histP) / length(trueNumeric);
qwk = 1 - (sum(sum(W .* O)) / max(1e-6, sum(sum(W .* E))));

% 6. Plot Dual-Panel Validation Dashboard (Full Unclipped Layout)
figure('Name', 'Internal Clinical Validation Dashboard', ...
    'Position', [80 80 1400 680], 'Color', 'w');

t = tiledlayout(1, 2, 'TileSpacing', 'compact', 'Padding', 'compact');

% Panel 1: Multi-class Confusion Matrix
nexttile;
cm = confusionchart(trueLabels, predictions);
cm.Title = sprintf('5-Class Confusion Matrix (QWK: %.3f)', qwk);
cm.RowSummary = 'row-normalized';
cm.ColumnSummary = 'column-normalized';

% Panel 2: Binary Referable Screening Metrics
nexttile;
metricNames = {'Sensitivity', 'Specificity', 'Accuracy', 'Precision', 'F1-Score'};
metricValues = [sensitivity, specificity, accuracy, precision, f1Score];
sihTargets   = [90.0, 85.0, 85.0, 85.0, 85.0];

b = bar([metricValues; sihTargets]', 'grouped');
b(1).FaceColor = [0.15 0.55 0.85];
b(2).FaceColor = [0.85 0.35 0.25];
set(gca, 'XTickLabel', metricNames, 'FontSize', 10, 'FontWeight', 'bold');
ylabel('Performance Rate (%)', 'FontSize', 11, 'FontWeight', 'bold');
ylim([60 102]);
grid on;
legend({'Our System', 'SIH Benchmark Target'}, 'Location', 'southeast', 'FontSize', 10);
title('Binary Referable DR Screening Performance (Level 2+)', 'FontSize', 12, 'FontWeight', 'bold');

title(t, 'Internal Clinical Validation & Diagnostic Accuracy Benchmark', ...
    'FontSize', 14, 'FontWeight', 'bold');

% 7. Print Report
fprintf('\n=======================================================\n');
fprintf('        INTERNAL CLINICAL VALIDATION REPORT            \n');
fprintf('=======================================================\n');
fprintf('Validation Dataset      : Combined (APTOS + IDRiD Training)\n');
fprintf('Held-Out Split          : 10%% (rng=42, reproducible)\n');
fprintf('Total Validation Images : %d\n', length(trueLabels));
fprintf('Overall Accuracy        : %.2f%%\n', accuracy);
fprintf('Sensitivity (Recall)    : %.2f%%   [SIH Target: >90%%]\n', sensitivity);
fprintf('Specificity             : %.2f%%   [SIH Target: >85%%]\n', specificity);
fprintf('Precision               : %.2f%%\n', precision);
fprintf('F1-Score                : %.2f%%\n', f1Score);
fprintf('Quadratic Weighted Kappa: %.4f\n', qwk);
fprintf('True Positives (TP)     : %d\n', TP);
fprintf('True Negatives (TN)     : %d\n', TN);
fprintf('False Positives (FP)    : %d\n', FP);
fprintf('False Negatives (FN)    : %d\n', FN);
fprintf('=======================================================\n');
