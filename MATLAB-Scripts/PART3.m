%% STEP 3: MULTI-SEVERITY GRAD-CAM EXPLAINABILITY DASHBOARD (FULL UNCLIPPED DISPLAY)
clear; clc; close all;

scriptDir = fileparts(mfilename('fullpath'));
projectRoot = fileparts(scriptDir);

% Load trained model
modelPath = fullfile(projectRoot, 'dr_resnet50_model.mat');
if ~isfile(modelPath), modelPath = 'dr_resnet50_model.mat'; end
load(modelPath, 'drNet');
inputSize = drNet.Layers(1).InputSize(1:2);

% Pick one representative image from EACH severity class
sortedDir = fullfile(projectRoot, 'Datasets', '1 APTOS 2019 Blindness Detection', 'sorted_train_images');
classNames = {'0_NoDR', '1_Mild', '2_Moderate', '3_Severe', '4_Proliferative'};
clinicalNames = {'No DR (Healthy)', 'Mild NPDR', 'Moderate NPDR', 'Severe NPDR', 'Proliferative DR'};

figure('Name', 'Explainable AI — Multi-Severity Grad-CAM Dashboard', ...
    'Position', [30 40 1550 750], 'Color', 'w');

% Modern tiledlayout: strictly preserves image proportions without clipping top/bottom
t = tiledlayout(2, 5, 'TileSpacing', 'compact', 'Padding', 'compact');

% Store images and heatmaps for two-row display
rawImages = cell(1, 5);
scoreMaps = cell(1, 5);
predLabels = cell(1, 5);
confidences = zeros(1, 5);

for c = 1:5
    classDir = fullfile(sortedDir, classNames{c});
    imgFiles = dir(fullfile(classDir, '*.png'));
    if isempty(imgFiles), continue; end
    
    raw = imread(fullfile(classDir, imgFiles(1).name));
    % Scale preserving aspect ratio
    scale = 400 / size(raw, 1);
    rawScaled = imresize(raw, scale);
    rawImages{c} = rawScaled;
    
    resizedForNet = imresize(rawScaled, inputSize);
    [pred, scores] = classify(drNet, resizedForNet);
    predLabels{c} = pred;
    confidences(c) = max(scores) * 100;
    
    sMap = gradCAM(drNet, resizedForNet, pred);
    scoreMaps{c} = imresize(sMap, [size(rawScaled, 1), size(rawScaled, 2)]);
end

% Row 1: Original Fundus Photos
for c = 1:5
    nexttile(c);
    imshow(rawImages{c});
    title(sprintf('True: %s', clinicalNames{c}), 'FontSize', 10, 'FontWeight', 'bold');
    axis image;
    axis off;
end

% Row 2: Grad-CAM Explainability Attention Maps
for c = 1:5
    nexttile(c + 5);
    imshow(rawImages{c});
    hold on;
    imagesc(scoreMaps{c}, 'AlphaData', 0.45);
    colormap(gca, 'jet');
    title(sprintf('AI: %s (%.1f%%)', string(predLabels{c}), confidences(c)), ...
        'FontSize', 10, 'FontWeight', 'bold');
    axis image;
    axis off;
end

title(t, 'Explainable AI for Diabetic Retinopathy — Grad-CAM Attention Across All 5 Severity Levels', ...
    'FontSize', 13, 'FontWeight', 'bold');

fprintf('\n=================================================================\n');
fprintf('         GRAD-CAM EXPLAINABILITY REPORT                          \n');
fprintf('=================================================================\n');
fprintf('Architecture          : ResNet-50 Transfer Learning\n');
fprintf('XAI Method            : Gradient-weighted Class Activation Mapping\n');
fprintf('Severity Classes Shown: 5 (NoDR, Mild, Moderate, Severe, PDR)\n');
fprintf('Display Mode          : Full Unclipped Aspect Ratio\n');
fprintf('Clinical Purpose      : 30-second ophthalmologist validation aid\n');
fprintf('=================================================================\n');
