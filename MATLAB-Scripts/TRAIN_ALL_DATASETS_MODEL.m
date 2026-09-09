%% MASTER TRAINING PIPELINE: RESNET-50 ON ALL TRAINING DATASETS (4,075 IMAGES)
clear; clc; close all;

rng(42); % Fixed seed for strict experimental reproducibility

scriptDir = fileparts(mfilename('fullpath'));
projectRoot = fileparts(scriptDir);

fprintf('=================================================================\n');
fprintf('   HIGH-ACCURACY PRODUCTION TRAINING ON COMBINED DATASETS        \n');
fprintf('=================================================================\n');

%% 1. LOAD COMBINED DATASET (APTOS + IDRiD TRAIN = 4,075 IMAGES)
dataPath = fullfile(projectRoot, 'Datasets', 'combined_training_dataset');
imds = imageDatastore(dataPath, ...
    'IncludeSubfolders', true, ...
    'LabelSource', 'foldernames');

labelCounts = countEachLabel(imds);
fprintf('Total Training Pool Distribution (%d images):\n', sum(labelCounts.Count));
disp(labelCounts);

%% 2. INVERSE-FREQUENCY CLINICAL LOSS WEIGHTING
counts = double(labelCounts.Count);
totalSamples = sum(counts);
classFrequencies = counts / totalSamples;
% Inverse frequency with smoothing
weights = (1 ./ (classFrequencies + 1e-3));
weights = weights / mean(weights); % Normalize mean to 1.0

fprintf('Class Penalty Weights in Loss Function:\n');
for c = 1:height(labelCounts)
    fprintf('  • %-16s : %.2f\n', string(labelCounts.Label(c)), weights(c));
end

%% 3. STRATIFIED TRAIN / VALIDATION SPLIT (90% Train, 10% Val)
% 90% for training (3,667 images), 10% for internal validation (408 images)
[imdsTrain, imdsVal] = splitEachLabel(imds, 0.9, 'randomized');

%% 4. PRETRAINED RESNET-50 BACKBONE
net = resnet50;
inputSize = net.Layers(1).InputSize(1:2); % [224 224]

%% 5. CLINICAL DATA AUGMENTATION (Full Rotational & Scale Invariance)
augmenter = imageDataAugmenter( ...
    'RandXReflection', true, ...
    'RandYReflection', true, ...
    'RandRotation', [-30, 30], ...
    'RandXScale', [0.88 1.12], ...
    'RandYScale', [0.88 1.12]);

augTrain = augmentedImageDatastore(inputSize, imdsTrain, 'DataAugmentation', augmenter);
augVal   = augmentedImageDatastore(inputSize, imdsVal);

%% 6. NETWORK SURGERY FOR 5-CLASS DR GRADING
lgraph = layerGraph(net);
newLayers = [
    fullyConnectedLayer(5, 'Name', 'fc_dr', ...
        'WeightLearnRateFactor', 10, ...
        'BiasLearnRateFactor', 10)
    softmaxLayer('Name', 'softmax_dr')
    classificationLayer('Name', 'output_dr', ...
        'Classes', labelCounts.Label, ...
        'ClassWeights', weights)
];
lgraph = replaceLayer(lgraph, 'fc1000', newLayers(1));
lgraph = replaceLayer(lgraph, 'fc1000_softmax', newLayers(2));
lgraph = replaceLayer(lgraph, 'ClassificationLayer_fc1000', newLayers(3));

%% 7. TRAINING OPTIONS (GPU ACCELERATED WITH LR DECAY)
options = trainingOptions('adam', ...
    'InitialLearnRate', 1e-4, ...
    'LearnRateSchedule', 'piecewise', ...
    'LearnRateDropFactor', 0.3, ...
    'LearnRateDropPeriod', 3, ...
    'MaxEpochs', 10, ...
    'MiniBatchSize', 32, ...
    'ValidationData', augVal, ...
    'ValidationFrequency', 30, ...
    'Shuffle', 'every-epoch', ...
    'Plots', 'none', ... % Headless console logging for background execution
    'Verbose', true, ...
    'ExecutionEnvironment', 'gpu');

%% 8. LAUNCH TRAINING
fprintf('\nStarting training on GPU (%d batches per epoch)...\n', ceil(length(imdsTrain.Files)/32));
tic;
drNet = trainNetwork(augTrain, lgraph, options);
trainTimeSec = toc;

fprintf('\nTraining completed in %.1f minutes (%.1f seconds).\n', trainTimeSec/60, trainTimeSec);

%% 9. SAVE TRAINED PRODUCTION MODEL
savePath = fullfile(projectRoot, 'dr_resnet50_model.mat');
save(savePath, 'drNet');
fprintf('Saved production model weights to: %s\n', savePath);

% Copy to MATLAB-Scripts for convenience
copyfile(savePath, fullfile(scriptDir, 'dr_resnet50_model.mat'));

fprintf('=================================================================\n');
fprintf(' MASTER TRAINING COMPLETE — ACCURACY OPTIMIZED ACROSS ALL DATASETS\n');
fprintf('=================================================================\n');
