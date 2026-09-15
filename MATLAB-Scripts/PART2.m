%% STEP 2: TRAIN RESNET-50 ON APTOS DATASET (CLASS-WEIGHTED TRANSFER LEARNING)
clear; clc; close all;

rng(42); % Fixed random seed for reproducibility

% 1. Load image datastore from sorted folders
dataPath = fullfile('..', 'Datasets', '1 APTOS 2019 Blindness Detection', 'sorted_train_images');
imds = imageDatastore(dataPath, ...
    'IncludeSubfolders', true, ...
    'LabelSource', 'foldernames');

% 2. Count images and calculate Inverse Frequency Class Weights
labelCounts = countEachLabel(imds);
fprintf('Original Class Distribution:\n');
disp(labelCounts);

% Class weights: Higher penalty for missing minority severe classes
counts = double(labelCounts.Count);
totalSamples = sum(counts);
classFrequencies = counts / totalSamples;
weights = (1 ./ (classFrequencies + 1e-3));
weights = weights / mean(weights); % Normalize mean weight to 1.0

fprintf('Calculated Clinical Loss Weights:\n');
for c = 1:height(labelCounts)
    fprintf('  • Class %-15s : Weight = %.2f\n', string(labelCounts.Label(c)), weights(c));
end

% 3. Split 80% Train, 20% Validation
[imdsTrain, imdsVal] = splitEachLabel(imds, 0.8, 'randomized');

% 4. Load Pretrained ResNet-50
net = resnet50;
inputSize = net.Layers(1).InputSize(1:2); % [224 224]

% 5. Data Augmentation (Rotation, Reflection, Zoom)
augmenter = imageDataAugmenter( ...
    'RandXReflection', true, ...
    'RandYReflection', true, ...
    'RandRotation', [-25, 25], ...
    'RandXScale', [0.9 1.1], ...
    'RandYScale', [0.9 1.1]);

augTrain = augmentedImageDatastore(inputSize, imdsTrain, 'DataAugmentation', augmenter);
augVal   = augmentedImageDatastore(inputSize, imdsVal);

% 6. Replace final layers with Class-Weighted Classification Layer
lgraph = layerGraph(net);
newLayers = [
    fullyConnectedLayer(5, 'Name', 'fc_dr', 'WeightLearnRateFactor', 10, 'BiasLearnRateFactor', 10)
    softmaxLayer('Name', 'softmax_dr')
    classificationLayer('Name', 'output_dr', 'Classes', labelCounts.Label, 'ClassWeights', weights)
];
lgraph = replaceLayer(lgraph, 'fc1000', newLayers(1));
lgraph = replaceLayer(lgraph, 'fc1000_softmax', newLayers(2));
lgraph = replaceLayer(lgraph, 'ClassificationLayer_fc1000', newLayers(3));

% 7. Training Options (12 Epochs with Learning Rate Decay)
options = trainingOptions('adam', ...
    'InitialLearnRate', 1e-4, ...
    'LearnRateSchedule', 'piecewise', ...
    'LearnRateDropFactor', 0.3, ...
    'LearnRateDropPeriod', 4, ...
    'MaxEpochs', 12, ...
    'MiniBatchSize', 32, ...
    'ValidationData', augVal, ...
    'ValidationFrequency', 25, ...
    'Shuffle', 'every-epoch', ...
    'Plots', 'training-progress', ...
    'Verbose', true, ...
    'ExecutionEnvironment', 'auto');

% 8. Train!
fprintf('\nTraining ResNet-50 with Class-Weighted Loss on GPU...\n');
drNet = trainNetwork(augTrain, lgraph, options);

% 9. Save trained model
scriptDir = fileparts(mfilename('fullpath'));
projectRoot = fileparts(scriptDir);
savePath = fullfile(projectRoot, 'dr_resnet50_model.mat');
save(savePath, 'drNet');
fprintf('\nSUCCESS! Model saved to: %s\n', savePath);
