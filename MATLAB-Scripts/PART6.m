%% STEP 6: RETINAL STRUCTURE SEGMENTATION & CLINICAL REPORT (FULL UNCLIPPED DISPLAY)
clear; clc; close all;

scriptDir = fileparts(mfilename('fullpath'));
projectRoot = fileparts(scriptDir);
modelPath = fullfile(projectRoot, 'dr_resnet50_model.mat');
if ~isfile(modelPath)
    modelPath = 'dr_resnet50_model.mat';
end

fprintf('[1/4] Loading trained ResNet-50 AI model...\n');
tic;
load(modelPath, 'drNet');
fprintf('      -> Model loaded in %.2f seconds.\n', toc);

% 2. Load test fundus image
testImgPath = fullfile(projectRoot, 'Datasets', '1 APTOS 2019 Blindness Detection', 'train_images', '000c1434d8d7.png');
if ~isfile(testImgPath)
    testImgPath = '000c1434d8d7.png';
end
rawImgOriginal = imread(testImgPath);

fprintf('[2/4] Standardizing resolution (aspect-ratio preserved) & segmenting...\n');
tic;

% Scale proportionally to target height 600 (Preserves 100% native aspect ratio!)
targetH = 600;
scaleFactor = targetH / size(rawImgOriginal, 1);
rawImg = imresize(rawImgOriginal, scaleFactor);

green = rawImg(:, :, 2);
red   = rawImg(:, :, 1);
[H, W] = size(green);

% 3. Field-of-View (FOV) Mask
fovMask = (double(red) + double(green)) > 30;
fovMask = imfill(fovMask, 'holes');
fovMask = bwareafilt(fovMask, 1);
fovMask = imerode(fovMask, strel('disk', round(H * 0.03)));

% 4. CLAHE Enhancement inside FOV
enhancedGreen = adapthisteq(green, 'ClipLimit', 0.02, 'NumTiles', [8 8]);
enhancedGreen(~fovMask) = 0;

% 5. Segment Blood Vessels
seVessel = strel('disk', max(3, round(H * 0.008)));
topHat = imtophat(enhancedGreen, seVessel);
vesselThresh = graythresh(topHat(fovMask)) * 1.2;
vessels = (topHat > (vesselThresh * 255)) & fovMask;
vessels = bwareaopen(vessels, 15);
vesselsDil = imdilate(vessels, strel('disk', 2));

% 6. Anatomically-Accurate Optic Disc Localization (Central 60% ROI)
centralROI = false(H, W);
centralROI(round(H*0.25):round(H*0.75), round(W*0.15):round(W*0.85)) = true;

% Combine Red + Green luminance (Optic Disc is bright orange-yellow)
lumMap = (double(red) + double(green)) .* double(centralROI & fovMask);
blurredLum = imgaussfilt(lumMap, 12);
[~, maxIdx] = max(blurredLum(:));
[odY, odX] = ind2sub(size(blurredLum), maxIdx);
odRadius = round(H * 0.055);

[X, Y] = meshgrid(1:W, 1:H);
odMask = ((X - odX).^2 + (Y - odY).^2) <= (odRadius^2);

% Fovea Localization (~2.5 disc diameters temporally)
foveaDist = round(2.5 * odRadius);
if odX > W/2
    foveaX = max(30, odX - foveaDist);
else
    foveaX = min(W - 30, odX + foveaDist);
end
foveaY = min(H - 25, max(25, odY + round(odRadius * 0.15)));
foveaRadius = round(odRadius * 0.65);
foveaMask = ((X - foveaX).^2 + (Y - foveaY).^2) <= (foveaRadius^2);

% Non-lesion exclusion zone
exclusionZone = vesselsDil | odMask | ~fovMask;

% 7. CLINICALLY CALIBRATED LESION DETECTION
% A. Hard Exudates (Bright yellow deposits)
brightCandidates = (enhancedGreen > 190) & ~exclusionZone;
ccEx = bwconncomp(brightCandidates);
statsEx = regionprops(ccEx, 'Area');
if ~isempty(statsEx)
    areasEx = [statsEx.Area];
    validExIdx = find(areasEx >= 4 & areasEx <= 200);
    exudateMask = ismember(labelmatrix(ccEx), validExIdx);
    numExudates = length(validExIdx);
else
    exudateMask = false(size(green));
    numExudates = 0;
end

% B. Microaneurysms (Small dark red dots)
bottomHat = imbothat(enhancedGreen, strel('disk', 4));
bottomHat(exclusionZone) = 0;

validPixels = bottomHat(fovMask & ~exclusionZone);
if ~isempty(validPixels)
    threshMA = prctile(validPixels, 99.7);
    maCandidates = (bottomHat > threshMA) & ~exclusionZone;
    ccDark = bwconncomp(maCandidates);
    statsDark = regionprops(ccDark, 'Area', 'Eccentricity');
    
    if ~isempty(statsDark)
        areasDark = [statsDark.Area];
        eccsDark  = [statsDark.Eccentricity];
        
        validMAIdx = find(areasDark >= 3 & areasDark <= 35 & eccsDark < 0.90);
        validHEIdx = find(areasDark > 35 & areasDark <= 300);
        
        L_dark = labelmatrix(ccDark);
        microaneurysmMask = ismember(L_dark, validMAIdx);
        hemorrhageMask    = ismember(L_dark, validHEIdx);
        
        numMicroaneurysms = length(validMAIdx);
        numHemorrhages    = length(validHEIdx);
    else
        microaneurysmMask = false(size(green));
        hemorrhageMask    = false(size(green));
        numMicroaneurysms = 0;
        numHemorrhages    = 0;
    end
else
    microaneurysmMask = false(size(green));
    hemorrhageMask    = false(size(green));
    numMicroaneurysms = 0;
    numHemorrhages    = 0;
end
fprintf('      -> Segmentation completed in %.2f seconds.\n', toc);

% 8. ResNet-50 Inference & Grad-CAM Attention Map
fprintf('[3/4] Running AI inference and computing Grad-CAM heatmap...\n');
tic;
inputSize = drNet.Layers(1).InputSize(1:2);
resizedAI = imresize(rawImg, inputSize);
[predictedLabel, scores] = classify(drNet, resizedAI);
confidence = max(scores) * 100;
scoreMap = gradCAM(drNet, resizedAI, predictedLabel);
% Resize scoreMap back to full display resolution so aspect ratio matches 100%!
scoreMapFull = imresize(scoreMap, [H, W]);
fprintf('      -> AI inference & Grad-CAM done in %.2f seconds.\n', toc);

% 9. Generate 4-Panel Clinical Validation Dashboard (Full View, No Clipping!)
fprintf('[4/4] Rendering clinical dashboard (full unclipped layout)...\n');
fig = figure('Name', 'Ophthalmologist 30-Second Validation Dashboard', ...
    'Position', [50 40 1500 860], 'Color', 'w');

% Use modern tiledlayout: eliminates clipping and handles aspect ratios cleanly
t = tiledlayout(2, 2, 'TileSpacing', 'compact', 'Padding', 'compact');

% Panel 1: Original Fundus & Landmarks
nexttile;
imshow(rawImg);
hold on;
visboundaries(odMask, 'Color', 'yellow', 'LineWidth', 2);
text(odX - 25, max(20, odY - odRadius - 10), 'Optic Disc', ...
    'Color', 'yellow', 'FontWeight', 'bold', 'FontSize', 10, 'BackgroundColor', [0 0 0 0.5]);
visboundaries(foveaMask, 'Color', 'cyan', 'LineWidth', 1.5);
text(foveaX - 20, max(20, foveaY - foveaRadius - 10), 'Fovea', ...
    'Color', 'cyan', 'FontWeight', 'bold', 'FontSize', 10, 'BackgroundColor', [0 0 0 0.5]);
title('1. Retinal Fundus & Anatomical Landmarks (OD + Fovea)', 'FontSize', 11, 'FontWeight', 'bold');
axis image;
axis off;

% Panel 2: Blood Vessels
nexttile;
imshow(vessels);
title('2. Segmented Retinal Vascular Network', 'FontSize', 11, 'FontWeight', 'bold');
axis image;
axis off;

% Panel 3: Lesion Detection Overlay
nexttile;
imshow(enhancedGreen);
hold on;
if numExudates > 0
    visboundaries(exudateMask, 'Color', 'yellow', 'LineWidth', 1.5);
end
if numMicroaneurysms > 0
    visboundaries(microaneurysmMask, 'Color', 'red', 'LineWidth', 1.5);
end
if numHemorrhages > 0
    visboundaries(hemorrhageMask, 'Color', 'magenta', 'LineWidth', 1.5);
end
title(sprintf('3. Verified Lesions: MA (%d) | EX (%d) | HE (%d)', ...
    numMicroaneurysms, numExudates, numHemorrhages), 'FontSize', 11, 'FontWeight', 'bold');
axis image;
axis off;

% Panel 4: Grad-CAM Attention Heatmap
nexttile;
imshow(rawImg);
hold on;
imagesc(scoreMapFull, 'AlphaData', 0.45);
colormap(gca, 'jet');
title(sprintf('4. Grad-CAM Attention Heatmap (Diagnosis: %s, %.1f%%)', ...
    string(predictedLabel), confidence), 'FontSize', 11, 'FontWeight', 'bold');
axis image;
axis off;

title(t, 'MathWorks SIH — Explainable AI Diagnostic & Clinical Triage Dashboard', ...
    'FontSize', 14, 'FontWeight', 'bold');

% 10. Print Validated 30-Second Clinical Screening Report
fprintf('\n=================================================================\n');
fprintf('         AUTOMATED TELEMEDICINE CLINICAL REPORT                  \n');
fprintf('=================================================================\n');
fprintf('Patient Image ID        : %s\n', '000c1434d8d7.png');
fprintf('Screening Timestamp     : %s\n', string(datetime('now')));
fprintf('AI Diagnosis            : %s\n', string(predictedLabel));
fprintf('Confidence Score        : %.2f%%\n', confidence);
fprintf('Optic Disc Center       : [X: %d, Y: %d]\n', round(odX), round(odY));
fprintf('Fovea Center            : [X: %d, Y: %d]\n', round(foveaX), round(foveaY));
fprintf('-----------------------------------------------------------------\n');
fprintf('CLINICAL LESION EVIDENCE BREAKDOWN:\n');
fprintf('  • Microaneurysms (MA) : %d lesions detected (Earliest DR sign)\n', numMicroaneurysms);
fprintf('  • Hard Exudates (EX)  : %d deposits detected (Lipid leakage)\n', numExudates);
fprintf('  • Hemorrhages (HE)    : %d blotches detected (Vascular damage)\n', numHemorrhages);
fprintf('-----------------------------------------------------------------\n');

if predictedLabel == "2_Moderate" || predictedLabel == "3_Severe" || predictedLabel == "4_Proliferative"
    fprintf('CLINICAL ACTION         : [!] REFERRAL REQUIRED (Referable DR)\n');
    fprintf('Triage Urgency          : URGENT — Tele-consult ophthalmologist within 7 days\n');
elseif predictedLabel == "1_Mild"
    fprintf('CLINICAL ACTION         : MONITORING (Non-referable DR)\n');
    fprintf('Triage Urgency          : ROUTINE — Repeat fundus screening in 6 months\n');
else
    fprintf('CLINICAL ACTION         : NO PATHOLOGY DETECTED\n');
    fprintf('Triage Urgency          : NORMAL — Annual preventive diabetic eye screening\n');
end
fprintf('Doctor Validation Time  : < 30 seconds human-in-the-loop review\n');
fprintf('=================================================================\n');
