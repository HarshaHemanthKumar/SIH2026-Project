%% STEP 5: IMAGE QUALITY ASSESSMENT & ADAPTIVE ENHANCEMENT (UNCLIPPED DISPLAY)
clear; clc; close all;

scriptDir = fileparts(mfilename('fullpath'));
projectRoot = fileparts(scriptDir);

% 1. Locate sample image
sampleImgPath = fullfile(projectRoot, 'Datasets', '1 APTOS 2019 Blindness Detection', 'train_images', '000c1434d8d7.png');
if ~isfile(sampleImgPath)
    sampleImgPath = '000c1434d8d7.png';
end
rawImg = imread(sampleImgPath);

% 2. Run Image Quality Assessment
[qualityStatus, qualityScore, metrics, feedback] = evaluateFundusQuality(rawImg);

% 3. Apply Adaptive Enhancement if Borderline or Adequate
if strcmp(qualityStatus, 'ADEQUATE') || strcmp(qualityStatus, 'BORDERLINE')
    enhancedImg = applyAdaptiveEnhancement(rawImg);
else
    enhancedImg = rawImg; % Cannot enhance ungradeable
end

% 4. Visualize Results (Full Unclipped Display with TiledLayout)
figure('Name', 'Image Quality Assessment & Enhancement', ...
    'Position', [80 80 1450 650], 'Color', 'w');

t = tiledlayout(1, 3, 'TileSpacing', 'compact', 'Padding', 'compact');

% Panel 1: Raw Input
nexttile;
imshow(rawImg);
title(sprintf('1. Input Raw Fundus\nStatus: %s (Score: %d/3)', qualityStatus, qualityScore), ...
    'FontSize', 11, 'FontWeight', 'bold');
axis image;
axis off;

% Panel 2: Green Channel
nexttile;
imshow(rawImg(:,:,2));
title(sprintf('2. Green Channel\nSharpness: %.1f | Illumination: %.1f', ...
    metrics.sharpness, metrics.brightness), 'FontSize', 11, 'FontWeight', 'bold');
axis image;
axis off;

% Panel 3: Enhanced Result or Recapture Warning
nexttile;
if strcmp(qualityStatus, 'UNGRADEABLE')
    imshow(zeros(size(rawImg)));
    text(50, round(size(rawImg,1)*0.5), sprintf('REJECTED!\n\nOperator Feedback:\n%s', strjoin(feedback, '\n')), ...
        'Color', 'red', 'FontSize', 12, 'FontWeight', 'bold');
    title('3. Rejection & Recapture Notice', 'FontSize', 11, 'FontWeight', 'bold');
else
    imshow(enhancedImg);
    title('3. CLAHE + Illumination Normalized', 'FontSize', 11, 'FontWeight', 'bold');
end
axis image;
axis off;

title(t, 'MathWorks SIH — Automated Retinal Image Quality Gate & Adaptive Enhancement', ...
    'FontSize', 14, 'FontWeight', 'bold');

% 5. Print CLI Report
fprintf('\n=======================================================\n');
fprintf('           IMAGE QUALITY ASSESSMENT REPORT             \n');
fprintf('=======================================================\n');
fprintf('Quality Decision : %s (Score: %d/3)\n', qualityStatus, qualityScore);
fprintf('Focus/Sharpness  : %.2f (Threshold: > 50)\n', metrics.sharpness);
fprintf('Brightness Mean  : %.2f (Valid Range: 30 - 210)\n', metrics.brightness);
fprintf('Contrast (StdDev): %.2f (Threshold: > 25)\n', metrics.contrast);
if ~isempty(feedback)
    fprintf('\nOperator Feedback:\n');
    for i = 1:length(feedback)
        fprintf('  [!] %s\n', feedback{i});
    end
else
    fprintf('\nStatus: Image is clinically adequate for AI diagnosis.\n');
end
fprintf('=======================================================\n');


%% ================= HELPER FUNCTIONS ================= %%

function [status, score, metrics, feedback] = evaluateFundusQuality(img)
    gray = rgb2gray(img);

    % 1. Sharpness via Laplacian Variance
    lapKernel = fspecial('laplacian', 0.2);
    lapImg = imfilter(double(gray), lapKernel, 'replicate');
    sharpness = var(lapImg(:));

    % 2. Brightness & Contrast
    brightness = mean(gray(:));
    contrast = std(double(gray(:)));

    metrics.sharpness = sharpness;
    metrics.brightness = brightness;
    metrics.contrast = contrast;

    score = 0;
    feedback = {};

    if sharpness >= 50
        score = score + 1;
    else
        feedback{end+1} = 'Image is out of focus / blurry. Please hold steady and refocus.';
    end

    if brightness >= 30 && brightness <= 210
        score = score + 1;
    else
        feedback{end+1} = 'Improper lighting (underexposed or overexposed). Adjust camera flash.';
    end

    if contrast >= 25
        score = score + 1;
    else
        feedback{end+1} = 'Low contrast. Clean camera lens or dilate pupil.';
    end

    if score == 3
        status = 'ADEQUATE';
    elseif score == 2
        status = 'BORDERLINE';
    else
        status = 'UNGRADEABLE';
    end
end

function enhanced = applyAdaptiveEnhancement(img)
    % 1. Convert to LAB color space
    lab = rgb2lab(img);

    % 2. Apply CLAHE on Luminance channel
    L = lab(:,:,1) / 100;
    L_clahe = adapthisteq(L, 'ClipLimit', 0.02, 'NumTiles', [8 8]);
    lab(:,:,1) = L_clahe * 100;
    enhanced = lab2rgb(lab);

    % 3. Illumination Normalization on HSV
    hsv = rgb2hsv(enhanced);
    V = hsv(:,:,3);
    bg = imgaussfilt(V, 25);
    V_norm = V ./ (bg + 0.05);
    V_norm = V_norm / max(V_norm(:));
    hsv(:,:,3) = V_norm;

    enhanced = hsv2rgb(hsv);
    enhanced = im2uint8(enhanced);
end
