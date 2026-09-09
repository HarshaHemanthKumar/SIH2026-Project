%% STEP 1: ORGANIZE APTOS IMAGES INTO 5 SEVERITY FOLDERS
clear; clc;

% Dataset paths
datasetBase = 'C:\Users\hk3202\workspace\Projects\SIH2026-Project\Datasets\1 APTOS 2019 Blindness Detection';
csvPath = fullfile(datasetBase, 'train.csv');
sourceImgDir = fullfile(datasetBase, 'train_images');
targetDir = fullfile(datasetBase, 'sorted_train_images');

% Class names
classNames = {'0_NoDR', '1_Mild', '2_Moderate', '3_Severe', '4_Proliferative'};

% Create class folders
for i = 1:length(classNames)
    folder = fullfile(targetDir, classNames{i});
    if ~exist(folder, 'dir')
        mkdir(folder);
    end
end

% Read labels CSV
fprintf('Reading train.csv...\n');
dataTable = readtable(csvPath);

% Move/Copy images into respective class folders
total = height(dataTable);
fprintf('Sorting %d images into class folders...\n', total);

for i = 1:total
    imgId = dataTable.id_code{i};
    label = dataTable.diagnosis(i); % 0 to 4

    src = fullfile(sourceImgDir, [imgId, '.png']);
    dst = fullfile(targetDir, classNames{label + 1}, [imgId, '.png']);

    if exist(src, 'file') && ~exist(dst, 'file')
        copyfile(src, dst);
    end

    if mod(i, 500) == 0
        fprintf('Progress: %d / %d images sorted\n', i, total);
    end
end

fprintf('Done! Images sorted in: %s\n', targetDir);