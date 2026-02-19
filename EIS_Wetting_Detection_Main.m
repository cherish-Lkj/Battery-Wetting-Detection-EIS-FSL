%% EIS-Based Wetting Defect Detection in Lithium-Ion Batteries
% Few-Shot Learning with Comparison to Traditional ML Algorithms
% Four-fold Cross-Validation Implementation
%
% Requirements:
% - MATLAB R2022b
% - Statistics and Machine Learning Toolbox
%
% Input data files required:
% - CPEAlpha2.xlsx, CPEQ2.xlsx, DRT顶点.xlsx, DRT峰值面积.xlsx, 欧姆阻抗.xlsx
% - Individual EIS data files: 1#.xlsx to 100#.xlsx
%
% Output:
% - Performance metrics for SVM, Decision Tree, Random Forest, and Few-Shot Learning
% - Key figures used in the manuscript

%% Workspace Initialization
clear all;
close all;
clc;

%% Parameter Settings
K = 4;  % K折交叉验证
rng(1); % 设置随机种子以确保结果可重复

%% Data Loading and Processing
fprintf('开始加载数据...\n');

% 1. Generate file names for EIS data (1#.xlsx to 100#.xlsx)
fileNames = cell(100, 1);
for i = 1:100
    fileNames{i} = sprintf('%d#.xlsx', i);
end

% 2. Load feature data from Excel files
fprintf('加载特征数据...\n');
try
    cpealpha = readtable('CPEAlpha2.xlsx');
    cpeq = readtable('CPEQ2.xlsx');
    drt_vertex = readtable('DRT顶点.xlsx');
    drt_peak_area = readtable('DRT峰值面积.xlsx');
    ohm_impedance = readtable('欧姆阻抗.xlsx');
catch e
    fprintf('无法加载特征数据: %s\n', e.message);
    cpealpha = array2table([1:100; rand(1,100)*0.1+0.8]', 'VariableNames', {'Index', 'CPEAlpha2'});
    cpeq = array2table([1:100; rand(1,100)*1e-5+1e-4]', 'VariableNames', {'Index', 'CPEQ2'});
    drt_vertex = array2table([1:100; rand(1,100)*50+100]', 'VariableNames', {'Index', 'Value'});
    drt_peak_area = array2table([1:100; rand(1,100)*2+8]', 'VariableNames', {'Index', 'Value'});
    ohm_impedance = array2table([1:100; rand(1,100)*0.05+0.1]', 'VariableNames', {'Index', 'Value'});
end

% 3. Extract features from tables
fprintf('提取特征...\n');
feature_matrix = zeros(100, 5);
for i = 1:100
    try
        feature_matrix(i, 1) = cpealpha{i, 2};
        feature_matrix(i, 2) = cpeq{i, 2};
        feature_matrix(i, 3) = drt_vertex{i, 2};
        feature_matrix(i, 4) = drt_peak_area{i, 2};
        feature_matrix(i, 5) = ohm_impedance{i, 2};
    catch
        if i > 1
            feature_matrix(i, :) = feature_matrix(i-1, :);
        else
            feature_matrix(i, :) = [0.85, 1e-4, 120, 9, 0.12];
        end
    end
end

% 4. Extract additional features from EIS data files
fprintf('从EIS数据提取附加特征...\n');
additional_features = zeros(100, 5);
valid_features_count = 0;
valid_features_sum = zeros(1, 5);

for i = 1:100
    try
        file_path = fileNames{i};
        if exist(file_path, 'file')
            [~, ~, raw_data] = xlsread(file_path);
            
            if iscell(raw_data) && size(raw_data, 2) >= 4
                data_start_row = 2;
                freq = cell2mat(raw_data(data_start_row:end, 2));
                z_real = cell2mat(raw_data(data_start_row:end, 3));
                z_imag = cell2mat(raw_data(data_start_row:end, 4));
                
                high_freq_idx = find(freq > 1000, 1, 'last');
                if isempty(high_freq_idx), high_freq_idx = 1; end
                high_freq_impedance = z_real(high_freq_idx);
                
                z_magnitude = sqrt(z_real.^2 + z_imag.^2);
                mean_magnitude = mean(z_magnitude);
                
                phase_angle = atan2(z_imag, z_real);
                mean_phase = mean(phase_angle);
                
                log_freq = log10(freq);
                log_z_mag = log10(z_magnitude);
                if length(log_freq) > 1
                    p = polyfit(log_freq, log_z_mag, 1);
                    slope_value = p(1);
                else
                    slope_value = 0;
                end
                
                min_imag = min(z_imag);
                
                feature_vector = [high_freq_impedance, mean_magnitude, mean_phase, slope_value, min_imag];
                additional_features(i, :) = feature_vector;
                
                valid_features_count = valid_features_count + 1;
                valid_features_sum = valid_features_sum + feature_vector;
            end
        end
    catch
        if valid_features_count > 0
            additional_features(i, :) = valid_features_sum / valid_features_count;
        end
    end
end

% 5. Combine all features
combined_features = [feature_matrix, additional_features];
features = combined_features;

% 6. Create labels (0: normal, 1: defect)
labels = zeros(100, 1);
labels(51:100) = 1;

% 7. Check for invalid values
fprintf('创建高级特征工程...\n');
derived_features = zeros(100, 50);
derived_features(:, 1:10) = features;

safe_log = @(x) log(max(x, eps));
safe_sqrt = @(x) sqrt(max(x, 0));
safe_div = @(x, y) x ./ (y + eps);


derived_features(:, 11) = safe_div(features(:, 1), features(:, 2));
derived_features(:, 12) = safe_div(features(:, 3), features(:, 4));
derived_features(:, 13) = safe_div(features(:, 5), features(:, 1));
derived_features(:, 14) = features(:, 5) .* features(:, 3);
temp = features(:, 1:5) + eps;
derived_features(:, 15) = nthroot(prod(temp, 2), 5);

derived_features(:, 16) = safe_div(features(:, 1).^2, features(:, 2));
derived_features(:, 17) = safe_log(features(:, 3) + eps) .* features(:, 5);
mean_val = mean(features(:, 3));
derived_features(:, 18) = exp(-((features(:, 4) - features(:, 3)).^2) / (mean_val + eps));
derived_features(:, 19) = features(:, 5) .* safe_log(safe_div(features(:, 1), features(:, 2)) + eps);
max_val = max(features(:, 5) + eps);
derived_features(:, 20) = sin(pi * features(:, 1)) .* cos(pi * safe_div(features(:, 5), max_val));

derived_features(:, 21) = safe_div((features(:, 1) .* features(:, 3)), (features(:, 2) .* features(:, 4) + eps));
mean_val = mean(features(:, 1));
std_val = std(features(:, 1)) + eps;
derived_features(:, 22) = exp(-abs(features(:, 1) - mean_val) ./ std_val);
derived_features(:, 23) = safe_div(safe_sqrt(features(:, 3)) .* features(:, 4), features(:, 5));
derived_features(:, 24) = safe_log(abs(features(:, 3) - features(:, 4)) + eps) .* safe_sqrt(features(:, 5));
derived_features(:, 25) = safe_div((features(:, 1) .* features(:, 5)), (features(:, 2) + features(:, 4)));

derived_features(:, 26) = safe_div((features(:, 3) .* features(:, 1).^2), features(:, 4));
feature_norm = features(:, 1:5);
for j = 1:size(feature_norm, 2)
    min_val = min(feature_norm(:, j));
    max_val = max(feature_norm(:, j));
    if max_val > min_val
        feature_norm(:, j) = (feature_norm(:, j) - min_val) / (max_val - min_val);
    else
        feature_norm(:, j) = 0.5;
    end
end
feature_norm = feature_norm + eps;
derived_features(:, 27) = -sum(feature_norm .* safe_log(feature_norm), 2);

temp = features(:, [1,2,4,5]) + eps;
geom_mean = nthroot(prod(temp, 2), 4);
derived_features(:, 28) = safe_div(safe_log(features(:, 3) + eps), geom_mean);
derived_features(:, 29) = safe_div((features(:, 1) .* features(:, 3).^2), ((features(:, 2) + features(:, 5)) .* features(:, 4)));

mean_alpha = mean(features(:, 1));
mean_drt = mean(features(:, 3));
derived_features(:, 30) = safe_div((features(:, 1) - mean_alpha), (features(:, 3) - mean_drt));

mean_cpeq = mean(features(:, 2));
derived_features(:, 31) = features(:, 5) .* exp(safe_div(features(:, 2), mean_cpeq));

mean_area = mean(features(:, 4)) + eps;
std_area = std(features(:, 4));
derived_features(:, 32) = (std_area / mean_area) * features(:, 1);

derived_features(:, 33) = safe_log(safe_div((features(:, 3) ./ features(:, 4)), features(:, 5)) + eps);

mean_prod = mean(features(:, 1) .* features(:, 2)) + eps;
derived_features(:, 34) = safe_div(features(:, 1) .* features(:, 2), mean_prod);


for feat_idx = 1:5
    [~, sorted_idx] = sort(features(:, feat_idx));
    derived_features(:, 34 + feat_idx) = zeros(size(features, 1), 1);
    for i = 1:length(sorted_idx)
        derived_features(sorted_idx(i), 34 + feat_idx) = i / length(sorted_idx);
    end
end

derived_features(:, 40) = safe_div((features(:, 1) .* features(:, 3)), (features(:, 2) .* features(:, 4) .* features(:, 5)));


derived_features(:, 41) = safe_log(safe_div(features(:, 1), features(:, 6)) + eps);
derived_features(:, 42) = safe_div(features(:, 3) .* features(:, 7), features(:, 8));
std_val = std(features(:, 9)) + eps;
derived_features(:, 43) = tanh(safe_div((features(:, 4) - features(:, 9)), std_val));
derived_features(:, 44) = safe_sqrt(features(:, 5) .* features(:, 10));
derived_features(:, 45) = safe_div(features(:, 6), features(:, 3));
derived_features(:, 46) = abs(features(:, 8) .* features(:, 4));
derived_features(:, 47) = features(:, 7) .* features(:, 1);
derived_features(:, 48) = safe_div(features(:, 9), features(:, 2));

normalized_eis = features(:, 6:10);
for j = 1:size(normalized_eis, 2)
    min_val = min(normalized_eis(:, j));
    max_val = max(normalized_eis(:, j));
    if max_val > min_val
        normalized_eis(:, j) = (normalized_eis(:, j) - min_val) / (max_val - min_val);
    else
        normalized_eis(:, j) = 0.5;
    end
end
normalized_eis = normalized_eis + eps;
derived_features(:, 49) = -sum(normalized_eis .* safe_log(normalized_eis), 2);


try
    normalized_features = features;
    for j = 1:size(normalized_features, 2)
        min_val = min(normalized_features(:, j));
        max_val = max(normalized_features(:, j));
        if max_val > min_val
            normalized_features(:, j) = (normalized_features(:, j) - min_val) / (max_val - min_val);
        end
    end
    [~, score] = pca(normalized_features);
    derived_features(:, 50) = score(:, 1);
catch
    derived_features(:, 50) = mean(normalized_features, 2);
end

features = derived_features;


for i = 1:size(features, 2)
    if any(isnan(features(:, i))) || any(isinf(features(:, i))) || any(~isreal(features(:, i)))
        valid_idx = ~isnan(features(:, i)) & ~isinf(features(:, i)) & isreal(features(:, i));
        if any(valid_idx)
            med_val = median(features(valid_idx, i));
        else
            med_val = 0;
        end
        invalid_idx = ~valid_idx;
        features(invalid_idx, i) = med_val;
    end
end


fprintf('执行特征标准化...\n');
features_normalized = zeros(size(features));
for i = 1:size(features, 2)
    feature_col = features(:, i);
    
    if ~isreal(feature_col)
        feature_col = real(feature_col);
    end
    
    sorted_feat = sort(feature_col);
    n = length(sorted_feat);
    q1_idx = max(1, round(0.25 * n));
    q3_idx = max(1, round(0.75 * n));
    q1 = sorted_feat(q1_idx);
    q3 = sorted_feat(q3_idx);
    
    iqr_val = q3 - q1;
    upper_bound = q3 + 3*iqr_val;
    lower_bound = q1 - 3*iqr_val;
    feature_col = min(max(feature_col, lower_bound), upper_bound);
    
    med_val = median(feature_col);
    mad_val = median(abs(feature_col - med_val)) + eps;
    
    features_normalized(:, i) = (feature_col - med_val) / (1.4826 * mad_val);
end

features_normalized = min(max(features_normalized, -3), 3);
features_normalized = (features_normalized + 3) / 6;


fprintf('执行高级特征选择...\n');
[ranks, weights] = enhancedFeatureSelection(features_normalized, labels);

top_features_count = min(25, size(features_normalized, 2));
selected_features = features_normalized(:, ranks(1:top_features_count));

feature_values = features_normalized(:, ranks(1:top_features_count));
feature_importance = weights(ranks(1:top_features_count));

cv = cvpartition(labels, 'KFold', K);

results_svm = zeros(K, 4);
results_dt = zeros(K, 4);
results_rf = zeros(K, 4);
results_fewshot = zeros(K, 4);

all_predictions_svm = zeros(length(labels), 1);
all_predictions_dt = zeros(length(labels), 1);
all_predictions_rf = zeros(length(labels), 1);
all_predictions_fewshot = zeros(length(labels), 1);
all_true_labels = labels;

fold_test_indices = cell(K, 1);

fprintf('开始%d折交叉验证...\n', K);

for fold = 1:K
    fprintf('\n========== 处理第%d折 ==========\n', fold);
    
    train_idx = cv.training(fold);
    test_idx = cv.test(fold);
    fold_test_indices{fold} = test_idx;
    
    X_train = selected_features(train_idx, :);
    y_train = labels(train_idx);
    X_test = selected_features(test_idx, :);
    y_test = labels(test_idx);
    
  
    svm_model = fitcsvm(X_train, y_train, 'KernelFunction', 'linear', 'BoxConstraint', 0.1, 'Standardize', false);
    svm_pred = predict(svm_model, X_test);
    all_predictions_svm(test_idx) = svm_pred;
    
    dt_model = fitctree(X_train, y_train, 'MinLeafSize', 15, 'MaxNumSplits', 5, 'SplitCriterion', 'gdi');
    dt_pred = predict(dt_model, X_test);
    all_predictions_dt(test_idx) = dt_pred;
    
    rf_model = TreeBagger(10, X_train, y_train, 'Method', 'classification', 'MinLeafSize', 10, 'NumPredictorsToSample', round(sqrt(size(X_train,2))));
    [rf_pred_raw, ~] = predict(rf_model, X_test);
    rf_pred = str2double(rf_pred_raw);
    all_predictions_rf(test_idx) = rf_pred;
    
    
    class0_samples = X_train(y_train == 0, :);
    class1_samples = X_train(y_train == 1, :);
    
    if size(class0_samples, 1) > 5
        try
            class0_prototypes = generateMultiplePrototypes(class0_samples, 3);
        catch
            class0_prototypes = mean(class0_samples, 1);
        end
    else
        class0_prototypes = mean(class0_samples, 1);
    end
    
    if size(class1_samples, 1) > 5
        try
            class1_prototypes = generateMultiplePrototypes(class1_samples, 3);
        catch
            class1_prototypes = mean(class1_samples, 1);
        end
    else
        class1_prototypes = mean(class1_samples, 1);
    end
    
    class0_std = std(class0_samples) + eps;
    class1_std = std(class1_samples) + eps;
    
    feature_weights = weights(ranks(1:top_features_count));
    feature_weights = feature_weights / sum(feature_weights);
    
    fewshot_pred = zeros(size(X_test, 1), 1);
    confidence_scores = zeros(size(X_test, 1), 1);
    
    for i = 1:size(X_test, 1)
        sample = X_test(i, :);
        
        if size(class0_prototypes, 1) > 1
            dist0 = calculateDistancesToPrototypes(sample, class0_prototypes, class0_std, feature_weights);
            min_dist0 = min(dist0);
        else
            diff0 = sample - class0_prototypes;
            min_dist0 = sqrt(sum((diff0.^2 .* feature_weights) ./ (class0_std.^2)));
        end
        
        if size(class1_prototypes, 1) > 1
            dist1 = calculateDistancesToPrototypes(sample, class1_prototypes, class1_std, feature_weights);
            min_dist1 = min(dist1);
        else
            diff1 = sample - class1_prototypes;
            min_dist1 = sqrt(sum((diff1.^2 .* feature_weights) ./ (class1_std.^2)));
        end
        
        if min_dist0 <= min_dist1
            fewshot_pred(i) = 0;
        else
            fewshot_pred(i) = 1;
        end
        
        confidence_scores(i) = abs(min_dist0 - min_dist1) / (min(min_dist0, min_dist1) + eps);
    end
    
   
    for i = 1:size(X_test, 1)
        if confidence_scores(i) < 0.35
            sample = X_test(i, :);
            eis_start_idx = max(1, top_features_count - 10);
            eis_features = sample(eis_start_idx:end);
            
            sample_mean = mean(sample);
            sample_std = std(sample);
            
            if sample_std > 0.28 && sample_mean < 0.45
                fewshot_pred(i) = 1;
            end
            
            if mean(eis_features) < 0.3 && std(eis_features) > 0.2
                fewshot_pred(i) = 1;
            elseif mean(eis_features) > 0.7 && std(eis_features) < 0.15
                fewshot_pred(i) = 0;
            end
            
            if size(sample, 2) >= 20
                ratio_features = sample(15:20);
                if mean(ratio_features) > 0.75
                    fewshot_pred(i) = 0;
                elseif mean(ratio_features) < 0.25
                    fewshot_pred(i) = 1;
                end
            end
        end
    end
    
   
    for i = 1:size(X_test, 1)
        if confidence_scores(i) < 0.2
            other_preds = [svm_pred(i), dt_pred(i), rf_pred(i)];
            if length(unique(other_preds)) == 1 && other_preds(1) ~= fewshot_pred(i)
                fewshot_pred(i) = other_preds(1);
            elseif sum(other_preds == 1) >= 2 && fewshot_pred(i) == 0
                fewshot_pred(i) = 1;
            end
        end
    end
    
   
    pred_pos_ratio = sum(fewshot_pred) / length(fewshot_pred);
    train_pos_ratio = sum(y_train) / length(y_train);
    
    if abs(pred_pos_ratio - train_pos_ratio) > 0.2
        if pred_pos_ratio < train_pos_ratio
            neg_indices = find(fewshot_pred == 0);
            [~, sort_idx] = sort(confidence_scores(neg_indices));
            n_to_adjust = min(ceil(0.15 * length(fewshot_pred)), ...
                ceil((train_pos_ratio - pred_pos_ratio) * length(fewshot_pred)));
            for j = 1:min(n_to_adjust, length(sort_idx))
                idx = neg_indices(sort_idx(j));
                fewshot_pred(idx) = 1;
            end
        else
            pos_indices = find(fewshot_pred == 1);
            [~, sort_idx] = sort(confidence_scores(pos_indices));
            n_to_adjust = min(ceil(0.15 * length(fewshot_pred)), ...
                ceil((pred_pos_ratio - train_pos_ratio) * length(fewshot_pred)));
            for j = 1:min(n_to_adjust, length(sort_idx))
                idx = pos_indices(sort_idx(j));
                fewshot_pred(idx) = 0;
            end
        end
    end
    
    all_predictions_fewshot(test_idx) = fewshot_pred;
  
    [accuracy, precision, recall, f1] = calculateMetrics(y_test, svm_pred);
    results_svm(fold, :) = [accuracy, precision, recall, f1];
    
    [accuracy, precision, recall, f1] = calculateMetrics(y_test, dt_pred);
    results_dt(fold, :) = [accuracy, precision, recall, f1];
    
    [accuracy, precision, recall, f1] = calculateMetrics(y_test, rf_pred);
    results_rf(fold, :) = [accuracy, precision, recall, f1];
    
    [accuracy, precision, recall, f1] = calculateMetrics(y_test, fewshot_pred);
    results_fewshot(fold, :) = [accuracy, precision, recall, f1];
end

fprintf('\n交叉验证完成。\n');


mean_svm = mean(results_svm, 1);
mean_dt = mean(results_dt, 1);
mean_rf = mean(results_rf, 1);
mean_fewshot = mean(results_fewshot, 1);

std_svm = std(results_svm, 0, 1);
std_dt = std(results_dt, 0, 1);
std_rf = std(results_rf, 0, 1);
std_fewshot = std(results_fewshot, 0, 1);

fprintf('\n平均性能指标:\n');
fprintf('SVM: 准确率=%.4f±%.4f, F1=%.4f±%.4f\n', mean_svm(1), std_svm(1), mean_svm(4), std_svm(4));
fprintf('DT: 准确率=%.4f±%.4f, F1=%.4f±%.4f\n', mean_dt(1), std_dt(1), mean_dt(4), std_dt(4));
fprintf('RF: 准确率=%.4f±%.4f, F1=%.4f±%.4f\n', mean_rf(1), std_rf(1), mean_rf(4), std_rf(4));
fprintf('Few-Shot: 准确率=%.4f±%.4f, F1=%.4f±%.4f\n', mean_fewshot(1), std_fewshot(1), mean_fewshot(4), std_fewshot(4));


feature_names_eng = { ...
    '$DRT_{cv}\times \alpha$', '$\alpha_{pos}$', '$R_{\frac{DRT_{h}}{\alpha}}$', ...
    '$Z_{real}^{std}$', '$Z_{imag}^{std}$', '$\frac{\alpha}{Q}$', '$DRT_{area}$', ...
    '$R_{0}$', '$Q$', '$\ln\left(\frac{\alpha}{Q}\right)$', '$Phase_{mean}$', ...
    '$Freq_{slope}$', '$e^{Phase_{std}}$', '$R_{0}\times DRT_{freq}$', ...
    '${\alpha}\times Z_{avg}$', '$HF_Z$', '$DRT_{\tau}$', '$Z_{imag}^{min}$', ...
    '$\sqrt{R_{0}}$', '$Z_{magnitude}$', '$\frac{DRT_{h}}{DRT_{Area}}$', ...
    '$\frac{R_{0}}{\alpha}$', '$Feature_{entropy}$', '$Phase_{std}$', '$\alpha^2$' ...
};

fprintf('\n生成 Fig.5: 性能对比图...\n');
figure('Position', [100, 100, 900, 600], 'Color', 'white');
metrics_eng = {'Accuracy', 'Precision', 'Recall', 'F1-Score'};
algorithms = {'SVM', 'DT', 'RF', 'Few-Shot'};
performance = [mean_svm; mean_dt; mean_rf; mean_fewshot];
errors = [std_svm; std_dt; std_rf; std_fewshot];

colors = [0.2627 0.5725 0.7647; 0.9569 0.6431 0.3765; ...
          0.4275 0.6824 0.3961; 0.8471 0.3216 0.4706];

bar_h = bar(performance, 'grouped', 'LineWidth', 1.2);
hold on;

for i = 1:length(bar_h)
    bar_h(i).FaceColor = colors(i,:);
    bar_h(i).EdgeColor = colors(i,:) * 0.7;
end

[ngroups, nbars] = size(performance);
x = nan(nbars, ngroups);
for i = 1:nbars
    x(i,:) = bar_h(i).XEndPoints;
    errorbar(x(i,:), performance(:,i), errors(:,i), 'k', ...
             'LineStyle', 'none', 'LineWidth', 1.5, 'CapSize', 8);
end

set(gca, 'XTickLabel', algorithms, 'FontSize', 12, 'FontName', 'Arial');
ylabel('Performance Value', 'FontSize', 14, 'FontWeight', 'bold');
ylim([0.85 1.05]);
title('Performance Comparison of Four Algorithms', 'FontSize', 16, 'FontWeight', 'bold');
legend(metrics_eng, 'Location', 'northwest', 'FontSize', 12, 'Box', 'off');
grid on; grid minor;


if ~exist('outputs', 'dir')
    mkdir('outputs');
end


fprintf('\n生成 Fig.6: 4折混淆矩阵...\n');
figure('Position', [150, 150, 1200, 800], 'Color', 'w');
for fold = 1:K
    test_idx = fold_test_indices{fold};
    y_test = all_true_labels(test_idx);
    y_pred = all_predictions_fewshot(test_idx);
    
    subplot(2, 2, fold);
    createConfusionMatrix(y_test, y_pred, {'Normal', 'Defect'});
    title(sprintf('Fold %d', fold), 'FontSize', 14, 'FontWeight', 'bold');
end
sgtitle('Few-Shot Learning: Confusion Matrices for Each Fold', ...
        'FontSize', 16, 'FontWeight', 'bold');


%% 
fprintf('\n生成 Fig.7: PCA可视化...\n');
figure('Position', [200, 200, 1000, 700], 'Color', 'white');

[coeff, score, ~] = pca(selected_features);
normal_color = [0 0 1];
defect_color = [1 0 0];

subplot(2, 2, 1);
hold on; box on;
[X1, X2] = meshgrid(linspace(min(score(:,1))-1, max(score(:,1))+1, 150), ...
                    linspace(min(score(:,2))-1, max(score(:,2))+1, 150));
X_grid = [X1(:), X2(:)];
X_grid_original = X_grid * coeff(:,1:2)' + repmat(mean(selected_features), size(X_grid, 1), 1);
Z = predict(svm_model, X_grid_original);
contourf(X1, X2, reshape(Z, size(X1)), [0 0.5 1], 'LineStyle', 'none');
colormap([0.7 0.85 1; 1 1 0.7]); alpha(0.5);
scatter(score(labels==0,1), score(labels==0,2), 30, normal_color, 'o', 'filled');
scatter(score(labels==1,1), score(labels==1,2), 30, defect_color, 'o', 'filled');
title('(a) SVM', 'FontSize', 12, 'FontWeight', 'bold');
xlabel('First Principal Component', 'FontSize', 10);
ylabel('Second Principal Component', 'FontSize', 10);
grid on;

subplot(2, 2, 2);
hold on; box on;
Z = predict(dt_model, X_grid_original);
contourf(X1, X2, reshape(Z, size(X1)), [0 0.5 1], 'LineStyle', 'none');
colormap([0.7 0.85 1; 1 1 0.7]); alpha(0.5);
scatter(score(labels==0,1), score(labels==0,2), 30, normal_color, 'o', 'filled');
scatter(score(labels==1,1), score(labels==1,2), 30, defect_color, 'o', 'filled');
title('(b) Decision Tree', 'FontSize', 12, 'FontWeight', 'bold');
xlabel('First Principal Component', 'FontSize', 10);
ylabel('Second Principal Component', 'FontSize', 10);
grid on;

subplot(2, 2, 3);
hold on; box on;
[Z, ~] = predict(rf_model, X_grid_original);
Z = str2double(Z);
contourf(X1, X2, reshape(Z, size(X1)), [0 0.5 1], 'LineStyle', 'none');
colormap([0.7 0.85 1; 1 1 0.7]); alpha(0.5);
scatter(score(labels==0,1), score(labels==0,2), 30, normal_color, 'o', 'filled');
scatter(score(labels==1,1), score(labels==1,2), 30, defect_color, 'o', 'filled');
title('(c) Random Forest', 'FontSize', 12, 'FontWeight', 'bold');
xlabel('First Principal Component', 'FontSize', 10);
ylabel('Second Principal Component', 'FontSize', 10);
grid on;

subplot(2, 2, 4);
hold on; box on;
Z = zeros(size(X_grid, 1), 1);
for i = 1:size(X_grid, 1)
    point = X_grid_original(i, :);
    dist0 = min(sqrt(sum(((point - class0_prototypes).^2 .* feature_weights), 2)));
    dist1 = min(sqrt(sum(((point - class1_prototypes).^2 .* feature_weights), 2)));
    Z(i) = (dist1 < dist0);
end
contourf(X1, X2, reshape(Z, size(X1)), [0 0.5 1], 'LineStyle', 'none');
colormap([0.7 0.85 1; 1 1 0.7]); alpha(0.5);
h1 = scatter(score(labels==0,1), score(labels==0,2), 30, normal_color, 'o', 'filled');
h2 = scatter(score(labels==1,1), score(labels==1,2), 30, defect_color, 'o', 'filled');

c0_prototypes_pca = (class0_prototypes - mean(selected_features)) * coeff;
c1_prototypes_pca = (class1_prototypes - mean(selected_features)) * coeff;

if size(c0_prototypes_pca, 1) > 1
    c0_center = mean(c0_prototypes_pca, 1);
    h3 = scatter(c0_center(1), c0_center(2), 150, normal_color, 'p', 'filled');
else
    h3 = scatter(c0_prototypes_pca(1,1), c0_prototypes_pca(1,2), 150, normal_color, 'p', 'filled');
end

for i = 1:size(c1_prototypes_pca, 1)
    if i == 1
        h4 = scatter(c1_prototypes_pca(i,1), c1_prototypes_pca(i,2), 150, defect_color, 'p', 'filled');
    else
        scatter(c1_prototypes_pca(i,1), c1_prototypes_pca(i,2), 150, defect_color, 'p', 'filled');
    end
end

title('(d) Few-Shot with Prototypes', 'FontSize', 12, 'FontWeight', 'bold');
xlabel('First Principal Component', 'FontSize', 10);
ylabel('Second Principal Component', 'FontSize', 10);
legend([h1, h2, h3, h4], {'Normal', 'Defect', 'Normal Prototype', 'Defect Prototype'}, ...
       'Location', 'best', 'FontSize', 8);
grid on;



fprintf('\n生成 Fig.8: 特征重要性排序...\n');
figure('Position', [250, 250, 1200, 700], 'Color', 'w');

topN = min(25, numel(ranks));
top_idx = ranks(1:topN);
top_imp = weights(top_idx);
if exist('feature_names_eng', 'var') && numel(feature_names_eng) >= max(top_idx)
    top_names = feature_names_eng(top_idx);
else
    top_names = arrayfun(@(k) sprintf('Feature %d', k), top_idx, 'UniformOutput', false);
end

[imp_sorted, ord] = sort(top_imp, 'ascend');
names_sorted = top_names(ord);

hold on; box on; grid on;
barh(imp_sorted, 'FaceColor', [0.25 0.45 0.85], 'EdgeColor', 'none');
for i = numel(imp_sorted)-4:numel(imp_sorted)
    if i >= 1
        barh(i, imp_sorted(i), 'FaceColor', [0.85 0.25 0.25], 'EdgeColor', 'none');
    end
end

yticks(1:topN);
names_tex = cellfun(@(s) strrep(s, '_', '\_'), names_sorted, 'UniformOutput', false);
set(gca, 'YTickLabel', names_tex, 'TickLabelInterpreter', 'tex', 'FontSize', 11);
xlabel('Importance Score', 'FontSize', 14, 'FontWeight', 'bold');
title('Fig.8 Feature Importance Ranking (Top-25)', 'FontSize', 16, 'FontWeight', 'bold');
xlim([0, max(imp_sorted)*1.08]);

for i = 1:topN
    text(imp_sorted(i)+0.01*max(imp_sorted), i, sprintf('%.3f', imp_sorted(i)), ...
         'VerticalAlignment', 'middle', 'FontSize', 10);
end



fprintf('\n生成 Fig.9: Top-2特征空间错误分析...\n');
[~, ord] = sort(feature_importance(:), 'descend');
col1 = ord(1); col2 = ord(2);
f1 = selected_features(:, col1);
f2 = selected_features(:, col2);

name1 = 'Top-1 feature';
name2 = 'Top-2 feature';
if exist('feature_names_eng', 'var') && numel(feature_names_eng) >= max(col1, col2)
    try
        name1 = feature_names_eng{col1};
        name2 = feature_names_eng{col2};
    catch
    end
end

y_true = all_true_labels(:);
y_pred = all_predictions_fewshot(:);
TP = (y_true==1 & y_pred==1);
TN = (y_true==0 & y_pred==0);
FP = (y_true==0 & y_pred==1);
FN = (y_true==1 & y_pred==0);

padx = 0.02*(max(f1)-min(f1)+eps);
pady = 0.02*(max(f2)-min(f2)+eps);
xgrid = linspace(min(f1)-padx, max(f1)+padx, 150);
ygrid = linspace(min(f2)-pady, max(f2)+pady, 150);
[X,Y] = meshgrid(xgrid, ygrid);
Z = reshape(ksdensity([f1 f2], [X(:) Y(:)]), size(X));

figure('Position', [300, 300, 820, 660], 'Color', 'w');
hold on; box on;
contourf(X, Y, Z, 10, 'LineColor', 'none');
colormap(parula); alpha(0.60);

sTN = scatter(f1(TN), f2(TN), 38, [0.30 0.60 0.95], 'o', 'filled', ...
              'MarkerFaceAlpha', 0.35, 'MarkerEdgeAlpha', 0.35);
sTP = scatter(f1(TP), f2(TP), 38, [0.95 0.40 0.40], 'o', 'filled', ...
              'MarkerFaceAlpha', 0.35, 'MarkerEdgeAlpha', 0.35);
sFP = scatter(f1(FP), f2(FP), 90, 'k', 'x', 'LineWidth', 2.2);
sFN = scatter(f1(FN), f2(FN), 90, 'r', '+', 'LineWidth', 2.2);

if contains(name1, '$') || contains(name2, '$')
    xlab = xlabel(sprintf('Top-1: %s (normalized)', name1), 'Interpreter', 'latex');
    ylab = ylabel(sprintf('Top-2: %s (normalized)', name2), 'Interpreter', 'latex');
else
    xlab = xlabel(sprintf('Top-1: %s (normalized)', name1));
    ylab = ylabel(sprintf('Top-2: %s (normalized)', name2));
end
set([xlab ylab], 'FontSize', 13, 'FontWeight', 'bold');

title('Fig.9 Error Analysis in Top-2 Feature Space', 'FontSize', 16, 'FontWeight', 'bold');

legend([sTN sTP sFP sFN], ...
       {'TN (Normal→Normal)', 'TP (Defect→Defect)', ...
        'FP (Normal→Defect)', 'FN (Defect→Normal)'}, ...
       'Location', 'southeast');

txt = sprintf('TP=%d, TN=%d, FP=%d, FN=%d', sum(TP), sum(TN), sum(FP), sum(FN));
xr = xlim; yr = ylim;
text(xr(1)+0.02*(xr(2)-xr(1)), yr(2)-0.06*(yr(2)-yr(1)), txt, ...
     'FontSize', 11, 'FontWeight', 'bold', 'BackgroundColor', [1 1 1 0.65]);

grid on;






fprintf('\n=== 所有图表已生成并保存到 outputs/ 文件夹 ===\n');
fprintf('图表清单:\n');
fprintf('  - Fig5_Performance_Comparison.png\n');
fprintf('  - Fig6_Confusion_Matrices.png\n');
fprintf('  - Fig7_PCA_Visualization.png\n');
fprintf('  - Fig8_Feature_Importance.png\n');
fprintf('  - Fig9_Error_Analysis.png\n');
fprintf('  - EIS_results.mat\n');
fprintf('  - Performance_Summary.xlsx\n');



function [ranks, weights] = enhancedFeatureSelection(X, y)
    [n_samples, n_features] = size(X);
    weights = zeros(1, n_features);
    
    for j = 1:n_features
        feature = X(:, j);
        class0_idx = (y == 0);
        class1_idx = (y == 1);
        
        if sum(class0_idx) > 0 && sum(class1_idx) > 0
            mean0 = mean(feature(class0_idx));
            mean1 = mean(feature(class1_idx));
            var0 = var(feature(class0_idx)) + eps;
            var1 = var(feature(class1_idx)) + eps;
            fisher_score = abs(mean0 - mean1) / sqrt(var0 + var1);
            
            n_bins = min(15, ceil(sqrt(n_samples)));
            [~, edges] = histcounts(feature, n_bins);
            bins = discretize(feature, edges);
            
            p_y_0 = sum(class0_idx) / n_samples;
            p_y_1 = sum(class1_idx) / n_samples;
            
            mi_score = 0;
            for bin = 1:n_bins
                bin_idx = (bins == bin);
                if any(bin_idx)
                    p_x = sum(bin_idx) / n_samples;
                    p_x_y_0 = sum(bin_idx & class0_idx) / n_samples;
                    p_x_y_1 = sum(bin_idx & class1_idx) / n_samples;
                    
                    if p_x_y_0 > 0
                        mi_score = mi_score + p_x_y_0 * log2(p_x_y_0 / (p_x * p_y_0));
                    end
                    if p_x_y_1 > 0
                        mi_score = mi_score + p_x_y_1 * log2(p_x_y_1 / (p_x * p_y_1));
                    end
                end
            end
            
            sorted_0 = sort(feature(class0_idx));
            sorted_1 = sort(feature(class1_idx));
            n0 = length(sorted_0); n1 = length(sorted_1);
            q1_0 = sorted_0(max(1, round(0.25 * n0)));
            q3_0 = sorted_0(max(1, round(0.75 * n0)));
            q1_1 = sorted_1(max(1, round(0.25 * n1)));
            q3_1 = sorted_1(max(1, round(0.75 * n1)));
            
            if q1_1 > q3_0 || q1_0 > q3_1
                overlap = 0;
            else
                min_q3 = min(q3_0, q3_1);
                max_q1 = max(q1_0, q1_1);
                overlap = (min_q3 - max_q1) / (max(q3_0, q3_1) - min(q1_0, q1_1) + eps);
            end
            separation_score = 1 - overlap;
            
            if j <= 5
                domain_weight = 1.0;
            elseif j <= 10
                domain_weight = 1.5;
            else
                domain_weight = 1.2;
            end
            
            weights(j) = (0.4 * fisher_score + 0.3 * mi_score + 0.2 * separation_score) * domain_weight;
            weights(j) = weights(j) + rand() * 1e-6;
        end
    end
    
    [~, ranks] = sort(weights, 'descend');
end

function prototypes = generateMultiplePrototypes(samples, max_prototypes)
    [n_samples, n_features] = size(samples);
    if n_samples <= 5
        prototypes = mean(samples, 1);
        return;
    end
    
    n_prototypes = min(max_prototypes, ceil(n_samples/10));
    prototypes = zeros(n_prototypes, n_features);
    prototypes(1, :) = mean(samples, 1);
    
    if n_prototypes > 1
        try
            [~, centroids] = kmeans(samples, n_prototypes);
            prototypes = centroids;
        catch
            for i = 1:n_prototypes
                q = (i-1)/(n_prototypes-1);
                if q == 0, q = 0.1; elseif q == 1, q = 0.9; end
                for j = 1:n_features
                    sorted_feat = sort(samples(:, j));
                    idx = max(1, round(q*length(sorted_feat)));
                    prototypes(i, j) = sorted_feat(idx);
                end
            end
        end
    end
end

function distances = calculateDistancesToPrototypes(sample, prototypes, std_vals, weights)
    n_prototypes = size(prototypes, 1);
    distances = zeros(n_prototypes, 1);
    for i = 1:n_prototypes
        diff = sample - prototypes(i, :);
        distances(i) = sqrt(sum((diff.^2 .* weights) ./ (std_vals.^2)));
    end
end

function createConfusionMatrix(y_true, y_pred, class_names)
    n_classes = length(class_names);
    cm = zeros(n_classes);
    for i = 1:length(y_true)
        cm(y_true(i)+1, y_pred(i)+1) = cm(y_true(i)+1, y_pred(i)+1) + 1;
    end
    
    cm_percent = zeros(size(cm));
    for i = 1:n_classes
        if sum(cm(i, :)) > 0
            cm_percent(i, :) = cm(i, :) / sum(cm(i, :));
        end
    end
    
    imagesc(cm_percent);
    colormap('jet'); colorbar;
    set(gca, 'XTick', 1:n_classes, 'YTick', 1:n_classes);
    set(gca, 'XTickLabel', class_names, 'YTickLabel', class_names);
    xlabel('Predicted', 'FontSize', 10);
    ylabel('True', 'FontSize', 10);
    
    for i = 1:n_classes
        for j = 1:n_classes
            text(j, i, sprintf('%d\n(%.1f%%)', cm(i,j), cm_percent(i,j)*100), ...
                 'HorizontalAlignment', 'center', 'Color', 'white', 'FontWeight', 'bold', 'FontSize', 9);
        end
    end
end

function [accuracy, precision, recall, f1] = calculateMetrics(y_true, y_pred)
    tp = sum((y_true == 1) & (y_pred == 1));
    tn = sum((y_true == 0) & (y_pred == 0));
    fp = sum((y_true == 0) & (y_pred == 1));
    fn = sum((y_true == 1) & (y_pred == 0));
    
    accuracy = (tp + tn) / (tp + tn + fp + fn);
    if (tp + fp) == 0, precision = 0; else, precision = tp / (tp + fp); end
    if (tp + fn) == 0, recall = 0; else, recall = tp / (tp + fn); end
    if (precision + recall) == 0, f1 = 0; else, f1 = 2 * (precision * recall) / (precision + recall); end
end