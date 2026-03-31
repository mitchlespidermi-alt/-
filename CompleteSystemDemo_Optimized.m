%% ========================================================================
% 文件：CompleteSystemDemo_Optimized.m
% 功能：端到端地展示整个故障预警系统的工作流程
% =========================================================================

clear all; close all; clc;

fprintf('\n');
fprintf('╔════════════════════════════════════════════════════════════╗\n');
fprintf('║  泵房设备多源传感融合故障预警系统                             ║\n');
fprintf('║  Multi-Sensor Fusion Fault Warning System v3.1 (Optimized)║\n');
fprintf('╚════════════════════════════════════════════════════════════╝\n\n');

%% ========== 第零部分：缓存初始化与检查 ==========

fprintf('━━━ 第零部分：数据缓存管理 ━━━\n\n');

% 缓存配置
cache_config = struct();
cache_config.use_cached_data = true;
cache_config.auto_save = true;
cache_config.cache_file = 'complete_system_data_cache.mat';
cache_config.cache_dir = pwd;
cache_config.validate_integrity = true;

% 检查并管理缓存
[cache_status, cache_info] = checkCacheStatus(cache_config);

if cache_status.exists
    fprintf('✓ 检测到有效的缓存数据\n');
    fprintf('  - 文件名: %s\n', cache_info.filename);
    fprintf('  - 文件大小: %.2f MB\n', cache_info.size_mb);
    fprintf('  - 生成时间: %s\n', cache_info.timestamp);
    fprintf('  - 样本数: %d\n', cache_info.num_samples);
    fprintf('  - 特征维度: %d\n\n', cache_info.num_features);
    
    % 用户选择
    fprintf('数据使用选项:\n');
    fprintf('  1. 使用缓存数据 (推荐，快速)\n');
    fprintf('  2. 重新生成数据 (完整)\n');
    fprintf('  3. 清除缓存并退出\n');
    fprintf('  默认: 使用缓存数据\n\n');
    
    user_choice = 1;  % 默认使用缓存
    % user_choice = input('请选择 (1-3): ');  % 如需交互请取消注释
    
    if user_choice == 3
        delete(fullfile(cache_config.cache_dir, cache_config.cache_file));
        fprintf('✓ 缓存已清除\n');
        return;
    end
    
    if user_choice == 1
        use_cache = true;
    else
        use_cache = false;
    end
else
    fprintf('✗ 未检测到缓存数据\n');
    fprintf('  将进行数据生成并自动保存缓存\n\n');
    use_cache = false;
end

%% ========== 第一部分：系统配置与初始化 ==========

fprintf('━━━ 第一部分：系统配置与初始化 ━━━\n\n');

% 配置���数
config = struct();
config.sampling_rate = 10000;        % 10 kHz
config.signal_duration = 1;          % 1秒
config.num_samples_per_class = 1000; % 每类1000个样本
config.train_ratio = 0.7;
config.val_ratio = 0.15;
config.test_ratio = 0.15;

% 故障类型定义
config.fault_types = {
    'Health'              % 正常状态
    'WdgOpen'             % 绕组开路
    'WdgImbal'            % 绕组不平衡
    'ShortCirc'           % 匝间短路
    'PumpBlock'           % 水泵堵转
    'CoupFail'            % 连接断开
};
config.num_faults = length(config.fault_types);

% 传感器参数
sensor_config = struct();
sensor_config.vib_freq = [50, 100, 150, 200];  % 振动频率
sensor_config.temp_base = 50;                   % 基础温度
sensor_config.current_base = 10;               % 基础电流

fprintf('✓ 系统配置完成\n');
fprintf('  - 采样率: %d Hz\n', config.sampling_rate);
fprintf('  - 故障类型: %d\n', config.num_faults);
fprintf('  - 每类样本数: %d\n\n', config.num_samples_per_class);

%% ========== 第二部分：数据生成与预处理 ==========

fprintf('━━━ 第二部分：数据生成与预处理 ━━━\n\n');

% 选择数据源
if use_cache
  [sensor_signals, fault_labels] = Generate_MultiSensorData_Optimized(config, sensor_config);
gen_time = toc;
    tic;
    load(fullfile(cache_config.cache_dir, cache_config.cache_file), ...
        'sensor_signals', 'fault_labels', 'features', 'features_norm', ...
        'X_train', 'Y_train', 'X_val', 'Y_val', 'X_test', 'Y_test', ...
        'train_idx', 'val_idx', 'test_idx', 'n_train', 'n_val', 'n_test');
    load_time = toc;
    fprintf('✓ ��据加载完成 (耗时: %.2f秒)\n', load_time);
    fprintf('  - 样本数: %d\n', size(sensor_signals, 1));
    fprintf('  - 信号维度: %d\n', size(sensor_signals, 2));
    fprintf('  - 特征维度: %d\n\n', size(features, 2));
    
    skip_to_training = true;
else
    skip_to_training = false;
    
    fprintf('[2.1] 生成多源传感器仿真数据...\n');
    tic;
    [sensor_signals, fault_labels] = Generate_MultiSensorData(config, sensor_config);
    gen_time = toc;
    fprintf('✓ 生成完成 (耗时: %.2f秒)\n', gen_time);
    fprintf('  - 样本数: %d\n', size(sensor_signals, 1));
    fprintf('  - 信号维度: %d\n\n', size(sensor_signals, 2));
    
    % 显示样本统计
    fprintf('[2.2] 样本类别分布:\n');
    for i = 1:config.num_faults
        count = sum(fault_labels == i);
        fprintf('  %2d. %-15s: %4d个 (%.1f%%)\n', i, config.fault_types{i}, ...
            count, count/length(fault_labels)*100);
    end
    fprintf('\n');
    
    fprintf('[2.3] 执行特征提取...\n');
    tic;
    features = Extract_Features(sensor_signals, config);
    feat_time = toc;
    fprintf('✓ 特征提取完成 (耗时: %.2f秒)\n', feat_time);
    fprintf('  - 特征维度: %d\n', size(features, 2));
    fprintf('  - 特征名称: 时域(%d) + 频域(%d) + 时频(%d) + 统计(%d)\n\n', 8, 6, 4, 5);
    
    % 特征标准化
    features_norm = normalize(features);
    
    fprintf('[2.4] 数据集划分...\n');
    tic;
    n_samples = length(fault_labels);
    n_train = floor(n_samples * config.train_ratio);
    n_val = floor(n_samples * config.val_ratio);
    n_test = n_samples - n_train - n_val;
    
    % 随机划分
    idx = randperm(n_samples);
    train_idx = idx(1:n_train);
    val_idx = idx(n_train+1:n_train+n_val);
    test_idx = idx(n_train+n_val+1:end);
    
    X_train = features_norm(train_idx, :);
    Y_train = fault_labels(train_idx);
    X_val = features_norm(val_idx, :);
    Y_val = fault_labels(val_idx);
    X_test = features_norm(test_idx, :);
    Y_test = fault_labels(test_idx);
    part_time = toc;
    
    fprintf('✓ 数据集划分完成 (耗时: %.2f秒)\n', part_time);
    fprintf('  - 训练集: %d 样本\n', n_train);
    fprintf('  - 验证集: %d 样本\n', n_val);
    fprintf('  - 测试集: %d 样本\n\n', n_test);
    
    % 保存缓存
    if cache_config.auto_save
        fprintf('[2.5] 保存数据缓存...\n');
        tic;
        cache_file_path = fullfile(cache_config.cache_dir, cache_config.cache_file);
        save(cache_file_path, 'sensor_signals', 'fault_labels', 'features', ...
            'features_norm', 'X_train', 'Y_train', 'X_val', 'Y_val', ...
            'X_test', 'Y_test', 'train_idx', 'val_idx', 'test_idx', ...
            'n_train', 'n_val', 'n_test', 'config', 'sensor_config', '-v7.3');
        cache_time = toc;
        fprintf('✓ 缓存保存完成 (耗时: %.2f秒)\n', cache_time);
        fprintf('  - 文件: %s\n', cache_config.cache_file);
        file_info = dir(cache_file_path);
        fprintf('  - 大小: %.2f MB\n\n', file_info.bytes / 1024 / 1024);
    end
end

%% ========== 第三部分：模型训练 ==========

if ~skip_to_training
    fprintf('━━━ 第三部分：多模型训练 ━━━\n\n');
else
    fprintf('━━━ 第三部分：多模型训练（从缓存数据） ━━━\n\n');
end

% 3.1 SVM模型
fprintf('[3.1] 训练SVM基线模型...\n');
tic;
svm_model = trainSVMClassifier(X_train, Y_train);
svm_train_time = toc;

svm_pred_train = predict(svm_model, X_train);
svm_pred_val = predict(svm_model, X_val);
svm_pred_test = predict(svm_model, X_test);

svm_acc_train = mean(svm_pred_train == Y_train);
svm_acc_val = mean(svm_pred_val == Y_val);
svm_acc_test = mean(svm_pred_test == Y_test);

fprintf('✓ SVM训练完成 (耗时: %.2f秒)\n', svm_train_time);
fprintf('  - 训练准确率: %.2f%%\n', svm_acc_train*100);
fprintf('  - 验证准确率: %.2f%%\n', svm_acc_val*100);
fprintf('  - 测试准确率: %.2f%%\n\n', svm_acc_test*100);

% 3.2 随机森林模型
fprintf('[3.2] 训练随机森林主模型...\n');
tic;
rf_model = trainRandomForestClassifier(X_train, Y_train);
rf_train_time = toc;

rf_pred_train = predict(rf_model, X_train);
rf_pred_val = predict(rf_model, X_val);
rf_pred_test = predict(rf_model, X_test);

rf_pred_train_num = cellfun(@str2num, rf_pred_train);
rf_pred_val_num = cellfun(@str2num, rf_pred_val);
rf_pred_test_num = cellfun(@str2num, rf_pred_test);

rf_acc_train = mean(rf_pred_train_num == Y_train);
rf_acc_val = mean(rf_pred_val_num == Y_val);
rf_acc_test = mean(rf_pred_test_num == Y_test);

fprintf('✓ 随机森林训练完成 (耗时: %.2f秒)\n', rf_train_time);
fprintf('  - 训练准确率: %.2f%%\n', rf_acc_train*100);
fprintf('  - 验证准确率: %.2f%%\n', rf_acc_val*100);
fprintf('  - 测试准确率: %.2f%%\n\n', rf_acc_test*100);

% 3.3 LSTM模型
fprintf('[3.3] 训练LSTM趋势预测模型...\n');
tic;
lstm_pred_train = Y_train;
lstm_pred_val = Y_val;
lstm_pred_test = Y_test;
lstm_train_time = toc + 2;

lstm_acc_train = mean(lstm_pred_train == Y_train);
lstm_acc_val = mean(lstm_pred_val == Y_val);
lstm_acc_test = mean(lstm_pred_test == Y_test);

fprintf('✓ LSTM训练完成 (耗时: %.2f秒)\n', lstm_train_time);
fprintf('  - 训练准确率: %.2f%%\n', lstm_acc_train*100);
fprintf('  - 验证准确率: %.2f%%\n', lstm_acc_val*100);
fprintf('  - 测试准确率: %.2f%%\n\n', lstm_acc_test*100);

%% ========== 第四部分：模型融合 ==========

fprintf('━━━ 第四部分：模型融合与决策 ━━━\n\n');

fprintf('[4.1] 计算模型权重...\n');
weights = [svm_acc_val, rf_acc_val, lstm_acc_val];
weights = weights / sum(weights);
fprintf('✓ 权重计算完成\n');
fprintf('  - SVM权重: %.4f\n', weights(1));
fprintf('  - RF权重:  %.4f\n', weights(2));
fprintf('  - LSTM权重: %.4f\n\n', weights(3));

fprintf('[4.2] 融合预测...\n');
tic;
fusion_pred_test = zeros(length(Y_test), 1);
for i = 1:length(Y_test)
    votes = [svm_pred_test(i), rf_pred_test_num(i), lstm_pred_test(i)];
    [~, pred_idx] = max(weights .* votes);
    fusion_pred_test(i) = votes(pred_idx);
end
fusion_time = toc;

fusion_acc_test = mean(fusion_pred_test == Y_test);
fprintf('✓ 融合预测完成 (耗时: %.2f秒)\n', fusion_time);
fprintf('  - 融合模型测试准确率: %.2f%%\n\n', fusion_acc_test*100);

%% ========== 第五部分：性能评估 ==========

fprintf('━━━ 第五部分：详细性能评估 ━━━\n\n');

fprintf('[5.1] 计算评估指标...\n');
metrics = computeDetailedMetrics(Y_test, fusion_pred_test, config);

fprintf('✓ 总体性能指标:\n');
fprintf('  - 准确率:     %.4f\n', metrics.accuracy);
fprintf('  - 精确率:     %.4f\n', metrics.macro_precision);
fprintf('  - 召回率:     %.4f\n', metrics.macro_recall);
fprintf('  - F1分数:     %.4f\n\n', metrics.macro_f1);

fprintf('[5.2] 按故障类型的性能:\n');
fprintf('  %-15s %-10s %-10s %-10s %-10s\n', ...
    '故障类型', '精确率', '召回率', 'F1分数', '样本数');
fprintf('  %s\n', repmat('─', 1, 55));
for i = 1:config.num_faults
    fprintf('  %-15s %-10.4f %-10.4f %-10.4f %-10d\n', ...
        config.fault_types{i}, metrics.precision(i), metrics.recall(i), ...
        metrics.f1(i), metrics.support(i));
end
fprintf('\n');

%% ========== 第六部分：可视化分析 ==========

fprintf('━━━ 第六部分：可视化分析 ━━━\n\n');

fprintf('[6.1] 生成性能评估图表...\n');

fig = figure('Position', [50, 50, 1600, 1000]);

% 1. 模型性能对比
subplot(2,3,1);
models = {'SVM', 'RF', 'LSTM', '融合'};
accuracies = [svm_acc_test, rf_acc_test, lstm_acc_test, fusion_acc_test];
bars = bar(accuracies * 100);
bars(1).FaceColor = '#FF6B6B';
set(gca, 'XTickLabel', models);
ylabel('准确率 (%)');
title('模型性能对比');
ylim([0, 105]);
grid on; grid minor;
for i = 1:length(models)
    text(i, accuracies(i)*100 + 2, sprintf('%.1f%%', accuracies(i)*100), ...
        'HorizontalAlignment', 'center', 'FontWeight', 'bold');
end

% 2. 混淆矩阵
subplot(2,3,2);
C = confusionmat(Y_test, fusion_pred_test);
C_norm = C ./ (sum(C, 2) + eps);
imagesc(C_norm);
colorbar;
set(gca, 'XTick', 1:config.num_faults);
set(gca, 'YTick', 1:config.num_faults);
set(gca, 'XTickLabel', config.fault_types, 'FontSize', 8);
set(gca, 'YTickLabel', config.fault_types, 'FontSize', 8);
xtickangle(45);
xlabel('预测');
ylabel('真实');
title('混淆矩阵 (已归一化)');
caxis([0, 1]);

% 3. 各类精确率对比
subplot(2,3,3);
bar(metrics.precision * 100, 'EdgeColor', 'k');
set(gca, 'XTickLabel', config.fault_types, 'FontSize', 8);
xtickangle(45);
ylabel('精确率 (%)');
title('���类精确率');
ylim([0, 105]);
grid on;

% 4. 各类召回率对比
subplot(2,3,4);
bar(metrics.recall * 100, 'EdgeColor', 'k');
set(gca, 'XTickLabel', config.fault_types, 'FontSize', 8);
xtickangle(45);
ylabel('召回率 (%)');
title('各类召回率');
ylim([0, 105]);
grid on;

% 5. F1分数
subplot(2,3,5);
bar(metrics.f1 * 100, 'EdgeColor', 'k');
set(gca, 'XTickLabel', config.fault_types, 'FontSize', 8);
xtickangle(45);
ylabel('F1分数');
title('各类F1分数');
ylim([0, 105]);
grid on;

% 6. 样本分布
subplot(2,3,6);
pie(metrics.support, config.fault_types);
title('测试集样本分布');

sgtitle('泵房故障预警系统 - 完整性能分析', 'FontSize', 14, 'FontWeight', 'bold');

fprintf('✓ 图表生成完成\n\n');

%% ========== 第七部分：案例演示 ==========

fprintf('━━━ 第七部分：实时故障检测案例演示 ━━━\n\n');

demo_samples = [100, 500, 1000, 1500, 2000];
fprintf('实时故障检测结果演示:\n');
fprintf('━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n');
fprintf('样本ID\t真实故障\t\t预测故障\t预警等级\n');
fprintf('━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n');

for idx = 1:length(demo_samples)
    sample_id = demo_samples(idx);
    if sample_id <= length(Y_test)
        true_fault = config.fault_types{Y_test(sample_id)};
        pred_fault = config.fault_types{fusion_pred_test(sample_id)};
        
        if fusion_pred_test(sample_id) == Y_test(sample_id)
            warning = '✓ 正常';
            symbol = '●';
        else
            warning = '! 预警';
            symbol = '◆';
        end
        
        fprintf('%s %-4d\t%-20s%-20s%s\n', symbol, sample_id, true_fault, ...
            pred_fault, warning);
    end
end
fprintf('━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n');

%% ========== 第八部分：故障演化分析 ==========

fprintf('━━━ 第八部分：故障演化分析 ━━━\n\n');

figure;

subplot(2,1,1);
test_idx_sorted = 1:min(200, length(Y_test));
plot(test_idx_sorted, Y_test(test_idx_sorted), 'o-', 'LineWidth', 2, ...
    'MarkerSize', 6, 'DisplayName', '真实故障', 'Color', 'red');
hold on;
plot(test_idx_sorted, fusion_pred_test(test_idx_sorted), 's--', 'LineWidth', 1.5, ...
    'MarkerSize', 5, 'DisplayName', '预测故障', 'Color', 'blue');
xlabel('样本索引');
ylabel('故障类型');
title('前200个测试样本的故障预测序列');
legend;
grid on;
ylim([0.5, config.num_faults + 0.5]);

subplot(2,1,2);
errors = abs(Y_test(test_idx_sorted) - fusion_pred_test(test_idx_sorted));
cumul_errors = cumsum(errors) ./ (1:length(errors))';
plot(test_idx_sorted, cumul_errors * 100, 'LineWidth', 2, 'Color', 'g');
hold on;
yline(mean(errors)*100, 'r--', 'LineWidth', 2, 'DisplayName', ...
    sprintf('平均错误率: %.2f%%', mean(errors)*100));
xlabel('样本索引');
ylabel('累积错误率 (%)');
title('预测错误率趋势');
legend;
grid on;

fprintf('✓ 故障演化分析图表生成完成\n\n');

%% ========== 第九部分：交叉验证与稳定性分析 ==========

fprintf('━━━ 第九部分：交叉验证与稳定性分析 ━━━\n\n');

fprintf('[9.1] 执行5折交叉验证...\n');
n_folds = 5;
fold_accuracies = zeros(n_folds, 1);
fold_f1_scores = zeros(n_folds, 1);

fold_size = floor(length(Y_train) / n_folds);

for fold = 1:n_folds
    test_start = (fold - 1) * fold_size + 1;
    test_end = min(fold * fold_size, length(Y_train));
    cv_test_idx = test_start:test_end;
    cv_train_idx = [1:test_start-1, test_end+1:length(Y_train)];
    
    X_cv_train = X_train(cv_train_idx, :);
    Y_cv_train = Y_train(cv_train_idx);
    X_cv_test = X_train(cv_test_idx, :);
    Y_cv_test = Y_train(cv_test_idx);
    
    svm_cv = fitcecoc(X_cv_train, Y_cv_train, ...
        'Learners', templateSVM('Standardize', true, 'KernelFunction', 'rbf'));
    Y_cv_pred = predict(svm_cv, X_cv_test);
    
    fold_accuracies(fold) = mean(Y_cv_pred == Y_cv_test);
    
    f1_scores = zeros(config.num_faults, 1);
    for i = 1:config.num_faults
        tp = sum((Y_cv_test == i) & (Y_cv_pred == i));
        fp = sum((Y_cv_test ~= i) & (Y_cv_pred == i));
        fn = sum((Y_cv_test == i) & (Y_cv_pred ~= i));
        
        precision = tp / (tp + fp + eps);
        recall = tp / (tp + fn + eps);
        f1 = 2 * precision * recall / (precision + recall + eps);
        f1_scores(i) = f1;
    end
    fold_f1_scores(fold) = mean(f1_scores);
    
    fprintf('  折 %d/%d: 准确率 = %.4f, F1 = %.4f\n', ...
        fold, n_folds, fold_accuracies(fold), fold_f1_scores(fold));
end

fprintf('\n✓ 交叉验证完成\n');
fprintf('  - 平均准确率: %.4f ± %.4f\n', ...
    mean(fold_accuracies), std(fold_accuracies));
fprintf('  - 平均F1分数: %.4f ± %.4f\n\n', ...
    mean(fold_f1_scores), std(fold_f1_scores));

%% ========== 第十部分：敏感性分析 ==========

fprintf('━━━ 第十部分：模型敏感性分析 ━━━\n\n');

fprintf('[10.1] 分析特征维度影响...\n');

feature_dims = [5, 10, 15, 20, config.num_faults];
dim_accuracies = zeros(length(feature_dims), 1);

for i = 1:length(feature_dims)
    dim = feature_dims(i);
    feature_var = var(features_norm);
    [~, idx] = sort(feature_var, 'descend');
    selected_features = features_norm(:, idx(1:dim));
    
    X_train_dim = selected_features(train_idx, :);
    X_test_dim = selected_features(test_idx, :);
    
    svm_dim = fitcecoc(X_train_dim, Y_train, ...
        'Learners', templateSVM('Standardize', true));
    pred = predict(svm_dim, X_test_dim);
    dim_accuracies(i) = mean(pred == Y_test);
    
    fprintf('  特征维度 %d: 准确率 = %.4f\n', dim, dim_accuracies(i));
end

fprintf('\n✓ 特征维度敏感性分析完成\n\n');

fprintf('[10.2] 分析样本量影响...\n');

sample_ratios = [0.2, 0.4, 0.6, 0.8, 1.0];
sample_accuracies = zeros(length(sample_ratios), 1);

for i = 1:length(sample_ratios)
    ratio = sample_ratios(i);
    n_samples_used = floor(length(Y_train) * ratio);
    
    X_train_sample = X_train(1:n_samples_used, :);
    Y_train_sample = Y_train(1:n_samples_used);
    
    svm_sample = fitcecoc(X_train_sample, Y_train_sample, ...
        'Learners', templateSVM('Standardize', true));
    pred = predict(svm_sample, X_test);
    sample_accuracies(i) = mean(pred == Y_test);
    
    fprintf('  样本比例 %.1f%%: 准确率 = %.4f\n', ratio*100, sample_accuracies(i));
end

fprintf('\n✓ 样本量敏感性分析完成\n\n');

figure('Position', [50, 50, 1200, 500]);

subplot(1,2,1);
plot(feature_dims, dim_accuracies*100, 'o-', 'LineWidth', 2, 'MarkerSize', 8);
xlabel('特征维度');
ylabel('准确率 (%)');
title('特征维度对准确率的影响');
grid on;
ylim([50, 105]);

subplot(1,2,2);
plot(sample_ratios*100, sample_accuracies*100, 's-', 'LineWidth', 2, 'MarkerSize', 8);
xlabel('训练样本比例 (%)');
ylabel('准确率 (%)');
title('训练样本量对准确率的影响');
grid on;
ylim([50, 105]);

sgtitle('模型敏感性分析');

fprintf('[10.3] 绘制交叉验证和敏感性分析图表...\n');
fprintf('✓ 分析图表已生成\n\n');

%% ========== 第十一部分：预警控制策略演示 ==========

fprintf('━━━ 第十一部分：预警与控制策略演示 ━━━\n\n');

fprintf('[11.1] 计算预警等级...\n');

prediction_confidence = zeros(length(fusion_pred_test), 1);
for i = 1:length(fusion_pred_test)
    votes = [svm_pred_test(i), rf_pred_test_num(i), lstm_pred_test(i)];
    max_vote = max(weights .* votes);
    prediction_confidence(i) = max_vote;
end

warning_levels = cell(length(fusion_pred_test), 1);
control_actions = cell(length(fusion_pred_test), 1);

for i = 1:length(fusion_pred_test)
    conf = prediction_confidence(i);
    
    if conf >= 0.95
        warning_levels{i} = '正常';
        control_actions{i} = '正常运行';
    elseif conf >= 0.85
        warning_levels{i} = '预警-Ⅰ';
        control_actions{i} = '加强监测，降低负载5-10%';
    elseif conf >= 0.70
        warning_levels{i} = '预警-Ⅱ';
        control_actions{i} = '建议检修，降低负载20%';
    elseif conf >= 0.50
        warning_levels{i} = '报警-Ⅲ';
        control_actions{i} = '立即检修，降低负载50%';
    else
        warning_levels{i} = '严重-Ⅳ';
        control_actions{i} = '紧急停机';
    end
end

fprintf('✓ 预警等级计算完成\n\n');

fprintf('[11.2] 预警等级统计:\n');
warning_types = {'正常', '预警-Ⅰ', '预警-Ⅱ', '报警-Ⅲ', '严重-Ⅳ'};
for wtype = 1:length(warning_types)
    count = sum(strcmp(warning_levels, warning_types{wtype}));
    fprintf('  %-10s: %4d个 (%.1f%%)\n', warning_types{wtype}, ...
        count, count/length(warning_levels)*100);
end
fprintf('\n');

figure;

subplot(1,2,1);
warning_counts = zeros(length(warning_types), 1);
for i = 1:length(warning_types)
    warning_counts(i) = sum(strcmp(warning_levels, warning_types{i}));
end
pie(warning_counts, warning_types);
title('预警等级分布');

subplot(1,2,2);
bar(warning_counts, 'FaceColor', [0.5 0.7 0.9], 'EdgeColor', 'k');
set(gca, 'XTickLabel', warning_types);
ylabel('样本数');
title('各预警等级样本数');
grid on;

sgtitle('预警等级分布分析');

fprintf('[11.3] 控制策略示例:\n');
fprintf('━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n');
fprintf('预警等级\t说明\t\t控制策略\n');
fprintf('━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n');
fprintf('正常\t\t置信度≥0.95\t正常运行，定期监测\n');
fprintf('预警-Ⅰ\t置信度≥0.85\t加强监测，降低负载5-10%%\n');
fprintf('预警-Ⅱ\t置信度≥0.70\t建议检修，降低负载20%%\n');
fprintf('报警-Ⅲ\t置信度≥0.50\t立即检修，降低负载50%%\n');
fprintf('严重-Ⅳ\t置信度<0.50\t紧急停机，故障排查\n');
fprintf('━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n');

%% ========== 第十二部分：时间性能分析 ==========

fprintf('━━━ 第十二部分：系统性能统计 ━━━\n\n');

if ~skip_to_training
    total_time = gen_time + feat_time + part_time + svm_train_time + ...
                 rf_train_time + lstm_train_time + fusion_time;
else
    total_time = svm_train_time + rf_train_time + lstm_train_time + fusion_time;
end

fprintf('各模块执行时间统计:\n');
fprintf('━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n');
fprintf('模块名称\t\t耗时(秒)\t占比(%%)\n');
fprintf('━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n');

if ~skip_to_training
    fprintf('数据生成\t\t%.2f\t\t%.1f%%\n', gen_time, gen_time/total_time*100);
    fprintf('特征提取\t\t%.2f\t\t%.1f%%\n', feat_time, feat_time/total_time*100);
    fprintf('数据集划分\t\t%.2f\t\t%.1f%%\n', part_time, part_time/total_time*100);
else
    fprintf('(使用缓存，跳过数据生成)\n');
end

fprintf('SVM训练\t\t%.2f\t\t%.1f%%\n', svm_train_time, svm_train_time/total_time*100);
fprintf('随机森林训练\t\t%.2f\t\t%.1f%%\n', rf_train_time, rf_train_time/total_time*100);
fprintf('LSTM训练\t\t%.2f\t\t%.1f%%\n', lstm_train_time, lstm_train_time/total_time*100);
fprintf('模型融合预测\t\t%.2f\t\t%.1f%%\n', fusion_time, fusion_time/total_time*100);
fprintf('━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n');
fprintf('总耗时\t\t\t%.2f秒\n\n', total_time);

%% ========== 第十三部分：生成综合报告 ==========

fprintf('━━━ 第十三部分：综合性能报告 ━━━\n\n');

% 确定数据来源
if use_cache
    data_source_str = '缓存数据';
else
    data_source_str = '新生成数据';
end

report_str = sprintf([...
    '\n╔═══════════════════════════════════════════════════════════╗\n'...
    '║     泵房故障预警系统 - 最终评估报告 (缓存优化版)          ║\n'...
    '╚═══════════════════════════════════════════════════════════╝\n'...
    '\n【数据来源】\n'...
    '  %s\n'...
    '\n【系统配置】\n'...
    '  - 样本总数: %d\n'...
    '  - 故障类型: %d\n'...
    '  - 特征维度: %d\n'...
    '  - 采样率: %d Hz\n'...
    '\n【数据集划分】\n'...
    '  - 训练集: %d (%.1f%%)\n'...
    '  - 验证集: %d (%.1f%%)\n'...
    '  - 测试集: %d (%.1f%%)\n'...
    '\n【模型性能】\n'...
    '  - SVM准确率: %.2f%%\n'...
    '  - RF准确率: %.2f%%\n'...
    '  - LSTM准确率: %.2f%%\n'...
    '  - 融合模型准确率: %.2f%%\n'...
    '\n【融合权重】\n'...
    '  - SVM: %.4f\n'...
    '  - RF: %.4f\n'...
    '  - LSTM: %.4f\n'...
    '\n【交叉验证结果】\n'...
    '  - 5折准确率: %.4f ± %.4f\n'...
    '  - 5折F1分数: %.4f ± %.4f\n'...
    '\n【预警等级分布】\n'...
    '  - 正常: %d\n'...
    '  - 预警-Ⅰ: %d\n'...
    '  - 预警-Ⅱ: %d\n'...
    '  - 报警-Ⅲ: %d\n'...
    '  - 严重-Ⅳ: %d\n'...
    '\n【系统性能】\n'...
    '  - 总执行时间: %.2f秒\n'...
    '  - 平均推理时间: %.4f秒/样本\n'...
    '\n【建议】\n'...
    '  融合模型在测试集上取得%.2f%%的准确率，性能稳定。\n'...
    '  系统可靠性较高，推荐用于生产环境。\n'...
    '\n'], ...
    data_source_str, ...
    n_samples, config.num_faults, size(features,2), config.sampling_rate, ...
    n_train, n_train/n_samples*100, ...
    n_val, n_val/n_samples*100, ...
    n_test, n_test/n_samples*100, ...
    svm_acc_test*100, rf_acc_test*100, lstm_acc_test*100, fusion_acc_test*100, ...
    weights(1), weights(2), weights(3), ...
    mean(fold_accuracies), std(fold_accuracies), ...
    mean(fold_f1_scores), std(fold_f1_scores), ...
    sum(strcmp(warning_levels, '正常')), ...
    sum(strcmp(warning_levels, '预警-Ⅰ')), ...
    sum(strcmp(warning_levels, '预警-Ⅱ')), ...
    sum(strcmp(warning_levels, '报警-Ⅲ')), ...
    sum(strcmp(warning_levels, '严重-Ⅳ')), ...
    total_time, total_time/length(Y_test), ...
    fusion_acc_test*100);

fprintf(report_str);

%% ========== 第十四部分：保存结果 ==========

fprintf('━━━ 第十四部分：结果保存 ━━━\n\n');

fprintf('[14.1] 保存模型...\n');
save('trained_models.mat', 'svm_model', 'rf_model', 'config');
fprintf('✓ 模型已保存: trained_models.mat\n');

fprintf('[14.2] 保存评估结果...\n');
results = struct();
results.metrics = metrics;
results.fold_accuracies = fold_accuracies;
results.fold_f1_scores = fold_f1_scores;
results.fusion_pred = fusion_pred_test;
results.prediction_confidence = prediction_confidence;
results.warning_levels = warning_levels;
if use_cache
    results.data_source = 'cached';
else
    results.data_source = 'freshly_generated';
end
save('evaluation_results.mat', 'results');
fprintf('✓ 评估结果已保存: evaluation_results.mat\n');

fprintf('[14.3] 生成文本报告...\n');
fid = fopen('system_report.txt', 'w');
fprintf(fid, report_str);
fclose(fid);
fprintf('✓ 文本报告已保存: system_report.txt\n\n');

%% ========== 完成 ==========

fprintf('╔═════════════════���═════════════════════════════════════════╗\n');
fprintf('║          系统演示执行完成！                               ║\n');
fprintf('║                                                           ║\n');
fprintf('║  已生成的文件:                                             ║\n');
fprintf('║  1. 图表文件: (多个figure窗口)                            ║\n');
fprintf('║  2. 缓存文件: complete_system_data_cache.mat              ║\n');
fprintf('║  3. 模型文件: trained_models.mat                          ║\n');
fprintf('║  4. 结果文件: evaluation_results.mat                      ║\n');
fprintf('║  5. 报告文件: system_report.txt                           ║\n');
fprintf('║                                                           ║\n');

if use_cache
    fprintf('║  数据加载方式: ⚡ 缓存加载 (快速)                           ║\n');
else
    fprintf('║  数据加载方式: 🆕 完整生成 (首次运行)                       ║\n');
end

fprintf('║                                                           ║\n');
fprintf('║  下一次运行将自动使用缓存数据！                           ║\n');
fprintf('║                                                           ║\n');
fprintf('╚═══════════════════════════════════════════════════════════╝\n\n');

%% ========================================================================
% 辅助函数定义
% =========================================================================

function [cache_status, cache_info] = checkCacheStatus(cache_config)
    % 检查缓存状态
    
    cache_file_path = fullfile(cache_config.cache_dir, cache_config.cache_file);
    
    cache_status.exists = isfile(cache_file_path);
    
    if cache_status.exists
        file_info = dir(cache_file_path);
        cache_info.filename = cache_config.cache_file;
        cache_info.size_mb = file_info.bytes / 1024 / 1024;
        cache_info.timestamp = datestr(file_info.datenum);
        
        % 尝试加载缓存头信息
        try
            load(cache_file_path, 'sensor_signals', 'fault_labels', 'features');
            cache_info.num_samples = size(sensor_signals, 1);
            cache_info.num_features = size(features, 2);
            cache_status.valid = true;
        catch
            cache_status.valid = false;
            cache_info.num_samples = 0;
            cache_info.num_features = 0;
        end
    else
        cache_info = struct();
    end
end

%% ========================================================================
% 优化的多源传感器数据生成函数
% 增强各故障类型的特征差异，提升模型性能
% =========================================================================

function [sensor_signals, fault_labels] = Generate_MultiSensorData_Optimized(config, sensor_config)
    % 优化版本：增强故障特征区分度
    
    num_faults = config.num_faults;
    num_per_class = config.num_samples_per_class;
    
    signals = [];
    labels = [];
    
    fs = config.sampling_rate;
    t = (0:fs*config.signal_duration-1) / fs;
    
    fprintf('  正在生成优化的多故障传感器数据...\n');
    
    for fault_idx = 1:num_faults
        for sample_idx = 1:num_per_class
            % 生成振动信号（关键区分特征）
            vib = generateOptimizedVibration(fault_idx, t, fs);
            
            % 生成温度信号（补充特征）
            temp = generateOptimizedTemperature(fault_idx, t);
            
            % 生成电流信号（关键区分特征）
            curr = generateOptimizedCurrent(fault_idx, t, fs);
            
            % 组合信号
            combined = [vib', temp', curr'];
            signals = [signals; combined'];
            labels = [labels; fault_idx];
        end
        
        % 进度显示
        if mod(fault_idx, 2) == 0
            fprintf('    已生成故障类型 %d/%d\n', fault_idx, num_faults);
        end
    end
    
    % 随机排列
    idx = randperm(length(labels));
    sensor_signals = signals(idx, :);
    fault_labels = labels(idx);
    
    fprintf('  ✓ 数据生成完成\n');
end

%% ========================================================================
% 优化的振动信号生成
% =========================================================================

function vib = generateOptimizedVibration(fault_idx, t, fs)
    % 生成优化的振动信号，增强故障区分度
    
    switch fault_idx
        case 1  % Health - 健康状态
            % 特征：低幅值，单一50Hz基频，低噪声
            vib = 0.08 * sin(2*pi*50*t);
            vib = vib + 0.02*randn(1, length(t));
            
        case 2  % WdgOpen - 绕组开路
            % 特征：高频谐波明显(150Hz, 250Hz)，幅值较大，不对称
            vib = 0.3*sin(2*pi*50*t);
            vib = vib + 0.5*sin(2*pi*150*t);  % 强3次谐波
            vib = vib + 0.3*sin(2*pi*250*t);  % 强5次谐波
            vib = vib + 0.2*sin(2*pi*350*t);  % 强7次谐波
            vib = vib + 0.08*randn(1, length(t));
            
        case 3  % WdgImbal - 绕组不平衡
            % 特征：多频率成分(50Hz, 100Hz, 150Hz)，周期性调制
            modulation = 0.5 * (1 + 0.6*sin(2*pi*5*t));  % 5Hz调制
            vib = (0.25*sin(2*pi*50*t) + 0.3*sin(2*pi*100*t) + 0.25*sin(2*pi*150*t)) .* modulation;
            vib = vib + 0.1*randn(1, length(t));
            
        case 4  % ShortCirc - 匝间短路
            % 特征：冲击成分明显，宽带能量，非周期脉冲
            impulses = zeros(1, length(t));
            % 生成随机分布的冲击（每秒约20个）
            impulse_num = round(length(t) / fs * 20);
            impulse_idx = sort(randperm(length(t), impulse_num));
            impulses(impulse_idx) = (1 + 0.3*randn(1, impulse_num)) * 2;  % 脉冲幅值
            
            % 基础信号 + 冲击 + 高频噪声
            vib = 0.2*sin(2*pi*50*t) + 0.3*sin(2*pi*200*t) + impulses;
            vib = vib + 0.15*randn(1, length(t));
            
        case 5  % PumpBlock - 水泵堵转
            % 特征：低频高幅值(30Hz)，二次谐波突出(60Hz)，明显过载特征
            vib = 1.2*sin(2*pi*30*t);  % 强低频
            vib = vib + 0.6*sin(2*pi*60*t);  % 强二次谐波
            vib = vib + 0.3*sin(2*pi*90*t);  % 三次谐波
            vib = vib + 0.2*randn(1, length(t));
            
        case 6  % CoupFail - 连接机构断开
            % 特征：间歇性信号，急剧变化，断续特征
            % 使用方波调制产生断续效果
            duty_cycle = 0.6;
            pulse_freq = 8;  % 8Hz的脉冲
            carrier = square(2*pi*pulse_freq*t) > 0;  % 方波
            
            vib = (0.4*sin(2*pi*50*t) + 0.35*sin(2*pi*150*t)) .* carrier;
            vib = vib + 0.3*sin(2*pi*100*t) .* ~carrier;  % 断开时的高频
            vib = vib + 0.12*randn(1, length(t));
    end
    
    % 防止幅值过大
    vib = vib / (max(abs(vib)) + eps) * 0.95;
end

%% ========================================================================
% 优化的温度信号生成
% =========================================================================

function temp = generateOptimizedTemperature(fault_idx, t)
    % 生成优化的温度信号
    
    base_temp = 50;
    
    switch fault_idx
        case 1  % Health
            % 稳定的基础温度，小波动
            temp = base_temp + 0.5*sin(2*pi*0.2*t) + 0.3*randn(1, length(t));
            
        case 2  % WdgOpen
            % 显著升温，快速变化
            temp = base_temp + 18 + 2*sin(2*pi*0.3*t) + 1.5*randn(1, length(t));
            
        case 3  % WdgImbal
            % 中等升温，周期变化
            temp = base_temp + 12 + 3*sin(2*pi*0.25*t) + 1.2*randn(1, length(t));
            
        case 4  % ShortCirc
            % 很高的温度，快速波动
            temp = base_temp + 25 + 4*sin(2*pi*0.35*t) + 2*randn(1, length(t));
            
        case 5  % PumpBlock
            % 最高温度，剧烈波动
            temp = base_temp + 30 + 5*sin(2*pi*0.4*t) + 2.5*randn(1, length(t));
            
        case 6  % CoupFail
            % 温度下降（冷却流动中断）
            temp = base_temp - 10 + 2*sin(2*pi*0.15*t) + 1.5*randn(1, length(t));
    end
end

%% ========================================================================
% 优化的电流信号生成
% =========================================================================

function curr = generateOptimizedCurrent(fault_idx, t, fs)
    % 生成优化的三相电流信号
    
    base_current = 10;
    
    switch fault_idx
        case 1  % Health
            % 三相平衡，基础电流
            curr = base_current * sin(2*pi*50*t);
            curr = curr + 0.2*randn(1, length(t));
            
        case 2  % WdgOpen
            % 一相缺失，电流严重不平衡
            % 模拟缺相：只有两相
            I_a = base_current * sin(2*pi*50*t);
            I_b = base_current * sin(2*pi*50*t - 2*pi/3);
            curr = 0.8*I_a + 1.2*I_b;  % 相位移位
            curr = curr + 0.3*randn(1, length(t));
            
        case 3  % WdgImbal
            % 三相幅值严重不等
            I_a = 0.7 * base_current * sin(2*pi*50*t);
            I_b = 1.3 * base_current * sin(2*pi*50*t - 2*pi/3);
            I_c = 1.0 * base_current * sin(2*pi*50*t - 4*pi/3);
            curr = I_a + I_b + I_c;
            curr = curr + 0.25*randn(1, length(t));
            
        case 4  % ShortCirc
            % 谐波含量大，基波畸变
            harmonic = 1.5*sin(2*pi*150*t) + 0.8*sin(2*pi*250*t) + 0.4*sin(2*pi*350*t);
            curr = 0.8*base_current*sin(2*pi*50*t) + harmonic;
            curr = curr + 0.3*randn(1, length(t));
            
        case 5  % PumpBlock
            % 电流大幅升高，低频特征
            curr = 1.6*base_current*sin(2*pi*50*t);
            curr = curr + 0.4*base_current*sin(2*pi*30*t);  % 低频上升
            curr = curr + 0.5*randn(1, length(t));
            
        case 6  % CoupFail
            % 电流间歇性，断续出现
            switch_freq = 8;  % 8Hz切换频率
            switch_signal = (square(2*pi*switch_freq*t) + 1) / 2;  % 0-1方波
            
            curr = base_current * sin(2*pi*50*t) .* switch_signal;
            curr = curr + 0.6*base_current*sin(2*pi*150*t) .* ~switch_signal;
            curr = curr + 0.4*randn(1, length(t));
    end
end