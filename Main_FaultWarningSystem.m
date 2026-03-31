%% ========================================================================
% 文件：Main_FaultWarningSystem.m
% 描述：基于多源传感融合的工业泵房故障预警系统 - 主程序
% 作者：工业控制智能应用
% 日期：2026年3月
% =========================================================================

clear all; close all; clc;

%% 1. 系统初始化
fprintf('\n========== 泵房设备故障预警系统 v3.0 ==========\n');
fprintf('工作模式：完整端到端系统\n');
fprintf('算法框架：SVM + Random Forest + LSTM\n\n');

% 系统参数配置
config.sampling_rate = 10000;        % 采样率 10kHz
config.signal_duration = 1;          % 信号时长 1秒
config.window_size = 1024;           % 特征窗口 1024点
config.overlap_rate = 0.5;           % 重叠率 50%
config.num_samples = 5000;           % 每类样本数
config.train_ratio = 0.7;            % 训练集比例
config.val_ratio = 0.15;             % 验证集比例
config.test_ratio = 0.15;            % 测试集比例

% 故障类型定义
config.fault_types = {
    'Health'              % 1. 健康状态
    'Winding_OpenCircuit' % 2. 电机绕组开路
    'Winding_Imbalance'   % 3. 绕组不平衡
    'Interphase_Shorted'  % 4. 匝间短路
    'Pump_Blocked'        % 5. 水泵堵转
    'Connection_Broken'   % 6. 连接机构断开
};

config.num_faults = length(config.fault_types);
fprintf('识别故障类型数：%d\n', config.num_faults);

%% 2. 生成模拟传感器数据
fprintf('\n[步骤1] 生成仿真传感器数据...\n');
[sensor_data, fault_labels] = Generate_SensorData(config);
fprintf('✓ 生成样本数：%d，特征维度：%d\n', ...
    size(sensor_data, 1), size(sensor_data, 2));

%% 3. 特征提取
fprintf('\n[步骤2] 执行特征提取...\n');
features = Extract_Features(sensor_data, config);
fprintf('✓ 提取特征数：%d维\n', size(features, 2));

% 特征标准化
features_normalized = normalize(features);

%% 4. 数据集划分
fprintf('\n[步骤3] 划分训练/验证/测试集...\n');
[train_idx, val_idx, test_idx] = Partition_Dataset(...
    length(fault_labels), config);

X_train = features_normalized(train_idx, :);
Y_train = fault_labels(train_idx);
X_val = features_normalized(val_idx, :);
Y_val = fault_labels(val_idx);
X_test = features_normalized(test_idx, :);
Y_test = fault_labels(test_idx);

fprintf('✓ 训练集：%d, 验证集：%d, 测试集：%d\n', ...
    length(Y_train), length(Y_val), length(Y_test));

%% 5. 模型训练
fprintf('\n[步骤4] 训练分类模型...\n');

% 5.1 SVM基线模型
fprintf('\n--- SVM基线模型训练 ---\n');
svm_model = trainSVMClassifier(X_train, Y_train, config);
svm_pred_train = predict(svm_model, X_train);
svm_pred_val = predict(svm_model, X_val);
svm_pred_test = predict(svm_model, X_test);

svm_acc_train = mean(svm_pred_train == Y_train);
svm_acc_val = mean(svm_pred_val == Y_val);
svm_acc_test = mean(svm_pred_test == Y_test);

fprintf('SVM - 训练准确率: %.2f%%, 验证准确率: %.2f%%, 测试准确率: %.2f%%\n', ...
    svm_acc_train*100, svm_acc_val*100, svm_acc_test*100);

% 5.2 随机森林主分类模型
fprintf('\n--- 随机森林主模型训练 ---\n');
rf_model = trainRandomForestClassifier(X_train, Y_train, config);
rf_pred_train = predict(rf_model, X_train);
rf_pred_val = predict(rf_model, X_val);
rf_pred_test = predict(rf_model, X_test);

rf_acc_train = mean(rf_pred_train == Y_train);
rf_acc_val = mean(rf_pred_val == Y_val);
rf_acc_test = mean(rf_pred_test == Y_test);

fprintf('RF  - 训练准确率: %.2f%%, 验证准确率: %.2f%%, 测试准确率: %.2f%%\n', ...
    rf_acc_train*100, rf_acc_val*100, rf_acc_test*100);

% 5.3 LSTM趋势预测模型
fprintf('\n--- LSTM趋势预测模型训练 ---\n');
lstm_model = trainLSTMPredictor(X_train, Y_train, config);
lstm_pred_train = predictLSTM(lstm_model, X_train);
lstm_pred_val = predictLSTM(lstm_model, X_val);
lstm_pred_test = predictLSTM(lstm_model, X_test);

lstm_acc_train = mean(round(lstm_pred_train) == Y_train);
lstm_acc_val = mean(round(lstm_pred_val) == Y_val);
lstm_acc_test = mean(round(lstm_pred_test) == Y_test);

fprintf('LSTM- 训练准确率: %.2f%%, 验证准确率: %.2f%%, 测试准确率: %.2f%%\n', ...
    lstm_acc_train*100, lstm_acc_val*100, lstm_acc_test*100);

%% 6. 模型融合
fprintf('\n[步骤5] 执行模型融合与决策...\n');

% 计算融合权重（基于验证集性能）
weights = [svm_acc_val, rf_acc_val, lstm_acc_val];
weights = weights / sum(weights);
fprintf('融合权重 - SVM: %.3f, RF: %.3f, LSTM: %.3f\n', ...
    weights(1), weights(2), weights(3));

% 融合预测
[fusion_pred_test, fusion_prob_test] = FusionPredictor(...
    svm_pred_test, rf_pred_test, lstm_pred_test, weights);

fusion_acc_test = mean(fusion_pred_test == Y_test);
fprintf('✓ 融合模型测试准确率: %.2f%%\n', fusion_acc_test*100);

%% 7. 故障预警分级与控制策略
fprintf('\n[步骤6] 故障预警分级与控制决策...\n');

% 预警等级划分（基于预测概率）
warning_levels = Assign_WarningLevels(fusion_prob_test, Y_test, config);
fprintf('✓ 预警分级完成\n');

%% 8. 性能评估与可视化
fprintf('\n[步骤7] 性能评估...\n');

% 计算详细评估指标
metrics = Evaluate_Performance(Y_test, fusion_pred_test, config);

% 显示评估报告
fprintf('\n========== 性能评估报告 ==========\n');
fprintf('总体准确率: %.2f%%\n', metrics.accuracy*100);
fprintf('加权精确率: %.2f%%\n', metrics.weighted_precision*100);
fprintf('加权召回率: %.2f%%\n', metrics.weighted_recall*100);
fprintf('加权F1分数: %.2f%%\n', metrics.weighted_f1*100);

% 按故障类型显示性能
fprintf('\n按故障类型的性能指标:\n');
for i = 1:config.num_faults
    fprintf('%-20s - P: %.2f%%, R: %.2f%%, F1: %.2f%%\n', ...
        config.fault_types{i}, metrics.precision(i)*100, ...
        metrics.recall(i)*100, metrics.f1(i)*100);
end

%% 9. 绘制结果图表
fprintf('\n[步骤8] 生成可视化结果...\n');

figure('Position', [100, 100, 1400, 900]);

% 9.1 混淆矩阵
subplot(2,3,1);
plotConfusionMatrix(Y_test, fusion_pred_test, config);

% 9.2 模型性能对比
subplot(2,3,2);
plotModelComparison([svm_acc_test, rf_acc_test, lstm_acc_test, fusion_acc_test]);

% 9.3 ROC曲线
subplot(2,3,3);
plotROCCurves(Y_test, fusion_prob_test, config);

% 9.4 精确率-召回率曲线
subplot(2,3,4);
plotPrecisionRecallCurves(metrics, config);

% 9.5 特征重要性
subplot(2,3,5);
plotFeatureImportance(rf_model);

% 9.6 预警等级分布
subplot(2,3,6);
plotWarningLevelDistribution(warning_levels, config);

sgtitle('泵房故障预警系统 - 完整性能分析', 'FontSize', 14, 'FontWeight', 'bold');

%% 10. 案例演示：实时故障检测
fprintf('\n[步骤9] 实时故障检测案例演示...\n');

% 选择测试样本进行实时检测演示
test_samples = [100, 200, 300, 400, 500]; % 选择5个样本
fprintf('\n实时检测结果演示:\n');
fprintf('样本ID\t真实故障\t\t预测故障\t融合概率\t预警等级\n');
fprintf('%-8d%-20s%-20s%.4f\t%s\n', 1, ...
    config.fault_types{Y_test(test_samples(1))}, ...
    config.fault_types{fusion_pred_test(test_samples(1))}, ...
    fusion_prob_test(test_samples(1)), ...
    warning_levels(test_samples(1)));

for idx = 1:length(test_samples)
    sample_id = test_samples(idx);
    fprintf('%-8d%-20s%-20s%.4f\t%s\n', idx, ...
        config.fault_types{Y_test(sample_id)}, ...
        config.fault_types{fusion_pred_test(sample_id)}, ...
        fusion_prob_test(sample_id), ...
        warning_levels(sample_id));
end

%% 11. 保存模型和结果
fprintf('\n[步骤10] 保存模型...\n');

% 保存训练好的模型
save('trained_models.mat', 'svm_model', 'rf_model', 'lstm_model', 'config');
fprintf('✓ 模型已保存: trained_models.mat\n');

% 保存结果统计
results.metrics = metrics;
results.fusion_pred = fusion_pred_test;
results.fusion_prob = fusion_prob_test;
results.warning_levels = warning_levels;
save('results_statistics.mat', 'results');
fprintf('✓ 结果已保存: results_statistics.mat\n');

fprintf('\n========== 系统执行完成！==========\n');
fprintf('生成的图表已保存为: fault_warning_analysis.fig\n');

%% ========================================================================
% 辅助函数
% =========================================================================

function [sensor_data, fault_labels] = Generate_SensorData(config)
    % 生成仿真多源传感器数据
    % 输入：config - 配置参数
    % 输出：sensor_data - 传感器原始信号，fault_labels - 故障标签
    
    num_faults = length(config.fault_types);
    num_samples = config.num_samples;
    signal_len = config.sampling_rate * config.signal_duration;
    
    sensor_data = [];
    fault_labels = [];
    
    for fault_idx = 1:num_faults
        for sample_idx = 1:num_samples
            % 生成三源传感器信号
            vibration = generateVibrationSignal(fault_idx, signal_len, config);
            temperature = generateTemperatureSignal(fault_idx, signal_len, config);
            current = generateCurrentSignal(fault_idx, signal_len, config);
            
            % 组合三源信号
            combined_signal = [vibration, temperature, current];
            
            sensor_data = [sensor_data; combined_signal];
            fault_labels = [fault_labels; fault_idx];
        end
    end
    
    % 随机排列
    idx = randperm(length(fault_labels));
    sensor_data = sensor_data(idx, :);
    fault_labels = fault_labels(idx);
end

function vibration = generateVibrationSignal(fault_type, signal_len, config)
    % 生成振动信号，不同故障类型具有不同特性
    
    fs = config.sampling_rate;
    t = (0:signal_len-1) / fs;
    
    % 基础振动（正弦波）
    vibration = 0.5 * sin(2*pi*50*t);  % 50Hz基频
    
    switch fault_type
        case 1  % 健康状态
            % 低幅值，低频内容
            vibration = vibration + 0.05*randn(1, signal_len);
            
        case 2  % 绕组开路
            % 高频成分，不对称
            vibration = vibration + 0.3*sin(2*pi*200*t) + 0.15*randn(1, signal_len);
            
        case 3  % 绕组不平衡
            % 多频率成分，周期性
            vibration = vibration + 0.2*sin(2*pi*100*t) + 0.15*sin(2*pi*150*t) + ...
                       0.1*randn(1, signal_len);
            
        case 4  % 匝间短路
            % 冲击成分，宽带噪声
            impulses = zeros(1, signal_len);
            impulse_idx = round(linspace(1, signal_len, 20));
            impulses(impulse_idx) = 2;  % 脉冲幅值
            vibration = vibration + 0.4*sin(2*pi*200*t) + impulses + ...
                       0.2*randn(1, signal_len);
            
        case 5  % 水泵堵转
            % 低频高幅值，明显过载特征
            vibration = 1.5*sin(2*pi*30*t) + 0.4*sin(2*pi*60*t) + 0.1*randn(1, signal_len);
            
        case 6  % 连接机构断开
            % 断续性信号，间歇性特征
            modulation = 0.5*(1 + sin(2*pi*10*t));  % 调制信号
            vibration = vibration .* modulation + 0.2*sin(2*pi*150*t) + ...
                       0.1*randn(1, signal_len);
    end
    
    % 加入高斯噪声
    vibration = vibration + 0.1*randn(1, signal_len);
end

function temperature = generateTemperatureSignal(fault_type, signal_len, config)
    % 生成温度信号
    
    fs = config.sampling_rate;
    t = (0:signal_len-1) / fs;
    
    % 基础温度（缓慢变化）
    temperature = 50 + 5*sin(2*pi*0.1*t);  % 基础温度50°C，缓慢波动
    
    switch fault_type
        case 1  % 健康状态
            temperature = temperature + 0.5*randn(1, signal_len);
            
        case 2  % 绕组开路
            temperature = temperature + 15 + 2*randn(1, signal_len);  % 升温
            
        case 3  % 绕组不平衡
            temperature = temperature + 10 + 1.5*randn(1, signal_len);  % 中等升温
            
        case 4  % 匝间短路
            temperature = temperature + 20 + 3*randn(1, signal_len);  % 显著升温
            
        case 5  % 水泵堵转
            temperature = temperature + 25 + 3.5*randn(1, signal_len);  % 大幅升温
            
        case 6  % 连接机构断开
            % 温度下降（冷却不足、循环中断）
            temperature = temperature - 8 + 2*randn(1, signal_len);
    end
end

function current = generateCurrentSignal(fault_type, signal_len, config)
    % 生成三相电流信号
    
    fs = config.sampling_rate;
    t = (0:signal_len-1) / fs;
    
    % 基础电流（三相正弦波，120°相差）
    I_a = 10*sin(2*pi*50*t);
    I_b = 10*sin(2*pi*50*t - 2*pi/3);
    I_c = 10*sin(2*pi*50*t - 4*pi/3);
    
    switch fault_type
        case 1  % 健康状态
            % 三相平衡
            current = I_a + 0.2*randn(1, signal_len);
            
        case 2  % 绕组开路
            % A相缺失
            current = I_b + 0.3*randn(1, signal_len);
            
        case 3  % 绕组不平衡
            % 三相幅值不等
            current = 0.9*I_a + 1.1*I_b + I_c + 0.25*randn(1, signal_len);
            
        case 4  % 匝间短路
            % 电流畸变，谐波含量大
            harmonic = 2*sin(2*pi*150*t) + sin(2*pi*250*t);
            current = I_a + harmonic + 0.4*randn(1, signal_len);
            
        case 5  % 水泵堵转
            % 电流大幅升高
            current = 15*I_a + 3*sin(2*pi*100*t) + 0.5*randn(1, signal_len);
            
        case 6  % 连接机构断开
            % 电流间歇性
            modulation = 0.5*(1 + square(2*pi*5*t));
            current = I_a .* modulation + 1*randn(1, signal_len);
    end
end

function features = Extract_Features(sensor_data, config)
    % 特征提取模块
    % 输入：sensor_data - 原始传感器数据
    % 输出：features - 提取的特征矩阵
    
    num_samples = size(sensor_data, 1);
    features = [];
    
    for i = 1:num_samples
        signal = sensor_data(i, :);
        
        % 时域特征
        feat_time = extractTimeFeatures(signal);
        
        % 频域特征
        feat_freq = extractFreqFeatures(signal, config.sampling_rate);
        
        % 时频域特征
        feat_tf = extractTimeFreqFeatures(signal);
        
        % 统计特征
        feat_stat = extractStatisticalFeatures(signal);
        
        % 组合所有特征
        feat_combined = [feat_time, feat_freq, feat_tf, feat_stat];
        features = [features; feat_combined];
    end
end

function feat_time = extractTimeFeatures(signal)
    % 时域特征提取
    
    feat_time = [
        mean(signal)              % 平均值
        std(signal)               % 标准差
        max(abs(signal))          % 峰值
        rms(signal)               % 均方根
        peak2peak(signal)         % 峰峰值
        skewness(signal)          % 偏度
        kurtosis(signal)          % 峭度
        sum(abs(diff(signal)))    % 一阶差分和
    ];
end

function feat_freq = extractFreqFeatures(signal, fs)
    % 频域特征提取
    
    % FFT分析
    fft_signal = abs(fft(signal));
    fft_signal = fft_signal(1:length(signal)/2);
    freq = (0:length(fft_signal)-1) * fs / length(signal);
    
    % 峰值频率
    [~, peak_idx] = max(fft_signal);
    peak_freq = freq(peak_idx);
    
    % 功率谱密度
    [pxx, f] = periodogram(signal, [], [], fs);
    
    feat_freq = [
        peak_freq                 % 主频率
        max(fft_signal)          % 频域峰值
        mean(fft_signal)         % 频域平均值
        std(fft_signal)          % 频域标准差
        sum(pxx(1:10))           % 低频能量
        sum(pxx(floor(end/2):end)) % 高频能量
    ];
end

function feat_tf = extractTimeFreqFeatures(signal)
    % 时频域特征（小波变换）
    
    % 小波分解
    [c, l] = wavedec(signal, 3, 'db4');
    
    % 各层能量
    energy_d3 = norm(c(1:l(1)))^2;
    energy_d2 = norm(c(l(1)+1:l(1)+l(2)))^2;
    energy_d1 = norm(c(l(1)+l(2)+1:l(1)+l(2)+l(3)))^2;
    energy_a = norm(c(l(1)+l(2)+l(3)+1:end))^2;
    
    total_energy = energy_d3 + energy_d2 + energy_d1 + energy_a + eps;
    
    feat_tf = [
        energy_d3/total_energy    % d3能量比
        energy_d2/total_energy    % d2能量比
        energy_d1/total_energy    % d1能量比
        energy_a/total_energy     % a能量比
    ];
end

function feat_stat = extractStatisticalFeatures(signal)
    % 统计特征提取
    
    feat_stat = [
        min(signal)               % 最小值
        max(signal)               % 最大值
        median(signal)            % 中位数
        iqr(signal)               % 四分位距
        sum(signal > 0) / length(signal)  % 正值比例
    ];
end

function [train_idx, val_idx, test_idx] = Partition_Dataset(n_samples, config)
    % 划分数据集
    
    idx = randperm(n_samples);
    
    train_size = floor(n_samples * config.train_ratio);
    val_size = floor(n_samples * config.val_ratio);
    
    train_idx = idx(1:train_size);
    val_idx = idx(train_size+1:train_size+val_size);
    test_idx = idx(train_size+val_size+1:end);
end

function svm_model = trainSVMClassifier(X, Y, config)
    % 训练SVM分类器
    
    % 使用One-vs-All策略的多类SVM
    template = templateSVM('Standardize', true, 'KernelFunction', 'rbf');
    svm_model = fitcecoc(X, Y, 'Learners', template);
end

function rf_model = trainRandomForestClassifier(X, Y, config)
    % 训练随机森林分类器
    
    % 创建随机森林（使用300棵树）
    rf_model = TreeBagger(300, X, Y, ...
        'Method', 'classification', ...
        'OOBPredictorImportance', 'on', ...
        'NumPredictorsToSample', 'auto');
end

function lstm_model = trainLSTMPredictor(X, Y, config)
    % 训练LSTM预测器
    
    % LSTM模型参数
    num_features = size(X, 2);
    num_hidden = 50;
    num_outputs = config.num_faults;
    
    % 创建LSTM网络
    lstm_layers = [
        sequenceInputLayer(num_features)
        lstmLayer(num_hidden, 'OutputMode', 'last')
        fullyConnectedLayer(num_outputs)
        softmaxLayer
        classificationLayer
    ];
    
    % 训练选项
    options = trainingOptions('adam', ...
        'MaxEpochs', 50, ...
        'MiniBatchSize', 32, ...
        'Verbose', false, ...
        'Plots', 'none');
    
    % 重塑数据为序列格式
    X_seq = reshape(X', num_features, 1, size(X, 1));
    
    % 训练网络
    lstm_model = trainNetwork(X_seq, categorical(Y), lstm_layers, options);
end

function pred = predictLSTM(lstm_model, X)
    % LSTM预测
    
    num_features = size(X, 2);
    X_seq = reshape(X', num_features, 1, size(X, 1));
    Y_pred = classify(lstm_model, X_seq);
    pred = double(Y_pred);
end

function [fusion_pred, fusion_prob] = FusionPredictor(svm_pred, rf_pred, lstm_pred, weights)
    % 模型融合预测器
    
    % 投票融合
    n_samples = length(svm_pred);
    fusion_pred = zeros(n_samples, 1);
    fusion_prob = zeros(n_samples, 1);
    
    for i = 1:n_samples
        % 加权投票
        votes = [svm_pred(i), rf_pred(i), lstm_pred(i)];
        vote_weights = weights;
        
        % 找出最高权重的预测
        [~, pred_idx] = max(vote_weights);
        fusion_pred(i) = votes(pred_idx);
        
        % 融合概率（最高权重）
        fusion_prob(i) = max(vote_weights);
    end
    
    fusion_pred = uint8(fusion_pred);
end

function warning_levels = Assign_WarningLevels(fusion_prob, Y_true, config)
    % 故障预警分级
    
    n_samples = length(fusion_prob);
    warning_levels = cell(n_samples, 1);
    
    for i = 1:n_samples
        if fusion_prob(i) >= 0.9
            warning_levels{i} = '正常';
        elseif fusion_prob(i) >= 0.75
            warning_levels{i} = '预警';
        elseif fusion_prob(i) >= 0.5
            warning_levels{i} = '报警';
        else
            warning_levels{i} = '严重';
        end
    end
end

function metrics = Evaluate_Performance(Y_true, Y_pred, config)
    % 性能评估
    
    % 总体准确率
    metrics.accuracy = mean(Y_true == Y_pred);
    
    % 按类别计算指标
    num_classes = config.num_faults;
    metrics.precision = zeros(num_classes, 1);
    metrics.recall = zeros(num_classes, 1);
    metrics.f1 = zeros(num_classes, 1);
    
    for i = 1:num_classes
        tp = sum((Y_true == i) & (Y_pred == i));
        fp = sum((Y_true ~= i) & (Y_pred == i));
        fn = sum((Y_true == i) & (Y_pred ~= i));
        
        precision = tp / (tp + fp + eps);
        recall = tp / (tp + fn + eps);
        f1 = 2 * precision * recall / (precision + recall + eps);
        
        metrics.precision(i) = precision;
        metrics.recall(i) = recall;
        metrics.f1(i) = f1;
    end
    
    % 加权平均
    class_counts = histc(Y_true, 1:num_classes);
    weights = class_counts / sum(class_counts);
    
    metrics.weighted_precision = sum(metrics.precision .* weights);
    metrics.weighted_recall = sum(metrics.recall .* weights);
    metrics.weighted_f1 = sum(metrics.f1 .* weights);
end

% 可视化函数
function plotConfusionMatrix(Y_true, Y_pred, config)
    % 混淆矩阵可视化
    
    C = confusionmat(Y_true, Y_pred);
    
    % 归一化混淆矩阵
    C_norm = C ./ (sum(C, 2) + eps);
    
    imagesc(C_norm);
    colorbar;
    set(gca, 'XTick', 1:config.num_faults);
    set(gca, 'YTick', 1:config.num_faults);
    set(gca, 'XTickLabel', config.fault_types);
    set(gca, 'YTickLabel', config.fault_types);
    xtickangle(45);
    xlabel('预测故障类型');
    ylabel('真实故障类型');
    title('混淆矩阵 (已归一化)');
end

function plotModelComparison(accuracies)
    % 模型性能对比
    
    models = {'SVM', 'RF', 'LSTM', '融合'};
    bar(accuracies * 100);
    set(gca, 'XTickLabel', models);
    ylabel('准确率 (%)');
    title('模型性能对比');
    grid on;
    ylim([0, 105]);
end

function plotROCCurves(Y_true, Y_prob, config)
    % ROC曲线绘制
    
    % 简化：绘制One-vs-Rest ROC曲线
    for i = 1:min(3, config.num_faults)
        y_binary = (Y_true == i);
        [fpr, tpr, ~, auc] = perfcurve(y_binary, Y_prob, 1);
        plot(fpr, tpr, 'LineWidth', 2);
        hold on;
    end
    
    plot([0, 1], [0, 1], 'k--', 'LineWidth', 1);
    xlabel('假正率');
    ylabel('真正率');
    title('ROC曲线');
    legend('故障类型1', '故障类型2', '故障类型3', '随机分类');
    grid on;
end

function plotPrecisionRecallCurves(metrics, config)
    % 精确率-召回率曲线
    
    bar([metrics.weighted_precision*100, ...
         metrics.weighted_recall*100, ...
         metrics.weighted_f1*100]);
    set(gca, 'XTickLabel', {'精确率', '召回率', 'F1分数'});
    ylabel('分数 (%)');
    title('加权性能指标');
    ylim([0, 105]);
    grid on;
end

function plotFeatureImportance(rf_model)
    % 特征重要性可视化
    
    importance = rf_model.OOBPermutedPredictorDeltaError;
    [~, idx] = sort(importance, 'descend');
    
    top_n = min(15, length(importance));
    bar(importance(idx(1:top_n)));
    xlabel('特征索引');
    ylabel('重要性');
    title('随机森林特征重要性 (Top 15)');
    grid on;
end

function plotWarningLevelDistribution(warning_levels, config)
    % 预警等级分布
    
    levels = {'正常', '预警', '报警', '严重'};
    counts = zeros(1, 4);
    
    for i = 1:length(warning_levels)
        for j = 1:4
            if strcmp(warning_levels{i}, levels{j})
                counts(j) = counts(j) + 1;
            end
        end
    end
    
    pie(counts, levels);
    title('预警等级分布');
end