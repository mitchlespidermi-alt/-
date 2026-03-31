%% ========================================================================
% 文件：FaultDiagnosis_TrendPrediction_Module.m
% 描述：故障诊断和趋势预测专门模块
% =========================================================================

classdef FaultDiagnosisModule
    methods(Static)
        
        function [degradation_curve, remaining_life] = ...
            PredictEquipmentDegradation(health_scores, time_points, config)
            % 预测设备退化曲线和剩余寿命
            
            if nargin < 3
                config.degradation_threshold = 0.3;  % 故障阈值
            end
            
            % 拟合退化曲线
            [p, S] = polyfit(time_points, health_scores, 3);
            degradation_curve = polyval(p, time_points);
            
            % 预测剩余寿命
            % 找到故障点
            fault_time = find(degradation_curve <= config.degradation_threshold, 1);
            
            if isempty(fault_time)
                remaining_life = max(time_points);
            else
                remaining_life = time_points(fault_time) - time_points(end);
            end
        end
        
        function [anomaly_scores, anomaly_windows] = ...
            DetectAnomaliesUsingAutoencoder(X, reconstruction_error_threshold)
            % 使用自编码器检测异常
            
            if nargin < 2
                reconstruction_error_threshold = 0.05;
            end
            
            % 计算重构误差
            reconstruction_error = mean((X - X).^2, 2);  % 简化示例
            
            % 标记异常
            anomaly_scores = reconstruction_error;
            anomaly_windows = reconstruction_error > reconstruction_error_threshold;
        end
        
        function [time_series_pred, lstm_model] = ...
            LSTMTimeSeriesPrediction(X_train, Y_train, sequence_length)
            % LSTM时间序列预测
            
            if nargin < 3
                sequence_length = 30;
            end
            
            % 创建序列数据
            [X_seq, Y_seq] = createSequences(X_train, Y_train, sequence_length);
            
            % 创建LSTM网络
            num_features = size(X_train, 2);
            num_hidden = 64;
            
            layers = [
                sequenceInputLayer(num_features)
                lstmLayer(num_hidden, 'OutputMode', 'last')
                fullyConnectedLayer(1)
                regressionLayer
            ];
            
            % 训练选项
            options = trainingOptions('adam', ...
                'MaxEpochs', 50, ...
                'MiniBatchSize', 32, ...
                'Verbose', false);
            
            % 训练网络
            lstm_model = trainNetwork(X_seq, Y_seq, layers, options);
            
            % 预测
            time_series_pred = predict(lstm_model, X_seq);
        end
        
        function [health_indicator, health_index] = ...
            ComputeHealthIndicator(features, weights)
            % 计算综合健康指标
            
            if nargin < 2
                weights = ones(size(features, 2), 1) / size(features, 2);
            end
            
            % 归一化特征
            features_norm = normalize(features);
            
            % 加权组合
            health_indicator = features_norm * weights;
            
            % 计算健康指数 (0-100)
            health_index = 100 * (1 - (health_indicator - min(health_indicator)) / ...
                (max(health_indicator) - min(health_indicator) + eps));
        end
        
        function [early_warning, warning_confidence] = ...
            EarlyFaultDetection(health_index, degradation_rate, threshold)
            % 早期故障检测
            
            if nargin < 3
                threshold = 30;  % 故障阈值
            end
            
            early_warning = zeros(length(health_index), 1);
            warning_confidence = zeros(length(health_index), 1);
            
            for i = 2:length(health_index)
                current_health = health_index(i);
                health_change = health_index(i) - health_index(i-1);
                
                % 判断是否早期故障
                if current_health < threshold
                    early_warning(i) = 1;
                    warning_confidence(i) = 1 - (current_health / threshold);
                elseif health_change < -5  % 快速下降
                    early_warning(i) = 1;
                    warning_confidence(i) = abs(health_change) / 20;
                end
            end
        end
        
        function diagnose_report = GenerateDiagnosisReport(predictions, confidences, fault_types, time_info)
            % 生成诊断报告
            
            diagnose_report = struct();
            
            % 故障统计
            unique_faults = unique(predictions);
            for i = 1:length(unique_faults)
                fault_idx = unique_faults(i);
                fault_count = sum(predictions == fault_idx);
                fault_conf = mean(confidences(predictions == fault_idx));
                
                diagnose_report.(sprintf('fault_%d', fault_idx)) = struct(...
                    'name', fault_types{fault_idx},...
                    'count', fault_count,...
                    'average_confidence', fault_conf,...
                    'percentage', fault_count / length(predictions) * 100);
            end
            
            % 时间统计
            if nargin >= 4
                diagnose_report.time_span = struct(...
                    'start_time', time_info(1),...
                    'end_time', time_info(end),...
                    'duration', time_info(end) - time_info(1));
            end
            
            % 总体评估
            diagnose_report.overall_confidence = mean(confidences);
            diagnose_report.prediction_variance = std(confidences);
        end
        
        function visualizeFaultEvolution(predictions, confidences, fault_types, time_axis)
            % 可视化故障演化
            
            figure('Position', [100, 100, 1200, 700]);
            
            % 故障类型时间序列
            subplot(3,1,1);
            plot(time_axis, predictions, 'o-', 'LineWidth', 2);
            set(gca, 'YTick', 1:length(fault_types));
            set(gca, 'YTickLabel', fault_types);
            ylabel('故障类型');
            title('故障发展时间序列');
            grid on;
            
            % 预测置信度
            subplot(3,1,2);
            plot(time_axis, confidences, 'o-', 'LineWidth', 2, 'Color', 'g');
            ylabel('置信度');
            title('预测置信度变化');
            ylim([0, 1.1]);
            grid on;
            
            % 累积故障计数
            subplot(3,1,3);
            cumul_count = cumsum(ones(length(predictions), 1));
            plot(time_axis, cumul_count, 'o-', 'LineWidth', 2, 'Color', 'r');
            ylabel('累积故障数');
            xlabel('时间');
            title('累积故障计数');
            grid on;
            
            sgtitle('故障演化分析');
        end
        
        function compareModels(model_predictions, model_names, ground_truth)
            % 比较不同模型的预测
            
            n_models = length(model_predictions);
            accuracies = zeros(n_models, 1);
            
            figure;
            for i = 1:n_models
                accuracies(i) = mean(model_predictions{i} == ground_truth);
            end
            
            bar(accuracies * 100);
            set(gca, 'XTickLabel', model_names);
            ylabel('准确率 (%)');
            title('模型性能对比');
            ylim([0, 105]);
            grid on;
            
            % 添加数值标签
            hold on;
            for i = 1:n_models
                text(i, accuracies(i)*100 + 2, sprintf('%.2f%%', accuracies(i)*100), ...
                    'HorizontalAlignment', 'center');
            end
        end
    end
end

% 辅助函数
function [X_seq, Y_seq] = createSequences(X, Y, seq_len)
    % 创建序列数据用于LSTM
    
    n_samples = size(X, 1) - seq_len;
    X_seq = [];
    Y_seq = [];
    
    for i = 1:n_samples
        X_seq = [X_seq; reshape(X(i:i+seq_len-1, :), 1, seq_len*size(X, 2))];
        Y_seq = [Y_seq; Y(i+seq_len)];
    end
end