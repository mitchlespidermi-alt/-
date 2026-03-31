%% ========================================================================
% 文件：FusionAndDecision_Module.m
% 描述：多模型融合和决策制定模块
% =========================================================================

classdef FusionAndDecisionModule
    methods(Static)
        
        function [fusion_pred, fusion_conf, control_action] = ...
            FusionDecisionSystem(svm_pred, rf_pred, lstm_pred, ...
                                svm_prob, rf_prob, lstm_prob, ...
                                weights, config)
            % 完整的融合决策系统
            
            n_samples = length(svm_pred);
            
            % 初始化输出
            fusion_pred = zeros(n_samples, 1);
            fusion_conf = zeros(n_samples, 1);
            control_action = cell(n_samples, 1);
            
            for i = 1:n_samples
                % 第一步：模型加权投票
                votes = [svm_pred(i), rf_pred(i), lstm_pred(i)];
                confidences = [svm_prob(i), rf_prob(i), lstm_prob(i)];
                
                % 加权融合
                weighted_votes = votes .* weights;
                weighted_conf = confidences .* weights;
                
                % 决策（选择权重最大的预测）
                [max_conf, pred_idx] = max(weighted_conf);
                fusion_pred(i) = votes(pred_idx);
                fusion_conf(i) = max_conf;
                
                % 第二步：预警分级
                warning_level = FusionAndDecisionModule.assignWarningLevel(...
                    fusion_conf(i), fusion_pred(i));
                
                % 第三步：生成控制策略
                control_action{i} = FusionAndDecisionModule.generateControlAction(...
                    fusion_pred(i), warning_level, config);
            end
            
            fusion_pred = uint8(fusion_pred);
        end
        
        function warning_level = assignWarningLevel(confidence, fault_type)
            % 预警分级策略
            
            if confidence >= 0.95
                warning_level = 0;  % 正常运行
            elseif confidence >= 0.85
                warning_level = 1;  % 一级预警（可继续运行，加强监测）
            elseif confidence >= 0.70
                warning_level = 2;  % 二级预警（建议检修）
            elseif confidence >= 0.50
                warning_level = 3;  % 三级报警（需要立即检修）
            else
                warning_level = 4;  % 四级严重（立即停机）
            end
        end
        
        function control_action = generateControlAction(fault_type, warning_level, config)
            % 生成控制策略
            
            switch warning_level
                case 0  % 正常
                    control_action = struct(...
                        'action', 'Normal operation',...
                        'frequency', 'Normal',...
                        'load', 'Normal',...
                        'alert_msg', 'System running normally');
                    
                case 1  % 一级预警
                    control_action = struct(...
                        'action', 'Enhanced monitoring',...
                        'frequency', 'Reduce 5-10%',...
                        'load', 'Reduce 5-10%',...
                        'alert_msg', 'Warning: Enhanced monitoring active');
                    
                case 2  % 二级预警
                    control_action = struct(...
                        'action', 'Maintenance required',...
                        'frequency', 'Reduce 20%',...
                        'load', 'Reduce 20%',...
                        'alert_msg', 'Warning: Schedule maintenance soon');
                    
                case 3  % 三级报警
                    control_action = struct(...
                        'action', 'Immediate maintenance',...
                        'frequency', 'Reduce 50%',...
                        'load', 'Reduce 50%',...
                        'alert_msg', 'Alarm: Immediate maintenance required');
                    
                case 4  % 四级严重
                    control_action = struct(...
                        'action', 'Emergency shutdown',...
                        'frequency', 'Stop',...
                        'load', 'Stop',...
                        'alert_msg', 'Critical: Equipment shutdown');
            end
            
            % 根据故障类型添加特定建议
            switch fault_type
                case 2  % 绕组开路
                    control_action.maintenance_suggestion = ...
                        'Check motor winding connections and insulation';
                case 3  % 绕组不平衡
                    control_action.maintenance_suggestion = ...
                        'Rebalance or replace motor windings';
                case 4  % 匝间短路
                    control_action.maintenance_suggestion = ...
                        'Replace motor or repair windings';
                case 5  % 水泵堵转
                    control_action.maintenance_suggestion = ...
                        'Clean pump inlet and outlet; check for blockages';
                case 6  % 连接机构断开
                    control_action.maintenance_suggestion = ...
                        'Inspect and tighten motor-pump coupling';
                otherwise
                    control_action.maintenance_suggestion = ...
                        'Perform routine maintenance';
            end
        end
        
        function [adaptive_weights, weight_history] = ...
            AdaptiveWeightLearning(model_performance, iteration_num)
            % 自适应权重学习
            
            % 使用指数加权移动平均
            decay_factor = 0.95;
            
            % 初始化权重
            if iteration_num == 1
                adaptive_weights = model_performance / sum(model_performance);
                weight_history = adaptive_weights;
            else
                % 更新权重
                normalized_perf = model_performance / sum(model_performance);
                adaptive_weights = decay_factor * adaptive_weights + ...
                                  (1 - decay_factor) * normalized_perf;
                weight_history = [weight_history; adaptive_weights];
            end
        end
        
        function reliability = evaluateSystemReliability(predictions, ground_truth, window_size)
            % 评估系统可靠性
            
            if nargin < 3
                window_size = 50;
            end
            
            n_samples = length(predictions);
            reliability = zeros(floor(n_samples/window_size), 1);
            
            for i = 1:length(reliability)
                start_idx = (i-1)*window_size + 1;
                end_idx = min(i*window_size, n_samples);
                
                window_pred = predictions(start_idx:end_idx);
                window_true = ground_truth(start_idx:end_idx);
                
                reliability(i) = mean(window_pred == window_true);
            end
        end
        
        function variance = assessPredictionVariance(svm_prob, rf_prob, lstm_prob)
            % 评估预测方差（模型间的一致性）
            
            n_samples = length(svm_prob);
            variance = zeros(n_samples, 1);
            
            for i = 1:n_samples
                probs = [svm_prob(i), rf_prob(i), lstm_prob(i)];
                variance(i) = std(probs);
            end
        end
        
        function visualizeFusionProcess(svm_pred, rf_pred, lstm_pred, fusion_pred, Y_true)
            % 可视化融合过程
            
            figure('Position', [100, 100, 1400, 600]);
            
            % 选择前100个样本展示
            n_show = min(100, length(fusion_pred));
            idx = 1:n_show;
            
            subplot(2,1,1);
            plot(idx, svm_pred(idx), 'o-', 'LineWidth', 1.5, 'MarkerSize', 4);
            hold on;
            plot(idx, rf_pred(idx), 's-', 'LineWidth', 1.5, 'MarkerSize', 4);
            plot(idx, lstm_pred(idx), '^-', 'LineWidth', 1.5, 'MarkerSize', 4);
            plot(idx, fusion_pred(idx), '*-', 'LineWidth', 2, 'MarkerSize', 8, 'Color', 'k');
            plot(idx, Y_true(idx), 'LineWidth', 2, 'Color', 'red');
            
            xlabel('样本索引');
            ylabel('预测故障类型');
            legend('SVM', 'RF', 'LSTM', '融合', '真实值');
            title('单个模型与融合预测对比');
            grid on;
            
            % 预测错误分析
            subplot(2,1,2);
            errors_svm = abs(svm_pred(idx) - Y_true(idx));
            errors_rf = abs(rf_pred(idx) - Y_true(idx));
            errors_lstm = abs(lstm_pred(idx) - Y_true(idx));
            errors_fusion = abs(fusion_pred(idx) - Y_true(idx));
            
            plot(idx, errors_svm, 'o-', 'LineWidth', 1.5);
            hold on;
            plot(idx, errors_rf, 's-', 'LineWidth', 1.5);
            plot(idx, errors_lstm, '^-', 'LineWidth', 1.5);
            plot(idx, errors_fusion, '*-', 'LineWidth', 2, 'MarkerSize', 8, 'Color', 'k');
            
            xlabel('样本索引');
            ylabel('预测错误');
            legend('SVM', 'RF', 'LSTM', '融合');
            title('预测错误对比');
            grid on;
            
            sgtitle('模型融合过程分析');
        end
    end
end