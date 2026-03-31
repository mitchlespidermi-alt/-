%% ========================================================================
% 文件：ExperimentAndEvaluation_Module.m
% 描述：完整的实验和评估框架
% =========================================================================

classdef ExperimentAndEvaluationModule
    methods(Static)
        
        function [report, detailed_metrics] = ...
            CompletePerformanceEvaluation(Y_true, Y_pred, Y_prob, fault_types)
            % 完整的性能评估
            
            n_classes = length(fault_types);
            
            % 基础指标
            report.accuracy = mean(Y_true == Y_pred);
            
            % 每类指标
            detailed_metrics.precision = zeros(n_classes, 1);
            detailed_metrics.recall = zeros(n_classes, 1);
            detailed_metrics.f1 = zeros(n_classes, 1);
            detailed_metrics.support = zeros(n_classes, 1);
            detailed_metrics.auc = zeros(n_classes, 1);
            
            for i = 1:n_classes
                tp = sum((Y_true == i) & (Y_pred == i));
                fp = sum((Y_true ~= i) & (Y_pred == i));
                fn = sum((Y_true == i) & (Y_pred ~= i));
                tn = sum((Y_true ~= i) & (Y_pred ~= i));
                
                % 计算指标
                precision = tp / (tp + fp + eps);
                recall = tp / (tp + fn + eps);
                f1 = 2 * precision * recall / (precision + recall + eps);
                
                detailed_metrics.precision(i) = precision;
                detailed_metrics.recall(i) = recall;
                detailed_metrics.f1(i) = f1;
                detailed_metrics.support(i) = tp + fn;
                
                % 计算AUC
                y_binary = (Y_true == i);
                if length(unique(y_binary)) > 1
                    [~, ~, ~, auc] = perfcurve(y_binary, Y_prob, 1);
                    detailed_metrics.auc(i) = auc;
                end
            end
            
            % 宏平均和微平均
            detailed_metrics.macro_precision = mean(detailed_metrics.precision);
            detailed_metrics.macro_recall = mean(detailed_metrics.recall);
            detailed_metrics.macro_f1 = mean(detailed_metrics.f1);
            
            weights = detailed_metrics.support / sum(detailed_metrics.support);
            detailed_metrics.weighted_precision = sum(detailed_metrics.precision .* weights);
            detailed_metrics.weighted_recall = sum(detailed_metrics.recall .* weights);
            detailed_metrics.weighted_f1 = sum(detailed_metrics.f1 .* weights);
            detailed_metrics.weighted_auc = sum(detailed_metrics.auc .* weights);
            
            % 混淆矩阵
            detailed_metrics.confusion_matrix = confusionmat(Y_true, Y_pred);
            
            % 生成报告字符串
            report.summary = sprintf(...
                ['=== 性能评估报告 ===\n',...
                '总体准确率: %.4f\n',...
                '宏平均F1: %.4f\n',...
                '加权平均F1: %.4f\n',...
                '加权平均AUC: %.4f\n'],...
                report.accuracy,...
                detailed_metrics.macro_f1,...
                detailed_metrics.weighted_f1,...
                detailed_metrics.weighted_auc);
            
            % 按类别详细报告
            report.detailed = sprintf('%-20s %-10s %-10s %-10s %-10s\n', ...
                '故障类型', '精确率', '召回率', 'F1分数', '样本数');
            for i = 1:n_classes
                report.detailed = [report.detailed, sprintf(...
                    '%-20s %-10.4f %-10.4f %-10.4f %-10d\n',...
                    fault_types{i},...
                    detailed_metrics.precision(i),...
                    detailed_metrics.recall(i),...
                    detailed_metrics.f1(i),...
                    detailed_metrics.support(i))];
            end
        end
        
        function results = CrossValidationExperiment(features, labels, config, n_folds)
            % K折交叉验证实验
            
            if nargin < 4
                n_folds = 5;
            end
            
            n_samples = length(labels);
            fold_size = floor(n_samples / n_folds);
            
            results.fold_accuracies = zeros(n_folds, 1);
            results.fold_f1_scores = zeros(n_folds, 1);
            results.all_predictions = [];
            results.all_true_labels = [];
            
            for fold = 1:n_folds
                % 划分测试集
                test_start = (fold - 1) * fold_size + 1;
                test_end = min(fold * fold_size, n_samples);
                test_idx = test_start:test_end;
                train_idx = [1:test_start-1, test_end+1:n_samples];
                
                X_train = features(train_idx, :);
                Y_train = labels(train_idx);
                X_test = features(test_idx, :);
                Y_test = labels(test_idx);
                
                % 训练模型
                svm_model = fitcecoc(X_train, Y_train, ...
                    'Learners', templateSVM('Standardize', true));
                
                % 评估
                Y_pred = predict(svm_model, X_test);
                accuracy = mean(Y_pred == Y_test);
                
                % 计算F1
                f1 = mean(ExperimentAndEvaluationModule.computeF1(...
                    Y_test, Y_pred, max(Y_test)));
                
                results.fold_accuracies(fold) = accuracy;
                results.fold_f1_scores(fold) = f1;
                
                results.all_predictions = [results.all_predictions; Y_pred];
                results.all_true_labels = [results.all_true_labels; Y_test];
                
                fprintf('折 %d/%d 完成 - 准确率: %.4f, F1: %.4f\n', ...
                    fold, n_folds, accuracy, f1);
            end
            
            % 总体统计
            results.mean_accuracy = mean(results.fold_accuracies);
            results.std_accuracy = std(results.fold_accuracies);
            results.mean_f1 = mean(results.fold_f1_scores);
            results.std_f1 = std(results.fold_f1_scores);
            
            fprintf('\n交叉验证结果:\n');
            fprintf('平均准确率: %.4f ± %.4f\n', results.mean_accuracy, results.std_accuracy);
            fprintf('平均F1分数: %.4f ± %.4f\n', results.mean_f1, results.std_f1);
        end
        
        function sensitivity_results = SensitivityAnalysis(features, labels, config)
            % 模型对超参数的敏感性分析
            
            % 测试不同参数范围
            param_ranges.n_trees = [50, 100, 200, 300, 500];
            param_ranges.n_features = [5, 10, 20, 30, 50];
            param_ranges.tree_depth = [5, 10, 15, 20, 30];
            
            sensitivity_results.accuracies = [];
            sensitivity_results.params_tested = [];
            
            % 测试随机森林树的数量
            fprintf('测试随机森林树的数量...\n');
            for n_trees = param_ranges.n_trees
                rf = TreeBagger(n_trees, features, labels, 'Method', 'classification');
                pred = predict(rf, features);
                pred = str2double(pred);
                acc = mean(pred == labels);
                
                sensitivity_results.accuracies = [sensitivity_results.accuracies; acc];
                sensitivity_results.params_tested = [sensitivity_results.params_tested; ...
                    {sprintf('n_trees=%d', n_trees)}];
                
                fprintf('  树数=%d: 准确率=%.4f\n', n_trees, acc);
            end
            
            % 绘制敏感性曲线
            figure;
            plot(param_ranges.n_trees, sensitivity_results.accuracies(1:length(param_ranges.n_trees)), ...
                'o-', 'LineWidth', 2);
            xlabel('随机森林树的数量');
            ylabel('准确率');
            title('模型对树数的敏感性');
            grid on;
        end
        
        function f1_scores = computeF1(Y_true, Y_pred, n_classes)
            % 计算F1分数
            
            f1_scores = zeros(n_classes, 1);
            for i = 1:n_classes
                tp = sum((Y_true == i) & (Y_pred == i));
                fp = sum((Y_true ~= i) & (Y_pred == i));
                fn = sum((Y_true == i) & (Y_pred ~= i));
                
                precision = tp / (tp + fp + eps);
                recall = tp / (tp + fn + eps);
                f1_scores(i) = 2 * precision * recall / (precision + recall + eps);
            end
        end
        
        function visualizeExperimentResults(cv_results, sensitivity_results)
            % 可视化实验结果
            
            figure('Position', [100, 100, 1200, 500]);
            
            subplot(1,2,1);
            errorbar(1:length(cv_results.fold_accuracies), ...
                cv_results.fold_accuracies, ...
                cv_results.std_accuracy * ones(size(cv_results.fold_accuracies)), ...
                'o-', 'LineWidth', 2);
            xlabel('折数');
            ylabel('准确率');
            title('K折交叉验证结果');
            grid on;
            ylim([0, 1.1]);
            
            subplot(1,2,2);
            bar(sensitivity_results.accuracies);
            xlabel('参数配置');
            ylabel('准确率');
            title('敏感性分析结果');
            grid on;
            
            sgtitle('实验结果可视化');
        end
    end
end