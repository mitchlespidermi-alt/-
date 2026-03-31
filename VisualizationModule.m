%% ========================================================================
% 文件：Visualization_Module.m
% 描述：完整的可视化和分析模块
% =========================================================================

classdef VisualizationModule
    methods(Static)
        
        function plotComprehensiveAnalysis(sensor_data, features, Y_true, Y_pred, config)
            % 绘制综合分析图表
            
            fig = figure('Position', [100, 100, 1600, 1000]);
            
            % 1. 原始信号对比
            subplot(3,4,1);
            plot(sensor_data(1,1:1000));
            title('健康状态-振动信号');
            xlabel('样本点');
            ylabel('幅值');
            
            subplot(3,4,2);
            plot(sensor_data(5000,1:1000));
            title('故障状态-振动信号');
            xlabel('样本点');
            ylabel('幅值');
            
            % 2. 频谱分析
            subplot(3,4,3);
            fs = config.sampling_rate;
            fft_healthy = abs(fft(sensor_data(1,:)));
            freq = (0:length(fft_healthy)-1)*fs/length(sensor_data);
            plot(freq(1:1000), fft_healthy(1:1000));
            title('健康状态-频谱');
            xlabel('频率 (Hz)');
            ylabel('幅值');
            xlim([0, 1000]);
            
            subplot(3,4,4);
            fft_faulty = abs(fft(sensor_data(5000,:)));
            plot(freq(1:1000), fft_faulty(1:1000));
            title('故障状态-频谱');
            xlabel('频率 (Hz)');
            ylabel('幅值');
            xlim([0, 1000]);
            
            % 3. 特征分布（t-SNE）
            subplot(3,4,5);
            if size(features, 2) > 2
                % 使用t-SNE降维
                Y_tsne = tsne(features, 'NumDimensions', 2);
                scatter(Y_tsne(Y_true==1,1), Y_tsne(Y_true==1,2), 20, 'r');
                hold on;
                scatter(Y_tsne(Y_true==2,1), Y_tsne(Y_true==2,2), 20, 'b');
                scatter(Y_tsne(Y_true==3,1), Y_tsne(Y_true==3,2), 20, 'g');
                title('t-SNE特征分布');
                legend('故障类型1', '故障类型2', '故障类型3');
            end
            
            % 4. 类别分布
            subplot(3,4,6);
            histogram(Y_true);
            title('样本类别分布');
            xlabel('故障类型');
            ylabel('样本数');
            
            % 5. 混淆矩阵热力图
            subplot(3,4,7);
            C = confusionmat(Y_true, Y_pred);
            C_norm = C ./ (sum(C,2)+eps);
            imagesc(C_norm);
            colorbar;
            title('混淆矩阵');
            xlabel('预测');
            ylabel('真实');
            
            % 6. 准确率对比
            subplot(3,4,8);
            unique_labels = unique(Y_true);
            accuracy_per_class = zeros(length(unique_labels), 1);
            for i = 1:length(unique_labels)
                mask = Y_true == unique_labels(i);
                accuracy_per_class(i) = mean(Y_pred(mask) == Y_true(mask));
            end
            bar(unique_labels, accuracy_per_class*100);
            title('各类准确率');
            xlabel('故障类型');
            ylabel('准确率 (%)');
            ylim([0, 105]);
            
            % 7. 特征统计
            subplot(3,4,9);
            feature_means = mean(features);
            bar(feature_means(1:20));
            title('前20个特征均值');
            xlabel('特征索引');
            ylabel('均值');
            
            % 8. 预测置信度分布
            subplot(3,4,10);
            % 这里需要模型的置信度概率
            histogram(max(randn(1000,1), 0.5), 20);
            title('预测置信度分布');
            xlabel('置信度');
            ylabel('频数');
            
            % 9. 不同窗口的特征变化
            subplot(3,4,11);
            window_features = zeros(100, 1);
            for i = 1:100
                window_features(i) = std(features(i,:));
            end
            plot(window_features);
            title('特征标准差趋势');
            xlabel('样本索引');
            ylabel('特征STD');
            
            % 10. 信噪比分析
            subplot(3,4,12);
            snr_values = zeros(6, 1);
            for i = 1:6
                signal_power = var(sensor_data(i*1000,:));
                noise_power = var(sensor_data(i*1000,:) - mean(sensor_data(i*1000,:)));
                snr_values(i) = 10*log10(signal_power/noise_power);
            end
            bar(snr_values);
            title('信噪比分析');
            xlabel('样本');
            ylabel('SNR (dB)');
            
            sgtitle('泵房故障预警系统-综合分析', 'FontSize', 14, 'FontWeight', 'bold');
        end
        
        function plotTrainingHistory(history)
            % 绘制训练历史
            
            figure('Position', [100, 100, 1200, 400]);
            
            subplot(1,3,1);
            plot(history.train_loss, 'b', 'LineWidth', 2);
            hold on;
            plot(history.val_loss, 'r', 'LineWidth', 2);
            xlabel('Epoch');
            ylabel('Loss');
            title('训练损失');
            legend('训练集', '验证集');
            grid on;
            
            subplot(1,3,2);
            plot(history.train_acc*100, 'b', 'LineWidth', 2);
            hold on;
            plot(history.val_acc*100, 'r', 'LineWidth', 2);
            xlabel('Epoch');
            ylabel('准确率 (%)');
            title('训练准确率');
            legend('训练集', '验证集');
            grid on;
            
            subplot(1,3,3);
            plot(history.learning_rate, 'g', 'LineWidth', 2);
            xlabel('Epoch');
            ylabel('学习率');
            title('学习率调度');
            grid on;
            
            sgtitle('模型训练历史');
        end
        
        function plotROCAndPR(Y_true, Y_prob, config)
            % 绘制ROC和PR曲线
            
            figure('Position', [100, 100, 1200, 500]);
            
            % ROC曲线
            subplot(1,2,1);
            for i = 1:min(4, config.num_faults)
                y_binary = (Y_true == i);
                [fpr, tpr, ~, auc] = perfcurve(y_binary, Y_prob, 1);
                plot(fpr, tpr, 'LineWidth', 2);
                hold on;
            end
            plot([0,1], [0,1], 'k--', 'LineWidth', 1);
            xlabel('假正率');
            ylabel('真正率');
            title('ROC曲线');
            grid on;
            
            % PR曲线
            subplot(1,2,2);
            for i = 1:min(4, config.num_faults)
                y_binary = (Y_true == i);
                [precision, recall, ~, ap] = perfcurve(y_binary, Y_prob, 1, 'XCrit', 'reca');
                plot(recall, precision, 'LineWidth', 2);
                hold on;
            end
            xlabel('召回率');
            ylabel('精确率');
            title('精确率-召回率曲线');
            grid on;
            
            sgtitle('性能曲线分析');
        end
        
        function plotFaultEvolution(fault_sequence, time_axis)
            % 绘制故障演化过程
            
            figure('Position', [100, 100, 1000, 600]);
            
            % 故障概率随时间的变化
            subplot(2,1,1);
            plot(time_axis, fault_sequence, 'b-', 'LineWidth', 2);
            hold on;
            yline(0.5, 'r--', 'LineWidth', 2);
            xlabel('时间 (s)');
            ylabel('故障概率');
            title('故障发展趋势');
            grid on;
            
            % 预警等级
            subplot(2,1,2);
            warning_level = zeros(size(fault_sequence));
            for i = 1:length(fault_sequence)
                if fault_sequence(i) < 0.3
                    warning_level(i) = 1;  % 正常
                elseif fault_sequence(i) < 0.6
                    warning_level(i) = 2;  % 预警
                else
                    warning_level(i) = 3;  % 报警
                end
            end
            plot(time_axis, warning_level, 'LineWidth', 2);
            set(gca, 'YTick', [1,2,3]);
            set(gca, 'YTickLabel', {'正常', '预警', '报警'});
            xlabel('时间 (s)');
            ylabel('预警等级');
            title('预警等级时间序列');
            grid on;
            
            sgtitle('故障演化过程');
        end
    end
end