%% Declare necesary variables

tempVarsPath = 'F:\MATLAB\Project 3\TempVars\';
savePath = 'F:\MATLAB\Project 3\Project 3.1\Run3\';
dataPath = 'F:\MATLAB\Project 3\Run4\data.mat';
range = 1992:2009;
numSubsampleDays = 7;
numPredictionSteps = 20;
%xlbls = ['118°W'; '114°W'; '110°W'; '106°W'; '102°W'];
%ylbls = ['22.9°N'; '26.9°N'; '30.9°N'];
xlbls = ['94°W'; '90°W'; '86°W'; '82°W'; '78°W'];
ylbls = ['22.9°N'; '26.9°N'; '30.9°N';];
segLength = 100;

if exist(strcat(savePath, 'progress.mat'), 'file') == 2
    load(strcat(savePath, 'progress.mat'));
    tempProg2 = tempProg;
else
    tempProg2 = 0;
end

if exist(dataPath, 'file') == 2
    load(dataPath);
    x = data{1};
    xLengthBase = size(x, 1);
    yLengthBase = size(x, 2);
    xLengthOver = floor((floor(xLengthBase / segLength) - 1) * segLength + segLength / 2 + mod(xLengthBase, segLength) / 2);
    yLengthOver = floor((floor(yLengthBase / segLength) - 1) * segLength + segLength / 2 + mod(yLengthBase, segLength) / 2);
    clear x
else
    DataPreparation;
    save(strcat(savePath, 'data.mat'), 'data');
end

numTimeStepsTrain = floor(0.9*numel(data));

xBound = floor(xLengthBase / segLength) + 1;
yBound = floor(yLengthBase / segLength) + 1;

PredReconstructionBase = zeros(xLengthBase, yLengthBase, numPredictionSteps);
PredReconstructionFull = zeros(xLengthBase, yLengthBase, numPredictionSteps);
PredReconstructionOver = zeros(xLengthOver, yLengthOver, numPredictionSteps);

opts = trainingOptions('adam', ...
    'MaxEpochs',250, ...
    'GradientThreshold',1, ...
    'InitialLearnRate',0.005, ...
    'LearnRateSchedule','piecewise', ...
    'LearnRateDropPeriod',125, ...
    'LearnRateDropFactor',0.2, ...
    'Verbose',0);

%% Load in preran data
if exist(strcat(savePath, 'coeffs.mat'), 'file') == 2
    load(strcat(savePath, 'coeffs.mat'));
else
    coeffs = zeros(numPredictionSteps, (numel(data) - numPredictionSteps - numTimeStepsTrain) + 1);
end
if exist(strcat(savePath, 'rmses.mat'), 'file') == 2
    load(strcat(savePath, 'rmses.mat'));
else
    rmses = zeros(numPredictionSteps, (numel(data) - numPredictionSteps - numTimeStepsTrain) + 1);
end


%% Preprocessing and calculating V's for base

for i = 1:xBound
    for j = 1:yBound
        fprintf("Generating base EOFs phase %d-%d: ", i, j);
        xRegion = int32(i - 1) * segLength + 1:min(int32(i) * segLength, xLengthBase);
        yRegion = int32(j - 1) * segLength + 1:min(int32(j) * segLength, yLengthBase);
        
        XData = zeros(numel(xRegion) * numel(yRegion), numel(data));
        for k = 1:numel(data)
            temp = data{k};
            temp = temp(xRegion, yRegion);
            temp = temp(:);
            XData(:,k) = temp;
        end
        clear temp;
        
        %Remove nans
        XData = XData(any(~isnan(XData), 2), :)';
        
        %Decomposition
        [~, ~, v] = svd(XData);
        save(strcat(tempVarsPath, num2str(i), '-', num2str(j), 'BaseV.mat'), 'v', '-v7.3');
        clear v;
        fprintf("Done\n");
    end
end
fprintf("\n");

%% Preprocessing and calculating V's for overlapping
for i = 1:xBound - 1
    for j = 1:yBound - 1
        fprintf("Generating overlap EOFs phase %d-%d: ", i, j);
        xRegion = (int32(i - 1) * segLength + 1:min(int32(i) * segLength, xLengthOver)) + segLength / 2;
        yRegion = (int32(j - 1) * segLength + 1:min(int32(j) * segLength, yLengthOver)) + segLength / 2;
        
        XData = zeros(numel(xRegion) * numel(yRegion), numel(data));
        for k = 1:numel(data)
            temp = data{k};
            temp = temp(xRegion, yRegion);
            temp = temp(:);
            XData(:,k) = temp;
        end
        clear temp;
        
        %Remove nans
        XData = XData(any(~isnan(XData), 2), :)';
        
        %Decomposition
        [~, ~, v] = svd(XData);
        save(strcat(tempVarsPath, num2str(i), '-', num2str(j), 'OverV.mat'), 'v', '-v7.3');
        clear v;
        fprintf("Done\n");
    end
end
fprintf("\n");

%% Start Algorithm!!!
for ss = tempProg2:(numel(data) - numPredictionSteps - numTimeStepsTrain)
    tempProg = ss;
    fprintf("RUNNING ITERATION %d OUT OF %d:\n\n", (ss + 1), (numel(data) - numPredictionSteps - numTimeStepsTrain) + 1);
    save(strcat(savePath, 'progress.mat'), 'tempProg');
    numTimeStepsTest = numel(data) - (numTimeStepsTrain + ss);
    
    %% Network training of base
    for i = 1:xBound
        for j = 1:yBound
            fprintf("Initializing base phase %d-%d: ", i, j);
            xRegion = int32(i - 1) * segLength + 1:min(int32(i) * segLength, xLengthBase);
            yRegion = int32(j - 1) * segLength + 1:min(int32(j) * segLength, yLengthBase);
            
            XData = zeros(numel(xRegion) * numel(yRegion), numel(data));
            for k = 1:numel(data)
                temp = data{k};
                temp = temp(xRegion, yRegion);
                temp = temp(:);
                XData(:,k) = temp;
            end
            clear temp;
            
            XData(isnan(XData)) = 0;
            
            %Remove zeros whilst storing rows
            tempZ = [];
            for k = 1:size(XData, 1)
                if max(abs(XData(k,:))) == 0
                    tempZ = cat(1, tempZ, k);
                end
            end
            XData = XData(any(XData, 2), :)';
            
            %Decomposition
            load(strcat(tempVarsPath, num2str(i), '-', num2str(j), 'BaseV.mat'));
            XData = XData / v';
            
            %Transpose for training purposes
            XData = XData';
            
            XTrain = XData(:,1:numTimeStepsTrain + ss);
            YTrain = XData(:,2:numTimeStepsTrain + ss + 1);
            
            inputSize = size(XData, 1);
            if inputSize < 1
                PredReconstructionBase(xRegion, yRegion, 1) = nan;
                fprintf("Done\n");
                continue
            end
            numHiddenUnits = 500;
            numHiddenUnits1 = 300;
            layers = [ ...
                sequenceInputLayer(inputSize)
                lstmLayer(numHiddenUnits)
                lstmLayer(numHiddenUnits1)
                fullyConnectedLayer(inputSize)
                regressionLayer];
            
            net = trainNetwork(XTrain,YTrain,layers,opts);
            net = predictAndUpdateState(net,XTrain);
            
            %Predict first day
            [net,YPred] = predictAndUpdateState(net,YTrain(:,end));
            
            %Store net
            save(strcat(tempVarsPath, num2str(i), '-', num2str(j), 'BaseNet.mat'), 'net');
            clear net
            
            %Full reconstruct
            recon = YPred(:, 1)';
            recon = recon * v';
            PredFullrecon = recon';
            for k = 1:numel(tempZ)
                PredFullrecon = cat(1, PredFullrecon(1:tempZ(k) - 1), nan, PredFullrecon(tempZ(k):end));
            end
            PredFullrecon = reshape(PredFullrecon, [xRegion(end) - xRegion(1) + 1, yRegion(end) - yRegion(1) + 1]);
            PredReconstructionBase(xRegion, yRegion, 1) = PredFullrecon;
            clear recon;
            fprintf("Done\n");
        end
    end
    
    %% Network training of overlap
    for i = 1:xBound - 1
        for j = 1:yBound - 1
            fprintf("Initializing overlap phase %d-%d: ", i, j);
            xRegion = (int32(i - 1) * segLength + 1:min(int32(i) * segLength, xLengthOver)) + segLength / 2;
            yRegion = (int32(j - 1) * segLength + 1:min(int32(j) * segLength, yLengthOver)) + segLength / 2;
            
            XData = zeros(numel(xRegion) * numel(yRegion), numel(data));
            for k = 1:numel(data)
                temp = data{k};
                temp = temp(xRegion, yRegion);
                temp = temp(:);
                XData(:,k) = temp;
            end
            clear temp;
            
            XData(isnan(XData)) = 0;
            
            %Remove zeros whilst storing rows
            tempZ = [];
            for k = 1:size(XData, 1)
                if max(abs(XData(k,:))) == 0
                    tempZ = cat(1, tempZ, k);
                end
            end
            XData = XData(any(XData, 2), :)';
            
            %Decomposition
            load(strcat(tempVarsPath, num2str(i), '-', num2str(j), 'OverV.mat'));
            XData = XData / v';
            
            %Transpose for training purposes
            XData = XData';
            
            XTrain = XData(:,1:numTimeStepsTrain + ss);
            YTrain = XData(:,2:numTimeStepsTrain + ss + 1);
            
            inputSize = size(XData, 1);
            if inputSize < 1
                PredReconstructionOver(xRegion - segLength / 2, yRegion - segLength / 2, 1) = nan;
                fprintf("Done\n");
                continue
            end
            numHiddenUnits = 500;
            numHiddenUnits1 = 300;
            layers = [ ...
                sequenceInputLayer(inputSize)
                lstmLayer(numHiddenUnits)
                lstmLayer(numHiddenUnits1)
                fullyConnectedLayer(inputSize)
                regressionLayer];
            
            net = trainNetwork(XTrain,YTrain,layers,opts);
            net = predictAndUpdateState(net,XTrain);
            
            %Predict first day
            [net,YPred] = predictAndUpdateState(net,YTrain(:,end));
            
            %Store net
            save(strcat(tempVarsPath, num2str(i), '-', num2str(j), 'OverNet.mat'), 'net');
            clear net
            
            %Full reconstruct
            recon = YPred(:, 1)';
            recon = recon * v';
            PredFullrecon = recon';
            for k = 1:numel(tempZ)
                PredFullrecon = cat(1, PredFullrecon(1:tempZ(k) - 1), nan, PredFullrecon(tempZ(k):end));
            end
            PredFullrecon = reshape(PredFullrecon, [xRegion(end) - xRegion(1) + 1, yRegion(end) - yRegion(1) + 1]);
            PredReconstructionOver(xRegion - segLength / 2, yRegion - segLength / 2, 1) = PredFullrecon;
            clear recon;
            fprintf("Done\n");
        end
    end
    
    PredReconstructionOver(:,:,1) = smoothSeg(PredReconstructionOver(:,:,1), segLength, 4);
    PredReconstructionBase(:,:,1) = smoothSeg(PredReconstructionBase(:,:,1), segLength, 4);
    PredReconstructionFull(:,:,1) = overlapIntegrate(PredReconstructionBase(:,:,1), PredReconstructionOver(:,:,1), segLength);
    PredReconstructionFull(:,:,1) = smoothSeg(PredReconstructionFull(:,:,1), segLength / 2, 4);
    
    %% Forecasting
    for d = 2:numPredictionSteps
        fprintf("Forecasting week %d: ", d);
        
        %% Forecast base
        for i = 1:xBound
            for j = 1:yBound
                xRegion = int32(i - 1) * segLength + 1:min(int32(i) * segLength, xLengthBase);
                yRegion = int32(j - 1) * segLength + 1:min(int32(j) * segLength, yLengthBase);
                
                temp = PredReconstructionFull(xRegion, yRegion, d - 1);
                temp = temp(:);
                tempZ = [];
                for k = 1:size(temp, 1)
                    if isnan(temp(k))
                        tempZ = cat(1, tempZ, k);
                    end
                end
                temp = temp(any(~isnan(temp), 2), :)';
                
                %Deconstruct to u * s
                load(strcat(tempVarsPath, num2str(i), '-', num2str(j), 'BaseV.mat'));
                temp = temp / v';
                
                %Transpose for training purposes
                temp = temp';
                inputSize = size(temp, 1);
                if inputSize < 1
                    PredReconstructionBase(xRegion, yRegion, d) = nan;
                    continue
                end
                %Load in net
                load(strcat(tempVarsPath, num2str(i), '-', num2str(j), 'BaseNet.mat'));
                %Predict next step
                [net, YPred] = predictAndUpdateState(net, temp);
                %Store net
                save(strcat(tempVarsPath, num2str(i), '-', num2str(j), 'BaseNet.mat'), 'net');
                clear net
                
                %Full reconstruct
                recon = YPred';
                
                recon = recon * v';
                PredFullrecon = recon';
                for k = 1:numel(tempZ)
                    PredFullrecon = cat(1, PredFullrecon(1:tempZ(k) - 1), nan, PredFullrecon(tempZ(k):end));
                end
                PredFullrecon = reshape(PredFullrecon, [xRegion(end) - xRegion(1) + 1, yRegion(end) - yRegion(1) + 1]);
                PredReconstructionBase(xRegion, yRegion, d) = PredFullrecon;
                clear recon;
            end
        end
        
        %% Forecast overlap
        for i = 1:xBound - 1
            for j = 1:yBound - 1
                xRegion = (int32(i - 1) * segLength + 1:min(int32(i) * segLength, xLengthOver)) + segLength / 2;
                yRegion = (int32(j - 1) * segLength + 1:min(int32(j) * segLength, yLengthOver)) + segLength / 2;
                
                temp = PredReconstructionFull(xRegion, yRegion, d - 1);
                temp = temp(:);
                temp(isnan(temp)) = 0;
                tempZ = [];
                for k = 1:size(temp, 1)
                    if max(abs(temp(k,:))) == 0
                        tempZ = cat(1, tempZ, k);
                    end
                end
                temp = temp(any(temp, 2), :)';
                
                %Deconstruct to u * s
                load(strcat(tempVarsPath, num2str(i), '-', num2str(j), 'OverV.mat'));
                temp = temp / v';
                
                %Transpose for training purposes
                temp = temp';
                inputSize = size(temp, 1);
                if inputSize < 1
                    PredReconstructionOver(xRegion - segLength / 2, yRegion - segLength / 2, d) = nan;
                    continue
                end
                %Load in net
                load(strcat(tempVarsPath, num2str(i), '-', num2str(j), 'OverNet.mat'));
                %Predict next step
                [net, YPred] = predictAndUpdateState(net, temp);
                %Store net
                save(strcat(tempVarsPath, num2str(i), '-', num2str(j), 'OverNet.mat'), 'net');
                clear net
                
                %Full reconstruct
                recon = YPred';
                
                recon = recon * v';
                PredFullrecon = recon';
                for k = 1:numel(tempZ)
                    PredFullrecon = cat(1, PredFullrecon(1:tempZ(k) - 1), nan, PredFullrecon(tempZ(k):end));
                end
                PredFullrecon = reshape(PredFullrecon, [xRegion(end) - xRegion(1) + 1, yRegion(end) - yRegion(1) + 1]);
                PredReconstructionOver(xRegion - segLength / 2, yRegion - segLength / 2, d) = PredFullrecon;
                clear recon;
            end
        end
        
        PredReconstructionOver(:,:,d) = smoothSeg(PredReconstructionOver(:,:,d), segLength, 4);
        PredReconstructionBase(:,:,d) = smoothSeg(PredReconstructionBase(:,:,d), segLength, 4);
        PredReconstructionFull(:,:,d) = overlapIntegrate(PredReconstructionBase(:,:,d), PredReconstructionOver(:,:,d), segLength);
        PredReconstructionFull(:,:,d) = smoothSeg(PredReconstructionFull(:,:,d), segLength / 2, 4);
        
        fprintf("Done\n");
    end
    
    %% Calculate performance measures
    for i = 1:numPredictionSteps
        test1 = PredReconstructionFull(:,:,i);
        test2 = data{numTimeStepsTrain + i + ss};
        test1(isnan(test1)) = 0;
        test2(isnan(test2)) = 0;
        test1 = test1(:);
        test2 = test2(:);
        rmses(i, ss + 1) = sqrt(mean((test1-test2).^2));
        test1 = test1(any(test1, 2), :)';
        test2 = test2(any(test2, 2), :)';
        c = corrcoef(test1, test2);
        coeffs(i, ss + 1) = c(1, 2);
    end
    save(strcat(savePath, 'coeffs.mat'), 'coeffs');
    save(strcat(savePath, 'rmses.mat'), 'rmses');
    clear test1 test2;
    
    mkdir(strcat(savePath, num2str(ss)));
    save(strcat(savePath, num2str(ss), '\preds.mat'), 'PredReconstructionFull');
    figure('units','normalized','outerposition',[0 0 1 0.5], 'visible', 'off')
    for i = 1:numPredictionSteps
        subplot(1, 2, 1);
        surf(PredReconstructionFull(:,:,i)', 'edgecolor', 'none')
        axis([0, xLengthBase, 0, yLengthBase, -1, 1.5])
        view([0 0 10])
        title('Forecasted Week', 'FontSize', 14)
        set(gca, 'xtick', [100 * (1:floor(xLengthBase / 100))])
        set(gca, 'ytick', [100 * (1:floor(yLengthBase / 100))])
        xticklabels(xlbls);
        yticklabels(ylbls);
        set(gca, 'FontSize', 14);
        set(gca, 'LineWidth', 1.5);
        set(gca, 'TickDir', 'in');
        box on
        grid off
        set(gca, 'Layer', 'top')
        caxis([-0.4 0.9])
%         hold on
%         contour3(PredReconstructionFull(:,:,i)', 'LevelList', 0.45, 'Color', 'black')
        
        subplot(1, 2, 2);
        surf(data{numTimeStepsTrain + i + ss}', 'edgecolor', 'none')
        axis([0, xLengthBase, 0, yLengthBase, -1, 1.5])
        view([0 0 10])
        title('Actual Week', 'FontSize', 14)
        set(gca, 'xtick', [100 * (1:floor(xLengthBase / 100))])
        set(gca, 'ytick', [100 * (1:floor(yLengthBase / 100))])
        set(gca, 'TickDir', 'in');
        xticklabels(xlbls);
        yticklabels(ylbls);
        set(gca, 'FontSize', 14);
        set(gca, 'LineWidth', 1.5);
        box on
        grid off
        set(gca, 'Layer', 'top')
        caxis([-0.4 0.9])
%         hold on
%         contour3(data{numTimeStepsTrain + i + ss}', 'LevelList', 0.45, 'Color', 'black')
        
        
        % hp4 = get(subplot(1,2,2),'Position');
        % c = colorbar('southoutside', 'Position', [hp4(2)  hp4(2) - .1  (hp4(1)+hp4(2) * 2) 0.04]);
        c = colorbar('eastoutside');
        c.Label.String = 'Sea Surface Height (SSH) in meters (m)';
        c.FontSize = 14;
        set(c, 'LineWidth', 1.5);
        
        saveas(gcf, strcat(savePath, num2str(ss), '\', num2str(i), '.png'));
    end
    
end


%% Demonstrate overLappingIntegration (comment out all code above)
% a = zeros(540, 384);
% b = ones(470, 292);
% x = overlapIntegrate(a, b, 100);
% surf(x, 'edgecolor', 'none')

%% Overlapping Function
function y = overlapIntegrate(base, seg, segLength)

ret = base;
for i = 1:size(seg, 1)
    for j = 1:size(seg, 2)
        %Obtain corner of segment
        xCorner = min(floor((i - 1 + segLength / 2) / segLength) * segLength + 1, size(seg, 1));
        yCorner = min(floor((j - 1 + segLength / 2) / segLength) * segLength + 1, size(seg, 2));
        %Obtain center of segment
        xCenter = floor((i - 1) / segLength) * segLength + min(segLength, size(seg, 1) - floor((i - 1) / segLength) * segLength) / 2;
        yCenter = floor((j - 1) / segLength) * segLength + min(segLength, size(seg, 2) - floor((j - 1) / segLength) * segLength) / 2;
        
        f = base(i + segLength / 2, j + segLength / 2);
        g = seg(i, j);
        r1 = pdist([xCorner, yCorner; i, j], 'euclidean');
        r2 = pdist([xCenter, yCenter; i, j], 'euclidean');
        %Debug
        %fprintf("%.2f %.2f\n", r1, r2);
        %ret(i + segLength / 2, j + segLength / 2) = (g * r1 + f * r2) / (r1 + r2);
        
        sinWeight = sin((r1 / (r1 + r2)) * pi / 2);
        
        if isnan(f) && isnan(g)
            ret(i + segLength / 2, j + segLength / 2) = nan;
        end
        
        ret(i + segLength / 2, j + segLength / 2) = (1 - sinWeight) * f + (sinWeight) * g;
    end
end
y = ret;

end

%% Smoothing function
function y = smoothSeg(M, segLength, width)
xBound = (floor(size(M, 1) / segLength + 1));
yBound = (floor(size(M, 2) / segLength + 1));
ret = M;
for i = 1:xBound - 1
    xRegion = segLength * i - width + 1:segLength * i + width;
    
    for j = 1:size(M, 2)
        tempRegion = xRegion;
        while numel(tempRegion) > 0 && isnan(M(tempRegion(1), j))
            tempRegion = tempRegion(2:end);
        end
        if numel(tempRegion) == 0
            continue
        end
        while numel(tempRegion) > 0 && isnan(M(tempRegion(end), j))
            tempRegion = tempRegion(1:end - 1);
        end
        if numel(tempRegion) == 0
            continue
        end
        curve = linspace(1, 0, numel(tempRegion));
        
        for k = 1:numel(tempRegion)
            if isnan(M(tempRegion(k), j))
                continue
            end
            ret(tempRegion(k), j) = curve(k) * M(tempRegion(1), j) + (1 - curve(k)) * M(tempRegion(end), j);
        end
        
        
    end
    
    %         for j = 1:numel(xRegion)
    %             ret(xRegion(j), :) = curve(j) * ret(xRegion(1),:) + (1 - curve(j)) * ret(xRegion(end),:);
    %         end
end
for i = 1:yBound - 1
    yRegion = segLength * i - width + 1:segLength * i + width;
    
    for j = 1:size(M, 1)
        tempRegion = yRegion;
        while numel(tempRegion) > 0 && isnan(M(j, tempRegion(1)))
            tempRegion = tempRegion(2:end);
        end
        if numel(tempRegion) == 0
            continue
        end
        while numel(tempRegion) > 0 && isnan(M(j, tempRegion(end)))
            tempRegion = tempRegion(1:end - 1);
        end
        if numel(tempRegion) == 0
            continue
        end
        curve = linspace(1, 0, numel(tempRegion));
        
        for k = 1:numel(tempRegion)
            if isnan(M(j, tempRegion(k)))
                continue
            end
            ret(j, tempRegion(k)) = curve(k) * ret(j, tempRegion(1)) + (1 - curve(k)) * ret(j, tempRegion(end));
        end
        
        
    end
    
    %         for j = 1:numel(xRegion)
    %             ret(xRegion(j), :) = curve(j) * ret(xRegion(1),:) + (1 - curve(j)) * ret(xRegion(end),:);
    %         end
end
y = ret;
end