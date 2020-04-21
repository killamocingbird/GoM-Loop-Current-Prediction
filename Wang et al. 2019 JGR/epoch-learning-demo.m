%{

This portion of code shows the LSTM effectively learning through epochs.
This code is for visualization purposes only and is independent of the
main algorithm.

%}

clear; clc;
savePath = 'F:\MATLAB\Project 3\EpochTest2\';
tempPath = 'F:\MATLAB\Project 3\TempVars\';
load('F:\MATLAB\Project 3\Run4\data.mat')

numUpTo = 5;
xRegions = [201:300; 301:400];
yRegions = [101:200; 201:300];

%Generate PCs & EOFs
for ii = 1:size(xRegions, 1)
    for jj = 1:size(yRegions, 1)
        fprintf("Decomposing Region %d %d\n", ii, jj);
        xRegion = xRegions(ii, :);
        yRegion = yRegions(jj, :);
        XData = zeros(numel(xRegion) * numel(yRegion), numel(data));
        for k = 1:numel(data)
            temp = data{k};
            temp = temp(xRegion, yRegion);
            XData(:,k) = temp(:);
        end
        clear temp;
        
        %Remove nans
        tempZ = any(XData, 2);
        XData = XData(tempZ, :)';
        save(strcat(tempPath, 'z', num2str(ii), '-', num2str(jj), '.mat'), 'tempZ')
        
        %Decompose
        [u s v] = svd(XData);
        XData = u * s;
        XData = XData';
        save(strcat(tempPath, 'd', num2str(ii), '-', num2str(jj), '.mat'), 'XData')
        save(strcat(tempPath, 'v', num2str(ii), '-', num2str(jj), '.mat'), 'v');
    end
end


fprintf("Beginning Epoch Testing:\n")
for ee = 1:250
    fprintf("Epoch %d: ", ee);
    Reconstruction = zeros(200, 200);
    
    for ii = 1:size(xRegions, 1)
        for jj = 1:size(yRegions, 1)
            xRegion = xRegions(ii, :);
            yRegion = yRegions(jj, :);
            %Load everything
            load(strcat(tempPath, 'd', num2str(ii), '-', num2str(jj), '.mat'))
            load(strcat(tempPath, 'z', num2str(ii), '-', num2str(jj), '.mat'))
            load(strcat(tempPath, 'v', num2str(ii), '-', num2str(jj), '.mat'))
            
            XTrain = XData(:,1:numUpTo - 1);
            YTrain = XData(:,2:numUpTo);
            
            
            opts = trainingOptions('adam', ...
                'MaxEpochs',ee, ...
                'GradientThreshold',1, ...
                'InitialLearnRate',0.005, ...
                'LearnRateSchedule','piecewise', ...
                'LearnRateDropPeriod',125, ...
                'LearnRateDropFactor',0.2, ...
                'Verbose',0);
            
            inputSize = size(XData, 1);
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
            [net,YPred] = predictAndUpdateState(net,YTrain(:,end));
            recon = YPred(:, 1)';
            recon = recon * v';
            PredFullrecon = nan(numel(tempZ), 1);
            PredFullrecon(tempZ, :) = recon';
            Reconstruction(xRegion - 200, yRegion - 100) = reshape(PredFullrecon, [xRegion(end) - xRegion(1) + 1, yRegion(end) - yRegion(1) + 1]);
        end
    end
    
    
    figure('units','normalized','outerposition',[0 0 0.7 1], 'visible', 'off')
    surf(Reconstruction', 'edgecolor', 'none')
    axis([0, 200, 0, 200, -1, 1.5])
    axis off
    box on
    view([0 0 10])
    caxis([-0.4 0.9])
    fprintf("Done\n")
    saveas(gcf, strcat(savePath, num2str(ee), '.png'));
end






