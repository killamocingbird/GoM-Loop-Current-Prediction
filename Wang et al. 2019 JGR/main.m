%{

To work around RAM bottle necks, this program uses a temporary folder,
essentially using harddrive space as RAM. If you happen to have a significant
amount of RAM ( > 128gb) and would like to speed up the program, you may
remove instances where variables are saved and loaded while assigning new
names to them. 

If there are any errors in this code, please contact me at jlwang5@illinois.edu 

%}



%Path for temporary variables
tempVarsPath = 'F:\MATLAB\Project 3\TempVars\';

%Path for output of algorithm
savePath = 'F:\MATLAB\Project 3\Run5\';

%Range of years in data
range = 1992:2009;
data = cell(numel(range) * 365 + 3, 1);
n = 7;

%Number of steps (weeks) to predict per window
numPredictionSteps = 20;
xlbls = ['94°W'; '90°W'; '86°W'; '82°W'; '78°W'];
ylbls = ['22.3°N'; '25.7°N'; '29.1°N'];

if exist(strcat(savePath, 'data.mat'), 'file') == 2
    load(strcat(savePath, 'data.mat'));
    x = data{1};
    xLength = size(x, 1);
    yLength = size(x, 2);
    clear x
else
    fprintf("Loading in Data:\n");
    counter = 1;
    for i = range
        load(strcat('x', num2str(i), '.mat'));
        data(counter:counter + numel(x) - 1) = x;
        counter = counter + numel(x);
        fprintf("Year %d loaded\n", i);
    end
    sample = x{1};
    xLength = size(sample, 1);
    yLength = size(sample, 2);
    clear x sample;
    data = subsample(data', n);
    fprintf("Done\n\n");
    save(strcat(savePath, 'data.mat'), 'data');
end

segLength = 100;
numTimeStepsTrain = floor(0.9*numel(data));
xBound = floor(xLength / segLength) + 1;
yBound = floor(yLength / segLength) + 1;

%coeffs = zeros(numel(data) - int32(numel(data) * 0.9) + 1, xBound * yBound);
PredReconstruction = zeros(xLength, yLength, numPredictionSteps);
realPCs = zeros(xBound, yBound, numel(data) - numTimeStepsTrain);
if exist(strcat(savePath, 'predPCs.mat'), 'file') == 2
    load(strcat(savePath, 'predPCs.mat'));
else
    predPCs = zeros(xBound, yBound, numel(data) - numTimeStepsTrain, numPredictionSteps);
end

opts = trainingOptions('adam', ...
    'MaxEpochs',250, ...
    'GradientThreshold',1, ...
    'InitialLearnRate',0.005, ...
    'LearnRateSchedule','piecewise', ...
    'LearnRateDropPeriod',125, ...
    'LearnRateDropFactor',0.2, ...
    'Verbose',0); %, ...
%'Plots','training-progress');

if exist(strcat(savePath, 'coeffs.mat'), 'file') == 2
    load(strcat(savePath, 'coeffs.mat'));
else
    coeffs = zeros(numPredictionSteps, (numel(data) - numPredictionSteps - numTimeStepsTrain) + 1);
end
if exist(strcat(savePath, 'coeffs.mat'), 'file') == 2
    load(strcat(savePath, 'rmses.mat'));
else
    rmses = zeros(numPredictionSteps, (numel(data) - numPredictionSteps - numTimeStepsTrain) + 1);
end

%Get v for each path
for i = 1:xBound
    for j = 1:yBound
        fprintf("Generating EOFs phase %d-%d: ", i, j);
        xRegion = int32(i - 1) * segLength + 1:min(int32(i) * segLength, xLength);
        yRegion = int32(j - 1) * segLength + 1:min(int32(j) * segLength, yLength);
        
        XData = zeros(numel(xRegion) * numel(yRegion), numel(data));
        for k = 1:numel(data)
            temp = data{k};
            temp = temp(xRegion, yRegion);
            temp = temp(:);
            XData(:,k) = temp;
        end
        clear temp;
        
        %Remove nans
        XData = XData(any(XData, 2), :)';
        
        %Decomposition
        [u, s, v] = svd(XData);
        tempPC = u * s;
        realPCs(i, j, :) = tempPC(numTimeStepsTrain + 1:numel(data), 1)';
        save(strcat(tempVarsPath, num2str(i), '-', num2str(j), 'v.mat'), 'v', '-v7.3');
        clear u s v;
        fprintf("Done\n");
    end
end
fprintf("\n");

save(strcat(savePath, 'realPCs.mat'), 'realPCs');

for ss = 54:(numel(data) - numPredictionSteps - numTimeStepsTrain)
    fprintf("RUNNING ITERATION %d OUT OF %d:\n\n", (ss + 1), (numel(data) - numPredictionSteps - numTimeStepsTrain) + 1);
    numTimeStepsTest = numel(data) - (numTimeStepsTrain + ss);
    %Train all networks for all patches
    for i = 1:xBound
        for j = 1:yBound
            fprintf("Initializing phase %d-%d: ", i, j);
            xRegion = int32(i - 1) * segLength + 1:min(int32(i) * segLength, xLength);
            yRegion = int32(j - 1) * segLength + 1:min(int32(j) * segLength, yLength);
            
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
            
            %Decomposition
            load(strcat(tempVarsPath, num2str(i), '-', num2str(j), 'v.mat'));
            XData = XData / v';
            
            %Transpose for training purposes
            XData = XData';
            
            XTrain = XData(:,1:numTimeStepsTrain + ss);
            YTrain = XData(:,2:numTimeStepsTrain + ss + 1);
            
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
            
            %Predict first day
            [net,YPred] = predictAndUpdateState(net,YTrain(:,end));
            predPCs(i, j, ss + 1, 1) = YPred(1);
            
            %Store net
            save(strcat(tempVarsPath, num2str(i), '-', num2str(j), 'net.mat'), 'net');
            clear net
            
            %Full reconstruct
            recon = YPred(:, 1)';
            recon = recon * v';
            PredFullrecon = nan(numel(tempZ), 1);
            PredFullrecon(tempZ, :) = recon';
            PredFullrecon = reshape(PredFullrecon, [xRegion(end) - xRegion(1) + 1, yRegion(end) - yRegion(1) + 1]);
            PredReconstruction(xRegion, yRegion, 1) = PredFullrecon;
            clear recon;
            fprintf("Done\n");
        end
    end
    
    PredReconstruction(:,:,1) = smoothSeg(PredReconstruction(:,:,1), segLength, 5);
    
    for d = 2:numPredictionSteps
        fprintf("Predicting week %d: ", d);
        for i = 1:xBound
            for j = 1:yBound
                xRegion = int32(i - 1) * segLength + 1:min(int32(i) * segLength, xLength);
                yRegion = int32(j - 1) * segLength + 1:min(int32(j) * segLength, yLength);
                
                temp = PredReconstruction(xRegion, yRegion, d - 1);
                temp = temp(:);
                tempZ = any(temp, 2);
                temp = temp(tempZ, :)';
                
                %Deconstruct to u * s
                load(strcat(tempVarsPath, num2str(i), '-', num2str(j), 'v.mat'));
                temp = temp / v';
                
                %Transpose for training purposes
                temp = temp';
                
                %Load in net
                load(strcat(tempVarsPath, num2str(i), '-', num2str(j), 'net.mat'));
                %Predict next step
                [net, YPred] = predictAndUpdateState(net, temp);
                predPCs(i, j, ss + d, d) = YPred(1);
                %Store net
                save(strcat(tempVarsPath, num2str(i), '-', num2str(j), 'net.mat'), 'net');
                clear net
                
                %Full reconstruct
                recon = YPred';
                
                recon = recon * v';
                PredFullrecon = nan(numel(tempZ), 1);
                PredFullrecon(tempZ,:) = recon';
                PredFullrecon = reshape(PredFullrecon, [xRegion(end) - xRegion(1) + 1, yRegion(end) - yRegion(1) + 1]);
                PredReconstruction(xRegion, yRegion, d) = PredFullrecon;
                clear recon;
            end
        end
        PredReconstruction(:,:,d) = smoothSeg(PredReconstruction(:,:,d), segLength, 5 + floor(d / 2));
        fprintf("Done\n");
    end
    
    
    
    for i = 1:numPredictionSteps
        test1 = PredReconstruction(:,:,i);
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
    clear test1 test2;
    
    save(strcat(savePath, 'predPCs.mat'), 'predPCs');
    save(strcat(savePath, 'rmses.mat'), 'rmses');
    save(strcat(savePath, 'coeffs.mat'), 'coeffs');
    
    mkdir(strcat(savePath, num2str(ss)));
    save(strcat(savePath, num2str(ss), '\preds.mat'), 'PredReconstruction');
    figure('units','normalized','outerposition',[0 0 1 0.5], 'visible', 'off')
    for i = 1:numPredictionSteps
        subplot(1, 2, 1);
        surf(PredReconstruction(:,:,i)', 'edgecolor', 'none')
        axis([0, xLength, 0, yLength, -1, 1.5])
        view([0 0 10])
        title('Forecasted Week', 'FontSize', 14)
        set(gca, 'xtick', [100 * (1:5)])
        set(gca, 'ytick', [100 * (1:3)])
        xticklabels(xlbls);
        yticklabels(ylbls);
        set(gca, 'FontSize', 14);
        set(gca, 'LineWidth', 1.5);
        set(gca, 'TickDir', 'in');
        box on
        grid off
        set(gca, 'Layer', 'top')
        caxis([-0.4 0.9])
        
        subplot(1, 2, 2);
        surf(data{numTimeStepsTrain + i + ss}', 'edgecolor', 'none')
        axis([0, xLength, 0, yLength, -1, 1.5])
        view([0 0 10])
        title('Actual Week', 'FontSize', 14)
        set(gca, 'xtick', [100 * (1:5)])
        set(gca, 'ytick', [100 * (1:3)])
        set(gca, 'TickDir', 'in');
        xticklabels(xlbls);
        yticklabels(ylbls);
        set(gca, 'FontSize', 14);
        set(gca, 'LineWidth', 1.5);
        box on
        grid off
        set(gca, 'Layer', 'top')
        caxis([-0.4 0.9])
        
        
        % hp4 = get(subplot(1,2,2),'Position');
        % c = colorbar('southoutside', 'Position', [hp4(2)  hp4(2) - .1  (hp4(1)+hp4(2) * 2) 0.04]);
        c = colorbar('eastoutside');
        c.Label.String = 'Sea Surface Height (SSH) in meters (m)';
        c.FontSize = 14;
        set(c, 'LineWidth', 1.5);
        
        saveas(gcf, strcat(savePath, num2str(ss), '\', num2str(i), '.png'));
    end
    
end

rmsesAv = zeros(numPredictionSteps, 1);
coeffsAv = zeros(numPredictionSteps, 1);
for i = 1:numPredictionSteps
    rmsesAv(i) = sum(rmses(i,:)) / (ss + 1);
    coeffsAv(i) = sum(coeffs(i,:)) / (ss + 1);
end

figure('units','normalized','outerposition',[0 0 1 1], 'visible', false)
plot(coeffsAv, '*-');
xlabel('Forecasted Time Step in Weeks');
ylabel('Averaged Correlation Coefficient')
set(gca, 'fontsize', 14);
saveas(gcf, strcat(savePath, 'Coefs.png'));

figure('units','normalized','outerposition',[0 0 1 1], 'visible', false)
plot(rmsesAv, '*-');
xlabel('Forecasted Time Step in Weeks');
ylabel('Average RMSE')
set(gca, 'fontsize', 14);
saveas(gcf, strcat(savePath, 'RMSE.png'));

function y = subsample(matrix, n)
%Matrix = 1 * days
%n = subsample size
ret = cell(1, int32(floor(size(matrix, 2) / n)));
for i = int32(1:int32(floor(size(matrix, 2) / n)))
    ret(i) = matrix((i - 1) * n + randi(n));
end
y = ret;
end

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
