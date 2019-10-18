path = 'D:\MATLAB\Project 3\Run4\';
path2 = 'D:\MATLAB\Project 3\Project 3.1\Run3\';
dataPath = 'D:\MATLAB\Project 3\Run4\data.mat';

range = 0:53;
MxRegion = 201:400;
MyRegion = 101:300;
% xRegion = 1:541;
% yRegion = 1:385;
PxRegion = 101:400;
PyRegion = 101:300;

% path = 'D:\MATLAB\Project 3\AVISO Run1\';
% dataPath = 'D:\MATLAB\Project 3\AVISO Run1\data.mat';

% range = 0:64;
% MxRegion = 33:64;
% MyRegion = 17:48;
% PxRegion = 33:64;
% PyRegion = 17:48;

xRefPnts = [100, 150, 200, 250, 200, 250, 300];
% yRefPnts = [202, 202, 202, 202, 227, 227, 227];
yRefPnts = [234, 234, 234, 234, 268, 268, 268];
% xRefPnts = [16, 24, 32, 40, 32, 40, 48];
% yRefPnts = [28, 28, 28, 28, 44, 44, 44];

%Load in data
load(dataPath);
numTimeStepsTrain = floor(0.9*numel(data));

%Model performance measures
Mrmses = zeros(20, 1);
% Mccs = zeros(20, 1);

NMrmses = zeros(20, 1);
% NMccs = zeros(20, 1);

%Persistence performance measures
Prmses = zeros(20, 1);
% Pccs = zeros(20, 1);

counter = 0;
for ii = range
    fprintf("Iteration %d: ", ii);
    %Load in prediction
    load(strcat(path, num2str(ii), '\preds.mat'));
%     load(strcat(path2, num2str(ii), '\preds.mat'));
    %Load in persistence
    Pprediction = data{numTimeStepsTrain + ii};
    
    %Load in actual
    tempReal = data(numTimeStepsTrain + 1 + ii:numTimeStepsTrain + 20 + ii);
    
    
    
    %Find averages
%     averageReal = zeros(size(tempReal{1}));
%     averageModel = averageReal;
%     for jj = 1:20
%         %Find actual average
%         tempMat = tempReal{jj};
%         averageReal = averageReal + tempMat;
%         
%         %Find model average
%         averageModel = averageModel + PredReconstruction(:,:,jj);
%     end
%     averageReal = averageReal / 20;
%     averageModel = averageModel / 20;
    
    %Find model average
    
    
    %Loop through
    for jj = 1:20
        tempMat = tempReal{jj};
        %Model calculations
%          MCC = cc(tempMat(MxRegion, MyRegion), PredReconstruction(MxRegion, MyRegion, jj));
%          NMCC = cc(tempMat(MxRegion, MyRegion), PredReconstructionFull(MxRegion, MyRegion, jj));
%          MRMSE = rmse(tempMat(MxRegion, MyRegion), PredReconstruction(MxRegion, MyRegion, jj));
%          NMRMSE = rmse(tempMat(MxRegion, MyRegion), PredReconstructionFull(MxRegion, MyRegion, jj));
        
        distsReal = zeros(size(xRefPnts));
        Mdists = zeros(size(xRefPnts));
        Pdists = zeros(size(xRefPnts));
%         NMdists = zeros(size(xRefPnts));
        RealC = contourc(tempMat, [0.45 0.45]);
        MC = contourc(PredReconstruction(:,:,jj), [0.45 0.45]);
        PC = contourc(Pprediction, [0.45 0.45]);
%         NMC = contourc(PredReconstructionFull(:,:,jj), [0.45 0.45]);
        for kk = 1:numel(xRefPnts)
            refPnt = [xRefPnts(kk) yRefPnts(kk)];
            distsReal(kk) = disCon(refPnt, RealC, 0.45);
            Mdists(kk) = disCon(refPnt, MC, 0.45);
            Pdists(kk) = disCon(refPnt, PC, 0.45);
%             NMdists(kk) = disCon(refPnt, NMC, 0.45);
        end
        clear refPnt;
        
        ME = Mdists - distsReal;
        PE = Pdists - distsReal;
%         NME = NMdists - distsReal;
%         MCC = cc(distsReal, Mdists);
        MRMSE = rmse(distsReal, Mdists);
%         PCC = cc(distsReal, Pdists);
        PRMSE = rmse(distsReal, Pdists);
%         NMRMSE = rmse(distsReal, NMdists);
        

        Mrmses(jj) = Mrmses(jj) + MRMSE;
        
        %Persistence calculations
%         PCC = cc(tempMat(PxRegion, PyRegion), Pprediction(PxRegion, PyRegion));
%         PRMSE = rmse(tempMat(PxRegion, PyRegion), Pprediction(PxRegion, PyRegion));

%         Pccs(jj) = Pccs(jj) + PCC;
        Prmses(jj) = Prmses(jj) + PRMSE;
        PRMSES(ii + 1, jj) = PRMSE;
        
        
    end
        
    counter = counter + 1;
    fprintf("Done\n");
end

% Mccs = Mccs / counter;
Mrmses = Mrmses / counter;
% MrmsesSTD = std(MRMSES, 0, 1);

% NMccs = NMccs / counter;
% NMrmses = NMrmses / counter;
% MNrmsesSTD = std(NMRMSES, 0, 1);

% Pccs = Pccs / counter;
Prmses = Prmses / counter;
PrmsesSTD = std(PRMSES, 0, 1);

Mrmses = idx2km(Mrmses);
Prmses = idx2km(Prmses);
% NMrmses = idx2km(NMrmses);

figure
% yyaxis left
% hold on
% plot(NMccs, '--*')
% plot(Mccs, '-*')
% plot(Pccs, '--*')

% ylim([0 1])
% yyaxis right
hold on
% plot(Mrmses, '-*', 'LineWidth', 2)
% plot(NMrmses, '-*', 'LineWidth', 2)
% plot(Prmses, '--*', 'LineWidth', 2)

p1 = plot_areaerrorbar(reshape(Mrmses, [1 20]), MrmsesSTD);

options.handle     = figure(1);
options.color_area = [243 169 114]./255;    % Orange theme
options.color_line = [236 112  22]./255;
options.alpha      = 0.5;
options.line_width = 2;
options.error      = 'std';
options.type = '-*';

p2 = plot_areaerrorbar(reshape(Prmses, [1 20]), PrmsesSTD, options);

ylim([0 100])
xlim([1 20])
grid on
grid minor
% yyaxis left
% legend('Model CC','Persistence CC', 'Model RMSE', 'Persistence RMSE')
% legend('Nonoverlapping RMSE', 'Overlapping RMSE', 'Persistence RMSE')
legend([p1 p2], 'Model RMSE', 'Persistence RMSE')
% legend('Nonoverlapping RMSE', 'Overlapping RMSE', 'Persistence RMSE')
% set(findall(gca, 'Type', 'Line'),'LineWidth',2);
set(gca, 'FontSize', 18)
% ylabel('Correlation Coefficient (CC)')
% yyaxis right
ylabel('Root-Mean Square Error (RMSE) in kilometers (m)')
xlabel('Prediction Week')
box on



function y = rmse(a, b)
    %Remove nans
    a = a(~isnan(a));
    b = b(~isnan(b));
    y = sqrt(mean((a - b).^2));
end

function y = cc(a, b)
    a = a(~isnan(a));
    b = b(~isnan(b));
    ccs = corrcoef(a, b);
    y = ccs(1, 2);
end

function y = disCon(pnt, C, ht)
    
    Clat = C(2,:)';
    Clon = C(1,:)';
    [~,d] = dsearchn([Clon,Clat],pnt);
    y = d;
    
end

function y = idx2coords(pnt)
    y = (pnt - 1) / 25 + [-98 18.9];
end

function y = idx2km(n)
    y = n / 25 * 111;
end

function p = plot_areaerrorbar(data, er, options)

    % Default options
    if(nargin<3)
        options.handle     = figure(1);
        options.color_area = [128 193 219]./255;    % Blue theme
        options.color_line = [ 52 148 186]./255;
        %options.color_area = [243 169 114]./255;    % Orange theme
        %options.color_line = [236 112  22]./255;
        options.alpha      = 0.5;
        options.line_width = 2;
        options.error      = 'std';
        options.type = '-*';
    end
    if(isfield(options,'x_axis')==0), options.x_axis = 1:size(data,2); end
    options.x_axis = options.x_axis(:);
    
    % Computing the mean and standard deviation of the data matrix
    data_mean = data;
    data_std  = er;
    
    % Type of error plot
    switch(options.error)
        case 'std', error = data_std;
        case 'sem', error = (data_std./sqrt(size(data,1)));
        case 'var', error = (data_std.^2);
        case 'c95', error = (data_std./sqrt(size(data,1))).*1.96;
    end
    
    % Plotting the result
    figure(options.handle);
    p = plot(options.x_axis, data_mean, options.type, 'color', options.color_line, ...
        'LineWidth', options.line_width);
    hold on
    x_vector = [options.x_axis', fliplr(options.x_axis')];
    patch = fill(x_vector, [data_mean+error,fliplr(data_mean-error)], options.color_area);
    set(patch, 'edgecolor', 'none');
    set(patch, 'FaceAlpha', options.alpha);
%     set(patch, 'HandleVisibility', 'off');
    
%     hold off;
    
end
