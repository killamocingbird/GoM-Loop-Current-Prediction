
%{

This code stub calculates the frontal distance performance
measure as laid out by Oey et al. (reference in paper)

%}

path = 'D:\MATLAB\Project 3\Run4\';
% path2 = 'D:\MATLAB\Project 3\Project 3.1\Run3\';
dataPath = 'D:\MATLAB\Project 3\Run4\data.mat';

range = 0:53;
MxRegion = 201:400;
MyRegion = 101:300;
% xRegion = 1:541;
% yRegion = 1:385;
PxRegion = 101:400;
PyRegion = 101:300;

xRefPnts = [100, 150, 200, 250, 200, 250, 300];
yRefPnts = [234, 234, 234, 234, 268, 268, 268];

%Load in data
load(dataPath);
numTimeStepsTrain = floor(0.9*numel(data));

%Model performance measures
Mrmses = zeros(20, 1);
Mccs = zeros(20, 1);

%Persistence performance measures
Prmses = zeros(20, 1);
Pccs = zeros(20, 1);

counter = 0;
for ii = range
    fprintf("Iteration %d: ", ii);
    %Load in prediction
    load(strcat(path, num2str(ii), '\preds.mat'));
    %Load in persistence
    Pprediction = data{numTimeStepsTrain + ii};
    
    %Load in actual
    tempReal = data(numTimeStepsTrain + 1 + ii:numTimeStepsTrain + 20 + ii);
        
    %Loop through
    for jj = 1:20
        tempMat = tempReal{jj};
        %Model calculations
        MCC = cc(tempMat(MxRegion, MyRegion), PredReconstruction(MxRegion, MyRegion, jj));
        MRMSE = rmse(tempMat(MxRegion, MyRegion), PredReconstruction(MxRegion, MyRegion, jj));
        
        Mccs(jj) = Mccs(jj) + MCC;
        MCCS(ii + 1, jj) = MCC;
        Mrmses(jj) = Mrmses(jj) + MRMSE;
        MRMSES(ii + 1, jj) = MRMSE;
     
        %Persistence calculations
        PCC = cc(tempMat(PxRegion, PyRegion), Pprediction(PxRegion, PyRegion));
        PRMSE = rmse(tempMat(PxRegion, PyRegion), Pprediction(PxRegion, PyRegion));

        Pccs(jj) = Pccs(jj) + PCC;
        PCCS(ii + 1, jj) = PCC;
        Prmses(jj) = Prmses(jj) + PRMSE;
        PRMSES(ii + 1, jj) = PRMSE;
        
        
    end
        
    counter = counter + 1;
    fprintf("Done\n");
end

Mccs = reshape(Mccs / counter, [1 20]);
MccsSTD = std(MCCS, 0, 1);
Mrmses = reshape(Mrmses / counter, [1 20]);
MrmsesSTD = std(MRMSES, 0, 1);

Pccs = reshape(Pccs / counter, [1 20]);
PccsSTD = std(PCCS, 0, 1);
Prmses = reshape(Prmses / counter, [1 20]);
PrmsesSTD = std(PRMSES, 0, 1);


figure
yyaxis left
hold on
ylim([0 1])
ylabel('Correlation Coefficient (CC)')
p1 = plot_areaerrorbar(Mccs, MccsSTD);

        options2.handle     = figure(1);
        options2.color_area = [128 193 219]./255;    % Blue theme
        options2.color_line = [ 52 148 186]./255;
        %options.color_area = [243 169 114]./255;    % Orange theme
        %options.color_line = [236 112  22]./255;
        options2.alpha      = 0.5;
        options2.line_width = 2;
        options2.error      = 'std';
        options2.type = '--*';

p2 = plot_areaerrorbar(Pccs, PccsSTD, options2);

yyaxis right
hold on
ylim([0 1])
ylabel('Root-Mean Square Error (RMSE) in kilometers (km)')

        options3.handle     = figure(1);
        options3.color_area = [243 169 114]./255;    % Orange theme
        options3.color_line = [236 112  22]./255;
        options3.alpha      = 0.5;
        options3.line_width = 2;
        options3.error      = 'std';
        options3.type = '-*';

p3 = plot_areaerrorbar(Mrmses, MrmsesSTD, options3);

        options4.handle     = figure(1);
        options4.color_area = [243 169 114]./255;    % Orange theme
        options4.color_line = [236 112  22]./255;
        options4.alpha      = 0.5;
        options4.line_width = 2;
        options4.error      = 'std';
        options4.type = '--*';

p4 = plot_areaerrorbar(Prmses, PrmsesSTD, options4);

xlim([1 20])
grid on
grid minor
xlabel('Prediction Week')
legend([p1 p2 p3 p4], 'Model CC', 'Persistence CC', 'Model RMSE', 'Persistence RMSE')
set(gca, 'FontSize', 18)
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
