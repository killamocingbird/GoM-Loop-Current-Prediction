load('F:\MATLAB\Project 3\Run4\data.mat')
% tempVarsPath = 'F:\MATLAB\Project 3\TempVars\';
segLength = 100;
xLength = 541;
yLength = 385;
% 
%block 1
% xid = 2;
% yid = 2;  
% 
% xRegion = int32(xid - 1) * segLength + 1:min(int32(xid) * segLength, xLength);
% yRegion = int32(yid - 1) * segLength + 1:min(int32(yid) * segLength, yLength);
% 
% load(strcat(tempVarsPath, num2str(xid), '-', num2str(yid), 'v.mat'));

xRegion = 1:200;
yRegion = 101:300;
% numTimeStepsTrain = floor(0.9*numel(data));
numTimeStepsTrain = 0;
% shift = 74;
shift = 94;

% vs = [31.7776 16.3338 11.3032 6.5582 5.4091 3.9514]; 2008
% vs = [35.7500 23.5488 12.8223 8.7657 4.6063 2.9464]; 1992

%Load in area
Area = zeros(200 * 200, numTimeStepsTrain + 20 + shift);
for i = 1:numTimeStepsTrain + 20 + shift
    temp = data{i};
    temp = temp(xRegion, yRegion);
    Area(:,i) = temp(:);
end

%remove nans
zs = any(Area, 2);
Area = Area(zs, :)';

%Isolate SSH
data = data(numTimeStepsTrain + 1 + shift:numTimeStepsTrain + 20 + shift);
% data = data(1 + 25:25 + 20);

%Decompose
[u s v] = svd(Area);
PCs = u * s;

%Initial Day
initDate = datetime(1993, 10, 18);
initDate.Format = 'MM/dd/yy';

%Isolate PCs
PCs = PCs(numTimeStepsTrain + 1 + shift:numTimeStepsTrain + 20 + shift, :);
% PCs = PCs(1 + 25:25 + 20, :);

%Create Empty Templace for PCs
empty = zeros(1, size(PCs, 2));
vars = zeros(6, 20);
% wks = [2 4 6 8 10 12 14 16 18 20];
wks = [1:20];
figure
for ii = 1:numel(wks)
    i = wks(ii);
    subplot(7, numel(wks), ii);
    temp = data{i};
    surf(temp(xRegion,yRegion)', 'edgecolor', 'none')
    axis([0, 200, 0, 200, -1, 1.5])
    view([0 0 10])
    xticks([])
    yticks([])
    caxis([-0.3 0.8])
    box on
end
counter = numel(wks) + 1;
for i = 1:6
    for jj = 1:numel(wks)
        j = wks(jj);
        tempPCs = PCs(j, :);
        vars(i, j) = abs(tempPCs(i)) / sum(abs(tempPCs));
        temp = empty;
        temp(i) = tempPCs(i);
        tempMap = temp * v';
        map = nan(1, 200 * 200);
        map(zs) = tempMap;
        map = reshape(map, 200, 200);
        
        subplot(7, numel(wks), counter)
        surf(map', 'edgecolor', 'none')
        axis([0,200, 0, 200, -1, 1.5])
        view([0 0 10])
        xticks([])
        yticks([])
        caxis([-0.1 0.2])
        box on
        counter = counter + 1;
    end
end

%Amplitude
figure
plot(PCs(:,1:6))
xlim([1 20])
xticks([2:2:20])
xlabel('Date')
ylabel('Amplitude')
xticklabels(string(initDate + (14 * [0:9])))
% xticklabels({'10/13/08', '10/27/08', '11/10/08', '11/24/08', '12/08/08', '12/22/08','01/05/09','01/19/09','02/02/09','02/16/09'})
% xticklabels({'03/30/09','04/13/09','04/27/09','05/11/09','05/25/09','06/08/09','06/22/09','07/06/09','07/20/09','08/03/09'});
ylim([-30 60])
legend('PC 1','PC 2','PC 3','PC 4','PC 5','PC 6')
% title("Amplitude of First 6 PCs from May 24th 1993 - October 4th 1993")
title("Amplitude of First 6 PCs from October 11th 1993 - February 21st 1994")
% title("Amplitude of First 6 PCs from October 6th 2008 - February 16th 2009")
% title("Amplitude of First 6 PCs from March 16th 2009 - August 3rd 2009")
set(findall(gca, 'Type', 'Line'),'LineWidth',2);
set(gca, 'FontSize', 18)
grid on
grid minor

%Variance
figure
plot(vars')
xlim([1 20])
xticks([2:2:20])
xlabel('Date')
ylim([0 0.35])
ylabel('Percentage Variance Accounted For')
xticklabels(string(initDate + (14 * [0:9])))
% xticklabels({'10/13/08', '10/27/08', '11/10/08', '11/24/08', '12/08/08', '12/22/08','01/05/09','01/19/09','02/02/09','02/16/09'})
% xticklabels({'03/30/09','04/13/09','04/27/09','05/11/09','05/25/09','06/08/09','06/22/09','07/06/09','07/20/09','08/03/09'});
yticklabels(["0%"; "5%"; "10%"; "15%"; "20%"; "25%"; "30%"; "35%"; "40%"])
legend('PC 1','PC 2','PC 3','PC 4','PC 5','PC 6')
% title("Variance Accounted for by the First 6 PCs from May 24th 1993 - October 4th 1993")
title("Variance Accounted for by the First 6 PCs from October 11th 1993 - February 21st 1994")
% title("Variance Accounted for by the First 6 PCs from October 6th 2008 - February 16th 2009")
set(findall(gca, 'Type', 'Line'),'LineWidth',2);
set(gca, 'FontSize', 18)
grid on
grid minor

% vars = round(vars, 3);
% vars = cat(2, reshape(1:size(vars, 1), [size(vars, 1), 1]), vars);
% tbl = array2table(vars, 'VariableNames', {'PCNum', 'Wk1','Wk2','Wk3','Wk4','Wk5','Wk6','Wk7','Wk8','Wk9','Wk10','Wk11','Wk12','Wk13','Wk14','Wk15','Wk16','Wk17','Wk18','Wk19','Wk20',});


