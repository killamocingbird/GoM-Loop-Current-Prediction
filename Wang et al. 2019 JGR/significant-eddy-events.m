
%{

Code to show PCs across all data along with marking significant
eddy events

%}

%Load in data
load('D:\MATLAB\Project 3\Run4\data.mat')

%Define starting date
initTime = datetime(1992, 1, 1);
initTime.Format = 'MM/dd/yy';

%Define training period
numTimeStepsTrain = floor(0.9*numel(data));

%Define dates (x-axis)
dates = initTime + (7 * [0:numTimeStepsTrain - 1]);

%Define PCs to target
PCTargets = [2 3];

%Define area to look at
xRegion = 201:400;
yRegion = 101:300;

%Load in area
Area = zeros(200 * 200, numTimeStepsTrain);
for i = 1:numTimeStepsTrain
    temp = data{i};
    temp = temp(xRegion, yRegion);
    Area(:,i) = temp(:);
end
clear temp;

%remove nans
Area = Area(any(Area, 2), :)';

%Decompose
[u, s, ~] = svd(Area);
PCs = u * s;
clear u s;

%Isolate wanted PCs
PCs = PCs(:, PCTargets);

%Identiy shedding events
sheds = [7 20 33 48 65 78 153 182 207 239 276 297 377 390 404 425 439 456 470 492,...
597 610 631 670 681 694 708 738 805 820];

plot(PCs);
xlim([1 size(PCs, 1)])
hold on
plot(zeros(size(PCs, 1), 1), '--')
legend('PC 2', 'PC 3', 'Midline')
xticks(sheds)
xticklabels(string(dates(sheds)))
xtickangle(90)
xlabel('Dates of Significant Eddy Shedding Events')
ylabel('Amplitude')
grid on
grid minor
box on





