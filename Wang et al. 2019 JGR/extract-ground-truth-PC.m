%{

Code to extract nth PCs from ground truth data

%}

savePath = 'D:\MATLAB\Project 3\Run4\';
load('D:\MATLAB\Project 3\Run4\data.mat');
xLength = 541;
yLength = 385;
segLength = 100;
pc = 3;
range = 20:49;
numTimeStepsTrain = floor(0.9*numel(data));
realPCs = zeros(6, 4, 94);

for ii = 1:6
    for jj = 1:4
        fprintf("Generating EOFs phase %d-%d: ", ii, jj);
        xRegion = int32(ii - 1) * segLength + 1:min(int32(ii) * segLength, xLength);
        yRegion = int32(jj - 1) * segLength + 1:min(int32(jj) * segLength, yLength);
        
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
        realPCs(ii, jj, :) = tempPC(numTimeStepsTrain + 1:numel(data), pc)';
        clear u s v;
        fprintf("Done\n");
    end
end
save(strcat(savePath, 'realPCs', num2str(pc), '.mat'), 'realPCs');





