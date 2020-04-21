
savePath = 'C:\Users\Justin Wang\Documents\MATLAB\Project 3\Data\SSH\';

baseURL = 'http://tds.hycom.org/thredds/dodsC/datasets/GOMl0.04/expt_02.2/';
baseYear = 1900;
sateliteCycle = 1;
runCycle = '00';
% type = 'z3d/';
type = 'mlayer/';
% var = 'u';
varAbrev = 'mlt';
varName = 'mixed_layer_temperature';
% vers = '022GOMl0.04-';
vers = 'GOMl0.04';
errors = [];
for ii = 92:108
    tempdat = {};
    year = baseYear + ii;
    fprintf("Starting download of %d %s data:\n", year, varAbrev);
    lastDay = 365;
    if leapyear(year)
        lastDay = 366;
    end
    errShift = 0;
    for jj = 1:lastDay
        fprintf("Day %d: ", jj)
        cont = true;
        while cont
            try
%                 hold = ncread(strcat(baseURL, num2str(year), '-', num2str(sateliteCycle), '/', type, vers, num2str(year), '_', sprintf('%.3d', jj), '_', runCycle, '_', var, '.nc'), var);
                hold = ncread(strcat(baseURL, num2str(year), '-', num2str(sateliteCycle), '/', type, vers, num2str(year), '_', sprintf('%.3d', jj), varAbrev, '.nc'), varName);
                for kk = 1:size(hold, 3)
                    tempdat(kk, jj) = {hold(:,:,kk)};
                end
                fprintf("Done\n")
                cont = false;
            catch ME
%                 errShift = errShift + 1;
%                 errors(errShift) = jj;
                fprintf("Error\n")
            end
        end
        
    end
    for jj = 1:size(hold, 3)
        templayer = tempdat(jj, :);
        save(strcat(savePath, sprintf('Layer %d', jj), '\', upper(varAbrev), '_', num2str(year), '.mat'), 'templayer');
    end
    
end

% 1050 - 1150, 400 - 500
