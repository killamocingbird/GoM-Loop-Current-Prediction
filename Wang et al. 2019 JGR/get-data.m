%{

Main method to download data for experiments

%}

clc;
clear;
close all;

year = 1900;
for i = 92:109
    fprintf('Initializing download of year %d:\n', year + i)
    link = strcat('http://tds.hycom.org/thredds/dodsC/datasets/GOMl0.04/expt_02.2/', num2str(year + i), '-1/mlayer/GOMl0.04', num2str(year + i), '_');
    if (leapyear(year + i)); lastDay = 366; else lastDay = 365; end
    x = cell(lastDay, 1);
    for j = 1:lastDay
        s = strcat(link, sprintf('%03d', j), 'mlu.nc');
        x{j} = ncread(s, 'mlu');
        fprintf('Done - Day %d\n', j);
    end
    save(strcat('x', num2str(year + i), '.mat'), 'x'); 
    fprintf('\n');
end

function status = leapyear(year)
    if mod(year, 4) == 0
        if mod(year, 100) == 0
            if mod(year,400) == 0
                status = true;
            else
                status = false;
            end
        else
            status = true;
        end
    else
        status = false;
    end
end

