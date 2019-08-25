data = cell(numel(range) * 365 + 3, 1);
fprintf("Loading in Data:\n");
counter = 1;
for i = range
    load(strcat('x', num2str(i), '.mat'));
    data(counter:counter + numel(x) - 1) = x;
    counter = counter + numel(x);
    fprintf("Year %d loaded\n", i);
end
data = subsample(data', numSubsampleDays);
for i = 1:numel(data)
    x = data{i};
    data(i) = {x(2:end,2:end)};
end
xLengthBase = size(x, 1) - 1;
yLengthBase = size(x, 2) - 1;
xLengthOver = (floor(xLengthBase / segLength) - 1) * segLength + segLength / 2 + mod(xLengthBase, segLength) / 2;
yLengthOver = (floor(yLengthBase / segLength) - 1) * segLength + segLength / 2 + mod(yLengthBase, segLength) / 2;

clear x
fprintf("Done\n\n");



function y = subsample(matrix, n)
%Matrix = 1 * days
%n = subsample size
ret = cell(1, int32(floor(size(matrix, 2) / n)));
for i = int32(1:int32(floor(size(matrix, 2) / n)))
    ret(i) = matrix((i - 1) * n + randi(n));
end
y = ret;
end