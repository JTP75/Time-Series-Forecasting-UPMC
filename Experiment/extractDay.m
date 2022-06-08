function [idxList, wkd, dSz] = extractDay(weekday_num, idx)
m = size(weekday_num);

lowerBound = idx-60;
upperBound = idx+60;

if lowerBound < 1
    lowerBound = 1;
end
if upperBound > m
    upperBound = m;
end

wkd = weekday_num(idx);
daySearchInterval = lowerBound:upperBound;

idxList = find(weekday_num(daySearchInterval)==wkd);
idxList = sort(idxList) + lowerBound - 1;
[dSz,~] = size(idxList);