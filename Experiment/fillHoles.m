function newData = fillHoles(oldData)
M = length(oldData);

% impute first value, if necessary
if isnan(oldData(1))
    oldData(1) = directionalNonNaN(oldData,1,1);
end

% impute last value, if necessary
if isnan(oldData(M))
    oldData(M) = directionalNonNaN(oldData,M,-1);
end

consecutive_NaN_max = 0;
% impute linearly spaced values
i = 1;
while i <= M
    
    % finds NaNs, then counts num of consecutive NaNs in NaNCount
    NaNCount = 0;
    while i <= M && isnan(oldData(i))
        i=i+1;
        NaNCount = NaNCount + 1;
    end
    
    % update max nan count (for testing purposes: too large of a gap may be problematic)
    if NaNCount > consecutive_NaN_max
        consecutive_NaN_max = NaNCount;
        iMax = i;
    end
    
    % impute other values
    if NaNCount ~= 0
        imputedVals = round(linspace(oldData(i-NaNCount-1),oldData(i),NaNCount+2))';
        oldData(i-NaNCount-1:i) = imputedVals;
    end
    
    i=i+1;
end

fprintf('\n================================\nConsecutive NaN Max: %d at %d\n================================\n',...
    consecutive_NaN_max, iMax)

newData = oldData;
end

function NN = directionalNonNaN(arr,i,dir)
    if ~isnan(arr(i)) && i <= length(arr) && i >= 1
        NN = arr(i);
    elseif ( dir == 1 || dir == -1 ) && i <= length(arr) && i >= 1
        NN = directionalNonNaN(arr,i+dir,dir);
    else
        NN = NaN;
    end
end