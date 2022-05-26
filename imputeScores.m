function [scoreArrOut,scoreArrNaN] = imputeScores(timeArr,scoreArr,dayInterval,dayLen)
% function returns days as 48 point column vectors, filling in missing score values
M = 48;     % num of obs per day

% instantiate arrays
timeArrIn = zeros([M,1]);
timeArrIn(1:dayLen) = timeArr(dayInterval);
timeArrCorrect = (0:1/M:.99)';

scoreArrOut = zeros([M,1]) + NaN;
scoreArrOut(1:dayLen) = scoreArr(dayInterval);

% fit data to 24 hour (48 points) day, leaving NaNs for missing values
for i = 1:M
    if timeArrCorrect(i) >= timeArrIn(i)+.0001 || timeArrCorrect(i) <= timeArrIn(i)-.0001
        timeArrIn(i:M) = circshift(timeArrIn(i:M),1);
        scoreArrOut(i:M) = circshift(scoreArrOut(i:M),1);
    end
end
scoreArrNaN = scoreArrOut;

% impute linearly spaced values
i = 1;
while i <= M
    
    % finds NaNs, then counts num of consecutive NaNs in NaNCount
    NaNCount = 0;
    while (i <= M) && isnan(scoreArrOut(i))
        i=i+1;
        NaNCount = NaNCount + 1;
    end
    
    % impute first value, if necessary
    if i < NaNCount+2 && NaNCount ~= 0
        scoreArrOut(1) = scoreArrOut(i);
        NaNCount = NaNCount - 1;
    end
    
    % impute last value, if necessary
    if i >= M && NaNCount ~= 0
        scoreArrOut(M) = scoreArrOut(M-1);
        NaNCount = NaNCount - 1;
        i=i-1;
    end
    
    % impute other values
    if NaNCount ~= 0
        imputedVals = round(linspace(scoreArrOut(i-NaNCount-1),scoreArrOut(i),NaNCount+2))';
        scoreArrOut(i-NaNCount-1:i) = imputedVals;
    end
    
    clear impVals
    i=i+1;
end













