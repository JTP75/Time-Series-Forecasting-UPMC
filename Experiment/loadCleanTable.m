function newTable = loadCleanTable()

if ~exist('T','var')
    cd ..
    T = readtable('PUH NEDOC.xlsx');
    cd Experiment
end

score = T.MODIFIED_SCORE_VAL;                                               % score array (y)
dateArr = T.Date;                                                           % date as a datetime array
dateNum = datenum(T.Date);                                                  % date as a numeric array
timeNum = T.Time;                                                           % time as a numeric array
month = T.Month;                                                            % month of year

levelStr = {};
levelStr = T.NEDOC_LEVEL;                                                   % nedoc level as string

[mSmpls, ~] = size(score);                                                       % nsamples

for i = 1:mSmpls
    levelNum(i,1) = str2num(levelStr{i}(7));
end

[weekday_num, weekday_name] = weekday(dateArr);

newTable = table(dateArr, dateNum+timeNum, dateNum, timeNum, month, weekday_name, weekday_num, score, levelNum, 'VariableNames',...
                {'Date_Time_DTA','Date_Time','Date','Time','Month','WKD_Name','Weekday','Score','Level'});







