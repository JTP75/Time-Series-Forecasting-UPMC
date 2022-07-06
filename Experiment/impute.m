function TI = impute(TC, ptsPerDay)

% helper fcn approximate equality
app_eq = @(a,b) abs(a-b)<0.001;

% generate proper vectors (e.g. 288)
Date_Time_DTA = [];
Date_Time = [];
Date = [];
Time = [];
Month = [];
WKD_Name = [];
Weekday = [];
i = 1;
while i <= height(TC)
    
    zervec = zeros([ptsPerDay,1]);
    onevec = ones([ptsPerDay,1]);
    
    DDTA_FD = zervec + TC.Date_Time_DTA(i);
    D_FD = onevec * TC.Date(i);
    T_FD = (0:1/ptsPerDay:.999)';
    DT_FD = D_FD + T_FD;
    M_FD = onevec * TC.Month(i);
    WN_FD = zervec + TC.WKD_Name(i);
    W_FD = onevec * TC.Weekday(i);
    
    Date_Time_DTA = [Date_Time_DTA ; DDTA_FD];
    Date_Time = [Date_Time ; DT_FD];
    Date = [Date ; D_FD];
    Time = [Time ; T_FD];
    Month = [Month ; M_FD];
    WKD_Name = [WKD_Name ; WN_FD];
    Weekday = [Weekday ; W_FD];
    
    while i <= height(TC) && TC.Date(i) == D_FD(1)
        i = i+1;
    end
end

% impute score values
Score = zeros([length(Time),1]);
for i = 1:length(Time)
    clear idx
    dt = Date_Time(i);
    idx = find(app_eq(TC.Date_Time,dt));
    if isempty(idx)
        Score(i) = NaN;
    else
        idx = idx(1);
        Score(i) = TC.Score(idx);
    end
end
Score = fillHoles(Score);
Level = getLevel(Score);
TI = table(Date_Time_DTA, Date_Time, Date, Time, Month, WKD_Name, Weekday, Score, Level);
