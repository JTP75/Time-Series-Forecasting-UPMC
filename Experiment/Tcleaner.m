function TCLN = Tcleaner(TRAW)

if nargin == 0
    load('PUH_NEDOC_FULL_RAW.mat');
    TRAW = PUHNEDOCS1;
    clear PUHNEDOCS1
end
app_eq = @(a,b,tol) abs(a-b)<tol;

dtnum = datenum(TRAW.Date);
[w,d,~] = num2dt(dtnum);

Date_Time_DTA = d;
Date_Time = dtnum + TRAW.Time;
Date = dtnum;
Time = TRAW.Time;
Month = TRAW.Month;
WKD_Name = w;
Weekday = weekday(d);
Score = TRAW.SCORE_VAL;
Level = getLevel(Score);

TCLN = table(Date_Time_DTA, Date_Time, Date, Time, Month, WKD_Name, Weekday, Score, Level);

tcurr = 0;
tprev = -1;
i=1;
while i <= height(TCLN)
    tcurr = TCLN.Time(i);
    if app_eq(tcurr, tprev, 0.001)
        TCLN(i,:) = [];
    end
    tprev = tcurr;
    i=i+1;
end

TCLN = flip(TCLN);



