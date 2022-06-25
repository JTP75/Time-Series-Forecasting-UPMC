function [T_clean, T_imputed, T_day, cldf] = construct_tables(T_clean)

if nargin==0
    load('PUH_NEDOC_CLEANED.mat','T_clean');
end

m = forecastModel(T_clean,'temp',0);
m = m.setSplit(length(m.y));
m = m.createClSet;

% imputed table
[w,d,~] = num2dt(m.X_Imp(:,1) + m.X_Imp(:,2));

Date_Time_DTA = d;
Date_Time = m.X_Imp(:,1) + m.X_Imp(:,2);
Date = m.X_Imp(:,1);
Time = m.X_Imp(:,2);
Month = m.X_Imp(:,4);
WKD_Name = w;
Weekday = m.X_Imp(:,3);
Score = m.y_Imp;
Level = getLevel(m.y_Imp);

T_imputed = table(Date_Time_DTA, Date_Time, Date, Time, Month, WKD_Name, Weekday, Score, Level);

% day table
[w,d,~] = num2dt(m.X_cl(:,1));

Date_Time_DTA = d;
Date = m.X_cl(:,1);
Month = m.X_cl(:,3);
WKD_Name = w;
Weekday = m.X_cl(:,2);
Day_Class = m.y_cl;
Average_Score = mean(m.dayClass_DEF(m.dayClass,:),2);
Average_Level = getLevel(Average_Score);

T_day = table(Date_Time_DTA, Date, Month, WKD_Name, Weekday, Day_Class, Average_Score, Average_Level);

cldf = m.dayClass_DEF;



