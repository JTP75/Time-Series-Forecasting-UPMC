function [w,d,t] = num2dt(num)

w = datetime(num,'ConvertFrom','datenum','Format','eee');
d = datetime(num,'ConvertFrom','datenum','Format','M/d/yyyy');
t = datetime(num,'ConvertFrom','datenum','Format','HH:mm');