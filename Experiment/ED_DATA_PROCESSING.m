%% ======================================================================== CLEAR & CLOSE ================================

clc
clear all
close all

%% ======================================================================== READ TABLE ===================================

if ~exist('T','var')
    cd ..
    T = readtable('PUH NEDOC.xlsx');
    cd Experiment
end

%% ======================================================================== LOAD DATA ====================================

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

biasFeature = ones([mSmpls, 1]);

%% ======================================================================== FEATURE ENGINEERING ==========================

% value 1-7 indicating which day of the week it is
% monday 1, tuesday 2, wednesday 3, thursday 4, friday 5, saturday 6, sunday 7
% January 1st, 2020 was a wednesday

[weekday_num, weekday_name] = weekday(dateArr);
T(:,23) = array2table(weekday_num);

%% ======================================================================== IMPUTATION ===================================

M = 48;             % desired num of smpls per day
timeImp = [];
wkdNumImp = [];
wkdNameImp = [];
monthImp = [];
dtNumImp = [];
dtDtmImp = [];
scoreImp = [];

dayTmImp = ((1:M)' - 1)/M;

MTdayArr = zeros([M,1]);
MTstrArr = strings([M,3]);
MTdtmArr = NaT([M,1]);

i = 1;
while i <= mSmpls
    [dayInt,wkd,dayLen] = extractDay(weekday_num,i);
    dayScore = imputeScores(timeNum,score,dayInt,dayLen);
    dayMonth = month(i);
    dayDtNum = dateNum(i);
    for o = 1:M
        dayDtDtm(o,1) = dateArr(i);
    end
        
    timeImp = [ timeImp ; dayTmImp ];
    wkdNumImp = [ wkdNumImp ; MTdayArr+wkd ];
    wkdNameImp = [ wkdNameImp ; MTstrArr+getWeekday(wkd) ];
    monthImp = [ monthImp ; MTdayArr+dayMonth ];
    dtNumImp = [ dtNumImp ; MTdayArr+dayDtNum ];
    dtDtmImp = [ dtDtmImp ; dayDtDtm ];
    scoreImp = [ scoreImp ; dayScore ];
    
    i=i+dayLen;
end
[mImp,~] = size(scoreImp);
bfImp = ones([mImp,1]);

%% ======================================================================== FILL MATRICES AND SPLIT DATA =================

X = [biasFeature dateNum timeNum weekday_num month];
XImp = [bfImp dtNumImp timeImp wkdNumImp monthImp];
% X = [dateNum timeNum weekday_num month];
yreg = score;
ycls = getLevel(score);

t = dtNumImp + timeImp;

yregImp = scoreImp;
yclsImp = getLevel(scoreImp);

addend = 1000;
today = round(3*mSmpls/4 + addend);
tdImp = find(dtNumImp==dateNum(today), 1);
% data points before today are training, after are testing

%% ======================================================================== NEW DATASTORE FOR DLTBOX =====================

dsTable = table(t,scoreImp);
writetable(dsTable,'ImputedTSData.xlsx');
ds = datastore('ImputedTSData.xlsx');

%% ======================================================================== REGRESSORS ===================================


%% RANDOM FOREST OF FINE REGRESSION TREES
idcurr = today;
days = 9;

fineTreeEnsModel = fineTreeEns(X(1:today,:), yreg(1:today));
[dayIdcs,~,dayLen] = extractDay(weekday_num,idcurr);
idcurr = dayIdcs(dayLen) + 1;

figure(1)
for i = 1:days
    
    [dayIdcs,~,dayLen] = extractDay(weekday_num,idcurr);
    
    Xq = X(dayIdcs,:);
    yregq = yreg(dayIdcs);
    yclsq = ycls(dayIdcs);
    
    yp_FTE = fineTreeEnsModel.predictFcn(Xq);                               % predicted score
    ypCl_FTE = getLevel(yp_FTE);                                            % predicted level
    
    accFTE(i) = sum(ypCl_FTE == yclsq) / dayLen * 100;
    fprintf(['Day: ' weekday_name(idcurr,:) '\taccFTE = %.2f%%\n'],accFTE(i))
    
    subplot(ceil(sqrt(days)),ceil(sqrt(days)),i)
    plot(yregq)
    hold on
    plot(yp_FTE, 'r--')
    ttl = ...
        [weekday_name(idcurr,:) ', ' datestr(dateArr(idcurr)) ': Acc = '...
        num2str(accFTE(i)) '%' ', dayLen = ' num2str(dayLen)];
    title(ttl)
    legend('Actual','Fine Tree Ensemble')
    axis([1, dayLen, 0, 200])
    hold off
    
    idcurr = dayIdcs(dayLen) + 1;
    clear Xq yregq yclsq 
    
end

%% RANDOM FOREST OF FINE REGRESSION TREES (IMPUTED)
idcurr = tdImp;
days = 9;

fineTreeEnsModelImp = fineTreeEns(XImp(1:tdImp,:), yregImp(1:tdImp));
figure(2)
for i = 1:days
    
    [dayIdcs,~,dayLen] = extractDay(wkdNumImp,idcurr);
    
    Xq = XImp(dayIdcs,:);
    yregq = yregImp(dayIdcs);
    yclsq = yclsImp(dayIdcs);
    
    yp_FTE = fineTreeEnsModelImp.predictFcn(Xq);                            % predicted score
    ypCl_FTE = getLevel(yp_FTE);                                            % predicted level
    
    accFTE(i) = sum(ypCl_FTE == yclsq) / dayLen * 100;
    fprintf(strcat('Day: ', wkdNameImp(idcurr), '\taccFTE = ', num2str(accFTE(i)), '%%\n'))
    
    subplot(ceil(sqrt(days)),ceil(sqrt(days)),i)
    plot(yregq)
    hold on
    plot(yp_FTE, 'r--')
    ttl = strcat(wkdNameImp(idcurr), ', ', datestr(dtDtmImp(idcurr)), ': Acc = ', num2str(accFTE(i)), '%');
    title(ttl)
    legend('Actual','Fine Tree Ensemble Imputed')
    axis([1, dayLen, 0, 200])
    hold off
    
    idcurr = dayIdcs(dayLen) + 1;
    clear Xq yregq yclsq 
    
end

%% ======================================================================== TRIG REGERESSION =============================
% inspired by fourier series: large sum of sines and cosines

% reconstruct data with trig vals
nFtrs = 100;
X_COS = ones([mImp, nFtrs]);
for j = 2:nFtrs
    X_COS(:,j) = cos( (dtNumImp + timeImp) * 30000 * j / 30000 );
end

idcurr = tdImp;
days = 9;

beta = mvregress(X_COS(1:tdImp,:),yregImp(1:tdImp));
figure(3)
for i = 1:days
    
    [dayIdcs,~,dayLen] = extractDay(wkdNumImp,idcurr);
    
    Xq = X_COS(dayIdcs,:);
    yregq = yregImp(dayIdcs);
    yclsq = yclsImp(dayIdcs);
    
    yp_LTR = Xq * beta;                                                     % predicted score
    ypCl_LTR = getLevel(yp_LTR);                                            % predicted level
    
    accLTR(i) = sum(ypCl_LTR == yclsq) / dayLen * 100;
    fprintf(strcat('Day: ', wkdNameImp(idcurr), '\taccLTR = ', num2str(accLTR(i)), '%%\n'))
    
    subplot(ceil(sqrt(days)),ceil(sqrt(days)),i)
    plot(yregq)
    hold on
    plot(yp_LTR, 'r--')
    ttl = strcat(wkdNameImp(idcurr), ', ', datestr(dtDtmImp(idcurr)), ': Acc = ', num2str(accLTR(i)), '%');
    title(ttl)
    legend('Actual','Fine Tree Ensemble Imputed')
    axis([1, dayLen, 0, 200])
    hold off
    
    idcurr = dayIdcs(dayLen) + 1;
    clear Xq yregq yclsq 
    
end

%% ======================================================================== **RNN** ======================================

str = con2seq(scoreImp(1:tdImp)');
ttr = con2seq(t(1:tdImp)');
lrn_net = layrecnet(1:2,4);
lrn_net.trainFcn = 'trainbfg';
lrn_net.trainParam.show = 5;
lrn_net.trainParam.epochs = 50;
lrn_net = train(lrn_net,ttr,str);

%%
fprintf('\n=============\n\n')
idcurr = today;
days = 9;
yptot = cell(days,1);

RNNModel = lrn_net;
figure(4)
for i = 1:days
    
    [dayIdcs,wkd,dayLen] = extractDay(wkdNumImp,idcurr);
    
    tq = con2seq(t(dayIdcs)');
    sq = con2seq(scoreImp(dayIdcs)');
    
    yp_RNN = cell2mat(RNNModel(tq))';                                       % predicted score
    ypCl_RNN = getLevel(yp_RNN);                                            % predicted level
    
    accRNN(i) = sum(ypCl_RNN == getLevel(scoreImp(dayIdcs))) / dayLen * 100;
    fprintf(strcat('Day: ', wkdNameImp(idcurr), '\taccRNN = ', num2str(accRNN(i)), '%%\n'))
    
    subplot(ceil(sqrt(days)),ceil(sqrt(days)),i)
    plot(score(dayIdcs))
    hold on
    plot(yp_RNN, 'r--')
    ttl = strcat(wkdNameImp(idcurr), ', ', datestr(dtDtmImp(idcurr)), ': Acc = ', num2str(accRNN(i)), '%');
    title(ttl)
    legend('Actual','RNN')
    axis([1, dayLen, 0, 200])
    hold off
    
    idcurr = dayIdcs(dayLen) + 1;
    clear tq sq
    
    yptot{i} = yp_RNN;
    
end

%% ======================================================================== **ARIMA** ====================================

mdl_ARIMA_Template = arima('MALags',1:2,'SMALags',12);
mdl_ARIMA_Estimate = estimate(mdl_ARIMA_Template,scoreImp);
yp_ARIMA = forecast(mdl_ARIMA_Estimate,30,scoreImp);

%% ======================================================================== CLASSIFIERS ==================================

%% KNN

days = 9;
interval = 41;

idcurr = today;
k = 19;
knnModelCl = fitcknn(X(1:today,:), ycls(1:today), 'NumNeighbors', k);
for i = 1:days
    dayIdcs = idcurr : idcurr + interval;
    Xq = X(dayIdcs,:);
    yclsq = ycls(dayIdcs);
    
    yp_KNN = predict(knnModelCl,Xq);
    
    accKNN(i) = sum(yp_KNN == yclsq)/(interval+1)*100;
    fprintf(['Day: ' weekday_name(idcurr+5,:) '\taccKNN = %.2f%%\n'], accKNN(i))
    
    idcurr = idcurr + interval;
end
fprintf('\n')

%% ======================================================================== MUTUAL INFO ==================================

figure(2)
% weekday vs score

for wkd  = 1:7
    for i = 1:interval
        wkdIdcs = find(weekday_num==wkd);
        wkdMeanScore = mean(score(wkdIdcs));
    end
end


%% ======================================================================== NOTES ========================================


% goal: predict ~1 week
% put more weight on days closer to prediciton date
% nonlinear feature neural network

% mutual information - run on all features to see how output varies with
% each feature

% determine independent features
% what is a strong indicator of score


% average future days (predict saturday avg saturday)



% levels
% 1: 1-20
% 2: 21-60
% 3: 61-100
% 4: 101-140
% 5: 141-180
% 6: 181-200  <---- also level 5

%% ======================================================================== FUN TESTING ==================================

[dayIdcs, wkd, dSz] = extractDay(weekday_num, 13079);
timeIn = timeNum(dayIdcs);
scoreIn = score(dayIdcs);
timeOut = (0:1/48:.99)';

[scoreOut,scoreOutNaN] = imputeScores(timeNum,score,dayIdcs,dSz);

% idx = 2;
% len = 9;
% A = [120 130 140 135 125 110 0 0 0]
% A(idx:len) = circshift(A(idx:len),1)
% 
% idx = 7;
% A(idx:len) = circshift(A(idx:len),1)
% 
% idx = 8;
% A(idx:len) = circshift(A(idx:len),1)
% 
% 
% %                         \/
% %     \/                  \/
% % [120][130][140][135][125][110]
% % 
% % [120][130][140][135][125][110][ 0 ][ 0 ][ 0 ]
% % 
% % [120][130][NaN][140][135][125][NaN][NaN][110]
% % 
% % [120][130][135][140][135][125][120][115][110]
% 

%%
% 1. Create CUDAKernel object.
k = parallel.gpu.CUDAKernel('myfun.ptx','myfun.cu','entryPt1');

% 2. Set object properties.
k.GridSize = [8 1];
k.ThreadBlockSize = [16 1];

% 3. Call feval with defined inputs.
g1 = gpuArray(in1); % Input gpuArray.
g2 = gpuArray(in2); % Input gpuArray.

result = feval(k,g1,g2);









