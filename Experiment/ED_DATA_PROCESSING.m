%% ======================================================================== CLEAR & CLOSE ================================

clc
clearvars
close all

%% ======================================================================== CONSTRUCT TABLES =============================

[clean_table, imputed_table, day_table, SCDF_matrix] = construct_tables;
data_store = Nedoc_Data(clean_table, imputed_table, day_table, SCDF_matrix);
clear SCDF_matrix

%% ========================================================================= LOAD DATA TO DATASTORE =======================

data = Nedoc_Data(T_clean, T_imputed, T_day, cldf);

%% O======================================================================================================================O
%  |----------------------------------------------------------------------------------------------------------------------|
%% |======================================================================= REGRESSORS ===================================|
%  |----------------------------------------------------------------------------------------------------------------------|
%% O======================================================================================================================O


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
%% reg
lags = 6;
[crnet, mu, sig, Xr] = trainCRNet(data_store,lags);   % Xr is standardized lag matrix
y_prediction = predict( crnet, Xr, "ExecutionEnvironment",'gpu', "MiniBatchSize",lags );
y_prediction = sig .* y_prediction + mu;
y_prediction = [ones([lags,1]) ; y_prediction];
y_prediction = cast(y_prediction,"double");
%% optimize lags
% for lags = 1:100
%     for
%         
%     end
% end

%% cls
lags = 31;
[crnet, Xr] = trainCRNet_CLS(data_store,lags);   % Xr is standardized lag matrix
y_prediction = predict( crnet, Xr, "ExecutionEnvironment",'gpu', "MiniBatchSize",lags );
y_prediction = classifyProbs(y_prediction);
y_prediction = [ones([lags,1]) ; y_prediction];
y_prediction = cast(y_prediction,"double");

%% ======================================================================== **ARIMA** ====================================

scoreImp = myModel.y_Imp;
mdl_ARIMA_Template = arima('MALags',1:2,'SMALags',12);
mdl_ARIMA_Estimate = estimate(mdl_ARIMA_Template,scoreImp);
yp_ARIMA = forecast(mdl_ARIMA_Estimate,30,scoreImp);

%% ======================================================================== **CLASS TESTING** ============================

today = 30000;
fprintf('\n===================================================\n')
fprintf( ['Last day in training set: ' , datestr(T_clean.Date_Time_DTA(today,:)) , ' (index %d)\n\n'], today )

%% tree ens init and train

RFFT = forecastModel(T_clean,'Tree Ensemble Model 00',1);
RFFT = RFFT.setSplit(today);
RFFT = RFFT.selectModelFunctions(@fineTreeEns,@predictFcn);
RFFT = RFFT.train();

% RFFT = forecastModel(T_clean,'Tree Ensemble Model 01',0);
% RFFT = RFFT.setSplit(today);
% RFFT = RFFT.selectModelFunctions(@fineTreeEns01,@predictFcn);
% RFFT = RFFT.train();

%% tree ens predict and plot

RFFT = RFFT.pred(0);

[~,acc1] = RFFT.generateRegPlots(RFFT.tomorrow,9);
[~,acc2] = RFFT.generateRegPlots(RFFT.tomorrow+48*9,9);
fprintf('\n==============================\n')
fprintf('Accuracy_1 = %.2f%%, Accuracy_2 = %.2f%%\n',acc1,acc2);

%% clustering classifier init & train

CCFM = forecastModel(T_clean,'Clustered Day Classifier Model',0);
CCFM = CCFM.setSplit(today);
CCFM = CCFM.createClSet;

% dec = CCFM.getRespForClusters;
% CCFM.score_pred = dec;
% CCFM.level_pred = getLevel(dec);
% CCFM.generateRegPlots(CCFM.tomorrow,9);

%% clustering classifier pred & plot.

% boosted tree ens classifier
% mdl_BSTC = BoostedTreeEns_CLS(X_train,y_train);
% yp = mdl_BSTC.predictFcn(CCFM.X_cl);

% 50 x 100 x 50 NN classifier
% mdl = ThreeLayerNN_CLS(X_train,y_train);
% yp = mdl.predictFcn(CCFM.X_cl);

% naive-bayes optimized classifier (only 20 iters)
mdl = mdl_NBOC;

% fine tree
% mdl = mdl_FTRC;

yp = mdl.predictFcn(CCFM.X_cl);
clsacc = sum(yp==CCFM.y_cl)/length(yp);
yp_REG = CCFM.getRespForClusters(yp);
CCFM.score_pred = yp_REG;
CCFM.level_pred = getLevel(yp_REG);
[plotfig1,acc1] = CCFM.generateRegPlots(CCFM.tomorrow,9);
[plotfig2,acc2] = CCFM.generateRegPlots(CCFM.tomorrow+48*9,9);
fprintf('\n==============================\n')
fprintf('Day Class Accuracy = %.2f%%\n',clsacc*100);
fprintf('Accuracy_1 = %.2f%%, Accuracy_2 = %.2f%%\n',acc1,acc2);

%% HAC

X = CCFM.X_clust;

% dists = pdist(X,'cityblock');
% linkage_matrix = linkage(dists,'average');
% 
% CCC = cophenet(linkage_matrix,dists);       % Cophenetic Correlation Coefficient
% IC = inconsistent(linkage_matrix);          % inconsistency coefficients
% 
% dendy = figure('NumberTitle','off','Name','Dendrogram Figure');
% dendrogram(linkage_matrix);
% ttl = ['CCC = ', num2str(CCC)];
% axis([0 32 0 linkage_matrix(length(linkage_matrix),3)*1.1])
% title(ttl)
% 
% C = 100000;
% hold on
% plot(C*ones([100,1]),'k-.')
% hold off

% clList = cluster(linkage_matrix,'Cutoff',C,'Criterion','distance');
clList = kmeans(X,16,'distance','sqeuclidean','Replicates',50);

K = max(clList);
% disp('Num of Classes:')
% disp(K)

%% clustered days' plots

figure('NumberTitle','off','Name','Day-Class All Curves')
meanFork = zeros([K,48]);
for k = 1:K
    kidcs = find(clList==k);
    meanFork(k,:) = mean(X(kidcs,:),1);
    subplot(ceil(sqrt(K)),ceil(sqrt(K)),k)
    axis([1,48,0,200])
    hold on
    dayCount = 0;
    for i = 1:length(X)
        if clList(i) == k
            plot(X(i,:))
            dayCount = dayCount + 1;
        end
    end
    plot(meanFork(k,:),'k-','LineWidth',3)
    ttl = ['Num of Days Plotted = ' num2str(dayCount)];
    title(ttl)
    hold off
end

%% plotting days predictions

dayz = myModel.X_clust_all;
resp = myModel.dayClass;
labelDefs = myModel.dayClass_DEF;

npls = 9;
strt = 30;
figure('NumberTitle','off','Name','Clustered Day plots')
for n = 1:npls
    subplot(ceil(sqrt(npls)),ceil(sqrt(npls)),n)
    hold on
    axis([1,48,0,200]);
    plot(dayz(strt+n,:));
    plot(labelDefs(resp(strt+n),:))
    acc(n) = sum(getLevel(dayz(strt+n,:))==getLevel(labelDefs(resp(strt+n),:)))/48;
    ttl = ['acc = ' num2str(acc(n))];
    title(ttl)
    hold off
end

%% day mapping

for i = 1:575
    X_day = (1:48)';
    y_day(:,i) = X(i,:)';
    X_cos = [ones([48,1]),cos(pi/24 * X_day),cos(pi/24 * X_day + 5),cos(pi/24 * X_day + 10),cos(pi/24 * X_day + 20)];
    
    theta(:,i) = pinv(X_cos'*X_cos)*(X_cos'*y_day(:,i));
    ypred(:,i) = X_cos * theta(:,i);
    
    err(i) = sum((ypred(:,i)-y_day(:,i)).^2)/48;
end

[errs,idcs] = sort(err);
errp = [idcs',errs']


idx = 67
plot(y_day(:,idx))
hold on
plot(ypred(:,idx))
axis([1,48,0,200])
hold off

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










