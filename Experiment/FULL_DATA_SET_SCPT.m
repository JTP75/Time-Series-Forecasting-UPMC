%% ======================================================================== FULL DATA SET (288 obs per day) ===================
load FULL_IMPUTED.mat
ds_base = NedocData(TI_full,288);
ds_base = ds_base.setToday(0.95);
ds_base = ds_base.setPPD(48);
ds_ARC = ds_base;
ds_RNN0 = ds_base;
ppd = ds_base.PPD;
disp('.')

%% ======================================================================== PCA ===============================================

daymat = ds_base.getmats('all','days')';
meand = mean(daymat,2);
stdd = std(daymat,0,2);
daymat_std = (daymat - meand) ./ stdd;

covmat = daymat_std * daymat_std';
all_eigvals = flip(eig(covmat));
total_variance = sum(all_eigvals);
for k = 1:20
    captured_variance(k) = sum(all_eigvals(1:k))/total_variance; %#ok<SAGROW>
end
K = 9;
eigvals = all_eigvals(1:K);
[eigvecs,~] = eigs(covmat,K);
weights = eigvecs' * daymat;

%% view reconstr
y_reconstr = (eigvecs*weights)';
y_prediction = reshape(y_reconstr', [], 1);
ds_base = ds_base.pushResp(y_prediction,'PCA');
plotfig = ds_base.plot('View Transform', 'tmr-9', 25, 'Showmean', false)                       %#ok<NOPTS>
ds_base.getaccs('all')
ds_base = ds_base.popResp;
%%

ys = y';
mu0 = mean(ys,2);
sig0 = std(ys,0,2);
ys = (ys-mu0)./sig0;
[W,evects] = PCA(ys,0.90);
    
%% ======================================================================== TRAIN ARCRIN ======================================
% for modelnum = 1:20
M = height(ds_base.T_imp) / ppd;
lags = 1:14;
[crnet, mu1, sig1, Xcell] = trainARCRIN(ds_base, lags, 200, y', 'enablePCA', true);

%% ======================================================================== PREDICT & ADD TO STORE ============================
Wpred_std = predict( crnet, Xcell, "ExecutionEnvironment",'gpu', "MiniBatchSize",64 );
Wpred_std = cast(Wpred_std,"double");
Wpred_std = [zeros([M-size(Wpred_std,1) , size(Wpred_std,2)]) ; Wpred_std];
Wpred = sig1 .* Wpred_std + mu1;

daypred = Wpred;%*evects;
% daypred = (sig0 .* daypred + mu0)';
y_prediction = reshape(daypred, [], 1);

% lbl = ['Validation v' num2str(modelnum)];
lbl = 'PrePCA ARCRIN';
ds_ARC = ds_ARC.pushResp(y_prediction,lbl);
% end



%% ======================================================================== PROP =============================================
%% lode
[Xr,Yr,C,T] = dataprep_shell(ds_base,'Validate',false,'Lags',1:14,'PCApcnt',[0.90,0.90]);
NET_architectures;
%% trane
net = train_network(Xr,Yr,'Plot',false);
%% predic
yp = predict_net(net,Xr.all,C,T);
ds_ARC = ds_ARC.pushResp(yp,'RNN0 Bayesian Opt');

%% ======================================================================== PLOT =============================================
atestfig = ds_ARC.plot('RNN0 Predictor', 'tmr+9', 16, 'Showmean', false);
arcacc = ds_ARC.getaccs('test',1);
fprintf('\n===========================\n')
for i = 1:size(arcacc,1)
    fprintf([arcacc{i,1} ': acc = %.2f%%\n'],arcacc{i,2});
end

%% ======================================================================== ENS RNN0 =========================================
num_mdls = 20;

[Xr,Yr,C,T] = dataprep_shell(ds_base,'Validate',false,'Lags',1:14,'PCApcnt',[0.90,0.90]);
NET_architectures;

networks = {};
for mn = 1:num_mdls
    networks{end+1} = train_network(Xr,Yr,'Layers',nets.RNN0,'Fold',false); %#ok<SAGROW>
end

yps = {};
for mn = 1:num_mdls
    yps{end+1} = predict_net(networks{mn},Xr.all,C,T); %#ok<SAGROW>
    ds_RNN0 = ds_RNN0.pushResp(yps{end},['RNN0: ' num2str(mn)]);
end

%% ======================================================================== PLOT =============================================
plotfig = ds_RNN0.plot('RNN Predictor Ensemble', 'tmr', 16, 'Showmean', true);
RNNaccs = ds_RNN0.getaccs('test');
fprintf('\n===========================\n')
for i = 1:size(RNNaccs,1)
    fprintf([RNNaccs{i,1} ': acc = %.2f%%\n'],RNNaccs{i,2});
end

%% ======================================================================== OPTIMIZE RNN0 ====================================
% rnn0 is fast, accurate, and (relatively) easy to train. will now perform
% bayesian optimization:

OptVars = ...
[
    optimizableVariable('gru_size',[16 256],'Type','integer')
    optimizableVariable('lstm_size',[16 256],'Type','integer')
    optimizableVariable('bilstm_size',[16 256],'Type','integer')
    optimizableVariable('bilstm_dropfactor',[0.1,1],'Type','real')
    optimizableVariable('dropout_pcnt',[0 1],'Type','real')
    optimizableVariable('LR_init',[1e-5 1e-2],'Transform','log')
    optimizableVariable('LR_dropPeriod',[10 100],'Type','integer')
    optimizableVariable('LR_dropFactor',[0.1 1],'Type','real')
    optimizableVariable('grad_threshold',[0.1 1],'Type','real')
];

[Xr,Yr,C,T] = dataprep_shell(ds_base,'Validate',true,'Lags',1:14,'PCApcnt',[0.90,0.90]);
objfcn = makeObjFcn(Xr.train,Yr.train,Xr.valid,Yr.valid);

BayesObject = bayesopt(objfcn,OptVars, ...
    'MaxTime',10*60*60, ...
    'IsObjectiveDeterministic',false, ...
    'UseParallel',true);

%% ======================================================================== WEEK PREDICTOR ==================================
daymat = ds_base.getmats('all','days');
% predict the mean score on days within ~1 week in the future
lag = 1:14;
daymeans = mean(daymat,2);
lagmat = lagmatrix(daymeans,lag);

xtr = lagmat(1:round(length(lagmat)*0.95),:);
ytr = daymeans(1:round(length(lagmat)*0.95));
xal = lagmat;
yal = daymeans;

xtr = xtr(lag(end)+1:end,:);
ytr = ytr(lag(end)+1:end,:);
xal = xal(lag(end)+1:end,:);
yal = yal(lag(end)+1:end,:);

cent = struct('mu',mean(xtr),'sig',std(xtr));
xtr_std = (xtr - cent.mu) ./ cent.sig;
xal_std = (xal - cent.mu) ./ cent.sig;

Xr = mat2cellR(xal_std);
Xr_train = mat2cellR(xtr_std);
Yr = yal;
Yr_train = ytr;

%% trane
opts = trainingOptions( ...
    "adam", ...
    'MaxEpochs',            200, ...
    'GradientThreshold',    0.37957, ...
    'InitialLearnRate',     0.00144, ...
    'LearnRateSchedule',    "piecewise", ...
    'LearnRateDropPeriod',  93, ...
    'LearnRateDropFactor',  0.21244, ...
    'MiniBatchSize',        64, ...
    'Verbose',              true, ...
    'Shuffle',              "every-epoch", ...
    'ExecutionEnvironment', 'gpu' ...
);
wknet = train_network(Xr_train,Yr_train,'Options',opts);

%% pred
mpred = predict( wknet, Xr, "ExecutionEnvironment",'gpu', "MiniBatchSize",64 );
mpred = cast(mpred,"double");
mpred = [100*ones([lag(end),1]) ; mpred];





%%

err = RMSE_centered(ds_ARC.y(ds_ARC.today.i:end),ds_ARC.yp{1}(ds_ARC.today.i:end));
fprintf("error = %.4f\n",err)
mtlresp = ds_ARC.yp{1};
dlmwrite('mtlrsp.txt',mtlresp);




