%% ======================================================================== FULL DATA SET (288 obs per day) ===================
load FULL_IMPUTED.mat
ds_full = NedocData(TI_full,288);
ds_full = ds_full.setToday(0.95);
ds_full = ds_full.setPPD(48);
ARCRINpreds = ds_full;
ppd = ds_full.PPD;
disp('.')

%% ======================================================================== DAY CURVE TRANSFORM ===============================
% getmats
[~,yimp] = ds_full.getmats('all','time');

x = (1:ppd)';
y = zeros([ds_full.L/ppd,ppd]);
for i = 1:ppd:length(yimp)-(ppd-1)
    y((i+(ppd-1))/ppd,:) = yimp(i:i+(ppd-1));
end

% fits
transform = [x.^0,...
    sin(x*2*pi/ppd), cos(x*2*pi/ppd),...
    sin(x*2*pi/(ppd/2)), cos(x*2*pi/(ppd/2)),...
    sin(x*2*pi/(ppd/4)), cos(x*2*pi/(ppd/4)),...
    sin(x*2*pi/(ppd/8)), cos(x*2*pi/(ppd/8))     ];
y_trans = zeros([size(y,1),size(transform,2)]);

for i = 1:size(y,1)
    y_trans(i,:) = normalEqn(transform,y(i,:)');
end

% view transform
transformed_y = y_trans * transform';
y_prediction = reshape(transformed_y', [], 1);
ds_full = ds_full.pushResp(y_prediction,'trans');
[~,avgacc] = ds_full.plot('View Transform', 'tmr', 9)                       %#ok<NOPTS>
ds_full = ds_full.popResp;
    
%% ======================================================================== TRAIN ARCRIN ======================================
for modelnum = 1:20
M = height(ds_full.T_imp) / ppd;
lags = 1:14;
[crnet, mu, sig, transform, Xcell] = trainARCRIN(ds_full, lags, 200);         % Xr is standardized lag matrix

% %% ======================================================================== PREDICT & ADD TO STORE ============================
coeff_pred_std = predict( crnet, Xcell, "ExecutionEnvironment",'gpu', "MiniBatchSize",max(lags)*size(transform,2) );
coeff_pred_std = cast(coeff_pred_std,"double");
coeff_pred_std = [zeros([M-size(coeff_pred_std,1) , size(coeff_pred_std,2)]) ; coeff_pred_std];
coeff_pred = sig .* coeff_pred_std + mu;

daypred = coeff_pred * transform';
y_prediction = reshape(daypred', [], 1);

lbl = ['Validation v' num2str(modelnum)];
ARCRINpreds = ARCRINpreds.pushResp(y_prediction,lbl);
end

%% ======================================================================== PLOT =============================================
atestfig = ARCRINpreds.plot('ARCRIN Predictors', 'tmr', 16, 'ShowMean', true);
arcacc = ARCRINpreds.predictorAccs('test');
fprintf('\n===========================\n')
for i = 1:size(arcacc,1)
    fprintf([arcacc{i,1} ': acc = %.2f%%\n'],arcacc{i,2});
end





