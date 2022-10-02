function loss = nedocLoss_01(day_obs,day_pred)
% trying to find an ideal measure of performance
%
% params:
%
%       day_obs: double row vector
%       day_pred: double row vector
%
%       - day_pred and day_obs must have same shape
%       - must both be ROW VECTORS
%

if size(day_obs) ~= size(day_pred)
    error("nedocLoss(): arg dimensions mismatch")
elseif size(day_obs,1) ~= 1 || size(day_pred,1) ~= 1
    error("nedocLoss(): args must be row vectors")
end

MSE = @(y,yp) mean((y-yp).^2);
RMSE = @(y,yp) sqrt(MSE(y,yp));

mse = MSE(day_obs,day_pred);
rmse = RMSE(day_obs,day_pred);
dmse = MSE(diff(day_obs),diff(day_pred));
drmse = RMSE(diff(day_obs),diff(day_pred));

fprintf("%.2f\t%.2f\t%.2f\t%.2f\n",mse,rmse,dmse,drmse)








