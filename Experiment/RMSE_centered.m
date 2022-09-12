function error = RMSE_centered(actual,predicted)

a = (actual - mean(actual))/std(actual);
p = (predicted - mean(predicted))/std(predicted);

error = mean((a-p).^2);