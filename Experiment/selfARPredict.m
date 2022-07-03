function y_pred = selfARPredict(net, ds, Lag)
M = ds.L.i;
td = ds.today.i;

[~,y] = ds.getmats('all','imp','time');
[~,y_train] = ds.getmats('train','imp','time');

mu = mean(y_train);
sig = std(y_train);
y_train_std = (y_train - mu) / sig;
y_std = (y - mu) / sig;

XTrain = lagmatrix(y_train_std,Lag);
XTrain = XTrain(max(Lag)+1:end,:)';
XrTrain = cell(size(XTrain,2),1);
for i=1:size(XTrain,2)
    XrTrain{i,1} = XTrain(:,i);
end

yp_train = predict( net, XrTrain, "ExecutionEnvironment",'gpu', "MiniBatchSize",max(Lag) );
yp_train = [ones([(td - length(yp_train)),1]) ; yp_train];

y_pred = yp_train;

for i = (td+1):M
    inp_single = XrTrain{end};
    yp_single = predict( net, inp_single, "ExecutionEnvironment",'gpu', "MiniBatchSize",max(Lag) );
    y_pred = [y_pred ; yp_single]; %#ok<AGROW>
    lagmat = lagmatrix(y_pred,Lag);
    XrTrain{end+1} = lagmat(end,:)';
end

y_pred = cast(sig .* y_pred - mu,"double");
















