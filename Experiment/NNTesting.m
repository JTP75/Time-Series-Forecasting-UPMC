%% ======================================================================== NN TESTING ================================

clc
close all
clear all

%% ======================================================================== DATA LOADING & NN PARAMETERS ==============

DATA = load('HW7_Data.mat');
X = DATA.X;
y = DATA.y;
ybin = toBinary(y,3);
[m,n] = size(X);

WEIGHTS = load('HW7_weights_2.mat');
global thCAG;
thCAG = cell([2 3]);
thCAG{1,1} = WEIGHTS.Theta1;
thCAG{2,1} = size(WEIGHTS.Theta1);
thCAG{1,2} = WEIGHTS.Theta2;
thCAG{2,2} = size(WEIGHTS.Theta2);

K = 3;                                                                      % num of classes
L = 3;                                                                      % NN depth (num of layers)
layerSizes = [4,8,3];

if L ~= size(layerSizes)
    layerMismatch = MException('NN:LayerCountMismatch',...
        'Number of layers doesn''t match layer array size');
    throw(layerMismatch)
end

clear DATA WEIGHTS

%% ======================================================================== FORWARD PROPAGATION & PREDICTING ==========

[ypG,h,a,z] = predictNN(thCAG,X);
accG = sum(ypG==y)/m;
costG = nnCost(thCAG,X,y,K,0);

%% ======================================================================== BACK PROPAGATION & STOCHASTIC GD ==========

lam = 0;
alpha = 1;
MaxEpochs = 100;

thCA = sGD(layerSizes, X, y, lam, alpha, MaxEpochs);            % OOP approach may be better...
ypP = predictNN(thCA,X);
accP = sum(ypP==y)/m;



%% ======================================================================== COST PLOTTING =============================

thetaSpace = thCAG;
x = linspace(-10,10);

l = 1;
r = 1;
c = 1;
% thetaSpace{l}(r,c) = x;

%% ======================================================================== OOP APPROACH ==============================

%% toy set
fprintf('\n=============================================\n')

X_toy = [ 0,0 ; 0,1 ; 1,0 ; 1,1 ];
y_toy = [  0  ;  1  ;  1  ;  0  ];

myNetwork = Network();

myNetwork = myNetwork.addN(FCLayer(2,10));
myNetwork = myNetwork.addN(ActivationLayer(@sigmoid,@sigmoid));
% myNetwork = myNetwork.addN(FCLayer(10,10));
% myNetwork = myNetwork.addN(ActivationLayer(@tanh,@tanh_prime));
% myNetwork = myNetwork.addN(FCLayer(10,10));
% myNetwork = myNetwork.addN(ActivationLayer(@tanh,@tanh_prime));
% myNetwork = myNetwork.addN(FCLayer(10,10));
% myNetwork = myNetwork.addN(ActivationLayer(@tanh,@tanh_prime));
% myNetwork = myNetwork.addN(FCLayer(10,10));
% myNetwork = myNetwork.addN(ActivationLayer(@tanh,@tanh_prime));
% myNetwork = myNetwork.addN(FCLayer(10,10));
% myNetwork = myNetwork.addN(ActivationLayer(@tanh,@tanh_prime));
% myNetwork = myNetwork.addN(FCLayer(10,10));
% myNetwork = myNetwork.addN(ActivationLayer(@tanh,@tanh_prime));
myNetwork = myNetwork.addN(FCLayer(10,1));
myNetwork = myNetwork.addN(ActivationLayer(@tanh,@tanh_prime));

myNetwork = myNetwork.useN(@loss_mse, @loss_mse_prime);
myNetwork = myNetwork.fitN(X_toy, y_toy, 1000, 0.1);
output = myNetwork.predictN(X_toy);

fprintf('\n')
disp(output)

%% real set
fprintf('\n=============================================\n')

clear X_train y_train X_test y_test
[X_train,y_train,X_test,y_test] = splitData(X,y,0.8);
y_train_bin = toBinary(y_train,3);
y_test_bin = toBinary(y_test,3);

net = Network();

net = net.addN(FCLayer(4,20));
net = net.addN(ActivationLayer(@sigmoid,@sigmoid_prime));
net = net.addN(FCLayer(20,20));
net = net.addN(ActivationLayer(@sigmoid,@sigmoid_prime));
net = net.addN(FCLayer(20,20));
net = net.addN(ActivationLayer(@sigmoid,@sigmoid_prime));
net = net.addN(FCLayer(20,3));
net = net.addN(ActivationLayer(@sigmoid,@sigmoid_prime));

net = net.useN(@loss_mse, @loss_mse_prime);
net = net.fitN(X_train, y_train_bin, 1000, 0.1);
output = net.predictN(X_test);
pred = classifyProbs(output);

fprintf('\n')
disp([y_test,pred])
disp(sum(y_test==pred)/length(y_test))

%% matlab set for regression

outages = readtable('outages.csv');

region_label = outages.Region;
outtime_DT = outages.OutageTime;
resttime_DT = outages.RestorationTime;
nCustomers = outages.Customers;
cause_label = outages.Cause;

loss = outages.Loss;

region = labels2numbers(region_label);
cause = labels2numbers(cause_label);
outtime = datenum(outtime_DT);
resttime = datenum(resttime_DT);

REMOVE_NANS_MATRIX = [outtime,resttime,nCustomers,cause,region,loss];
m = length(loss);
newm = m;
i = 1;
j = 1;
while i < m
    if sum(isnan(REMOVE_NANS_MATRIX(i,:)))~=0
        newm = newm - 1;
    else
        NEW_REMOVE_NANS_MATRIX(j,:) = REMOVE_NANS_MATRIX(i,:);
        j = j + 1;
    end
    i = i + 1;
end

X = NEW_REMOVE_NANS_MATRIX(:,1:5);
y = NEW_REMOVE_NANS_MATRIX(:,6);
%%
regnet = Network();

regnet.addN(FCLayer(5,8));
regnet.addN(ActivationLayer(@relum,@relu_prime));
regnet.addN(FCLayer(8,1));
regnet.addN(ActivationLayer(@relum,@relu_prime));

regnet = regnet.useN(@loss_mse,@loss_mse_prime);
regnet = regnet.fitN(X,y,1000,0.1);
outpt = regnet.predictN(X);

























%% hardware playing

maxIterations = 500;
gridSize = 1000;
xlim = [-0.748766713922161, -0.748766707771757];
ylim = [ 0.123640844894862,  0.123640851045266];

%%

% Setup
t = tic();
x = linspace( xlim(1), xlim(2), gridSize );
y = linspace( ylim(1), ylim(2), gridSize );
[xGrid,yGrid] = meshgrid( x, y );
z0 = xGrid + 1i*yGrid;
count = ones( size(z0) );

% Calculate
z = z0;
for n = 0:maxIterations
    z = z.*z + z0;
    inside = abs( z )<=2;
    count = count + inside;
end
count = log( count );

% Show
cpuTime = toc( t );
fig = gcf;
fig.Position = [200 200 600 600];
imagesc( x, y, count );
colormap( [jet();flipud( jet() );0 0 0] );
axis off
title( sprintf( '%1.2fsecs (without GPU)', cpuTime ) );

%%

% Setup
t = tic();
x = gpuArray.linspace( xlim(1), xlim(2), gridSize );
y = gpuArray.linspace( ylim(1), ylim(2), gridSize );
[xGrid,yGrid] = meshgrid( x, y );
z0 = complex( xGrid, yGrid );
count = ones( size(z0), 'gpuArray' );

% Calculate
z = z0;
for n = 0:maxIterations
    z = z.*z + z0;
    inside = abs( z )<=2;
    count = count + inside;
end
count = log( count );

% Show
count = gather( count ); % Fetch the data back from the GPU
naiveGPUTime = toc( t );
imagesc( x, y, count )
axis off
title( sprintf( '%1.3fsecs (naive GPU) = %1.1fx faster', ...
    naiveGPUTime, cpuTime/naiveGPUTime ) )


