function [net,mu,sig,XrAll] = trainCRNet_DC(data_store,lags,ymat)
% ARCRIN: AutoRegressive Convolutional-Recurrent Integrated Network 


%% OPTIONS ================================================================
Lag = lags;
MiniBatchSize = 64;        % max(Lag) * size(ymat,2);
MaxEpochs = 150;
learningrate = 0.00311;     % 0.00611
solver = "adam";

%% LOAD DATA ==============================================================
td = data_store.today.d;
y = ymat;
y_train = ymat(1:td,:);
mu = mean(y_train);
sig = std(y_train);
y_train = (y_train - mu) ./ sig;
y = (y - mu) ./ sig;

%% PREPARE VARIABLES FOR AUTOREGRESSION ===================================
% use cell array to store feature vectors (lags * coefficients)
% each lag corresponds to one day

% Training data
XTrain = lagmatrix(y_train,Lag);                % <------ AUTOREGRESSIVE
XTrain = XTrain(max(Lag)+1:end,:)';
YTrain = y_train(max(Lag)+1:end,:)';
XrTrain = cell(size(XTrain,2),1);
YrTrain = zeros(size(YTrain,2),size(YTrain,1));
for i=1:size(XTrain,2)
    XrTrain{i,1} = XTrain(:,i);
    YrTrain(i,:) = YTrain(:,i);
end

% All data
XAll = lagmatrix(y,Lag);
XAll = XAll(max(Lag)+1:end,:)';
YAll = y(max(Lag)+1:end,:)';
XrAll = cell(size(XAll,2),1);
YrAll = zeros(size(YAll,2),size(YAll,1));
for i=1:size(XAll,2)
    XrAll{i,1} = XAll(:,i);
    YrAll(i,:) = YAll(:,i);
end

%% NETWORK ARCHITECTURE ===================================================
numFeatures = size(XTrain,1);% it depends on the roll back window (No of features, one output)
numResponses = size(y,2);FiltZise = 5;
% you can follow this template and create your own Architecture
layers = [...
    % Here input the sequence. No need to be modified
    sequenceInputLayer([numFeatures 1 1],'Name','input')
    sequenceFoldingLayer('Name','fold')
    
    % from here do your engeneering design of your CNN feature
    % extraction
    convolution2dLayer(FiltZise,32,'Padding','same','WeightsInitializer','he','Name','conv','DilationFactor',1); % <-- CONVOLVE
    batchNormalizationLayer('Name','bn')
    eluLayer('Name','elu')
    convolution2dLayer(FiltZise,32,'Padding','same','WeightsInitializer','he','Name','conv1','DilationFactor',2);
    eluLayer('Name','elu1')
    convolution2dLayer(FiltZise,32,'Padding','same','WeightsInitializer','he','Name','conv2','DilationFactor',4);
    eluLayer('Name','elu2')
    convolution2dLayer(FiltZise,32,'Padding','same','WeightsInitializer','he','Name','conv3','DilationFactor',8);
    eluLayer('Name','elu3')
    convolution2dLayer(FiltZise,32,'Padding','same','WeightsInitializer','he','Name','conv4','DilationFactor',16);
    eluLayer('Name','elu4')
    averagePooling2dLayer(1,'Stride',FiltZise,'Name','pool1')
    
    % here you finish your CNN design and next step is to unfold and
    % flatten. Keep this part like this
    sequenceUnfoldingLayer('Name','unfold')
    flattenLayer('Name','flatten')
    
    % from here the RNN design. Feel free to add or remove layers
    gruLayer(128,'Name','gru1','RecurrentWeightsInitializer','He','InputWeightsInitializer','He')
    lstmLayer(64,'Name','gru2','RecurrentWeightsInitializer','He','InputWeightsInitializer','He')          % <---- RECURRENT
    dropoutLayer(0.25,'Name','drop2')
    % this last part you must change the outputmode to last
    lstmLayer(32,'OutputMode',"last",'Name','bil4','RecurrentWeightsInitializer','He','InputWeightsInitializer','He')
    dropoutLayer(0.25,'Name','drop3')
    % here finish the RNN design
    
    % use a fully connected layer with one neuron because you will predict one step ahead
    fullyConnectedLayer(numResponses,'Name','fc')
    regressionLayer('Name','output')     ];

layers = layerGraph(layers);
layers = connectLayers(layers,'fold/miniBatchSize','unfold/miniBatchSize');     % <------ INTEGRATED NETWORK

%% TRAINING OPTIONS =======================================================
if gpuDeviceCount>0
    mydevice = 'gpu';
else
    mydevice = 'cpu';
end
options = trainingOptions...
(...
    solver, ...
    'MaxEpochs',            MaxEpochs, ...
    'GradientThreshold',    1, ...
    'InitialLearnRate',     learningrate, ...
    'LearnRateSchedule',    "piecewise", ...
    'LearnRateDropPeriod',  96, ...
    'LearnRateDropFactor',  0.25, ...
    'MiniBatchSize',        MiniBatchSize,...
    'Verbose',              false, ...
    'Shuffle',              "every-epoch",...
    'ExecutionEnvironment', mydevice,...
    'Plots',                'training-progress'...
);

%% TRAIN NETWORK (MAY TAKE A WHILE...) ====================================
% rng(0);
% net = trainNetwork(XrTrain,YrTrain,layers,options);
plot(net)






