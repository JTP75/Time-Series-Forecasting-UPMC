function [net, mean_vect, std_vect, transmat, Xr] = trainARCRIN(ds,Lag,MaxEpochs)
% takes day curve coeffs as ymat

%% OPTIONS ================================================================
MiniBatchSize = 64;         % max(Lag) * size(ymat,2);
learningrate = 0.004;
solver = "adam";

%% PREP DATA ==============================================================
[transmat,ymat] = transformDayCurves(ds);

td = floor(ds.today.i / ds.PPD);
ytr = ymat(1:td,:);
mean_vect = mean(ytr);
std_vect = std(ytr);
ymat = (ymat - mean_vect) ./ std_vect;
ytr = (ytr - mean_vect) ./ std_vect;

X = lagmatrix(ymat, Lag);
X = X(Lag(end)+1:end,:)';
y = ymat(Lag(end)+1:end,:)';

Xtr = lagmatrix(ytr, Lag);
Xtr = Xtr(Lag(end)+1:end,:)';
ytr = ytr(Lag(end)+1:end,:)';

Xr = mat2cellR(X);
Yr = y';
Xr_train = mat2cellR(Xtr);
Yr_train = ytr';

%% NETWORK ARCHITECTURE ===================================================
feat_count = length(Xr{1});
resp_count = size(Yr,2);
filt_size = 5;
layers = [...

    % input & fold
    sequenceInputLayer([feat_count 1 1],'Name','input')
    sequenceFoldingLayer('Name','fold')
    
        % convolution
        convolution2dLayer(filt_size,32,'Padding','same','WeightsInitializer','he','Name','conv0','DilationFactor',1);
        batchNormalizationLayer('Name','bn')
        eluLayer('Name','elu0')
        convolution2dLayer(filt_size,32,'Padding','same','WeightsInitializer','he','Name','conv1','DilationFactor',2);
        eluLayer('Name','elu1')
        convolution2dLayer(filt_size,32,'Padding','same','WeightsInitializer','he','Name','conv2','DilationFactor',4);
        eluLayer('Name','elu2')
        convolution2dLayer(filt_size,32,'Padding','same','WeightsInitializer','he','Name','conv3','DilationFactor',8);
        eluLayer('Name','elu3')
        convolution2dLayer(filt_size,32,'Padding','same','WeightsInitializer','he','Name','conv4','DilationFactor',16);
        eluLayer('Name','elu4')
        averagePooling2dLayer(1,'Stride',filt_size,'Name','pool1')
    
    % unfold & flatten
    sequenceUnfoldingLayer('Name','unfold')
    flattenLayer('Name','flatten')
    
        % recurrent
        gruLayer(128,'Name','gru1','RecurrentWeightsInitializer','He','InputWeightsInitializer','He')
        lstmLayer(64,'Name','gru2','RecurrentWeightsInitializer','He','InputWeightsInitializer','He')
        dropoutLayer(0.25,'Name','drop2')
        lstmLayer(32,'OutputMode',"last",'Name','bil4','RecurrentWeightsInitializer','He','InputWeightsInitializer','He')
        dropoutLayer(0.25,'Name','drop3')
    
    % output
    fullyConnectedLayer(resp_count,'Name','fc')
    regressionLayer('Name','output')
    
];

layers = layerGraph(layers);
layers = connectLayers(layers,'fold/miniBatchSize','unfold/miniBatchSize');

%% TRAINING OPTIONS =======================================================
if gpuDeviceCount>0
    mydevice = 'gpu';
else
    mydevice = 'cpu';
end
options = trainingOptions...
(...
    solver, ...                                     % adam
    'MaxEpochs',            MaxEpochs, ...          % max iters
    'GradientThreshold',    1, ...                  % gradient boundary
    'InitialLearnRate',     learningrate, ...       % initial learning rate
    'LearnRateSchedule',    "piecewise", ...        % how LR changes
    'LearnRateDropPeriod',  96, ...                 % how long between drops
    'LearnRateDropFactor',  0.25, ...               % lower LR by factor
    'MiniBatchSize',        MiniBatchSize,...       % nObs per iteration
    'Verbose',              true, ...               % whether to show info
    'Shuffle',              "every-epoch",...       % shuffle data
    'ExecutionEnvironment', mydevice...             % gpu
);

%% TRAIN NETWORK ==========================================================
net = trainNetwork(Xr_train, Yr_train, layers, options);

