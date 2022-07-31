function [net, mean_vect, std_vect, Xr, extra_arg_out1] = trainARCRIN(ds,Lag,MaxEpochs,y_input,varargin)
% takes day curve vals as y_input

PCAen = false;
PCA_pcnt = 0.90;
for arg_idx = 1:2:length(varargin)
    if strcmp('enablePCA',varargin{arg_idx})
        PCAen = varargin{arg_idx+1};
    elseif strcmp('PCApcnt',varargin{arg_idx})
        PCA_pcnt = varargin{arg_idx+1};
    end
end

%% OPTIONS ================================================================
MiniBatchSize = 64;
learningrate = 0.004;
solver = "adam";

%% PREP DATA ==============================================================
ymat = y_input';
td = floor(ds.today.i / ds.PPD);
y_trANDval = ymat(1:td,:);

% standardize data
mean_vect = mean(y_trANDval);
std_vect = std(y_trANDval);
ymat = (ymat - mean_vect) ./ std_vect;
y_trANDval = (y_trANDval - mean_vect) ./ std_vect;

% all data
X = lagmatrix(ymat, Lag);
X = X(Lag(end)+1:end,:)';
y = ymat(Lag(end)+1:end,:)';

% train and val
X_trANDval = lagmatrix(y_trANDval, Lag);
X_trANDval = X_trANDval(Lag(end)+1:end,:)';
y_trANDval = y_trANDval(Lag(end)+1:end,:)';

if PCAen
    [weights,eigen_vectors] = PCA(X_trANDval,PCA_pcnt);
    extra_arg_out1 = eigen_vectors';
    weights_all = extra_arg_out1 * X;
    
    X = weights_all;
    X_trANDval = weights;
end

% split train and valid
[Xtr, ytr, Xvl, yvl] = splitData(X_trANDval', y_trANDval', 0.8);

Xr = mat2cellR(X);
Yr = y';
Xr_valid = mat2cellR(Xvl');
yr_valid = yvl'';
Xr_train = mat2cellR(Xtr');
Yr_train = ytr'';

%% NETWORK ARCHITECTURE ===================================================

% feat_count = length(Xr{1});
% resp_count = size(Yr,2);
% filt_size = 5;
% nfilt = 32;
% 
% layers = [...
% 
%     % input & fold
%     sequenceInputLayer([feat_count 1 1],'Name','input')         % [63 1 1]
%     sequenceFoldingLayer('Name','fold')                         % [63 1]
%     
%     
%         % convolution
%         convolution2dLayer(filt_size,nfilt,'Padding','same','WeightsInitializer','he','Name','conv0','DilationFactor',1);
%         batchNormalizationLayer('Name','bn')
%         eluLayer('Name','elu0')
%         convolution2dLayer(filt_size,nfilt,'Padding','same','WeightsInitializer','he','Name','conv1','DilationFactor',2);
%         eluLayer('Name','elu1')
%         convolution2dLayer(filt_size,nfilt,'Padding','same','WeightsInitializer','he','Name','conv2','DilationFactor',4);
%         eluLayer('Name','elu2')
%         convolution2dLayer(filt_size,nfilt,'Padding','same','WeightsInitializer','he','Name','conv3','DilationFactor',8);
%         eluLayer('Name','elu3')
%         convolution2dLayer(filt_size,nfilt,'Padding','same','WeightsInitializer','he','Name','conv4','DilationFactor',16);
%         eluLayer('Name','elu4')
%         averagePooling2dLayer(1,'Stride',filt_size,'Name','pool1')
%     
%     % unfold & flatten
%     sequenceUnfoldingLayer('Name','unfold')
%     flattenLayer('Name','flatten')
%     
%         % recurrent
%         gruLayer(128,'Name','gru1','RecurrentWeightsInitializer','He','InputWeightsInitializer','He')
%         lstmLayer(64,'Name','gru2','RecurrentWeightsInitializer','He','InputWeightsInitializer','He')
%         dropoutLayer(0.25,'Name','drop2')
%         lstmLayer(32,'OutputMode',"last",'Name','bil4','RecurrentWeightsInitializer','He','InputWeightsInitializer','He')
%         dropoutLayer(0.25,'Name','drop3')
%     
%     % output
%     fullyConnectedLayer(resp_count,'Name','fc')
%     regressionLayer('Name','output')
%     
% ];
% 
% layers = layerGraph(layers);
% layers = connectLayers(layers,'fold/miniBatchSize','unfold/miniBatchSize');

% ORIGINAL ^^^^^^

feat_count = length(Xr{1});
resp_count = size(Yr,2);
cnn_filt_size = 5;
nfilt = 32;

layers = [...

    % input & fold
    sequenceInputLayer([feat_count 1 1],'Name','input')         % [126 1 1]
    sequenceFoldingLayer('Name','fold')                         % [126 1]
    
        % convolution
        convolution2dLayer(cnn_filt_size,nfilt,'Padding','same','WeightsInitializer','he','Name','conv0','DilationFactor',1)
        batchNormalizationLayer('Name','bn')
        eluLayer('Name','elu0')
%         convolution2dLayer(cnn_filt_size,nfilt,'Padding','same','WeightsInitializer','he','Name','conv1','DilationFactor',2)
%         eluLayer('Name','elu1')
        
        averagePooling2dLayer(1,'Stride',cnn_filt_size,'Name','pool1')
    
    % unfold & flatten
    sequenceUnfoldingLayer('Name','unfold')
    flattenLayer('Name','flatten')
    
        % recurrent
        gruLayer(128,'Name','gru1','RecurrentWeightsInitializer','He','InputWeightsInitializer','He')
        lstmLayer(64,'Name','gru2','RecurrentWeightsInitializer','He','InputWeightsInitializer','He')
        bilstmLayer(32,'Name','bil1','RecurrentWeightsInitializer','He','InputWeightsInitializer','He')
        dropoutLayer(0.25,'Name','drop2')
        
        bilstmLayer(16,'OutputMode',"last",'Name','bil2','RecurrentWeightsInitializer','He','InputWeightsInitializer','He')
        dropoutLayer(0.25,'Name','drop3')
    
    % output
    fullyConnectedLayer(resp_count,'Name','fcOut')
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
    'ValidationData',       {Xr_valid, yr_valid},...% validation set
    'ValidationFrequency',  30,...                  % validation freq
    'MaxEpochs',            MaxEpochs, ...          % max iters
    'GradientThreshold',    1, ...                  % gradient boundary
    'InitialLearnRate',     learningrate, ...       % initial learning rate
    'LearnRateSchedule',    "piecewise", ...        % how LR changes
    'LearnRateDropPeriod',  50, ...                 % how long between drops
    'LearnRateDropFactor',  0.25, ...               % lower LR by factor
    'MiniBatchSize',        MiniBatchSize,...       % nObs per iteration
    'Verbose',              true, ...               % whether to show info
    'Shuffle',              "every-epoch",...       % shuffle data
    'ExecutionEnvironment', mydevice);%,...            % gpu
%     'plots',                'training-progress'...  % plot
% );

%% TRAIN NETWORK ==========================================================
net = trainNetwork(Xr_train, Yr_train, layers, options);

