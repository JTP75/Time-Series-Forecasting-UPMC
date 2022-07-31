function [net,info] = train_network(Xr,Yr,varargin)
if isa(Xr,'struct')
    X_train = Xr.train;
    y_train = Yr.train;
else
    X_train = Xr;
    y_train = Yr;
end

% default args
solver = "adam";
default_layers = true;
default_options = true;

% process arguments
for arg_idx = 1:2:length(varargin)
    if strcmp('Layers',varargin{arg_idx})
        layers = varargin{arg_idx+1};
        default_layers = false;
    elseif strcmp('Options',varargin{arg_idx})
        options = varargin{arg_idx+1};
        default_options = false;
    end
end

% network architecture
feat_count = length(X_train{1});
resp_count = size(y_train,2);

if default_layers
    layers = [...
        
        sequenceInputLayer([feat_count 1 1],'Name','input')
        flattenLayer('Name','flatten')
        % below values from bayesOpt
        gruLayer(191,'Name','gru1','RecurrentWeightsInitializer','He','InputWeightsInitializer','He')
        lstmLayer(107,'Name','gru2','RecurrentWeightsInitializer','He','InputWeightsInitializer','He')
        dropoutLayer(0.22651,'Name','drop1')
        bilstmLayer(188,'Name','bil1','RecurrentWeightsInitializer','He','InputWeightsInitializer','He')
        bilstmLayer(188,'Name','bil2','RecurrentWeightsInitializer','He','InputWeightsInitializer','He')
        dropoutLayer(0.22651,'Name','drop2')
        bilstmLayer(41,'Name','bil3','RecurrentWeightsInitializer','He','InputWeightsInitializer','He')
        bilstmLayer(41,'Name','bil4','RecurrentWeightsInitializer','He','InputWeightsInitializer','He')
        dropoutLayer(0.22651,'Name','drop3')
        bilstmLayer(9,'OutputMode',"last",'Name','bil5','RecurrentWeightsInitializer','He','InputWeightsInitializer','He')
        dropoutLayer(0.22651,'Name','drop4')
        
        fullyConnectedLayer(resp_count,'Name','fcOut')
        regressionLayer('Name','output')
        
    ];
end
if isa(layers,'nnet.cnn.layer.Layer')
    layers = layerGraph(layers);
end

% training options
if gpuDeviceCount>0
    mydevice = 'gpu';
else
    mydevice = 'cpu';
end

if default_options
    options = trainingOptions...
    (...
        solver, ...                                             % adam
        'MaxEpochs',            200, ...                        % max iters
        'GradientThreshold',    0.37957, ...                    % gradient boundary
        'InitialLearnRate',     0.00144, ...                    % initial learning rate
        'LearnRateSchedule',    "piecewise", ...                % how LR changes
        'LearnRateDropPeriod',  93, ...                         % how long between drops
        'LearnRateDropFactor',  0.21244, ...                    % lower LR by factor
        'MiniBatchSize',        64,...                          % nObs per iteration
        'Verbose',              true,...                        % whether to show info
        'Shuffle',              "every-epoch",...               % shuffle data
        'ExecutionEnvironment', mydevice...                     % gpu
    );
end

[net,info] = trainNetwork(X_train, y_train, layers, options);



