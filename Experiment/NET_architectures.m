layer_array = {};

feat_count = length(Xr.train{1});
resp_count = size(Yr.train,2);
cnn_filt = 5;
nfilt = 32;

%% ARCHITECTURES ==========================================================

% ARC0
for d=1
    
    ARC0 = [...
        
        % Here input the sequence. No need to be modified
        sequenceInputLayer([feat_count 1 1],'Name','input')
        sequenceFoldingLayer('Name','fold')

        % from here do your engeneering design of your CNN feature
        % extraction
        convolution2dLayer(cnn_filt,nfilt,'Padding','same','WeightsInitializer','he','Name','conv','DilationFactor',1);
        batchNormalizationLayer('Name','bn')
        eluLayer('Name','elu')
        convolution2dLayer(cnn_filt,nfilt,'Padding','same','WeightsInitializer','he','Name','conv1','DilationFactor',2);
        eluLayer('Name','elu1')
        convolution2dLayer(cnn_filt,nfilt,'Padding','same','WeightsInitializer','he','Name','conv2','DilationFactor',4);
        eluLayer('Name','elu2')
        convolution2dLayer(cnn_filt,nfilt,'Padding','same','WeightsInitializer','he','Name','conv3','DilationFactor',8);
        eluLayer('Name','elu3')
        convolution2dLayer(cnn_filt,nfilt,'Padding','same','WeightsInitializer','he','Name','conv4','DilationFactor',16);
        eluLayer('Name','elu4')
        averagePooling2dLayer(1,'Stride',cnn_filt,'Name','pool1')

        % here you finish your CNN design and next step is to unfold and
        % flatten. Keep this part like this
        sequenceUnfoldingLayer('Name','unfold')
        flattenLayer('Name','flatten')

        % from here the RNN design. Feel free to add or remove layers
        gruLayer(128,'Name','gru1','RecurrentWeightsInitializer','He','InputWeightsInitializer','He')
        lstmLayer(64,'Name','gru2','RecurrentWeightsInitializer','He','InputWeightsInitializer','He')
        dropoutLayer(0.25,'Name','drop2')
        % this last part you must change the outputmode to last
        lstmLayer(32,'OutputMode',"last",'Name','bil4','RecurrentWeightsInitializer','He','InputWeightsInitializer','He')
        dropoutLayer(0.25,'Name','drop3')
        % here finish the RNN design

        % use a fully connected layer with one neuron because you will predict one step ahead
        fullyConnectedLayer(resp_count,'Name','fc')
        regressionLayer('Name','output')
        
    ];
end
layer_array{end+1} = 'ARC0';
layer_array{end+1} = ARC0;

% ARC1
for d=1
    
    ARC1 = [...
        
    % input & fold
    sequenceInputLayer([feat_count 1 1],'Name','input')
    sequenceFoldingLayer('Name','fold')
    
    % convolution
    convolution2dLayer(cnn_filt,nfilt,'Padding','same','WeightsInitializer','he','Name','conv0','DilationFactor',1)
    batchNormalizationLayer('Name','bn')
    eluLayer('Name','elu0')
    
    averagePooling2dLayer(1,'Stride',cnn_filt,'Name','pool1')
    
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

end
layer_array{end+1} = 'ARC1';
layer_array{end+1} = ARC1;

% RNN0
for d=1
    
    RNN0 = [...
        
        sequenceInputLayer([feat_count 1 1],'Name','input')
        flattenLayer('Name','flatten')
        
        gruLayer(128,'Name','gru1','RecurrentWeightsInitializer','He','InputWeightsInitializer','He')
        lstmLayer(128,'Name','gru2','RecurrentWeightsInitializer','He','InputWeightsInitializer','He')
        dropoutLayer(0.25,'Name','drop1')
        bilstmLayer(64,'Name','bil1','RecurrentWeightsInitializer','He','InputWeightsInitializer','He')
        bilstmLayer(64,'Name','bil2','RecurrentWeightsInitializer','He','InputWeightsInitializer','He')
        dropoutLayer(0.25,'Name','drop2')
        bilstmLayer(32,'Name','bil3','RecurrentWeightsInitializer','He','InputWeightsInitializer','He')
        bilstmLayer(32,'Name','bil4','RecurrentWeightsInitializer','He','InputWeightsInitializer','He')
        dropoutLayer(0.25,'Name','drop3')
        bilstmLayer(16,'OutputMode',"last",'Name','bil5','RecurrentWeightsInitializer','He','InputWeightsInitializer','He')
        dropoutLayer(0.25,'Name','drop4')
        
        fullyConnectedLayer(resp_count,'Name','fcOut')
        regressionLayer('Name','output')
        
    ];
    
end
layer_array{end+1} = 'RNN0';
layer_array{end+1} = RNN0;

% RNN1
for d=1
    
    RNN1 = [...
        
        sequenceInputLayer([feat_count 1 1],'Name','input')
        flattenLayer('Name','flatten')
        
        gruLayer(128,'Name','gru1','RecurrentWeightsInitializer','He','InputWeightsInitializer','He')
        lstmLayer(128,'Name','gru2','RecurrentWeightsInitializer','He','InputWeightsInitializer','He')
        dropoutLayer(0.25,'Name','drop1')
        bilstmLayer(64,'Name','bil1','RecurrentWeightsInitializer','He','InputWeightsInitializer','He')
        bilstmLayer(64,'Name','bil2','RecurrentWeightsInitializer','He','InputWeightsInitializer','He')
        dropoutLayer(0.25,'Name','drop2')
        bilstmLayer(32,'Name','bil3','RecurrentWeightsInitializer','He','InputWeightsInitializer','He')
        bilstmLayer(32,'Name','bil4','RecurrentWeightsInitializer','He','InputWeightsInitializer','He')
        dropoutLayer(0.25,'Name','drop3')
        bilstmLayer(32,'Name','bil5','RecurrentWeightsInitializer','He','InputWeightsInitializer','He')
        bilstmLayer(32,'Name','bil6','RecurrentWeightsInitializer','He','InputWeightsInitializer','He')
        dropoutLayer(0.25,'Name','drop4')
        bilstmLayer(16,'OutputMode',"last",'Name','bEnd','RecurrentWeightsInitializer','He','InputWeightsInitializer','He')
        dropoutLayer(0.25,'Name','dropEnd')
        
        fullyConnectedLayer(resp_count,'Name','fcOut')
        regressionLayer('Name','output')
        
    ];
    
end
layer_array{end+1} = 'RNN1';
layer_array{end+1} = RNN1;

% resnet50
for d=1
    RN50 = resnet50(feat_count,resp_count);
end
layer_array{end+1} = 'RN50';
layer_array{end+1} = RN50;

% xception
for d=1
    XPTN = xception(feat_count,resp_count);
end
layer_array{end+1} = 'XPTN';
layer_array{end+1} = XPTN;

%% FINAL STRUCT ===========================================================

nets = struct(layer_array{:});

for d=1
    layersEst = [...
        
        sequenceInputLayer([feat_count 1 1],'Name','input')
        flattenLayer('Name','flatten')
        % below values from bayesOpt
        gruLayer(229,'Name','gru1','RecurrentWeightsInitializer','He','InputWeightsInitializer','He')
        lstmLayer(116,'Name','gru2','RecurrentWeightsInitializer','He','InputWeightsInitializer','He')
        dropoutLayer(0.74135,'Name','drop1')
        bilstmLayer(126,'Name','bil1','RecurrentWeightsInitializer','He','InputWeightsInitializer','He')
        bilstmLayer(126,'Name','bil2','RecurrentWeightsInitializer','He','InputWeightsInitializer','He')
        dropoutLayer(0.74135,'Name','drop2')
        bilstmLayer(35,'Name','bil3','RecurrentWeightsInitializer','He','InputWeightsInitializer','He')
        bilstmLayer(35,'Name','bil4','RecurrentWeightsInitializer','He','InputWeightsInitializer','He')
        dropoutLayer(0.74135,'Name','drop3')
        bilstmLayer(10,'OutputMode',"last",'Name','bil5','RecurrentWeightsInitializer','He','InputWeightsInitializer','He')
        dropoutLayer(0.74135,'Name','drop4')
        
        fullyConnectedLayer(resp_count,'Name','fcOut')
        regressionLayer('Name','output')
        
    ];
    optionsEst = trainingOptions...
    (...
        "adam", ...                                             % adam
        'MaxEpochs',            200, ...                        % max iters
        'GradientThreshold',    0.93312, ...                    % gradient boundary
        'InitialLearnRate',     0.0019428, ...                  % initial learning rate
        'LearnRateSchedule',    "piecewise", ...                % how LR changes
        'LearnRateDropPeriod',  96, ...                         % how long between drops
        'LearnRateDropFactor',  0.13937, ...                    % lower LR by factor
        'MiniBatchSize',        64,...                          % nObs per iteration
        'Verbose',              false, ...                      % whether to show info
        'Shuffle',              "every-epoch",...               % shuffle data
        'ExecutionEnvironment', 'gpu'...                        % gpu
    );
end


