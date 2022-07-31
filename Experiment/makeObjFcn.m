function ObjFcn = makeObjFcn(XTrain,YTrain,XValidation,YValidation)
ObjFcn = @valErrorFun;
    function [valError,cons,fileName] = valErrorFun(optVars)
         
        feat_count = size(XTrain{1},1);
        resp_count = size(YTrain,2);
        
        % architecture optimizables
        szgru = optVars.gru_size;
        szlst = optVars.lstm_size;
        szbil = optVars.bilstm_size;
        dfbil = optVars.bilstm_dropfactor;
        dpcnt = optVars.dropout_pcnt;
        
        layers = [...
            
            sequenceInputLayer([feat_count 1 1],'Name','input')
            flattenLayer('Name','flatten')
        
            gruLayer(szgru,'Name','gru1','RecurrentWeightsInitializer','He','InputWeightsInitializer','He')
            lstmLayer(szlst,'Name','gru2','RecurrentWeightsInitializer','He','InputWeightsInitializer','He')
            dropoutLayer(dpcnt,'Name','drop1')
            bilstmLayer(szbil,'Name','bil1','RecurrentWeightsInitializer','He','InputWeightsInitializer','He')
            bilstmLayer(szbil,'Name','bil2','RecurrentWeightsInitializer','He','InputWeightsInitializer','He')
            dropoutLayer(dpcnt,'Name','drop2')
            bilstmLayer(round(szbil*dfbil),'Name','bil3','RecurrentWeightsInitializer','He','InputWeightsInitializer','He')
            bilstmLayer(round(szbil*dfbil),'Name','bil4','RecurrentWeightsInitializer','He','InputWeightsInitializer','He')
            dropoutLayer(dpcnt,'Name','drop3')
            bilstmLayer(round(szbil*(dfbil^2)),'OutputMode',"last",'Name','bil5','RecurrentWeightsInitializer','He','InputWeightsInitializer','He')
            dropoutLayer(dpcnt,'Name','drop4')
        
            fullyConnectedLayer(resp_count,'Name','fcOut')
            regressionLayer('Name','output')
        
        ];
    
    
        solver = "adam";
        
        % training options optimizables
        alpha = optVars.LR_init;
        lrndp = optVars.LR_dropPeriod;
        lrndf = optVars.LR_dropFactor;
        gthrs = optVars.grad_threshold;
        
        options = trainingOptions(...
            solver, ...
            'ValidationData',       {XValidation, YValidation},...
            'ValidationFrequency',  30,...
            'MaxEpochs',            200, ...
            'GradientThreshold',    gthrs, ...
            'InitialLearnRate',     alpha, ...
            'LearnRateSchedule',    "piecewise", ...
            'LearnRateDropPeriod',  lrndp, ...
            'LearnRateDropFactor',  lrndf, ...
            'MiniBatchSize',        64,...
            'Verbose',              true, ...
            'Shuffle',              "every-epoch",...
            'ExecutionEnvironment', 'gpu'...
        );
    
        [net,info] = train_network(XTrain,YTrain,'Layers',layers,'Options',options);
        
        valError = info.FinalValidationRMSE;
        fileName = "RNN0__" + num2str(valError) + "__.mat";
        save(fileName,'net','valError','layers','options')
        cons = [];
        
    end
end
