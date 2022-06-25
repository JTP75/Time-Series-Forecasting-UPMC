

classdef Network
    properties
        layers cell                                                         % array of layer objects
        loss_function
        loss_prime_function
        layerCount
    end
    methods
        function obj = Network()
            obj.layers = {};
            obj.loss_function = [];
            obj.loss_prime_function = [];
            obj.layerCount = 0;
        end
        function obj = addN(obj, layer)
            obj.layers{obj.layerCount + 1} = layer;
            obj.layerCount = obj.layerCount + 1;
        end
        function obj = useN(obj, J_function, gradJ_function)
            obj.loss_function = J_function;
            obj.loss_prime_function = gradJ_function;
        end
        function yp = predictN(obj, input_data)
            [m,~] = size(input_data);
            
            for i = 1:m
                output = input_data(i,:);
                for k = 1:obj.layerCount
                    [obj.layers{k},output] = obj.layers{k}.FProp(output);
                end
                yp(i,:) = output;
            end
        end
        function obj = fitN(obj, X_train, y_train, epochs, alpha)
            [m,~] = size(X_train);                                          % stochastic gradient descent
            for i = 1:epochs
                err = 0;
                for j = 1:m
                    output = X_train(j,:);                                  % FP
                    for k = 1:obj.layerCount
                        [obj.layers{k}, output] = obj.layers{k}.FProp(output);
                    end
                    
                    err = err + obj.loss_function(y_train(j,:), output);
                    
                    error = obj.loss_prime_function(y_train(j,:), output);  % BP
                    for k = obj.layerCount:-1:1
                        [obj.layers{k}, error] = obj.layers{k}.BProp(error, alpha);
                    end
                end
                err = err / m;
                [or,oc] = size(output);
                fprintf('%d/%d\t\terror = %f\t\toutput size  = (%d,%d)\n', i, epochs, err, or, oc)
            end
        end
    end
end