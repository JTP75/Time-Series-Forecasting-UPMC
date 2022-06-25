classdef FCLayer < Layer                                                    % fully connected layer
    
    properties
        weights
        bias
    end
    
    methods
        function obj = FCLayer(input_size, output_size)
            obj@Layer();
            obj.weights = rand([input_size,output_size])-0.5;
            obj.bias = zeros([1,output_size]);
        end
        function [obj, out] = FProp(obj, input_data)
            obj.input = input_data;
            obj.output = (obj.input)*(obj.weights) + obj.bias;
            out = obj.output;
        end
        function [obj, input_error] = BProp(obj, output_error, alpha)
            input_error = (output_error) * (obj.weights)';
            weights_error = (obj.input)' * (output_error);
            
            obj.weights = obj.weights - alpha * weights_error;
            obj.bias = obj.bias - alpha * output_error;
        end
    end
end










