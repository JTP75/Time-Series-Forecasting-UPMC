
classdef ActivationLayer < Layer
    properties
        activation_function
        activation_function_prime
    end
    methods
        function obj = ActivationLayer(a_func, a_func_prime)
            obj@Layer();
            obj.activation_function = a_func;
            obj.activation_function_prime = a_func_prime;
        end
        function [obj, out] = FProp(obj, input_data)
            obj.input = input_data;
            obj.output = obj.activation_function(obj.input);
            out = obj.output;
        end
        function [obj, a_prime] = BProp(obj, output_error, ~)
            a_prime = obj.activation_function_prime(obj.input) .* output_error;
        end
    end
end