classdef (Abstract) Layer                                                              % abstract base class
    
    properties
        input
        output
    end
    
    methods (Abstract)
        FProp(obj, input)
        BProp(obj, output_error, alpha)
    end
    
    methods
        function obj = Layer()
            obj.input = [];
            obj.output = [];
        end
    end
    
end