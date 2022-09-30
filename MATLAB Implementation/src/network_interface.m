classdef network_interface < handle
    properties(Access=protected)
        % network IO (same lengths, format, etc.)
        Xr      % input struct
        Yr      % output struct
        Yrp     % response struct
        
        % network objects
        architecture        % specifies layers
        options             % specifies options
        network             % network object (filled on call to train_net())
        performance_info    % performance info
    end
    methods(Access=public)
        function obj = network_interface()
            obj.Xr = DataStruct();
            obj.Yr = DataStruct();
            obj.Yrp = DataStruct();
        end
        function obj = train_net(obj)
            if isempty(obj.Xr) || isempty(obj.Yr)...
                    || isempty(obj.architecture) || isempty(obj.options)
                desc = "Class attributes Xr, Yr, architecture, and options"...
                    + " must be set prior to obj.train_net() call.";
                id = "NetworkInterface:ClsAttrsNotSet";
                throw(MException(id,desc));
            end
            [obj.network, obj.performance_info] = trainNetwork(...
                obj.Xr.train,...
                obj.Yr.train,...
                obj.architecture,...
                obj.options);
        end
        function response(obj,varargin)
            obj.Yrp.all = predict(obj.network,obj.Xr.all,varargin);
            obj.Yrp.train = predict(obj.network,obj.Xr.train,varargin);
            obj.Yrp.valid = predict(obj.network,obj.Xr.valid,varargin);
            obj.Yrp.test = predict(obj.network,obj.Xr.test,varargin);
            obj.postprocess;
        end
    end
    methods(Abstract,Access=public)
        obj = compile(obj,arch,opts,varargin)
    end
    methods(Abstract,Access=protected)
        obj = preprocess(obj,varargin)
        obj = postprocess(obj,varargin)
    end
end