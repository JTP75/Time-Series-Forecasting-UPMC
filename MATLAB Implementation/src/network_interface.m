classdef network_interface < handle
    properties(Access=public)
        % network IO (same lengths, format, etc.)
        Xr      % input struct
        Yr      % output struct
        Yrp     % response struct
        
        % network objects
        architecture        % specifies layers
        options             % specifies options
        network             % network object (filled on call to train_net())
        performance         % performance info (typically a struct of some variety)
        
        % function handles
        lossFcn
    end
    methods(Access=public)
        function obj = network_interface()
            obj.Xr = DataStruct();
            obj.Yr = DataStruct();
            obj.Yrp = DataStruct();
        end
        function obj = setLossFcn(obj,lf)
            if ~isa(lf,'function_handle')
                id = "network_interface:InvalidArgType";
                desc = "Arg should be of type function_handle.";
                throw(MException(id,desc))
            end
            obj.lossFcn = lf;
        end
        function obj = train_net(obj)
            if isempty(obj.Xr) || isempty(obj.Yr)...
                    || isempty(obj.architecture) || isempty(obj.options)
                desc = "Class attributes Xr, Yr, architecture, and options"...
                    + " must be set prior to obj.train_net() call.";
                id = "NetworkInterface:ClsAttrsNotSet";
                throw(MException(id,desc));
            end
            obj.network = trainNetwork(...
                obj.Xr.train,...
                obj.Yr.train,...
                obj.architecture,...
                obj.options);
        end
        function response(obj,varargin)
            if isempty(varargin)
                varargin = {"MiniBatchSize",64};
            end
            fprintf("Executing Forward Propagation Routine...\t")
            obj.Yrp.all = predict(obj.network,obj.Xr.all,varargin{:});
            obj.Yrp.train = predict(obj.network,obj.Xr.train,varargin{:});
            if ~isempty(obj.Xr.valid)
                obj.Yrp.valid = predict(obj.network,obj.Xr.valid,varargin{:});
            else
                obj.Yrp.valid = [];
            end
            obj.Yrp.test = predict(obj.network,obj.Xr.test,varargin{:});
            fprintf("Done!\n\n")
            obj.postprocess;
        end
    end
    methods(Abstract,Access=public)
        obj = compile(obj,arch,opts,varargin)
        fig = plot(obj,varargin)
        obj = assess(obj)
    end
    methods(Abstract,Access=protected)
        obj = preprocess(obj,varargin)
        obj = postprocess(obj,varargin)
    end
end