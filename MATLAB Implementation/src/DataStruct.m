classdef DataStruct < handle
    properties
        all
        train
        valid
        test
    end
    methods
        function obj = DataStruct(all,train,valid,test)
            if nargin > 0
                obj.all = all;
            end
            if nargin > 1
                obj.train = train;
            end
            if nargin > 2
                obj.valid = valid;
            end
            if nargin > 3
                obj.test = test;
            end
        end
        function obj = setsplits(obj,trvl,vlts)
            obj.train = obj.all(1:trvl,:);
            obj.valid = obj.all(trvl+1:vlts,:);
            obj.test = obj.all(vlts+1:end,:);
        end
    end
end