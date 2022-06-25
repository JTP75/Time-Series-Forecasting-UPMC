classdef edge2
    properties
        weight
        from_node
        to_node
    end
    
    methods
        function this = edge2()
            this.weight = [];
            this.from_node = [];
            this.to_node = [];
        end
    end
end