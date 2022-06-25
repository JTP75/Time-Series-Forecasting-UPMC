
classdef node2
    properties
        value
        delta
        in_edge
        out_edge
        bias
    end
    
    methods
        function this = node2()
            this.value = [];
            this.delta = [];
            this.in_edge = [];
            this.out_edge =[];
            this.bias = [];
        end
        function this = build_in_edge(this, linked_node)
            new_edge = edge2();
            new_edge.from_node = this;
            new_edge.to_node = linked_node;
            this.in_edge = [this.in_edge , new_edge];
            linked_node.out_edge  = [linked_node.out_edge , new_edge];
        end
        function this = build_out_edge(this, linked_node)
            new_edge = edge2();
            new_edge.from_node = linked_node;
            new_edge.to_node = this;
            this.out_edge = [this.out_edge , new_edge];
            linked_node.in_edge  = [linked_node.in_edge , new_edge];
        end
    end
end







