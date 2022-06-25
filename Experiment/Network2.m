
classdef Network2
    properties
        layers
        nodes
    end
    
    methods
        function this = Network2(layers_arg)
            this.layers = [];
            this.nodes = [];
            [numLayers,~] = size(layers_arg);
            
            for layerID = 1:numLayers
                for nodeID = 1:layers(layerID)
                    new_node = node2();
                    this.nodes = [this.nodes , new_node];
                    this.layers(layerID) = [this.layers(layerID) , new_node];
                    if layerID ~= 1
                        for prev_layer_node = this.layers(layerID-1)
                            new_node.build_in_edge(prev_layer_node);
                        end
                    end
                end
            end
        end
        function this = init_weights(this)
            for node = this.nodes
                for edge = node.in_edge
                    if isempty(edge.weight)
                        edge.weight = 2*rand - 1;
                    end
                end
                for edge = node.out_edge
                    if isempty(edge.weight)
                        edge.weight = 2*rand - 1;
                    end
                end
                if sum(ismember(node,this.layers(1)))==0
                    node.bias = 2*rand - 1;
                end
            end
        end
        function this = BP_prop(this, X, activationFunction)
            for input_layerNodeID = 1:this.layers(1)
                this.layers(1)
            end
            for
                
            end
        end
    end
end









