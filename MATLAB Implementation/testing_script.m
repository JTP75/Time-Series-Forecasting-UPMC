load('data/PUH_NEDOC_imputed_set.mat');
load('mdl/RNN_001/architecture.mat');
load('mdl/RNN_001/options.mat');
load('mdl/RNN_001/network.mat');

IFC = NedocInterface(TI_full,48);
IFC.setLossFcn(@nedocLoss_01);
IFC.compile(layers,opts);

IFC.network = net;
IFC.response();

















