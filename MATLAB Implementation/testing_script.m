load('data/PUH_NEDOC_imputed_set.mat');
load('mdl/RNN_001/architecture.mat');
load('mdl/RNN_001/options.mat');
load('mdl/RNN_001/network.mat');

ptor = NedocInterface(TI_full,48);
ptor.setLossFcn(@nedocLoss_01);
ptor.compile(layers,opts);

ptor.network = net;
ptor.response;

ptor.assess;
















