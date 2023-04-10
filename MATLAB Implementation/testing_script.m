%% load data set and construct interface
load('data/PUH_NEDOC_imputed_set.mat');
IFC = NedocInterface(TI_full);
IFC.compile();

%% load RNN_001
IFC.loadNet("RNN",1);
IFC.response();
IFC.plot("DateTimeRange",["12/1/2021","12/16/2021"]);

%% assessing loss fcn
IFC.setLossFcn(@nedocLoss_01);














