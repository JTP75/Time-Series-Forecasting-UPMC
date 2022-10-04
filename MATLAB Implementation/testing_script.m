%% nav to folder
cd("C:\Users\pacel\Desktop\Work\Pitt\UPMC ED\DataItems\MATLAB Implementation")
addpath(genpath(pwd))

%% load data set and construct interface
load('data/PUH_NEDOC_imputed_set.mat');
IFC = NedocInterface(TI_full,48);

%% load RNN_001
IFC.loadNet("RNN",1);

%% set loss fcn
IFC.setLossFcn(@nedocLoss_01);














