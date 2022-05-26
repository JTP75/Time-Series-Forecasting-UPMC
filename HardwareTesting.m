%% ======================================================================== GPU TESTING ======================================

Xgpu = gpuArray(XImp);
ygpu = gpuArray(yregImp);

thetaGPU = normalEqn(XImp,yregImp);
