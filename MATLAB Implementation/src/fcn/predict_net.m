function yp = predict_net(net,Xr,centers,transforms)

p0 = predict( net, Xr, "ExecutionEnvironment",'gpu', "MiniBatchSize",64 );
p1 = cast(p0,"double");
p2 = p1 * transforms.PCAR';
p3 = centers.sigR .* p2 + centers.muR;
p4 = [zeros([transforms.lag,size(p3,2)]) ; p3];
p5 = reshape(p4', [], 1);
yp = p5;