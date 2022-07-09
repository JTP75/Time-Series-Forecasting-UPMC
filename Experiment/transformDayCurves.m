function [transformMatrix, y_trans] = transformDayCurves(ds)

[~,yimp] = ds.getmats('all','time');
ppd = ds.PPD;

x = (1:ppd)';
y = zeros([ds.L/ppd,ppd]);
for i = 1:ppd:length(yimp)-(ppd-1)
    y((i+(ppd-1))/ppd,:) = yimp(i:i+(ppd-1));
end

% fits
transformMatrix = [x.^0,...
    sin(x*2*pi/ppd), cos(x*2*pi/ppd),...
    sin(x*2*pi/(ppd/2)), cos(x*2*pi/(ppd/2)),...
    sin(x*2*pi/(ppd/4)), cos(x*2*pi/(ppd/4)),...
    sin(x*2*pi/(ppd/8)), cos(x*2*pi/(ppd/8))     ];
y_trans = zeros([size(y,1),size(transformMatrix,2)]);

for i = 1:size(y,1)
    y_trans(i,:) = normalEqn(transformMatrix,y(i,:)');
end

