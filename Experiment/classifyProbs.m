function y = classifyProbs(ybin)
[~,y] = max(ybin,[],2);