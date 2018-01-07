function plotMean()
% samples are generated from standard normal distribution in powers of 10.
% Maximum power considered is 8.
maxPower = 8;
% creating 1-d vector of X and Y coordinates with size = maxPower.
X = zeros(1,maxPower);
Y = zeros(1,maxPower);
for i = 1:maxPower
    % no. of samples = 10^i
    numSamples = power(10,i)
    % first randomly generate the samples and calculate its mean by inbuilt 
    % mean function
    m = mean(randn(1,numSamples));
    % x coordinate will be the ith index and y coordinate the mean.
    X(1,i) = i;
    Y(1,i) = m;
end
% plot the graph.
ax = subplot(1,1,1);
ylim(ax,[-0.2 0.2]);
scatter(ax,X,Y);
refline(ax,[0,0]);