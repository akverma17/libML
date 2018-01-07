function test()
data = load('adult.mat');

xTrain = data.Xtr;
%xTrain = xTrain(1:10);
yTrain = data.ytr;
%yTrain = yTrain(1:10);

xTest = data.Xte;
yTest = data.yte; 

[alpha] = train_krr(transpose(xTrain), transpose(yTrain), 2e-05, 'gaussian', 0.001);
[ypredicted] = test_krr(alpha,transpose(xTrain),transpose(yTrain),transpose(xTest),2e-05,'gaussian',0.001);
%disp(ypredicted);
disp('test accuracy: ');
disp(calculateAccuracy(ypredicted, transpose(yTest)));

[ypredicted] = test_krr(alpha,transpose(xTrain),transpose(yTrain),transpose(xTrain),2e-05,'gaussian',0.001);
disp('train accuracy: ');
disp(calculateAccuracy(ypredicted, transpose(yTrain)));

% function to calculate the accuracy of each model.
function [accuracy] = calculateAccuracy(ypredicted , yTest)
numTestSamples = size(yTest,1);
% if the sign of predicted value and true label is same, increase the count
correct = 0;
for i = 1:numTestSamples
    if sign(ypredicted(i,1)) == sign(yTest(i,1))
        correct = correct + 1;
    end
end
accuracy = correct / numTestSamples;