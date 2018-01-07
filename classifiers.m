function classifiers()
% load the data
data = load('synth_data.mat');

xTrain = data.Xtrain;
yTrain = data.ytrain;

xTest = data.Xtest;
yTest = data.ytest; 

% calculate the w, bias for each model and calculate its accuracy
[wSvm1 , biasSvm1] = train_svm_primal(xTrain , yTrain , 0.01);
accuracySvm1 = calculateAccuracy(wSvm1 , biasSvm1 , xTest , yTest);

[wSvm2 , biasSvm2] = train_svm_primal(xTrain , yTrain , 100.0);
accuracySvm2 = calculateAccuracy(wSvm2 , biasSvm2 , xTest , yTest);

[wPerc , biasPerc] = train_perceptron(xTrain , yTrain);
accuracyPerc = calculateAccuracy(wPerc , biasPerc , xTest , yTest);

lambda = 1;
[wRR , biasRR] = train_rr(xTrain , yTrain , lambda);
accuracyRR = calculateAccuracy(wRR , biasRR , xTest , yTest);
 
% make array of positive and negative samples
xPos = [];
xNeg = [];
[numTrainingSamples , ~] = size(xTrain);
for i = 1 : numTrainingSamples
    if (yTrain(i) == 1)
        xPos(end + 1 , :) = xTrain(i , :);
    else
        xNeg(end + 1 , :) = xTrain(i , :);
    end    
end

% plot the points and the hyperplanes corresponding to each model.
figure(1)
p = scatter(xPos(:,1),xPos(:,2),10,'+','MarkerEdgeColor',[0.5 0 0],...
              'MarkerFaceColor',[0.7 0 0],...
              'LineWidth',1.5);

hold on;
p = scatter(xNeg(:,1),xNeg(:,2),10,'o','MarkerEdgeColor',[0 0.5 0],...
              'MarkerFaceColor',[0 0.7 0],...
              'LineWidth',1.5);

plot(wRR , biasRR , [69/255 140/255 1]);
plot(wSvm1 , biasSvm1 , [239/255 80/255 62/255 ])
plot(wSvm2 , biasSvm2 , [47/255 220/255 35/255 ])
plot(wPerc , biasPerc , [53/255 53/255 53/255 ])

% print accuracies of each model
disp(accuracyRR*100);
disp(accuracySvm1*100);
disp(accuracySvm2*100);
disp(accuracyPerc*100);
legend('Positive Sample' , 'Negative Sample ' , 'Ridge' , 'SVM-0.01' , 'SVM-100' , 'Perceptron' )
     
% function to plot the lines with specific color
function plot(w , bias , color)
a = w(1,1);
b = w(2,1);
fplot( @(x) -(a/b)*x - bias/b ,'Color' , color,'LineWidth',3);

% function to calculate the accuracy of each model.
function [accuracy] = calculateAccuracy(w , bias , xTest , yTest)
[numTestSamples , d] = size(xTest);
xTest_new = [xTest ones(numTestSamples , 1)];
w = [w;bias];
yPredict = xTest_new * w;
% if the sign of predicted value and true label is same, increase the count
correct = 0;
for i = 1:numTestSamples
    if sign(yPredict(i,1)) == sign(yTest(i,1))
        correct = correct + 1;
    end
end
accuracy = correct / numTestSamples;