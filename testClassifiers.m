function testClassifiers()

data = load('synth_data.mat');

xTrain = data.Xtrain;
yTrain = data.ytrain;

xTest = data.Xtest;
yTest = data.ytest;

[wRidge , biasRidge,accuracyRidge] = trainRidge(xTrain , yTrain,xTest,yTest);

[wSvm1 , biasSvm1 , accuracySvm1] = trainAndTestSVM(xTrain , yTrain ,...
 xTest , yTest , 0.01);

[wSvm2 , biasSvm2 , accuracySvm2] = trainAndTestSVM(xTrain , yTrain ,...
 xTest , yTest , 100);

[wPerc , biasPerc , accuracyPerc] = trainAndTestPerceptron(xTrain , yTrain,xTest,yTest);
 
 
xPositives = [];
xNegatives = [];
[numTrainingSamples , ~] = size(xTrain);
for i = 1 : numTrainingSamples
    if (yTrain(i) == 1)
        xPositives(end + 1 , :) = xTrain(i , :);
    else
        xNegatives(end + 1 , :) = xTrain(i , :);
    end
    
end
figure(1)
s = scatter(xPositives(:,1),xPositives(:,2),10,'+','MarkerEdgeColor',[0 .5 0],...
              'MarkerFaceColor',[0 .7 0],...
              'LineWidth',1.5);

hold on;
s = scatter(xNegatives(:,1),xNegatives(:,2),10,'o','MarkerEdgeColor',[0.5 0 0],...
              'MarkerFaceColor',[0.7 0 0],...
              'LineWidth',1.5);

plot(wRidge , biasRidge , [66/255 134/255 1]);
plot(wSvm1 , biasSvm1 , [244/255 75/255 66/255 ])
plot(wSvm2 , biasSvm2 , [45/255 229/255 30/255 ])
plot(wPerc , biasPerc , [50/255 50/255 50/255 ])
          
disp(accuracyRidge);
disp(accuracySvm1 );
disp(accuracySvm2);
disp(accuracyPerc);
legend('Postive Sample' , 'Nagative Sample ' , 'Ridge' , 'SVM-0.01' , 'SVM-100' , 'Perceptron' )
          
function plot(w , bias , color)
a = w(1,1);
b = w(2 , 1);

fplot( @(x) -(a/b)*x - bias/b ,'Color' , color,'LineWidth',3);

function [wSvm , biasSvm ,accuracySvm ]= trainAndTestSVM(xTrain , yTrain , xTest ,yTest , C)
[wSvm , biasSvm] = train_svm_primal(xTrain , yTrain , C);

accuracySvm = testModel(wSvm , biasSvm , xTest , yTest);

function [wPerc , biasPerc , accuracyPerc] = trainAndTestPerceptron(xTrain , yTrain , xTest ,yTest)
[wPerc , biasPerc] = train_perceptron(xTrain , yTrain);
accuracyPerc = testModel(wPerc , biasPerc , xTest , yTest);

function [wRidge , biasRidge,accuracy] = trainRidge(xTrain , yTrain , xTest ,yTest)
lambda = 1;
[wRidge , biasRidge] = train_rr(xTrain , yTrain , lambda);

%test the model
accuracy = testModel(wRidge , biasRidge , xTest , yTest);



function [accuracy] = testModel(w , bias , xTest , yTest)

%test the model
[numTestSamples , dimention] = size(xTest);
xTestAppended = horzcat( xTest , ones(numTestSamples , 1) );

wAppended = [w;bias];
yPredicted = xTestAppended * wAppended;

correct = 0;
for i = 1:numTestSamples
    if ( (yPredicted(i) <= 0 && yTest(i) == -1) || ( yPredicted(i) > 0 && yTest(i) == 1 ) )
        correct = correct + 1;
    end
end

accuracy = correct / numTestSamples;

