% function for plotting training and test error
function plotting()
    data = load('adult.mat');

    xTrain = data.Xtr;
    yTrain = data.ytr;

    xTest = data.Xte;
    yTest = data.yte;
    
    % calculate w values of all the iterations
    
    [w] = train_svm_sgd(transpose(xTrain),transpose(yTrain),0.1,1,10000);
    
    % calculate training and test 0-1 error and plot it
    
    trainError = calculateError(w,transpose(xTrain),transpose(yTrain));
    testError = calculateError(w,transpose(xTest),transpose(yTest));
    disp('trainError : ');
    disp(trainError((size(trainError,1)),1));
    disp('testError : ');
    disp(testError((size(testError,1)),1));
    
    numIter = zeros(size(w,1),1);
    for i=1:size(w,1)
        numIter(i,1) = i;
    end
    
    figure(2)
    p2 = scatter(numIter(:,1),trainError(:,1),10,'+','MarkerEdgeColor',[0.5 0 0],...
              'MarkerFaceColor',[0.7 0 0],...
              'LineWidth',1.5);
    figure(3)
    p3 = scatter(numIter(:,1),testError(:,1),10,'+','MarkerEdgeColor',[0.5 0 0],...
              'MarkerFaceColor',[0.7 0 0],...
              'LineWidth',1.5);
 
% function to calculate 0-1 error

function [error] = calculateError(w,x,y)
    numSamples = size(x,1);
    numIter = size(w,1);
    error = zeros(numIter,1);
    for j = 1:numIter
        for i=1:numSamples
            if (y(i,1)* dot(w(j,:),x(i,:))) < 0
                error(j,1) = error(j,1) + 1;
            end
        end
    end
    
    
    