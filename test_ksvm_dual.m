function [ypredicted] = test_ksvm_dual(alpha,Xtr,ytr,Xte,kernel,gamma)

% calculate the number of training and test samples.
numTrainSamples = size(Xtr,1);
numTestSamples = size(Xte,1);

% ypredicted = alpha * y * kernel(Xtr,Xte)
% kernel is computed sample by sample wise.
% kernel would be a matrix of size (numTestSamples,numTrainSamples)

ypredicted = zeros(numTestSamples,1);
if(strcmp(kernel,'linear') == 1)
    for i = 1 : numTestSamples
        for j = 1 : numTrainSamples
            ypredicted(i,1) = ypredicted(i,1) + (alpha(j,1)*ytr(j,1)*linearK(Xtr(j,:) , Xte(i,:)));
        end
        if(ypredicted(i,1) > 0)
            ypredicted(i,1) = 1;
        else
            ypredicted(i,1) = -1;
        end
    end
else
    for i = 1 : numTestSamples
        for j = 1 : numTrainSamples
            ypredicted(i,1) = ypredicted(i,1) + (alpha(j,1)*ytr(j,1)*gaussianK(Xtr(j,:) , Xte(i,:), gamma));
        end
        if(ypredicted(i,1) > 0)
            ypredicted(i,1) = 1;
        else
            ypredicted(i,1) = -1;
        end
    end
end

% function to calculate guassian kernel of two row vectors x1 and x2.
function [k] = gaussianK(x1,x2,gamma)
k = exp(-gamma*((norm(x1-x2))^2));

% function to calculate linear kernel of two row vectors x1 and x2.
function [k] = linearK(x1,x2)
k = dot(x1,x2);