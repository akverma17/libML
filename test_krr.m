function [ypredicted] = test_krr(alpha,Xtr,ytr,Xte,lambda,kernel,gamma)

% calculate the number of training and test samples.
numTrainSamples = size(Xtr,1);
numTestSamples = size(Xte,1);

% ypredicted = (Xte * transpose(Xtr) * alpha) / (m*lambda).
% Xte * transpose(Xtr) is done by kernels sample by sample wise.
% resultant would be a matrix of size (numTestSamples,numTrainSamples)
k = zeros(numTestSamples,numTrainSamples);
if(strcmp(kernel,'linear') == 1)
    for i = 1 : numTestSamples
        for j = 1 : numTrainSamples
            k(i,j) = linearK(Xte(i,:) , Xtr(j,:));
        end
    end
else
    for i = 1 : numTestSamples
        for j = 1 : numTrainSamples
            k(i,j) = gaussianK(Xte(i,:) , Xtr(j,:), gamma);
        end
    end
end

ypredicted = (k * alpha) / (numTrainSamples * lambda);

% If sign of ypredicted(i) is positive, sample is classified as +1 else -1.
for i = 1 : numTestSamples
    if(ypredicted(i,1) > 0)
        ypredicted(i,1) = 1;
    else
        ypredicted(i,1) = -1;
    end
end

% function to calculate guassian kernel of two row vectors x1 and x2.
function [k] = gaussianK(x1,x2,gamma)
k = exp(-gamma*((norm(x1-x2))^2));

% function to calculate linear kernel of two row vectors x1 and x2.
function [k] = linearK(x1,x2)
k = dot(x1,x2);