function yPred = test_mhinge_kernel_sgd(alpha , Xsv , Xte , p)

numTestSamples = size(Xte,2);

yPred = zeros(1 , numTestSamples);

for i = 1:numTestSamples
    
    % take each test sample and predict its class using Xsv, alpha.

    X = Xte( : , i);

    pred = predict(alpha , Xsv , X , p);

    yPred(i) = pred;

end

function pred = predict(alpha , Xsv , X , p)

%compute the kernel of the current sample with all previously updated
%samples and repeat it 10 times to efficiently multiply later.

k = power(transpose(Xsv) * X ,p);
 
k = repmat(k , 1 , 10);

%calculate the element wise product of alpha and kernel for each class, sum
%it across all rows and take the max sum class as the predicted class. 

y = alpha .* k;

y = sum(y , 1);

[maxVal , index] = max(y);

pred = index;
