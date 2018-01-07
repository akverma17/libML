function [alpha, Xsv] = train_mhinge_kernel_sgd(Xtr,ytr, Delta, p)
    
    numSamples = size(Xtr,2);
    k = 10;
    alpha = [];
    
    Xsv = [];
    
    for i = 1:numSamples
        eta = 1 / sqrt(i);
        
        % calculate the loss for each class
        
        loss = calculate_loss(alpha, Xtr, ytr, Delta, i, Xsv,p);
        
        % calculate the class which gives the max loss
        
        [val, idx] = max(loss);
        
        % if the max loss class is not equal to the current class, add the
        % current sample to list of update samples and add new alpha row to
        % existing alpha matrix by adding -eta value to max loss class
        % while rest classes remain 0
        
        if idx ~= ytr(i)
            Xsv = [Xsv Xtr(:,i)];
            new_alpha = zeros(1,k);
            new_alpha(idx) = -eta;
            alpha = [alpha ; new_alpha];
        end
    end
    
end
    
function [loss] = calculate_loss(alpha, Xtr, ytr, Delta, t, Xsv,p)
    
actualY = ytr(t);
loss = zeros(10,1);

% for the first round return

if (t == 1)
    return;
end

%compute the kernel of the current sample with all previously updated
%samples and repeat it 10 times to efficiently multiply later.

k = power(transpose(Xsv) * Xtr(:,t),p);

k = repmat(k , 1 , 10);

% for each class compute loss.

% for any class i loss first compute alpha_i(1:t-1,y`) - alpha_y(1:t-1,y)

diff_alpha = alpha - repmat(alpha(:,actualY),1,10);

% multiply diff_alpha with kernel element wise to calculate product.

product = diff_alpha .* k;

%each column of product holds summation part. Sum it across rows to get
%final sum for each class

loss = sum(product , 1);

% calculate the delta value for the current class

deltaVal  = Delta(currentY , :);

% add it to the loss to get final loss

loss = loss + deltaVal;

end
            