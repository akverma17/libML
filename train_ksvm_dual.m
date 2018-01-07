function [alpha] = train_ksvm_dual(X,y,C,kernel,gamma)

% calculate no. of samples as row and dimensions as col.

[row, col] = size(X);

% kernels are computed sample wise.
% resultant would be a matrix of size (row,row)

k = zeros(row,row);
if(strcmp(kernel,'linear') == 1)
    for i = 1 : row
        for j = 1 : row
            k(i,j) = linearK(X(i,:) , X(j,:));
        end
    end
else
    for i = 1 : row
        for j = 1 : row
            k(i,j) = gaussianK(X(i,:) , X(j,:), gamma);
        end
    end
end

% H matrix would be k(x(i),x(j)).* (y(i) * y(j))
% y(i) * y(j) can be computed by (y * transpose(y))
% k(x(i),x(j)) is calculated and stored in matrix k.

H = k .* (y * transpose(y));

% f is a column vector with values -1.

f = -ones(row,1);

% taking transpose of f to multiply with alpha

f = transpose(f);

% lower bound for alpha is 0

lb = zeros(row,1);

% upper bound for alpha is C

ub = C*ones(row,1);

% Aeq, Beq, A ,b are null

[alpha,obj] = quadprog(H,f,[],[],[],[],lb,ub);
disp('objective:');
disp(obj);

% adding 1 to number of support vectors if alpha(i) > 0.001
numSupportVectors = 0;
for i = 1 : row
    if alpha(i,1) > 0.001
        numSupportVectors = numSupportVectors + 1;
    end
end
disp('num: ');
disp(numSupportVectors);

% function to calculate guassian kernel of two row vectors x1 and x2.
function [k] = gaussianK(x1,x2,gamma)
k = exp(-gamma*((norm(x1-x2))^2));

% function to calculate linear kernel of two row vectors x1 and x2.
function [k] = linearK(x1,x2)
k = dot(x1,x2);