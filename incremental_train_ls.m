function [w] = incremental_train_ls(X,y)
% matrix A = transpose(X) * X
X_new = transpose(X) * X;
[numRows,numCols] = size(X_new);
% initial inv(A) = inv(A') where A' is a numCols * numCols matrix formed
% from A where numCols is the no. of columns in original A.
tempRow = inv(X_new(1:numCols,1:numCols));
% from row = numCols+1 onwards, inverse is updated according to Sherman
% Morrison formula
for i = numCols+1:numRows
    v = X_new(i,:);
    tempRow = tempRow - (tempRow * (v * transpose(v)) * tempRow) ./ (1 + transpose(v)*tempRow*v)
end
% at last, w = inv(transpose(X) * X) 
w = tempRow * transpose(X) * y;