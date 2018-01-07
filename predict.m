% load data
load cadata;
% make a 1-d vector of size 5 to store the errors of 5 LR.
x_train = zeros(1,5);
x_test = zeros(1,5);
% make a column of 1's.
[numSamples, numCol] = size(Xtrain);
col = ones(numSamples,1);

% first normalize all the data so that it lies between -1 and 1.
[Xtrain_norm, Xtest_norm] = normalizeAll(Xtrain, Xtest);
%[ytrain, ytest] = normalize(ytrain, ytest);

% for each k value, generate the poly features of both training and testing
% data and then calculate the error (y - (w_0 + wx))^2 for each sample and
% the sum the errors.
for i = 1:5
    Xtrain_poly = generate_poly_features(Xtrain_norm,i);
    [~,numCol] = size(Xtrain_poly);
    [w,w_0] = train_ls(Xtrain_poly,ytrain,1);
    X_temp = [Xtrain_poly col];
    w(numCol + 1) = w_0; 
    predictY = X_temp*w;
    loss = ytrain-predictY;
    loss = power(loss,2);
    disp(loss);
    x_train(1,i) = sum(loss(:));
    disp(x_train(1,i));
end    
[numSamples, numCol] = size(Xtest);
col = ones(numSamples,1);
for i = 1:5    
    Xtest_poly = generate_poly_features(Xtest_norm,i);
    [~,numCol] = size(Xtest_poly);
    [w,w_0] = train_ls(Xtest_poly,ytest,1);
    X_temp = [Xtest_poly col];
    w(numCol + 1) = w_0; 
    predictY = X_temp*w;
    loss = ytest-predictY;
    loss = power(loss,2);
    x_test(1,i) = sum(loss(:));
end
disp(x_train)
disp(x_test)
% plot both the errors together. Green line is the training error plot and
% red line is the test error plot.
xAxis = [1 2 3 4 5];
plot(xAxis , x_train,'-go',...
'LineWidth',2,...
'MarkerSize',10,...
'MarkerEdgeColor','b');
hold on;
plot(xAxis , x_test,'-ro',...
'LineWidth',2,...
'MarkerSize',10,...
'MarkerEdgeColor','b');