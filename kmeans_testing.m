data = load('mnist.mat');

xTrain = data.Xtr;
yTrain = data.ytr;

xTest = data.Xte;
yTest = data.yte;

plot_x = [2,4,6,8,10,12,14];
plot_error = zeros(1,7);
plot_purity = zeros(1,7);

for j = 1:7
    M = kmeans_fast(xTrain,plot_x(j),100);
    D = pdist2(xTrain,M);
    cluster_labels = zeros(length(M),10);
    sum = 0;
    purity = 0;
    for i = 1:length(D)
        [val,id] = min(D(i,:));
        cluster_labels(id,yTrain(i)+1) = cluster_labels(id,yTrain(i)+1) + 1;
        sum = sum+power(val,2);
    end
    disp(sum);
    for i = 1:length(cluster_labels)
        [val,id] = max(cluster_labels(i,:));
        purity = purity + val;
    end
    plot_error(1,j) = sum;
    plot_purity(1,j) = purity / length(D);
end

figure(1)
plot(plot_x,plot_error,'b-',plot_x,plot_error,'rx');

figure(2)
plot(plot_x,plot_purity,'b-',plot_x,plot_purity,'rx');