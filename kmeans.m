function [M] = kmeans(X,k,T)
    [numSamples, numDim] = size(X);
    centres = X(1:k,:);
    c = zeros(k,numDim);
    c_num = zeros(k,1);
    for i = 1:T
        for m = 1:numSamples
            min = norm(X(m,:)-centres(1,:));
            cluster = 1;
            for j = 2:k
                d = norm(X(m,:)-centres(j,:));
                if d < min
                    min = d;
                    cluster = j;
                end
            end
            c(cluster,:) = (c(cluster,:)*c_num(cluster,1) + X(m,:))/(c_num(cluster,1)+1);
            c_num(cluster,1) = c_num(cluster,1) + 1;
        end
        if isequal(c,centres) == 1
            break;
        else
            centres = c;
        end
        disp(i);
    end
    disp(i);
    M = centres;
end