function [M] = kmeans_fast(X,k,T)
    
    % calculating num of samples and dimensions of X
    
    [numSamples, numDim] = size(X);
    
    % initialize first k points as k centres
    
    centres = X(1:k,:);
    
    % another matrix similar to centres to check whether points have
    % finished changing their positions.
    
    c = zeros(k,numDim);
    
    % store the no. of points in each cluster
    
    c_num = zeros(k,1);
    for i = 1:T
        
        % calculate the distance of each point from the k centres and store
        % it in matrix D. Each row of D corresponds to a point and each
        % column stores its distance to k centres.
        
        D = pdist2(X,centres);
        
        % now calculate the nearest cluster for each point and update that
        % cluster centre in the dummy centre matrix c
        
        for j = 1:length(D)
            [~,cluster] = min(D(j,:));
            c(cluster,:) = (c(cluster,:)*c_num(cluster,1) + X(j,:))/(c_num(cluster,1)+1);
            c_num(cluster,1) = c_num(cluster,1) + 1;
        end
        
        % check if centres are changing. If yes then stop else assign
        % centres to this new matrix c and continue
        if isequal(c,centres) == 1
            break;
        else
            centres = c;
        end
    end
    
    % return centres
    
    M = centres;
    disp(i);
end