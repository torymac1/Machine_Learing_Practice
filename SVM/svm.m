function [alpha, objective, w, b] = svm(X,Y,C)

K = X'*X;
[alpha, fval] = dual_svm(K, Y, C);
objective = -fval;
w=X*diag(Y)*alpha;
num_of_sv = 0;
b=0;
for index=1:size(alpha, 1)
    if(alpha(index)>0.0001)
        num_of_sv = num_of_sv+1;
        b=b+Y(index)-K(index,:)*diag(Y)*alpha;
    end
end

b = b/num_of_sv;