%4.3.1
clear all;
N=250; 
d=80; 
k=10; 
sigma=10; %1 or 10
X=randn(d,N);
w=[10*sign(randn(k,1));zeros(d-k,1)];
b=0;
Y=X'*w+b+sigma*randn(N,1);
converge=0.005;
lambda_max = 2*norm((X*(Y-(sum(Y)/N))),Inf);
lambda=lambda_max;
result_w{10} = [];
result_lambda{10} = [];
for i=1:10
    result_lambda{i}=lambda;
        [w_res,b_res,obj_res] = my_lasso(X,Y, w, b ,lambda, converge);
    %result_w(:,:,i)=w_res;
    %result_b(i)=b_res;
    result_w{i} = w_res;
    lambda = lambda / 2;
end

precision=cellfun(@(x)sum(sign(x(x~=0))==sign(w(x~=0)))/sum(x~=0), result_w);
recall=cellfun(@(x)sum(sign(x(x~=0))==sign(w(x~=0)))/k, result_w);
plot_code_4_3([precision;recall]');

