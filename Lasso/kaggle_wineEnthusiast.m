%%
%train
D1 = load('kaggle_wine/trainData.txt');
trX = sparse(D1(:,2), D1(:,1), D1(:,3));
trLb = load('kaggle_wine/trainLabels.txt');
D2=load('kaggle_wine/valData.txt');
valX=sparse(D2(:,2),D2(:,1),D2(:,3));
valLb=load('kaggle_wine/valLabels.txt');

X = trX;
Y = trLb;
N=10000; 
d=3000; 
k=1000; 
sigma=10; %1 or 10
w=zeros(d,1);
b=0;
converge=0.005;
lambda_max = 2*norm((X*(Y-(sum(Y)/N))),Inf);
lambda=lambda_max;
steps = 14;
result_w{steps} = [];
result_b{steps} = [];
result_lambda{steps} = [];
for i=1:steps
    result_lambda{i}=lambda;
    if(i>1)
        [w_res,b_res,obj_res] = my_lasso(X,Y, result_w{i-1}, result_b{i-1} ,lambda, converge);
    else
        [w_res,b_res,obj_res] = my_lasso(X,Y, w, b ,lambda, converge);
    end
    result_b{i}=b_res;
    result_w{i} = w_res;
    result_lambda{i} = lambda;
    lambda = lambda / 2;
    
end
%%
%figure
lambdas = cell2mat(result_lambda);
nonzeros = cellfun(@(x)sum(x~=0), result_w);
RMSE_tr = cellfun(@(x,y)sqrt(mean((trX'*x+y-trLb).^2)), result_w,result_b);
RMSE_val = cellfun(@(x,y)sqrt(mean((valX'*x+y-valLb).^2)),result_w,result_b);
plot_code_4_4_1_1([RMSE_tr; RMSE_val]');
plot_code_4_4_1_2(nonzeros);
%%
%top feature
%feature = load('kaggle_wine/featureTypes.txt');
w_best = result_w{10};
w_best1 = w_best; 
w_best1(find(w_best1==0))=[];
[x1,y1] = sort(abs(w_best),'descend');
[x2,y2] = sort(abs(w_best1));
% D2=load('kaggle_wine/valData.txt');
% valX=sparse(D2(:,2),D2(:,1),D2(:,3));