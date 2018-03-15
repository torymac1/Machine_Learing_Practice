
clear;
% function [w,b,obj]=multiclass_svm1(X,Y,C,maxEpoch, eta_0,eta_1)

load('hw2data/q3_2_data');
X = trD;
Y = trLb;
C=160;
maxEpoch=5000;
eta_0 = 20;
eta_1 = 300;
d=size(X,1);
n=size(X,2);
w=zeros(d,10);
obj_all = zeros(maxEpoch,1);
% obj=0.5*(w'*w)+C*sum(max(1-Y.*(X'*w+b),0));
%%
for epoch=1:maxEpoch
    epoch
    eta=eta_0/(eta_1+epoch);
    permute=randperm(n);
    for p = 1:n
%         obj =  1/(2*n)*(w(:,1)'*w(:,1)+w(:,2)'*w(:,2))+C*()
        i = permute(p);
%         w_tmp = w;
        j = Y(i);
        w_tmp = w;
        w_tmp(:,j) = [];
        t = max(w_tmp'*X(:,i))-w(:,j)'*X(:,i)+1;
        if(t<=0)
            w(:,j) = w(:,j) - 1/n*w(:,j)*eta;
        else
            w(:,j) = w(:,j) - (1/n*w(:,j) - X(:,i)*C)*eta;
        end
 
        obj_all(epoch) = obj_all(epoch) + 1/(2*n)*(w(:,1)'*w(:,1)+w(:,2)'*w(:,2))+C*(max(t,0));
        
    end
    
end

%%
[~,svm_result] = max(valD'*w,[],2);

accuracy = sum(abs(svm_result - valLb)<0.1)/2120;

plot(1:1:maxEpoch,obj_all,'.-');
title('Objective Value');
xlabel('Epoch');ylabel('Objective value');
