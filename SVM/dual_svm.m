%Question 3.1.2
function [alpha, fval] = dual_svm(K,Y,C)

n = size(Y,1);
H = diag(Y)'*K*diag(Y); 
H = (H+H')/2;
H(abs(H)<2.3283e-4) = 0;
f = -ones(n,1);
A = [];
b = [];
Aeq = Y';
beq = 0;
lb = zeros(n,1);
ub = C*ones(n,1);


[alpha, fval] = quadprog(H,f,A,b,Aeq,beq,lb,ub);


end