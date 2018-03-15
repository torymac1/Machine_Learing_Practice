function [w,b,obj] = my_lasso(X,Y, w_int, b_int ,lambda, converge)
w = w_int;
b = b_int;

d=size(X,1);
N=size(X,2);

r=Y-(X'*w+b);
obj=r'*r+lambda*sum(abs(w));
a=2*sum(X.^2,2);
stop=0;

while stop==0
    r=Y-(X'*w+b); 
    r_increase=mean(r);
    b=b+r_increase; 
    r=r-r_increase; 
    for k=1:d
        c_k=w(k)*a(k)+2*X(k,:)*r; 
        w_k=w(k);                 
        if c_k < -lambda
            w(k)=(c_k + lambda) / a(k);
        elseif c_k > lambda
                w(k)=(c_k - lambda) / a(k);
        else
            w(k)=0;
        end
        r=r-(w(k)-w_k)*X(k,:)';  
    end
    obj(end+1)=r'*r+lambda*sum(abs(w));
    change=(obj(end-1)-obj(end))/obj(end-1);
    if change <= converge
        stop = 1;
    end
end
end

