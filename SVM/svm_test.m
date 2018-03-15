clear;
load('hw2data/q3_1_data');
C = 10;
[alpha, objective, w, b] = svm(double(trD), double(trLb), C);
svm_result = valD'*w+b;


corrects = sum((svm_result>0)&(valLb>0))+sum((svm_result<0)&(valLb<0));
accuracy = corrects/size(svm_result,1);
num_of_support  = sum(alpha > 0.001);
confusion_mat = [sum((svm_result>0)&(valLb>0)), sum((svm_result>0)&(valLb<0));
                 sum((svm_result<0)&(valLb<0)), sum((svm_result<0)&(valLb>0))];
