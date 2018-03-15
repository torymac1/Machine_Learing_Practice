clear;

load('hw2data/q3_1_data');

C=10;
maxEpoch=2000;
eta_0 = 1;
eta_1 = 100;
[w,b,obj]=multiclass_svm1(trD,trLb,C,maxEpoch,eta_0,eta_1);