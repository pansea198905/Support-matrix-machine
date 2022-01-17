%%支持矩阵机实测程序  采用1v1策略
%%
clear;
clc;
addpath('C:\matlab\bin\MATLAB程序')
cd('C:\matlab\bin\MATLAB程序\各类支持矩阵机程序包');
%% 加载数据  
% AHUT七类故障 每类160个样本
% 故障类型 正常 0.2mm滚动体 0.4mm滚动体 0.2mm外圈 0.3mm外圈 0.3mm内圈 0.4mm内圈 
load('.\dataset.mat')
%% input paramater
trainnum = s*0.8;%训练样本数
C = 0.1;%%j损伤惩罚项系数
tau = 0.0001;%%低秩系数

k = 5;%预将数据分成5份
m= size(data_all,3)/z;
%交叉验证,使用十折交叉验证  Kfold
%indices为 m 行一列数据，表示每个训练样本属于k份数据的哪一份
indices = crossvalind('Kfold', m, 5);

result=[];
for i=1:5
    test = (indices == i);
    % 取反，获取第i份训练数据的索引逻辑值
    train = ~test;
    %4份训练,1份测试
    train_data=[];
    test_data=[];
    X=[];
    for j=1:z
        temp1 = data_all(:,:,(j-1)*s+1:j*s);
        train_data1 = temp1(:,:,train);
        test_data1 = temp1(:,:,test);
        temp2 = cat(3,train_data1,test_data1);
        X = cat(3,X,temp2);
    end
    tic;
    %训练并测试
    acc = SMM_train(X,C,tau,trainnum,z,s);
    time = toc;
    fprintf('C=%g tau=%g \r acc = %.4f, time =%.4f\r\n',C,tau,acc,time);
    %%保存结果
    result(i,1) = acc;
    result(i,2) = time;
end
clearvars -except result
for ii=1:2
    result(6,ii) = mean(result(1:5,ii));
    result(7,ii) = std(result(1:5,ii));
end

