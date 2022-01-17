%%֧�־����ʵ�����  ����1v1����
%%
clear;
clc;
addpath('C:\matlab\bin\MATLAB����')
cd('C:\matlab\bin\MATLAB����\����֧�־���������');
%% ��������  
% AHUT������� ÿ��160������
% �������� ���� 0.2mm������ 0.4mm������ 0.2mm��Ȧ 0.3mm��Ȧ 0.3mm��Ȧ 0.4mm��Ȧ 
load('.\dataset.mat')
%% input paramater
trainnum = s*0.8;%ѵ��������
C = 0.1;%%j���˳ͷ���ϵ��
tau = 0.0001;%%����ϵ��

k = 5;%Ԥ�����ݷֳ�5��
m= size(data_all,3)/z;
%������֤,ʹ��ʮ�۽�����֤  Kfold
%indicesΪ m ��һ�����ݣ���ʾÿ��ѵ����������k�����ݵ���һ��
indices = crossvalind('Kfold', m, 5);

result=[];
for i=1:5
    test = (indices == i);
    % ȡ������ȡ��i��ѵ�����ݵ������߼�ֵ
    train = ~test;
    %4��ѵ��,1�ݲ���
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
    %ѵ��������
    acc = SMM_train(X,C,tau,trainnum,z,s);
    time = toc;
    fprintf('C=%g tau=%g \r acc = %.4f, time =%.4f\r\n',C,tau,acc,time);
    %%������
    result(i,1) = acc;
    result(i,2) = time;
end
clearvars -except result
for ii=1:2
    result(6,ii) = mean(result(1:5,ii));
    result(7,ii) = std(result(1:5,ii));
end

