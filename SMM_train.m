function acc=SMM_train(data_all,C,tau,trainnum,z,s)
%% 测试样本
X_test=[];
y_test=[];
for i=1:z
    temp1=data_all(:,:,(i-1)*s+trainnum+1:i*s);
    temp2=i*ones(s-trainnum,1);
    X_test=cat(3,X_test,temp1);%%测试样本
    y_test=[y_test;temp2];%%测试标签
end
%% 开始训练并测试
a=0;
fprintf('开始训练\r ');
fprintf('共%g个子分类器, 已完成 ',(z*(z-1))/2);
for number1=1:(z-1)
    for number2=(number1+1):z
        a=a+1;
        fprintf('%g ',a)
        %%训练样本
        X=cat(3,data_all(:,:,(number1-1)*s+1:(number1-1)*s+trainnum),data_all(:,:,(number2-1)*s+1:(number2-1)*s+trainnum));%训练样本
        y=[ones(trainnum,1);-1*ones(trainnum,1)];%训练标签
        %%训练
        [W,b]=SMM_fastADMM (X, y, C, tau) ;
        %%测试
        for i=1:size(X_test,3)
            temp = sign(trace(W'*X_test(:,:,i))+b);
            if (temp == 1)
                pre(i,:)=number1;
            else
                pre(i,:)=number2;
            end
        end
        predict(:,a)=pre;
    end
end
fprintf('\r');

y_pre=[];
for i1=1:(s-trainnum)*z
    table=tabulate(predict(i1,:));
    [maxCount,idx]=max(table(:,2));
    y_pre(i1,1)=table(idx);
end
acc=length(find(y_pre == y_test))/length(y_test);
end