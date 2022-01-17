function [W,b]=SMM_fastADMM (X, y, C, tau)
if (~exist('max_iter', 'var'))%%var 风险评价模型
    max_iter = 500;%%算法收敛最大迭r代次数=500
end
if (~exist('rho', 'var'))%%检测rho中的变量是否存在,如果不存在返回0，存在返回1
    rho = 1e-3;
end

addpath('.\libsvm-master')

n = size(X, 1);
d = size(X, 2);
s_km1 = zeros(n, d);
s_hatk = s_km1;
lambda_km1 = ones(n, d);
lambda_hatk = lambda_km1;

for i=1:size(X,3)
    for j=1:size(X,3)
        K(i,j)=(y(i,:)*y(j,:)*trace(X(:,:,i)'*X(:,:,j)))/(rho+1);%K in paper
    end
end
clear i j

for k=1: max_iter
    for i=1:size(X,3)
        f(i,:)=1-(y(i,:)*trace((lambda_hatk + rho * s_hatk)'*X(:,:,i)))/(rho+1);%q in paper
    end
    clear i
    %% ①quadprog求解alpha
%     sz1=size(K,1);
%     LB = zeros(sz1,1);
%     UB = C * ones(sz1,1);
%     Aeq=[];
%     for i=1:size(y,1)
%         Aeq=[Aeq;y'];
%     end
%     beq=zeros(sz1,1);
%     alpha=quadprog(K,-f,[],[],Aeq,beq,LB,UB);
    %% ②SOR求解alpha
    sz1=size(K,1);
    LB = zeros(sz1,1);
    UB = C * ones(sz1,1);
    alpha0=LB+(UB-LB).*randn(sz1,1);
    alpha=SMM_SOR(K,f,alpha0,0.5,LB,UB,0.05);
    
    %%更新wk
    xay=0;
    for i=1:size(X,3)
        xay=xay+X(:,:,i)'*(alpha(i,:)*y(i,:));
    end
    w_k = (lambda_hatk + rho * s_hatk + xay) / (rho + 1);
    sel = (alpha > 0) & (alpha <= C);
    %%计算b
    b=0;
    y1=y(sel,:);
    X1=X(:,:,sel);
    for i=1:size(y1,2)
        b =b+ (y1(i,:) - trace( w_k'*X1(:,:,i))) ;
    end
    b=b/ sum(sel);
    %%更新S
    S = shrinkage(rho*w_k - lambda_hatk, tau) / rho;
    s_k = S;
    lambda_hatk = lambda_hatk - rho * (w_k - s_k);
    
    W=w_k; 
end

end

