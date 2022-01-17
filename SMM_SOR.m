function bestalpha=SMM_SOR(Q,f,alpha0,t,lb,ub,smallvalue)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%       bestalpha=DALRMC_SOR(Q,f,alpha0,t,lb,ub,smallvalue)
%
%       Input:
%               Q     - Hessian matrix(Require positive definite).
%
%               t     - (0,2) Paramter to control training.
%
%               C     - Upper bound
%
%               smallvalue - Termination condition终止条件
%
%       Output:
%               bestalpha - Solutions of QPPs.
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Initailization
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
[m,n]=size(Q);
L=tril(Q);%%Q的下三角
E=diag(Q);
twinalpha=alpha0;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Compute alpha
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
for j=1:n
    %     i=i+1;
    twinalpha(j,1)=alpha0(j,1)-(t/E(j,1))*(Q(j,:)*twinalpha(:,1)-f(j,:)+L(j,:)*(twinalpha(:,1)-alpha0));
    if twinalpha(j,1)<lb(j,1)
        twinalpha(j,1)=lb(j,1);
    elseif twinalpha(j,1)>ub(j,1)
        twinalpha(j,1)=ub(j,1);
    else
        
    end
end

alpha=[alpha0,twinalpha];
while norm(alpha(:,2)-alpha(:,1))>smallvalue %%alpha第二列减第一列
    for j=1:n
        twinalpha(j,1)=alpha(j,2)-(t/E(j,1))*(Q(j,:)*twinalpha(:,1)-f(j,:)+L(j,:)*(twinalpha(:,1)-alpha(:,2)));
        if twinalpha(j,1)<lb(j,1)
            twinalpha(j,1)=lb(j,1);
        elseif twinalpha(j,1)>ub(j,1)
            twinalpha(j,1)=ub(j,1);
        else
            
        end
    end
    alpha(:,1)=[];%%第一列置空
    alpha=[alpha,twinalpha];
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Output
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
bestalpha=alpha(:,2);


