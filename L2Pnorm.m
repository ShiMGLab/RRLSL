% compute l_{2,p} norm
% ||A||_{2,p}
function v = L2Pnorm(A,p, epsilon)
% a: each column is a data
% d:  norm value
if nargin<3
    epsilon=0;
end

v=sum((sum(A.*A,2)+epsilon).^(p/2))^(1/p);