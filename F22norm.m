% compute squared F-norm
% ||A||_F^2
function d = F22norm(A)
% A: each column is a data
% d:   distance value

d=sum(sum(A.*A));

