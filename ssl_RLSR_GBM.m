clear;
load C:\Users\mingguang.shi\Desktop\SSL\DataTCGA\GBM\GBM\GBM_ProcessedData_t.mat;
load C:\Users\mingguang.shi\Desktop\SSL\DataTCGA\GBM\GBM\GBM_Clinical.mat;
GBM=GBMProcessedDatat;
GBM(:,1)=[];
GBM(1,:)=[];
%normalization
GBM=table2array(GBM);
GBM_N=(GBM-repmat(mean(GBM,2),1,size(GBM,2)))./repmat(std(GBM,0,2),1,size(GBM,2)); 
%mRNA=12042,miRNA=534,the sum is 12576
GBM_P=GBM_N(1:12576,:);
[m,q]=size(GBM_P);
%91 label samples
XL=GBM_P(:,1:q/3);
%91 unlabel samples 
XU=GBM_P(:,q/3+1:2*q/3);
%91 unlabel test samples 
XTU=GBM_P(:,2*q/3+1:q);
%91 label 
YL=GBMClinical.VarName2(1:q/3,:);
y=[YL;zeros(1,q/3)'];

%%%%%%%select the features from RLSR
p=1;
%gamma=1;
gamma=1;
%MAX_ITER=10;
MAX_ITER=3;
[ranked, theta,W,obj] = RLSR( XL, YL, XU, p,gamma, MAX_ITER);
ranked_number=100;
sf=find(ranked<=ranked_number);
xl=XL(sf,:);
xu=XU(sf,:);
xlu=[xl,xu]';
%%%%%%%Laplacian regularization 
n=2*q/3;
x2=sum(xlu.^2,2); 
hh=2*1^2;
k=exp(-(repmat(x2,1,n)+repmat(x2',n,1)-2*xlu*xlu')/hh);
w=k;
t=(k^2+1*eye(n)+10*k*(diag(sum(w))-w)*k)\(k*y);
%%%%%%%performance evaluation from training data 
tet = mean( t(q/3+1:2*q/3).*GBMClinical.VarName2(q/3+1:2*q/3,:) < 0 );
accuracytest=1-tet;
labelstest=GBMClinical.VarName2(q/3+1:2*q/3,:);
scorestest=t(q/3+1:2*q/3);
[Xtest,Ytest,Ttest,AUCtest] = perfcurve(labelstest,scorestest,1);

[PREC, TPR, FPR, THRESH]= prec_rec(scorestest, labelstest);
prec_rec(scorestest, labelstest, 'holdFigure', 1);
auprc = polyarea(TPR,FPR);
auprc;

%%%%%%%independent test data
xl=XL(sf,:);
xtu=XTU(sf,:); 
xltu=[xl,xtu]';
%%%%%%%Laplacian regularization 
n=2*q/3;
x2=sum(xltu.^2,2); 
hh=2*1^2;
k=exp(-(repmat(x2,1,n)+repmat(x2',n,1)-2*xltu*xltu')/hh);
w=k;
t=(k^2+1*eye(n)+10*k*(diag(sum(w))-w)*k)\(k*y);
%%%%%%%performance evaluation
tet = mean( t(q/3+1:2*q/3).*GBMClinical.VarName2(2*q/3+1:q,:) < 0 );
accuracytest=1-tet;
labelstest=GBMClinical.VarName2(2*q/3+1:q,:);
scorestest=t(q/3+1:2*q/3);
[Xtest,Ytest,Ttest,AUCtest] = perfcurve(labelstest,scorestest,1);
AUCtest;

[PREC, TPR, FPR, THRESH]= prec_rec( scorestest,labelstest);
prec_rec(scorestest, labelstest, 'holdFigure', 1);
auprc_test = polyarea(TPR,FPR);
auprc_test;
