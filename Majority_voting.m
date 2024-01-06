clc;
close all;
clear all;

load prediction_TNC_GRNN;    
load prediction_TNC_KNN;
load prediction_TNC_PNN;
load prediction_TNC_SVM;
load prediction_TNC_RF_60_20;
load prediction_TNC_FN_796485;
load prediction_TNC_PRN_811286;


  a=find(TNC_PNN==1)
  PNN_Pre(a)=-1;
  b=find(TNC_PNN==2)
  PNN_Pre(b)=1;
  
  a=find(TNC_FN==0)
  Pre_FN(a)=-1;
  b=find(TNC_FN==1)
  Pre_FN(b)=1;
  
  a=find(TNC_PRN==0)
  Pre_PRN(a)=-1;
  b=find(TNC_PRN==1)
  Pre_PRN(b)=1;


 A=TNC_GRNN';
 B=TNC_KNN';
 C=PNN_Pre';
 D=TNC_SVM';
 E=TNC_RF';
 F=Pre_FN';
 G=Pre_PRN';


TotalACC=0;
total=[];
Pred1=[];

c11=[];
c22=[];
c33=[];
c44=[];
c55=[];
c66=[];
c77=[];
c88=[];


c1=0;
c2=0;
c3=0;
c4=0;
c5=0;
c6=0;
c7=0;
c8=0;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%
T_P=0;
T_N=0;
F_N=0;
F_P=0;
Total_sen=0;
Total_spe=0;
TotalMCC=0;
Total_F_Measure=0;
G_mean=0;
G_mean1=0;
Total_G_mean=0;


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  Memb_labels(1:490)=-1;
Memb_labels(491:1081)=1;

for i=1:1081

 data1=[];
 data2=[];
 data3=[];
 data4=[];
 data5=[];
 data6=[];
 data7=[];
%  data8=[];


total=[];

 data1=A(i,1);
 data2=B(i,1);
 data3=C(i,1);
 data4=D(i,1);
 data5=E(i,1);
 data6=F(i,1);
 data7=G(i,1);
%  data8=H(i,1);

    
        
    total=data1;
    total=[total data2];
    total=[total data3];
    total=[total data4];
    total=[total data5];
    total=[total data6];
    total=[total data7];
%     total=[total data8];
    
   one=0;
   two=0;
%    three=0;
%    four=0;
%    five=0;
%    six=0;
%    seven=0;
%    eight=0;
    for l=1:2
        if (total(l)==-1)
            one=one+1;
        elseif (total(l)==1)
            two=two+1;
%         elseif (total(l)==3)
%             three=three+1;
%         elseif(total(l)==4)
%            four=four+1;
%         elseif(total(l)==5)
%            five=five+1;
%            elseif (total(l)==6)
%             six=six+1;
%         elseif(total(l)==7)
%            seven=four+1;
%         elseif(total(l)==5)
%            five=five+1;
        end
    end
    
    %ar=[one two three four five];
    ar=[one two];
    
     maximam=max(ar);
    if (ar(1)==maximam)
        Pred1(i)=-1;
    elseif (ar(2)==maximam)
        Pred1(i)=1;
%     elseif(ar(3)==maximam)
%         Pred1(i)=3;
        %e
        %elseif(ar(4)==maximam)
       % Pred1(i)=4;
        %elseif(ar(5)==maximam)
        %Pred1(i)=5;
    end
        
    
end
y=Pred1;
 Result=find(y==Memb_labels);
    Total_correct=size(Result,2);
    Accuracy=(Total_correct/1081)*100
    
for i=1:1081
    if(i<=490)
        if(y(i)==1)
            T_P=T_P+1;
        else
            F_N=F_N+1;
        end
    elseif(i>491 && i<=1081)
            if(y(i)==2)
                T_N=T_N+1;
            else
                F_P=F_P+1;
                
            end
    end
end
    TNR=T_N/(T_N+F_P);
    TPR=T_P/(T_P+F_N);
    
    G_mean=sqrt(TNR*TPR); 
    Total_G_mean=Total_G_mean +G_mean;
    
    Sensitivity=T_P/(T_P+F_N);
    Total_sen=Total_sen+Sensitivity;
    
    Specificity=T_N/(T_N+F_P);
    
    a=((T_P+F_P)*(T_P+F_N)*(T_N+F_P)*(T_N+F_N));
    b=sqrt(a);
    d=((T_P*T_N)-(F_P*F_N));
    MCC=d/b;
    
    gh=(T_P/(T_P+F_P));
    rh=(T_P/(T_P+F_N));
    F=(gh*rh)/(gh+rh);
    
    F_Measure=2*F;
    Total_F_Measure=Total_F_Measure+F_Measure;
    
    Total_spe=Total_spe+Specificity;
    
    TotalMCC=TotalMCC+MCC;
    
 for i=1:1081
    if(i<=490)
        if(y(i)==1)
            T_N=T_N+1;
        else
            F_P=F_P+1;
        end
    elseif(i>491 && i<=1081)
            if(y(i)==2)
                T_P=T_P+1;
            else
                F_N=F_N+1;
                
            end
    end
 end
    TNR=T_N/(T_N+F_P);
    TPR=T_P/(T_P+F_N);
    
    G_mean=sqrt(TNR*TPR); 
    Total_G_mean=Total_G_mean +G_mean;
    
    Sensitivity=T_P/(T_P+F_N);
    Total_sen=Total_sen+Sensitivity;
    
    Specificity=T_N/(T_N+F_P);
    
    a=((T_P+F_P)*(T_P+F_N)*(T_N+F_P)*(T_N+F_N));
    b=sqrt(a);
    d=((T_P*T_N)-(F_P*F_N));
    MCC=d/b;
    
    gh=(T_P/(T_P+F_P));
    rh=(T_P/(T_P+F_N));
    
    F=(gh*rh)/(gh+rh);
    F_Measure=2*F;
    
    Total_F_Measure=Total_F_Measure+F_Measure;
    
    Total_spe=Total_spe+Specificity;
    
    TotalMCC=TotalMCC+MCC;   

    
    G_mean1=Total_G_mean/2
    sensi=(Total_sen/2)*100
    speci=(Total_spe/2)*100
    MCCT=(TotalMCC/2)
    T_F_Measure=(Total_F_Measure/2)
    
    