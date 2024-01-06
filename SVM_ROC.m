%Jake knife test code

clc;
close all;
clear all;
addpath libsvm-3.20\matlab;
addpath libsvm-mat-2.88-1;
load Combined_Pse_TCP_PSSM_1075;

%data=xlsread('Combined_Pse_TPC_AAC_PSSM_NMBAC_1075_45.csv');
    %****************************************
  
Result=0; 
y=[];
Accuracy=0;
Total_Seq_train=1075;
DNA_labels=[];
Total_correct=0;
  c1=0;  c2=0; 

DNA_labels(1:525)=1;
DNA_labels(526:1075)=2;
  %+++++++++++++++++++++++++++++  train label
 Labelstem=[];
 Samplestem=[];
 Samplestem=Combined_Pse_TCP_PSSM_1075;
 Labelstem= DNA_labels';

 %++++++++++++++++++++++++++++++ best values for SVM
%    [bestacc, bestc, bestg]=SVMcgForClass(Labelstem, Samplestem)
%    option=[' -c ' num2str(bestc) ' -g ' num2str(bestg) ];
 
 
  for A=1:size(Samplestem,1)
       A
        if A==1
            Samples=Samplestem(A+1:end,:)';
            TestSample=Samplestem(A,:)';
            Labels=Labelstem(A+1:end,:)';
            TestLabel=Labelstem(A,:)';
        else
            s11=Samplestem(1:(A-1),: ); % Jackknifing 
            s22=Samplestem((A+1):end,:);
            Samples=[s11;s22]';

            TestSample=Samplestem(A,:)';
            l11=Labelstem(1:(A-1),: );
            l22=Labelstem((A+1):end,:);
            Labels=[l11;l22]';
            TestLabel=Labelstem(A,:)';
        end
              model = svmtrain (Labels' , Samples' ,'-t 2 -c 27.85 -g 0.0206');
             [Predict_label,accuracy, dec_values] = svmpredict(TestLabel, TestSample', model);
             y(A)=Predict_label;    
             auc=roc_curve(accuracy,Predict_label);
  end
   y2=y;
   Result=find(y==DNA_labels);
   Total_correct=size(Result,2);
   Accuracy=(Total_correct/Total_Seq_train)*100;
   
   %+++++++++ individual Accuracy
    for i=1:525
        if( y(i)==1)
            c1=c1+1;
        end
    end
    for i=526:1075
        if( y(i)==2)
            c2=c2+1;
        end
    end
   
    
    C1=(c1/525)*100
    C2=(c2/550)*100