%Jake knife test code
clc;
close all;
clear all;
addpath libsvm-3.20\matlab;

%data=xlsread('Combined_PSSM_DWT_SAAC_DPC_RFE_186_413.csv');
load DNA_PSSM_DWT_SAAC_DPC_186;
load model_DNA_PSSM_DWT_SAAC_DPC_1075_SVM_lnr;
    %****************************************

Result=0;
y=[];

Accuracy=0;
Total_Seq_train=186;
DNA_labels=[];
Total_correct=0;
  c1=0;  c2=0;
  
 DNA_labels(1:93)=1;
 DNA_labels(94:186)=2;
  %+++++++++++++++++++++++++++++  train label

 Labelstem=[];
 Samplestem=[];
Samplestem=DNA_PSSM_DWT_SAAC_DPC_186;
Labelstem= DNA_labels';

  for i=1:size(Samplestem,1)
         TestSample=Samplestem(i,:)';
         TestLabel=Labelstem(i,:)';
         
         [Predict_label,accuracy, dec_values] = svmpredict(TestLabel, TestSample', model);
         y(i)=Predict_label;      
  end 
   y2=y;
   Result=find(y==DNA_labels);
   Total_correct=size(Result,2);
   Accuracy=(Total_correct/Total_Seq_train)*100
    
   %+++++++++ individual Accuracy
    for i=1:93
        if( y(i)==1)
            c1=c1+1;
        end
    end
    for i=94:186
        if( y(i)==2)
            c2=c2+1;
        end
    end

    C1=(c1/93)*100
    C2=(c2/93)*100

