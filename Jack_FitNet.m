%Jake knife test code

clc;
close all;
clear all;
%addpath libsvm-mat-2.88-1;
%addpath libsvm-3.18\windows;

load fisher_DNA_1075_feature_900;
    %****************************************
  
Result=0;
y=[];
Accuracy=0;
Total_Seq_train=1075;
DNA_labels=[];
Total_correct=0;
  c1=0; 
  c2=0;
  
  
  
  DNA_labels(1:525)=0;
DNA_labels(526:1075)=1;

  %+++++++++++++++++++++++++++++  train label

   
 Labelstem=[];
 Samplestem=[];

 Samplestem=fisher_DNA_1075_feature_900';

Labelstem=DNA_labels';

Knn=5;
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
             
        %%%%%%%%%%%%%%%%%%%%%%%%%%% fitting network %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%         net = newfit(simplefitInputs,simplefitTargets,20);
%         net = train(net,simplefitInputs,simplefitTargets);
%         simplefitOutputs = sim(net,simplefitInputs);
        
        net = newfit(Samples,Labels,20) ;
        net = train(net,Samples,Labels);
        Y =sim(net, TestSample);
        FIT=round(Y);
        y(A) = FIT;
        
%         predicted_lbls = vec2ind(Y);
%         Mahlabadistane_PNN_2_kPCA_dipep(A,predicted_lbls )=1; % 1=true for % the class which has won
%         y(A)=predicted_lbls;
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
             
  end
  
   y2=y;
   Result=find(y==DNA_labels);
   Total_correct=size(Result,2);
   Accuracy=(Total_correct/Total_Seq_train)*100
    
   
   %+++++++++ individual Accuracy
    for i=1:525
        if( y(i)==0)
            c1=c1+1;
        end
    end
    for i=526:1075
        if( y(i)==1)
            c2=c2+1;
        end
    end
    
    C1=(c1/525)*100
    C2=(c2/550)*100
    
    
