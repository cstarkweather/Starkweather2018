function [weights] = trainOdorA_WithOmission(training_session,Weber_fraction,jitter,learning_rate,w)
flatten_belief=500;
nTrials=50; %number of trials per training session
clear ISIdistributionMatrix
x=[]; %series of observations during trials - will be filled in later

% Set hazard rate of transitioning OUT of the ITI
% 1/65 mimics task parameters
ITIhazard=(1/65);

% Create distribution of ISI's - 10% unassigned are omission trials
% Possible ISIs range from 5-13
ISIpdf=normpdf([1.2 1.4 1.6 1.8 2 2.2 2.4 2.6 2.8],2,0.5)/sum(normpdf([1.2 1.4 1.6 1.8 2 2.2 2.4 2.6 2.8],2,0.5));
full_ISIpdf = [zeros(1,5) ISIpdf zeros(1,15)];
blurred_ISIpdf = blur_vector(full_ISIpdf,jitter); %adjust probability distribution for perceived jitter in cue onset timings
ISIcdf=zeros(1,29);
ISIcdf(1)=blurred_ISIpdf(1);
for i=2:length(blurred_ISIpdf)
    ISIcdf(i)=ISIcdf(i-1) + blurred_ISIpdf(i);
end

ISIs =  find(round(full_ISIpdf*nTrials)>0);
ISIdistribution = round(full_ISIpdf*nTrials*0.9);
trialtype=[];
for i=1:length(ISIs)
    trialtype=[trialtype ones(1,ISIdistribution(ISIs(i)))*ISIs(i)];
end

trials = [trialtype zeros(1,round(nTrials*0.1))]; %all rewarded trials and 10% omission trials
ISIdistributionMatrix=trials(randperm(length(trials))); %interleaving all trial types
%%
% Calculate hazard rate of receiving reward after substates 5-14 (ISIhazard);
% Used later to create transition matrix
ISIhazard=zeros(1,29);
ISIhazard(1)=blurred_ISIpdf(1);
for i=2:length(blurred_ISIpdf)
    ISIhazard(i)=blurred_ISIpdf(i)/(1-ISIcdf(i-1));
end
ISIhazard(ISIhazard>1)=1;
ISIhazard(ISIhazard<0)=0;
%%

% Generate sequence of observations that corresponds to trials
% Observations:
%   Null-->1
%   Odor ON-->2
%   Reward-->3
for i=1:1:length(ISIdistributionMatrix)
    if ISIdistributionMatrix(i)>0 %reward delivery trials
        blurred_ISIdistributionMatrix(i)=round(normrnd(ISIdistributionMatrix(i),jitter));
        ISI = ones(1,blurred_ISIdistributionMatrix(i)-1);
        ITI = ones(1,geornd(ITIhazard)+10);
        trial=[2;ISI';3;ITI'];
        x=[x;trial];
    else %omission trials
        ITI = ones(1,(geornd(ITIhazard)+20));
        trial=[2;ones(10,1);ITI'];
        x=[x;trial];
    end
    ITIdistributionMatrix(i)=length(ITI);
end
x=[x;ones(30,1)];
% states:
% ISI = 1-29
% ITI = 30

%Fill out the transition matrix T
%T(x,y) is the probability of transitioning from sub-state x-->y
T=zeros(30,30);

%%
indices = find(ISIhazard);
for i=1:indices(end)
    T(i,1+i)=1-ISIhazard(i);
    T(i,30)=ISIhazard(i);
end


% Blur the Transition Matrix to account for Scalar Uncertainty
T_blur = blur_matrices_inputWeberFraction(T,Weber_fraction);
T_blur(isnan(T_blur))=0;

% ITI length is drawn from exponential distribution in task
% this is captured with single ITI substate with high self-transition
% probability
T_blur(30,30)=((1-ITIhazard)+ITIhazard*0.1);
T_blur(30,1)=(ITIhazard*0.9);
%%

%Fill out the observation matrix O
% O(x,y,:) = [a b c]
% a is the probability that observation 1 (null) was observed given that a
% transition from sub-state x-->y just occurred
% b is the probability that observation 2 (odor ON) was observed given that
% a transition from sub-state x-->y just occurred
% c is the probability that observation 2 (reward) was observed given that
% a transition from sub-state x-->y just occurred
O=zeros(30,30,3);

%stimulus onset
O(30,1,:) = [0 1 0];
O(30,30,2) = ITIhazard*0.1; %omission trial

%ITI
O(30,30,1) = (1-ITIhazard*0.1);

%ISI - any substate transitions within the ISI will have a null observation
for i=1:size(T_blur,1)-1
    clear nonzeroTindices;
    nonzeroTindices = find(T_blur(i,2:size(T_blur,1)-1)>0.0001);
    O(i,nonzeroTindices+1,1)=1;
end
O((T_blur(:,30))>0.0001,30,3)=1;


%% Run TD learning
results_training = TDlearning(x,O,T_blur,w,flatten_belief,Weber_fraction,learning_rate);
save(strcat('OdorA_90%_results_',num2str(training_session),'.mat'))

weights = (results_training.w(end,:))';
end