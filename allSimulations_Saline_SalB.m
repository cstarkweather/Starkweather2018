% Medial Prefrontal Cortex Shapes Dopamine Reward Prediction Errors Under
% State Uncertainty
% POMDP in the spirit of Daw et al. (2006), Rao (2010)
% Author: Clara Kwon Starkweather
%% Training - All Odors

clearvars

Weber = 0.05; %Amount of scalar timing uncertainty (fraction of sub-state length)
Jitter = 2; %Amount of temporal jitter in detecting odor onsets (fraction of sub-state length)

n_sessions=500; %number of training sessions
alphaDistribution=linspace(0.1,0.1,n_sessions); % learning rate used in each training session. May be tapered, kept constant, etc...
learned_weightsA90=zeros(30,1); %initializing weights
learned_weightsA100=zeros(30,1);
learned_weightsB90=zeros(30,1);
learned_weightsC90=zeros(30,1);

for iterations=1:n_sessions
    
    [ new_weightsA90]=trainOdorA_WithOmission(iterations,Weber,Jitter,alphaDistribution(iterations),learned_weightsA90);
    learned_weightsA90 = new_weightsA90;
    [ new_weightsA100]=trainOdorA_WithOUTOmission(iterations,Weber,Jitter,alphaDistribution(iterations),learned_weightsA100);
    learned_weightsA100 = new_weightsA100;
    [ new_weightsB90]=trainOdorB(iterations,Weber,Jitter,alphaDistribution(iterations),learned_weightsB90);
    learned_weightsB90 = new_weightsB90;
    [ new_weightsC90]=trainOdorC(iterations,Weber,Jitter,alphaDistribution(iterations),learned_weightsC90);
    learned_weightsC90 = new_weightsC90;

end

%%
plotValueandRPE(14,x,ISIdistributionMatrix,results_training)

%% Testing sessions - all odors - Put these in a new folder, separate Saline and SalB simulations

    clearvars

    Weber = 0.05;
    Jitter = 2;
    Randomness_Jitter = 0;
    Randomness_Weber = 0;

    n_simulations=10;

    load('OdorA_100%_results_500.mat')
    learned_weightsA100 = results_training.w(end,:);

    load('OdorA_90%_results_500.mat')
    learned_weightsA90 = results_training.w(end,:);

    load('OdorB_results_500.mat')
    learned_weightsB = results_training.w(end,:);

    load('OdorC_results_500.mat')
    learned_weightsC = results_training.w(end,:);

    belief_impaired_dist = [zeros(1,30)]; %0 for intact belief state; 1 for impaired belief state; 
    belief_impaired_dist=belief_impaired_dist(randperm(length(belief_impaired_dist)));

    for iterations=1:n_simulations

        Weber_fraction = Weber + Randomness_Weber*rand;
        jitter = Jitter + Randomness_Jitter*rand;

        belief_impaired = belief_impaired_dist(iterations);

        learning_rate = 0.1;

        SimulateOdorA_withoutOmission(iterations,Weber_fraction,jitter,belief_impaired,learned_weightsA100',learning_rate)

        SimulateOdorA_withOmission(iterations,Weber_fraction,jitter,belief_impaired,learned_weightsA90',learning_rate);

        SimulateOdorC(iterations,Weber_fraction,jitter,belief_impaired,learned_weightsC',learning_rate)

        SimulateOdorB(iterations,Weber_fraction,jitter,belief_impaired,learned_weightsB',learning_rate)

    end
%% Plots
    
    clearvars
    figure(1)
    A90_simulations = dir('A90*');
    A100_simulations = dir('A100*');
    B_simulations = dir('B*');
    C_simulations = dir('C*');

    [phasic_A90 tonic_A90 omission_A90]=Organize_for_Plotting(A90_simulations);
    [phasic_A100 tonic_A100]=Organize_for_Plotting(A100_simulations);
    [phasic_B tonic_B omission_B]=Organize_for_Plotting(B_simulations);
    [phasic_C tonic_C omission_C]=Organize_for_Plotting(C_simulations);

    subplot(3,3,5)
    for i=1:size(phasic_A90,1)
        e_A90(i) = std(phasic_A90(i,2,:))/sqrt(size(phasic_A90,3));
        e_A100(i) = std(phasic_A100(i,2,:))/sqrt(size(phasic_A100,3));
    end
    
    e_B = std(phasic_B(1,2,:))/sqrt(size(phasic_B,3));
    e_C = std(phasic_C(1,2,:))/sqrt(size(phasic_C,3));

    plot(1.2:0.2:2.8,(sum(phasic_A90(:,2,:),3)/size(B_simulations,1))','k.','Markersize',25)
    hold on
    errorbar(1.2:0.2:2.8,(sum(phasic_A90(:,2,:),3)/size(B_simulations,1))',e_A90)
    hold on
    plot([1.2 2.8],[(sum(phasic_B(1,2,:),3)/size(B_simulations,1)) (sum(phasic_C(1,2,:),3)/size(B_simulations,1))],'k.','Markersize',25)
    hold on
    errorbar([1.2 2.8],[(sum(phasic_B(1,2,:),3)/size(B_simulations,1)) (sum(phasic_C(1,2,:),3)/size(B_simulations,1))],[e_B e_C])
    ylim([0.2 0.8])
    
    subplot(3,3,4)
    plot(1.2:0.2:2.8,(sum(phasic_A100(:,2,:),3)/size(B_simulations,1))','k.','Markersize',25)
    hold on
    errorbar(1.2:0.2:2.8,(sum(phasic_A100(:,2,:),3)/size(B_simulations,1))',e_A90)
    hold on
    
    for i=1:size(phasic_A90,1)
        e_A90(i) = std(phasic_A90(i,1,:))/sqrt(size(phasic_A90,3));
        e_A100(i) = std(phasic_A100(i,1,:))/sqrt(size(phasic_A100,3));
    end

    subplot(3,3,7)
    plot(1.2:0.2:2.8,(sum(phasic_A100(:,1,:),3)/size(B_simulations,1))','k.','Markersize',25)
    hold on
    errorbar(1.2:0.2:2.8,(sum(phasic_A100(:,1,:),3)/size(B_simulations,1))',e_A100)
    hold on
    ylim([-0.1 0.05])
    
    subplot(3,3,8)
    plot(1.2:0.2:2.8,(sum(phasic_A90(:,1,:),3)/size(B_simulations,1))','k.','Markersize',25)
    hold on
    errorbar(1.2:0.2:2.8,(sum(phasic_A90(:,1,:),3)/size(B_simulations,1))',e_A90)
    hold on
    ylim([-0.25 0.05])

    subplot(3,3,2)

    plot(1:19,tonic_A90(1:19),'k')
    hold on
    plot(21:30,tonic_A90(21:end),'k')
    for i=1:size(phasic_A90,1)
        plot(i+10:i+12,sum(phasic_A90(i,:,:),3)/size(A90_simulations,1),'Color',[1-i/9 i/9 1])
        hold on
    end
    ylim([-0.2 1])

    subplot(3,3,1)
    plot(1:19,tonic_A100(1:19),'k')
    hold on
    plot(21:30,tonic_A100(21:end),'k')
    for i=1:size(phasic_A100,1)
        plot(i+10:i+12,sum(phasic_A100(i,:,:),3)/size(A100_simulations,1),'Color',[1-i/9 i/9 1])
        hold on
    end
    ylim([-0.2 0.8])

    subplot(3,3,9)
    plot(omission_A90,'Color',[0.5 0.5 0.9])
    hold on
    plot(omission_B,'Color',[0.1 0.8 0.5])
    hold on
    plot(omission_C,'Color',[0.9 0.5 0.5])
    ylim([-0.4 0.8])
    
    subplot(3,3,3)
    ft = fittype( 'poly1' );
    for i=1:length(A90_simulations)
        [fitresult] = fit((1.2:0.2:2.8)',phasic_A90(:,2,i), ft );
        slopessetavg1(i)=fitresult.p1;
    end
    histogram(slopessetavg1,'BinWidth',0.05);
    xlim([-0.6 0.6])
    
    subplot(3,3,6)
        ft = fittype( 'poly1' );
    for i=1:length(A100_simulations)
        [fitresult] = fit((1.2:0.2:2.8)',phasic_A100(:,2,i), ft );
        slopessetavg2(i)=fitresult.p1;
    end
    
    histogram(slopessetavg2,'BinWidth',0.05);
    xlim([-0.6 0.6])
    %%
    boxplot(slopessetavg1,'orientation','horizontal')
    xlim([-0.6 0.6])
    %% Go through simulations and extract RPEs - Saline or SalB folder

A90_simulations = dir('A90*');
A100_simulations = dir('A100*');
B_simulations = dir('B*');
C_simulations = dir('C*');

all_A90_RPEs=zeros(10,21,n_simulations);
all_A100_RPEs=zeros(9,21,n_simulations);
all_B_RPEs=zeros(2,21,n_simulations);
all_C_RPEs=zeros(2,21,n_simulations);

number_ISIs_B=zeros(n_simulations,15);
number_ISIs_C=zeros(n_simulations,15);
number_ISIs_A90=zeros(1,15);
number_ISIs_A100=zeros(1,15);

for simulation=1:n_simulations

    load(A90_simulations(simulation).name)
    A90_ISIs = unique(ISIdistributionMatrix);

    for ISI_index=1:length(A90_ISIs)
        cue_onsets = find(x==2);
        whereplot = [cue_onsets(ISIdistributionMatrix==A90_ISIs(ISI_index)):cue_onsets(ISIdistributionMatrix==A90_ISIs(ISI_index))+20];
        all_A90_RPEs(ISI_index,:,simulation) = (results.rpe(whereplot))';
        number_ISIs_A90(ISI_index) = number_ISIs_A90(ISI_index)+ sum(A90_ISIs(ISI_index) == ISIdistributionMatrix);
    end

    load(A100_simulations(simulation).name)
    A100_ISIs = unique(ISIdistributionMatrix);

    for ISI_index=1:length(A100_ISIs)
        cue_onsets = find(x==3);
        whereplot = [cue_onsets(ISIdistributionMatrix==A100_ISIs(ISI_index)):cue_onsets(ISIdistributionMatrix==A100_ISIs(ISI_index))+20];
        all_A100_RPEs(ISI_index,:,simulation) = (results.rpe(whereplot))';
        number_ISIs_A100(ISI_index) = number_ISIs_A100(ISI_index)+ sum(A100_ISIs(ISI_index) == ISIdistributionMatrix);
    end

    load(B_simulations(simulation).name)
    B_ISIs = unique(ISIdistributionMatrix);

    for ISI_index=1:length(B_ISIs)
        cue_onsets = find(x==2);
        whereplot = [cue_onsets(ISIdistributionMatrix==B_ISIs(ISI_index)):cue_onsets(ISIdistributionMatrix==B_ISIs(ISI_index))+20];
        all_B_RPEs(ISI_index,:,simulation) = (results.rpe(whereplot))';
        number_ISIs_B(simulation,ISI_index) = sum(B_ISIs(ISI_index) == ISIdistributionMatrix);
    end

    load(C_simulations(simulation).name)
    C_ISIs = unique(ISIdistributionMatrix);

    for ISI_index=1:length(C_ISIs)
        cue_onsets = find(x==2);
        whereplot = [cue_onsets(ISIdistributionMatrix==C_ISIs(ISI_index)):cue_onsets(ISIdistributionMatrix==C_ISIs(ISI_index))+20];
        all_C_RPEs(ISI_index,:,simulation) = (results.rpe(whereplot))';
        number_ISIs_C(simulation,ISI_index) = sum(C_ISIs(ISI_index) == ISIdistributionMatrix);
    end

end

% Plot RPEs - set to Saline or SalB folder

subplot(3,2,1)
plot((sum(all_C_RPEs(1,:,:),3)')/n_simulations,'Color',[0.5 0.5 0.5])
hold on
plot((sum(all_B_RPEs(1,:,:),3)')/n_simulations,'Color',[0 0 0])
hold on
plot((sum(all_A90_RPEs(1,:,:),3)')/n_simulations,'Color',[0.6 0.4 1])

subplot(3,2,3)
plot((sum(all_C_RPEs(2:end,:,:),3)')/n_simulations,'Color',[0.5 0.5 0.5])
hold on
plot((sum(all_B_RPEs(2:end,:,:),3)')/n_simulations,'Color',[0 0 0])
hold on
plot((sum(all_A90_RPEs(2:end,:,:),3)')/n_simulations,'Color',[0.6 0.4 1])

subplot(3,2,5)
plot((sum(all_A100_RPEs(:,:,:),3)')/n_simulations,'Color',[0.6 0.4 1])

subplot(3,2,2)

for i=1:n_simulations
    temp = zeros(2,1);
    for j=2:length(C_ISIs)
    temp = [(all_B_RPEs(j,B_ISIs(j)+1,i))*number_ISIs_B(i,j)/sum(number_ISIs_B(i,2:end));(all_C_RPEs(j,C_ISIs(j)+1,i))*number_ISIs_C(i,j)/sum(number_ISIs_C(i,2:end))]+temp;
    end
    plot([1.2 2.8],temp,'Color',[0.6 0.6 0.6])
    hold on
end

subplot(3,2,4)
for i=1:n_simulations
    temp = (all_A90_RPEs(2:10,7:15,i));
    plot(1.2:0.2:2.8,diag(temp),'Color',[0.5 0.5 0.5])
    hold on
end

subplot(3,2,6)
for i=1:n_simulations
    temp = (all_A100_RPEs(1:9,7:15,i));
    plot(1.2:0.2:2.8,diag(temp),'Color',[0.5 0.5 0.5])
    hold on
end

%%
figure(1)
cueonsets=find(x==2);
whichISI=13;
cues=find((ISIdistributionMatrix==whichISI));
%cues=find(isnan(ISIdistributionMatrix));
lastcue=cues(1);
whereplot=cueonsets(lastcue):cueonsets(lastcue)+25;
for iterations=1%1:length(whereplot)
%plot(results_SalB.b(whereplot(i),:),'Color',[1-i/length(whereplot) i/length(whereplot) 1])
%plot(results.b(whereplot(iterations),:),'Color',[1-iterations/length(whereplot) iterations/length(whereplot) 1])
plot(results.value(whereplot));%,'Color',[1-iterations/length(whereplot) iterations/length(whereplot) 1])

%sum((results_SalB.b(i,:)))
hold on
% plot(results.rpe(whereplot(iterations)),'*','Color',[1-iterations/length(whereplot) iterations/length(whereplot) 1])
% hold on
end
ylim([0 1.2])
%HeatMap((results.b(whereplot,:)'),'ColorMap','redbluecmap')
%%
%% Value, valueprime, and RPE
%Code for Figure 6
    cueonsets=find(x==2);
    whichISI=10; %how long is the ISI for the trial type that you want to plot the value signal for? range: 5-13
    cueonsets=cueonsets(ISIdistributionMatrix==whichISI);
    %cueonsets=cueonsets(1); %  only look at trials after 2000 trials
    
    value=zeros(1,20);
    valueprime=zeros(1,20);
    rpe=zeros(1,20);
    for i=1:length(cueonsets)
        for j=1:20
            value(j)=results.value(cueonsets(i)+j-2)+value(j);
            valueprime(j)=results.value(cueonsets(i)+j-1)+valueprime(j);
            rpe(j)=results.rpe(cueonsets(i)+j-2)+rpe(j);
        end
    end
    
% plot value, value(t+1) and rpe
    subplot(3,1,1)
    plot(value/length(cueonsets),'k')
    ylim([0 1.5])
    hold on
    plot(valueprime/length(cueonsets),'Color',[0.5 0.5 0.5])
    title('Value [black] and Value(t+1) [grey]')
    
    subplot(3,1,2)
    plot((valueprime-value)/length(cueonsets),'k')
    ylim([-1 0.6])
    title('Value(t+1)-Value(t)')

    subplot(3,1,3)
    plot(rpe/length(cueonsets),'Color',[1-(whichISI-4)/9 (whichISI-4)/9 1])
    ylim([-0.5 0.7])
    title('TD error')
%%
figure(1)
bar(results_training.w(end,:))
xlim([0 31])
ylim([0 1.4])
%%
imagesc(T_blur)
%%
imagesc(O)