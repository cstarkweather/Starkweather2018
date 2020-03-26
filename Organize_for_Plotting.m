function [phasicRPE tonicRPE omissionRPE] = Organize_for_Plotting(simulations)

omissionRPE=zeros(1,30);
tonicRPE=zeros(1,30);
load(simulations(1).name)
unique_ISIs=unique(ISIdistributionMatrix(ISIdistributionMatrix>0));
phasicRPE=zeros(length(unique_ISIs),3,length(simulations));

for simulation=1:length(simulations)
    
load(simulations(simulation).name)
omission_ISI_indices=[];
unique_ISIs=unique(ISIdistributionMatrix(ISIdistributionMatrix>0));
unique_blurred_ISIs=unique(blurred_ISIdistributionMatrix(blurred_ISIdistributionMatrix>0));
x=[ones(6,1);x];
cueonsets = find(x==2);
results.rpe=[zeros(6,1); results.rpe];

noise = 12;
results.rpe=awgn(results.rpe,noise);

omission_ISI_indices = find(ISIdistributionMatrix==0);

temp=zeros(1,30);

if length(omission_ISI_indices)>0
for i=1:length(omission_ISI_indices)
    temp=results.rpe(cueonsets(omission_ISI_indices(i))-5:cueonsets(omission_ISI_indices(i))+24)'+temp;
end
else
end

if length(omission_ISI_indices)>0
omissionRPE = (temp/length(omission_ISI_indices)) + omissionRPE;
else
end


%%
temp_tonic=zeros(1,30);
bridge_mat=zeros(length(unique_blurred_ISIs),30);
for i=1:length(unique_blurred_ISIs)
    trial_indices = find(blurred_ISIdistributionMatrix==unique_blurred_ISIs(i));
    rewardonsets = cueonsets(trial_indices)'+blurred_ISIdistributionMatrix(blurred_ISIdistributionMatrix==unique_blurred_ISIs(i));
    
    bridge=zeros(1,30);
    
    for j=1:length(trial_indices)
    temp_tonic(1:min(unique_blurred_ISIs)+5)=results.rpe(cueonsets(trial_indices(j))-5:cueonsets(trial_indices(j))+min(unique_blurred_ISIs)-1)'+temp_tonic(1:min(unique_blurred_ISIs)+5);
    temp_tonic(max(unique_blurred_ISIs)+6:end)=results.rpe(cueonsets(trial_indices(j))+max(unique_blurred_ISIs)+1:cueonsets(trial_indices(j))+25)'+temp_tonic(max(unique_blurred_ISIs)+6:end);
    t_bridge=results.rpe(cueonsets(trial_indices(j))+min(unique_blurred_ISIs):cueonsets(trial_indices(j))+unique_blurred_ISIs(i)-1)';
    bridge(1:length(t_bridge))=t_bridge+bridge(1:length(t_bridge));
    end

    bridge_mat(i,:)=bridge/length(trial_indices);

end


for i=1:size(bridge_mat,2)
    final_bridge(i) = sum(bridge_mat(:,i))/length(find(bridge_mat(:,i)));
end

final_bridge=final_bridge(~isnan(final_bridge));

temp_tonic(1:min(unique_blurred_ISIs)+5)=temp_tonic(1:min(unique_blurred_ISIs)+5)/length(find(blurred_ISIdistributionMatrix));
temp_tonic(min(unique_blurred_ISIs)+6:max(unique_blurred_ISIs)+5)=final_bridge;
temp_tonic(max(unique_blurred_ISIs)+6:end)=temp_tonic(max(unique_blurred_ISIs)+6:end)/length(find(blurred_ISIdistributionMatrix));


tonicRPE = tonicRPE + temp_tonic;

for i=1:length(unique_ISIs)
    trial_indices = find(ISIdistributionMatrix==unique_ISIs(i));
    rewardonsets = cueonsets(trial_indices)'+blurred_ISIdistributionMatrix(ISIdistributionMatrix==unique_ISIs(i));
    phasicRPE(i,:,simulation) = ([temp_tonic(unique_ISIs(i)+5) sum(results.rpe(rewardonsets))/length(rewardonsets) sum(results.rpe(rewardonsets+1))/length(rewardonsets)]);
end

end

tonicRPE = tonicRPE/length(simulations);
omissionRPE = omissionRPE/length(simulations);
end