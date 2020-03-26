function blurred_dist = blur_vector(dist,jitter)

blurred_dist=zeros(1,29);
for i=1:length(dist)
    blurred_dist = blurred_dist+normpdf(1:29,i,jitter)*dist(i)/sum(normpdf(1:29,i,jitter));
end

end