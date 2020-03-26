function T_blur = blur_matrices_inputWeberFraction(T,weber)

T_blur = T;
for i=1:29
    p_dist=T(i,1:29);

    epochs=0.2:0.2:5.8;
    theta = weber;
    fun1 = @(tau,time)(p_dist(find(epochs==tau))*1/(time*theta*sqrt(2*pi))*exp(-(((tau-time).^2)/(2*theta^2*time^2))));
    %fun1 = @(tau,time)(p_dist(find(epochs==tau))*1/(time*theta*sqrt(2*pi))*exp(-(((tau-time).^2)/(2*theta^2*2^4))));

    for a=1:29
        
        integral1=0;
        
        for j=1:29
            temp1=fun1(epochs(j),epochs(a));
            integral1=integral1+temp1;
        end
        
        timesteps(a)=integral1;
        
    end
    
    T_blur(i,1:29)=timesteps/sum(timesteps)*sum(p_dist);


end

end