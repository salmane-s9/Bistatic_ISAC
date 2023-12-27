function [AoD_est,AoA_est,ToA_est] = Estim3D(y,f,q,Nt,Nr,N,K,dt,dr,lambda,delta_f,Mt,Mr,M)
% This method performs 3D ToA/AoA/AoD estimation when doppler is 0
%% Channel Estimation
HH = [];
for n = 1:N
    F = [];
    Y = [];
    for k = 1:K
        F = [F f{n,k}];
        Y = [Y y{n,k}];
    end
    H_est{n} = Y*F'*inv(F*F');
    HH = [HH H_est{n}(:)];
end

%% Sensing Estimation
c = @(tau) exp(-j*2*pi*[1:(N)].'*delta_f.*tau);

K_Nt = Nt - Mt + 1; % Mt <- Mp
K_Nr = Nr - Mr + 1; % Mr <- Np
K_N  = N  - M  + 1; 

% Step 1
for nt = 1:Nt
    x = HH([1:Nr] + (nt-1)*Nr,:); % Nr x N matrix
    x = x(:); % vectorize
    for m = 1:N
        xm = x([1:Nr] + (m-1)*Nr);
        for k = 1:K_Nr
            X{m}(:,k) = xm([1:Mr]+k-1);
        end
    end
    
    XX{nt} = [];
    for m = 1:M
        Xm = [];
        for k = 1:K_N
            Xm = [Xm X{k+m-1}];
        end
        XX{nt} = [XX{nt};Xm]; % Mr*M x K_Nr*K_N
    end

end
XXX = [];

for m = 1:Mt
    Xm = [];
    for k = 1:K_Nt
        Xm = [Xm XX{k+m-1}];
    end
    XXX=[XXX;Xm]; % Mt*Mr*M x K_Nt*K_Nr*K_N
end


% Step 2                                                
%  Mt*Mr*M x K_Nt*K_Nr*K_N
arr_l = [];
for kt = 1:K_Nt
    arr_l = [arr_l [[1:(K_Nr*(K_N-1))] + (kt-1)*K_Nr*(K_N)] ];
end
arr_r = arr_l + K_Nr;

X_l = XXX(:,arr_l);
X_r = XXX(:,arr_r);

% Step 3
[UU,SS,VV] = svd(X_l);
% Step 4+5
t_eig = eig(inv(SS(1:q,1:q))*UU(:,1:q)'*X_r*VV(:,1:q)); % 21a
% Step 6
ToA_est = sort(-(angle(t_eig)/(2*pi*delta_f)).'); % 27

% Step 7
C_est = c(ToA_est); % equation (33)
C_est_pinv = conj(C_est)*inv(C_est.'*conj(C_est)); % equation (32)
G_hat = HH*C_est_pinv;  % equation (31)

% Step 8
for ii = 1:q
    at_o_ar(:,ii) = G_hat(:,ii)/norm(G_hat(:,ii)); % equation (35)
end
% Step 9
TT(:,1) = kron(ones(Nt,1),[0:(Nr-1)].');
TT(:,2) = kron([0:(Nt-1)].',ones(Nr,1));
TT(:,3) = ones(Nt*Nr,1);
T_pinv = inv(TT'*TT)*TT';

for ii = 1:q
    e_i = T_pinv*unwrap(angle(at_o_ar(:,ii))); % top of equation (38)
    x1 = e_i(1)*lambda/(2*pi*dr);
    x2 = e_i(2)*lambda/(2*pi*dt);
    while x1 > 1
        x1 = x1 - 1;
    end
    while x1 < -1
        x1 = x1 + 1;
    end
    while x2 > 1
        x2 = x2 - 1;
    end
    while x2 < -1
        x2 = x2 + 1;
    end
    AoA_est(ii) = - asind(x1);
    AoD_est(ii) = - asind(x2);
end
