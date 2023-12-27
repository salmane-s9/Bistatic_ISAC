function [AoD_est_new,AoA_est_new,Ht_est] = Estim2D_new_single_peak(y,f,q,Nt,Nr,N,K,dt,dr,lambda,at,eigBasedEn,Mt,Mr,gamma)
% This method performs 2D AoA/AoD estimation assuming that ToA of all
% targets are distinguishable
q_aux = q;
q = 1;

%% Channel estimation
% step1: estimate the channel
H_est = [];
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

% step2: find the peaks for each target
index_hist = zeros(1,N);

for nt=1:Nt
    for nr=1:Nr
        H = zeros(N,1);
        for n = 1:N
            H_aux = H_est{n};
            H(n) = H_aux(nr,nt);
        end

        Ht = ifft(H); 

        if gamma == 1
            Ht = Ht(2:end);
        end

        [peaks, peakIndices] = findpeaks(abs(Ht));
        [~,index] = sort(abs(peaks),'descend'); 
        index_hist(peakIndices(index(1:q))) = index_hist(peakIndices(index(1:q)))+1;


    end
end

[~,indexes_found] = sort(index_hist,'descend'); 
indexes_found = indexes_found(1:q);

% step3: collect the peak for each target
Ht_est = zeros(Nr,Nt,q); % cannel in time domain
% collect peek in time domain
for nt=1:Nt
    for nr=1:Nr
        H = zeros(N,1);
        for n = 1:N
            H_aux = H_est{n};
            H(n) = H_aux(nr,nt);
        end

        Ht = ifft(H); 

        if gamma == 1
            Ht = Ht(2:end);
        end
        
        Ht_est(nr,nt,:) = Ht(indexes_found);

    end
end


q = q_aux;

% Ht_est is the input for the Single Target estimator sweeping over i in Ht_est(:,:,i) 
x = Ht_est(:);

% Nt <- M
% Nr <- N
K_Nt = Nt - Mt + 1; % Mt <- Mp
K_Nr = Nr - Mr + 1; % Mr <- Np

%% Algo 1
for m = 1:Nt
    xm = x([1:Nr] + (m-1)*Nr);
    for k = 1:K_Nr
        X{m}(:,k) = xm([1:Mr]+k-1);
    end
end


XX = [];
for m = 1:Mt
    Xm = [];
    for k = 1:K_Nt
        Xm = [Xm X{k+m-1}];
    end
    XX = [XX;Xm];
end

% Step 2
X_l = XX(:,1:(K_Nr*(K_Nt-1)));
X_r = XX(:,(K_Nr+1):(K_Nr*K_Nt));
if eigBasedEn
    % Step 3 - eigen based solution
    [UU,SS,VV] = svd(X_l);
    t_eig = eig(inv(SS(1:q,1:q))*UU(:,1:q)'*X_r*VV(:,1:q)); % 21a
    AoD_est = -asind(angle(t_eig)*lambda/(2*pi*dt)).'; % 27
else
    % Step 3 - 1D search over eigenvalues
    P  =[];
    thetaAxis = [-90:1:90];
    for theta = thetaAxis
        ll = exp(-j*2*pi*(dt/lambda)*sind(theta));
        XrmXl = X_r - ll*X_l;
        eigsXrmXl = svd(XrmXl);
        e= sort(eigsXrmXl,'descend');
        e3 = e(3);
        P  =[P e3];
    end
    f_spectrum = 1./P;
    if 0
        plot(thetaAxis,f_spectrum)
    end
    [thePeaks,idx] = findpeaks(f_spectrum);
    [peaks_srted,idx2] = sort(thePeaks,'descend');

    AoD_est = thetaAxis(idx(idx2(1:q)));
end

At_est = at(AoD_est); % equation (33)

% Step 4
Z = [];
for m = 1:Nt
    Z(:,m) = x([1:Nr] + (m-1)*Nr);
end
At_est_pinv = conj(At_est)*inv(At_est.'*conj(At_est)); % equation (32)
G_hat = Z*At_est_pinv;  % equation (31)
% Step 5
for ii = 1:q
    ar_est(:,ii) = G_hat(:,ii)/norm(G_hat(:,ii)); % equation (35)
end
% Step 6
TT(:,1) = [0:(Nr-1)].';
TT(:,2) = ones(Nr,1);
T_pinv = inv(TT'*TT)*TT';
for ii = 1:q
    e_i = T_pinv*unwrap(angle(ar_est(:,ii))); % top of equation (38)
    AoA_est(ii) = - asind(e_i(1)*lambda/(2*pi*dr));
end

AoA_est_new = AoA_est;
AoD_est_new = AoD_est;