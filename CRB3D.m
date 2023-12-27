function[CRB1] = CRB3D(alpha,f,sigma,Ar,At,Ctau,Dr,Dt,Dtau,q,Nr)
% This method computes the Cramér–Rao bound for 
% all the estimated parameters

s = f;
[N,K] = size(s);
Nt = size(At,1);

T_sig2_sig2 = Nr*N*K/sigma^4;
T_sig2_theta = zeros(1,q);
T_sig2_phi = zeros(1,q);
T_sig2_tau = zeros(1,q);
T_sig2_alphaB = zeros(1,q);
T_sig2_alphaT = zeros(1,q);

T_theta_theta = zeros(q,q);
T_theta_phi   = zeros(q,q);
T_theta_tau  = zeros(q,q);
T_theta_alphaB= zeros(q,q);
T_theta_alphaT= zeros(q,q);


T_phi_phi    = zeros(q,q);
T_phi_tau    = zeros(q,q);
T_phi_alphaB = zeros(q,q);
T_phi_alphaT = zeros(q,q);

T_tau_tau    = zeros(q,q);
T_tau_alphaB = zeros(q,q);
T_tau_alphaT = zeros(q,q);

T_alphaB_alphaB = zeros(q,q);
T_alphaB_alphaT = zeros(q,q);

T_alphaT_alphaT = zeros(q,q);



for ii = 1:q
    for jj = 1:q
        for nn = 1:N
            for kk = 1:K
                % w.r.t theta
                T_theta_theta(ii,jj)    = T_theta_theta(ii,jj)      + (2/sigma^2)*real( s{nn,kk}'*(alpha(ii)*Ctau(nn,ii)*Dr(:,ii)*At(:,ii).')'*(alpha(jj)*Ctau(nn,jj)*Dr(:,jj)*At(:,jj).')*s{nn,kk}  );
                T_theta_phi(ii,jj)      = T_theta_phi(ii,jj)        + (2/sigma^2)*real( s{nn,kk}'*(alpha(ii)*Ctau(nn,ii)*Dr(:,ii)*At(:,ii).')'*(alpha(jj)*Ctau(nn,jj)*Ar(:,jj)*Dt(:,jj).')*s{nn,kk}  );
                T_theta_tau(ii,jj)      = T_theta_tau(ii,jj)        + (2/sigma^2)*real( s{nn,kk}'*(alpha(ii)*Ctau(nn,ii)*Dr(:,ii)*At(:,ii).')'*(alpha(jj)*Dtau(nn,jj)*Ar(:,jj)*At(:,jj).')*s{nn,kk}  );
                T_theta_alphaB(ii,jj)   = T_theta_alphaB(ii,jj)     + (2/sigma^2)*real( s{nn,kk}'*(alpha(ii)*Ctau(nn,ii)*Dr(:,ii)*At(:,ii).')'*(          Ctau(nn,jj)*Ar(:,jj)*At(:,jj).')*s{nn,kk}  );
                T_theta_alphaT(ii,jj)   = T_theta_alphaT(ii,jj)     + (2/sigma^2)*real( s{nn,kk}'*(alpha(ii)*Ctau(nn,ii)*Dr(:,ii)*At(:,ii).')'*(        j*Ctau(nn,jj)*Ar(:,jj)*At(:,jj).')*s{nn,kk}  );
                % w.r.t phi
                T_phi_phi(ii,jj)        = T_phi_phi(ii,jj)          + (2/sigma^2)*real( s{nn,kk}'*(alpha(ii)*Ctau(nn,ii)*Ar(:,ii)*Dt(:,ii).')'*(alpha(jj)*Ctau(nn,jj)*Ar(:,jj)*Dt(:,jj).')*s{nn,kk}  );
                T_phi_tau(ii,jj)        = T_phi_tau(ii,jj)          + (2/sigma^2)*real( s{nn,kk}'*(alpha(ii)*Ctau(nn,ii)*Ar(:,ii)*Dt(:,ii).')'*(alpha(jj)*Dtau(nn,jj)*Ar(:,jj)*At(:,jj).')*s{nn,kk}  );
                T_phi_alphaB(ii,jj)     = T_phi_alphaB(ii,jj)       + (2/sigma^2)*real( s{nn,kk}'*(alpha(ii)*Ctau(nn,ii)*Ar(:,ii)*Dt(:,ii).')'*(          Ctau(nn,jj)*Ar(:,jj)*At(:,jj).')*s{nn,kk}  );
                T_phi_alphaT(ii,jj)     = T_phi_alphaT(ii,jj)       + (2/sigma^2)*real( s{nn,kk}'*(alpha(ii)*Ctau(nn,ii)*Ar(:,ii)*Dt(:,ii).')'*(        j*Ctau(nn,jj)*Ar(:,jj)*At(:,jj).')*s{nn,kk}  );
                % w.r.t tau
                T_tau_tau(ii,jj)        = T_tau_tau(ii,jj)          + (2/sigma^2)*real( s{nn,kk}'*(alpha(ii)*Dtau(nn,ii)*Ar(:,ii)*At(:,ii).')'*(alpha(jj)*Dtau(nn,jj)*Ar(:,jj)*At(:,jj).')*s{nn,kk}  );
                T_tau_alphaB(ii,jj)     = T_tau_alphaB(ii,jj)       + (2/sigma^2)*real( s{nn,kk}'*(alpha(ii)*Dtau(nn,ii)*Ar(:,ii)*At(:,ii).')'*(          Ctau(nn,jj)*Ar(:,jj)*At(:,jj).')*s{nn,kk}  );
                T_tau_alphaT(ii,jj)     = T_tau_alphaT(ii,jj)       + (2/sigma^2)*real( s{nn,kk}'*(alpha(ii)*Dtau(nn,ii)*Ar(:,ii)*At(:,ii).')'*(        j*Ctau(nn,jj)*Ar(:,jj)*At(:,jj).')*s{nn,kk}  );
                % w.r.t alphaB
                T_alphaB_alphaB(ii,jj)  = T_alphaB_alphaB(ii,jj)    + (2/sigma^2)*real( s{nn,kk}'*(          Ctau(nn,ii)*Ar(:,ii)*At(:,ii).')'*(          Ctau(nn,jj)*Ar(:,jj)*At(:,jj).')*s{nn,kk}  );
                T_alphaB_alphaT(ii,jj)  = T_alphaB_alphaT(ii,jj)    + (2/sigma^2)*real( s{nn,kk}'*(          Ctau(nn,ii)*Ar(:,ii)*At(:,ii).')'*(        j*Ctau(nn,jj)*Ar(:,jj)*At(:,jj).')*s{nn,kk}  );
                % w.r.t alphaT
                T_alphaT_alphaT(ii,jj)  = T_alphaT_alphaT(ii,jj)    + (2/sigma^2)*real( s{nn,kk}'*(        j*Ctau(nn,ii)*Ar(:,ii)*At(:,ii).')'*(        j*Ctau(nn,jj)*Ar(:,jj)*At(:,jj).')*s{nn,kk}  );

            end
        end
    end
end

T = [T_sig2_sig2 T_sig2_theta T_sig2_phi T_sig2_tau T_sig2_alphaB T_sig2_alphaT;
    T_sig2_theta.' T_theta_theta T_theta_phi T_theta_tau T_theta_alphaB T_theta_alphaT;
    T_sig2_phi.' T_theta_phi.' T_phi_phi T_phi_tau T_phi_alphaB T_phi_alphaT;
    T_sig2_tau.' T_theta_tau.' T_phi_tau.' T_tau_tau T_tau_alphaB T_tau_alphaT;
    T_sig2_alphaB.' T_theta_alphaB.' T_phi_alphaB.' T_tau_alphaB.' T_alphaB_alphaB T_alphaB_alphaT;
    T_sig2_alphaT.' T_theta_alphaT.' T_phi_alphaT.' T_tau_alphaT.' T_alphaB_alphaT.' T_alphaT_alphaT];

CRB = inv(T);

crb = diag(CRB);
CRB1.sig2    = crb(1);
CRB1.theta   = crb([1:q]+1);
CRB1.phi     = crb([1:q]+1+q);
CRB1.tau     = crb([1:q]+1+q+q);
CRB1.alphaB  = crb([1:q]+1+q+q+q);
CRB1.alphaT  = crb([1:q]+1+q+q+q+q);



