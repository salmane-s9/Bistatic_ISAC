% Figure 5:
% Performance comparison of MLP with the 2D Estimation algorithm 
% for both AoA and AoD angles for settings with two targets per peak
% Generating training data, 2D Estimator results and CRB bounds
% Classifying the number of targets in one peak.

% Creating Training Data
clc
clear all
close all

% ========= Channel Constants  ======================
c0              = 3e8;                                              % speed of light
fc              = 30e9;                                             % carrier frequency
lambda          = c0/fc;                                            % signal wavelength
d               = lambda/2;                                         % antenna spacing
gamma           = 1;                                                % binary variable indicating whether the direct link between base station and radar unit is there.

% ========== OFDM related Parameters ================
delta_f         = 15e3 * 2^6 ;                                      % OFDM subcarrier spacing
T               = 1 / delta_f;                                      % OFDM symbol duration
Tcp             = T / 4;                                            % cyclic OFDM prefix duration
Ts              = T + Tcp;                                          % total OFDM symbol duration
N               = 64;                                               % no. of OFDM subcarriers per symbol 
K               = 10;                                               % no. of OFDM symbols transmitted from BS
qam             = 1024;                                               % [4,16,64,..] QAM modulation order

% ========= SNR Range ===================

SNRdBTest = [-5, 5, 15, 25, 40];
nsnr_tests = length(SNRdBTest);
MC_ALT = 5000;
qTest = (1:1:5);
nq_tests = length(qTest);
for i_snr=1:nsnr_tests
    SNRdB = SNRdBTest(i_snr); 
    disp(['SNR: ', num2str(SNRdB), 'dB']);
    folderName = sprintf('./data/Bistatic_data_with_ToA_Nrtargets_classif/Bistatic_data_with_ToA_Nrtargets_classif_%d_dB/', SNRdB);
    if ~exist(folderName, 'dir')
        [status, msg, msgID] = mkdir(folderName);
    end
    for iq=1:nq_tests
        q = qTest(iq);
        disp(['Q: ', num2str(q), 'Targets']);
        for mc=1:MC_ALT
            clear GENDATA
            clear y
            clear data 
            clear f
        
            if mod(mc,100) == ~0 || mc==MC_ALT
                disp([num2str(mc-1),'/',num2str(MC_ALT)]);
            end
            % ========== Target Parameters ====================== 
            AoA_interval = [-5,5;25,35];
            AoD_interval = [-5,5;25,35];
            dist_interval = [3, 15];
            speed_interval = [0, 25];
    
            speed                  = 0*unifrnd(speed_interval(1),speed_interval(2), 1, q);                                      % velocity per Target (m/s)
    
            integer_delay = 3e8/(delta_f*N);        
            sum_distances = 5*integer_delay;
            distance_BS_Target = double(round(unifrnd(0, 0.8*sum_distances, 1, q), 3));
            distance_Target_radar = sum_distances - distance_BS_Target;            
            AnglesInterval = [0 75];
            step = (AnglesInterval(2) - AnglesInterval(1)) / q;
    
            for i=1:q
                AoA(i) = double(round(unifrnd((i-1)*step, i*step, 1, 1), 1));
                AoD(i) = double(round(unifrnd((i-1)*step, i*step, 1, 1), 1));
            end
    
            AoA              = AoA(1:q);                                        % Target AoAs relative to the radar
            AoD              = AoD(1:q);                                        % Target AoDs relative to the BS
            doppler_norm     = speed2dop(speed(1:q),lambda)*Ts;                 % normalized doppler per target (unitless)
            ToA_BS_Target    = range2time(distance_BS_Target(1:q), c0);         % delay between BS and targets (sec)
            ToA_Target_radar = range2time(distance_Target_radar(1:q), c0);      % delay between targets and radar unit (m)
            ToA_2W           = (ToA_BS_Target + ToA_Target_radar);                % two-way delay, i.e. the total ToA measured from source (BS) to sink (radar unit).
            PL_2W            = ones(1,q);
            
            AoA_BS_radar      = [60];   % AoA of BS w.r.t radar
            AoD_BS_radar      = [50];                                           % AoD of BS w.r.t radar
            distance_BS_radar = [0.5];                                          % distance b/w BS and radar
            ToA_BS_radar      = range2time(distance_BS_radar,c0);               % delay b/w BS and radar
            PL_BS_rad         = [1];
    
            % ========= Antenna configurations ===================
            Nt  = 8;                                                            % number of transmit antennas at the base station
            Nr  = 10;                                                           % number of receive antennas at the radar unit 
            dt  = lambda/2;                                                     % antenna spacing at the transmitter (BS) 
            dr  = lambda/2;                                                     % antenna spacing at the receiver (radar unit)
            at  = @(theta) exp(-j*2*pi*(dt/lambda)*[0:(Nt-1)].'*sind(theta));   % steering vector of the transmit array (at BS) following a uniform linear array (ULA) configuration
            ar  = @(theta) exp(-j*2*pi*(dr/lambda)*[0:(Nr-1)].'*sind(theta));   % steering vector of the receive array (at radar unit) following a uniform linear array (ULA) configuration
            At  = at(AoD);                                                      % transmit array manifold (this is a Nt x q matrix containing transmit steering vectors in its columns)
            Ar  = ar(AoA);                                                      % receive array manifold (this is a Nr x q matrix containing receive steering vectors in its columns)
            At0 = at(AoD_BS_radar);                                             % transmit array manifold due to the channel between BS and radar unit
            Ar0 = ar(AoA_BS_radar);                                             % receive array manifold due to the channel between BS and radar unit
            
            H_BS_radar  = (1/sqrt(2))*(randn(Nr,Nt) + j*randn(Nr,Nt));
            PP    = eye(Nr);
            % ========= Additional info
            plotEn = 0;
            %% Transmit signal
            
            for n = 1:N % loop over subcarriers
                for k = 1:K % loop over symbols
                    data{n,k}   = randi([0 qam - 1], Nt, 1);                        % generate random symbolic bits for QAM generation per symbol per subcarrier per Tx antenna
                    f{n,k}      = qammod(data{n,k}, qam, 'UnitAveragePower', true); % generate corresponding QAM info per symbol per subcarrier per Tx antenna
                end
            end
            Rf = zeros(Nt,Nt);
            for k1 = 1:size(f,1)
                for k2 = 1:size(f,2)
                    Rf = Rf + f{k1,k2}*f{k1,k2}';
                end
            end
            Rf = Rf/prod(size(f));
    
            %% Channel Generation
            initial_phase_offset = diag(exp(j*2*pi*randn(1,q)));
            initial_phase_offset = eye(q);
            for n = 1:N         % loop over OFDM subcarriers
                for k = 1:K     % loop over OFDM symbols
                    % BS -> radar
                    G0{n,k} = diag(PL_BS_rad.*                          ...             PL effect
                                  exp(-j*2*pi*n*delta_f.*ToA_BS_radar) ...              ToA effect
                                  ); 
                    H0{n,k} = Ar0*G0{n,k}*At0.';                                        % channel between BS and radar unit
            
                    % BS -> targets -> radar
                    G1{n,k} =    initial_phase_offset*...
                                  diag( PL_2W.*                          ...                 PL effect
                                  exp(-j*2*pi*n*delta_f.*ToA_2W).* ...                  ToA effect
                                  exp(+j*2*pi*k*Ts.*doppler_norm)  ...                  doppler effect 
                                 ); 
                    H1{n,k} = Ar*G1{n,k}*At.';                                          % channel due to targets
                    
                    % Overall channel
                    H{n,k} = gamma*H0{n,k} + H1{n,k};
                end 
            end
    
            %% Received signal 
            for n = 1:N         % loop over OFDM subcarriers
                for k = 1:K     % loop over OFDM symbols
                    Signal{n,k}  = H{n,k}*f{n,k}; % f is s in the manual
            
                    power_signal = sum(sum(abs(Signal{n,k}).^2))/(Nr*1);
                    temp_noise   = sqrtm(PP)*(randn(Nr,1)+j*randn(Nr,1))/sqrt(2);
                    power_noise  = sum(sum(abs(temp_noise).^2))/(Nr*1);
                    amp        = sqrt(power_signal/power_noise)/(10^(SNRdB/20));
                    Noise{n,k} = amp*temp_noise; % noise scale to attain the given SNR
                    %            signal        + noise
                    y{n,k}     = Signal{n,k} + Noise{n,k};
                end
            end
    
            % 2D estimation Multiple peaks
            eigBasedEn=1;
            Mt = round(Nt/2);
            Mr = round(Nr/2);
            [AoD_est_2D_peak,AoA_est_2D_peak, Ht_est_peak] = Estim2D_new_single_peak(y,f,q,Nt,Nr,N,K,dt,dr,lambda,at,eigBasedEn,Mt,Mr,gamma);
            
            GENDATA.GeneralParameters = struct('q',q,'mc',mc,'snr',SNRdB, 'Nr', Nr, 'Nt', Nt);
            GENDATA.Estimated2DParameters = struct('AoA_est_2D',AoA_est_2D_peak,'AoD_est_2D',AoD_est_2D_peak);
            GENDATA.TimeChannelEstimate = Ht_est_peak;
            GENDATA.TargetsParameters = struct('AoA',AoA,'AoD',AoD,'ToA_BS_Target', ToA_BS_Target, ...
                                                'distance_BS_Target', distance_BS_Target, ...
                                                'distance_Target_radar', distance_Target_radar, ...
                                                'ToA_Target_radar', ToA_Target_radar, 'doppler_norm', doppler_norm);
            
    
            save([folderName ,'GENDATA','_N_targets_',num2str(q),'_MC_', num2str(mc), '_SNR_',num2str(SNRdB), '_Nr_',num2str(Nr), '_Nt_',num2str(Nt), '.mat'],'GENDATA');
        end
    end
end

% Generating Test Data at various SNRs between -5 and 20

clc
clear all
close all

% ========= Channel Constants  ======================
c0              = 3e8;                                              % speed of light
fc              = 30e9;                                             % carrier frequency
lambda          = c0/fc;                                            % signal wavelength
d               = lambda/2;                                         % antenna spacing
gamma           = 1;                                                % binary variable indicating whether the direct link between base station and radar unit is there.

% ========== OFDM related Parameters ================
delta_f         = 15e3 * 2^6 ;                                      % OFDM subcarrier spacing
T               = 1 / delta_f;                                      % OFDM symbol duration
Tcp             = T / 4;                                            % cyclic OFDM prefix duration
Ts              = T + Tcp;                                          % total OFDM symbol duration
N               = 64;                                               % no. of OFDM subcarriers per symbol 
K               = 10;                                               % no. of OFDM symbols transmitted from BS
qam             = 1024;                                               % [4,16,64,..] QAM modulation order

% ========= SNR Range ===================
SNRdBTest = (-5:1:20);
nsnr_tests = length(SNRdBTest);
MC_ALT = 40; % Number of realisation per SNR and number of Targets
qTest = (1:1:5);
nq_tests = length(qTest);
for i_snr=1:nsnr_tests
    SNRdB = SNRdBTest(i_snr); 
    disp(['SNR: ', num2str(SNRdB), 'dB']);
    folderName = sprintf('./data/Bistatic_data_with_ToA_Nrtargets_classif_test/');
    if ~exist(folderName, 'dir')
        [status, msg, msgID] = mkdir(folderName);
    end
    for iq=1:nq_tests
        q = qTest(iq);
        disp(['Q: ', num2str(q), 'Targets']);
        for mc=1:MC_ALT
            clear GENDATA
            clear y
            clear data 
            clear f
        
            if mod(mc,100) == ~0 || mc==MC_ALT
                disp([num2str(mc-1),'/',num2str(MC_ALT)]);
            end
            % ========== Target Parameters ====================== 
            AoA_interval = [-5,5;25,35];
            AoD_interval = [-5,5;25,35];
            dist_interval = [3, 15];
            speed_interval = [0, 25];
    
            speed                  = 0*unifrnd(speed_interval(1),speed_interval(2), 1, q);                                      % velocity per Target (m/s)
    
            integer_delay = 3e8/(delta_f*N);        
            sum_distances = 5*integer_delay;
            distance_BS_Target = double(round(unifrnd(0, 0.8*sum_distances, 1, q), 3));
            distance_Target_radar = sum_distances - distance_BS_Target;

            AnglesInterval = [0 75];
            step = (AnglesInterval(2) - AnglesInterval(1)) / q;
    
            for i=1:q
                AoA(i) = double(round(unifrnd((i-1)*step, i*step, 1, 1), 1));
                AoD(i) = double(round(unifrnd((i-1)*step, i*step, 1, 1), 1));
            end
    
            AoA              = AoA(1:q);                                        % Target AoAs relative to the radar
            AoD              = AoD(1:q);                                        % Target AoDs relative to the BS
            doppler_norm     = speed2dop(speed(1:q),lambda)*Ts;                 % normalized doppler per target (unitless)
            ToA_BS_Target    = range2time(distance_BS_Target(1:q), c0);         % delay between BS and targets (sec)
            ToA_Target_radar = range2time(distance_Target_radar(1:q), c0);      % delay between targets and radar unit (m)
            ToA_2W           = (ToA_BS_Target + ToA_Target_radar);                % two-way delay, i.e. the total ToA measured from source (BS) to sink (radar unit).
            PL_2W            = ones(1,q);
            
            AoA_BS_radar      = [60];   % AoA of BS w.r.t radar
            AoD_BS_radar      = [50];                                           % AoD of BS w.r.t radar
            distance_BS_radar = [0.5];                                          % distance b/w BS and radar
            ToA_BS_radar      = range2time(distance_BS_radar,c0);               % delay b/w BS and radar
            PL_BS_rad         = [1];
    
            % ========= Antenna configurations ===================
            Nt  = 8;                                                            % number of transmit antennas at the base station
            Nr  = 10;                                                           % number of receive antennas at the radar unit 
            dt  = lambda/2;                                                     % antenna spacing at the transmitter (BS) 
            dr  = lambda/2;                                                     % antenna spacing at the receiver (radar unit)
            at  = @(theta) exp(-j*2*pi*(dt/lambda)*[0:(Nt-1)].'*sind(theta));   % steering vector of the transmit array (at BS) following a uniform linear array (ULA) configuration
            ar  = @(theta) exp(-j*2*pi*(dr/lambda)*[0:(Nr-1)].'*sind(theta));   % steering vector of the receive array (at radar unit) following a uniform linear array (ULA) configuration
            At  = at(AoD);                                                      % transmit array manifold (this is a Nt x q matrix containing transmit steering vectors in its columns)
            Ar  = ar(AoA);                                                      % receive array manifold (this is a Nr x q matrix containing receive steering vectors in its columns)
            At0 = at(AoD_BS_radar);                                             % transmit array manifold due to the channel between BS and radar unit
            Ar0 = ar(AoA_BS_radar);                                             % receive array manifold due to the channel between BS and radar unit
            
            H_BS_radar  = (1/sqrt(2))*(randn(Nr,Nt) + j*randn(Nr,Nt));
            PP    = eye(Nr);
            % ========= Additional info
            plotEn = 0;
            %% Transmit signal
            
            for n = 1:N % loop over subcarriers
                for k = 1:K % loop over symbols
                    data{n,k}   = randi([0 qam - 1], Nt, 1);                        % generate random symbolic bits for QAM generation per symbol per subcarrier per Tx antenna
                    f{n,k}      = qammod(data{n,k}, qam, 'UnitAveragePower', true); % generate corresponding QAM info per symbol per subcarrier per Tx antenna
                end
            end
            Rf = zeros(Nt,Nt);
            for k1 = 1:size(f,1)
                for k2 = 1:size(f,2)
                    Rf = Rf + f{k1,k2}*f{k1,k2}';
                end
            end
            Rf = Rf/prod(size(f));
    
            %% Channel Generation
            initial_phase_offset = eye(q);
            for n = 1:N         % loop over OFDM subcarriers
                for k = 1:K     % loop over OFDM symbols
                    % BS -> radar
                    G0{n,k} = diag(PL_BS_rad.*                          ...             PL effect
                                  exp(-j*2*pi*n*delta_f.*ToA_BS_radar) ...              ToA effect
                                  ); 
                    H0{n,k} = Ar0*G0{n,k}*At0.';                                        % channel between BS and radar unit
            
                    % BS -> targets -> radar
                    G1{n,k} =    initial_phase_offset*...
                                  diag( PL_2W.*                          ...                 PL effect
                                  exp(-j*2*pi*n*delta_f.*ToA_2W).* ...                  ToA effect
                                  exp(+j*2*pi*k*Ts.*doppler_norm)  ...                  doppler effect 
                                 ); 
                    H1{n,k} = Ar*G1{n,k}*At.';                                          % channel due to targets
                    
                    % Overall channel
                    H{n,k} = gamma*H0{n,k} + H1{n,k};
                end 
            end
    
            %% Received signal 
            for n = 1:N         % loop over OFDM subcarriers
                for k = 1:K     % loop over OFDM symbols
                    Signal{n,k}  = H{n,k}*f{n,k}; % f is s in the manual
            
                    power_signal = sum(sum(abs(Signal{n,k}).^2))/(Nr*1);
                    temp_noise   = sqrtm(PP)*(randn(Nr,1)+j*randn(Nr,1))/sqrt(2);
                    power_noise  = sum(sum(abs(temp_noise).^2))/(Nr*1);
                    amp        = sqrt(power_signal/power_noise)/(10^(SNRdB/20));
                    Noise{n,k} = amp*temp_noise; % noise scale to attain the given SNR
                    %            signal        + noise
                    y{n,k}     = Signal{n,k} + Noise{n,k};
                end
            end

            % 2D estimation Multiple peaks
            eigBasedEn=1;
            Mt = round(Nt/2);
            Mr = round(Nr/2);
            [AoD_est_2D_peak,AoA_est_2D_peak, Ht_est_peak] = Estim2D_new_single_peak(y,f,q,Nt,Nr,N,K,dt,dr,lambda,at,eigBasedEn,Mt,Mr,gamma);
            
            % Saving test data             
            GENDATA.GeneralParameters = struct('q',q,'mc',mc,'snr',SNRdB, 'Nr', Nr, 'Nt', Nt);
            GENDATA.Estimated2DParameters = struct('AoA_est_2D',AoA_est_2D_peak,'AoD_est_2D',AoD_est_2D_peak);
            GENDATA.TimeChannelEstimate = Ht_est_peak;
            GENDATA.TargetsParameters = struct('AoA',AoA,'AoD',AoD,'ToA_BS_Target', ToA_BS_Target, ...
                                                'distance_BS_Target', distance_BS_Target, ...
                                                'distance_Target_radar', distance_Target_radar, ...
                                                'ToA_Target_radar', ToA_Target_radar, 'doppler_norm', doppler_norm);
            
            
            save([folderName ,'GENDATA','_N_targets_',num2str(q),'_MC_', num2str(mc), '_SNR_',num2str(SNRdB), '_Nr_',num2str(Nr), '_Nt_',num2str(Nt), '.mat'],'GENDATA');
        end
    end
end