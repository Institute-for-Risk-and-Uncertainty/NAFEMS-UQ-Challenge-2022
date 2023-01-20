%% Challenge Problem 2:
clc; clear; 

%% Compute statistics of Log-evidence:

% Log-evidence statistics for Beta distribution assumption:
load('Cluster_NAFEMS_Problem_2a.mat', 'TEMCMC1_logevi', 'TEMCMC2_logevi', 'TEMCMC3_logevi')
mean_beta = mean([TEMCMC1_logevi, TEMCMC2_logevi, TEMCMC3_logevi]);
std_beta = std([TEMCMC1_logevi, TEMCMC2_logevi, TEMCMC3_logevi]);

% Log-evidence statistics for Normal distribution assumption:
load('Cluster_NAFEMS_Problem_2b.mat', 'TEMCMC1_logevi', 'TEMCMC2_logevi', 'TEMCMC3_logevi')
mean_norm = mean([TEMCMC1_logevi, TEMCMC2_logevi, TEMCMC3_logevi]);
std_norm = std([TEMCMC1_logevi, TEMCMC2_logevi, TEMCMC3_logevi]);

% Plot graph of the results:
figure;
fac = 1; % std factor
hold on; box on; grid on;
y_delta = std_beta; % error in the positive/negative y-direction
errorbar([1:length(mean_beta)], mean_beta, y_delta, y_delta, 'sr','MarkerSize',5,...
    'MarkerEdgeColor','red','MarkerFaceColor','red','linewidth',1.5);
y_delta = std_norm; % error in the positive/negative y-direction
errorbar([1:length(mean_norm)], mean_norm, y_delta, y_delta, 'sb','MarkerSize',5,...
    'MarkerEdgeColor','blue','MarkerFaceColor','blue','linewidth',1.5);
xlim([0.8 3.2]); xticks([1 2 3]); set(groot,'defaultAxesTickLabelInterpreter','latex'); xticklabels({'$D_p$', '$\epsilon$', '$L$'}); ylabel('Log model evidence'); set(gca, 'Fontsize', 18)
legend('Scaled Beta distribution', 'Normal distribution', 'linewidth', 2, 'location', 'southeast'); 

%% Define the Data set and key parameters:

Dp_data = [0.0032, 0.0039, 0.0037, 0.0035, 0.0031, 0.0040, 0.0038, 0.0038, 0.0040, 0.0037]; % [m]
epsilon_data = [0.375, 0.347, 0.329, 0.352, 0.388, 0.419, 0.404, 0.394, 0.352, 0.370]; % [~] dimensionless
L_data = [2.86, 3.13, 3.08, 3.12, 2.94, 2.90, 2.80, 3.05, 3.02, 3.04]; % [m]
rho = 1.225;         % [kg/m^3] Density of fluid air
mu_0 = 1.81e-05;     % [kg/(m.s)] Dynamic viscosity of fluid
vs_min = 0.35;       % [m/s] Minimum fluid velocity
delta_p_lim = 15250; % [Pa] Pressure drop limit

figure;
subplot(2,2,1)
hold on; box on; grid on;
[yp, xp] = ecdf(Dp_data); stairs(xp, yp, 'r', 'linewidth', 2)
xlabel('$D_p$ $[m]$', 'Interpreter', 'latex'); ylabel('CDF value'); 
title('$D_p$ Data', 'Interpreter', 'latex'); set(gca, 'Fontsize', 18)
subplot(2,2,2)
hold on; box on; grid on;
[yp, xp] = ecdf(epsilon_data); stairs(xp, yp, 'r', 'linewidth', 2)
xlabel('$\varepsilon$', 'Interpreter', 'latex'); ylabel('CDF value'); 
title('$\varepsilon$ Data', 'Interpreter', 'latex'); set(gca, 'Fontsize', 18)
subplot(2,2,3)
hold on; box on; grid on;
[yp, xp] = ecdf(L_data); stairs(xp, yp, 'r', 'linewidth', 2)
xlabel('$L$ $[m]$', 'Interpreter', 'latex'); ylabel('CDF value'); 
title('$L$ Data', 'Interpreter', 'latex'); set(gca, 'Fontsize', 18)

%% Define the Pressure-drop model Delta Pressure:

model = @(Dp,e,L) (((150*mu_0*L)./(Dp).^2).*(((1-e).^2)./e.^3).*vs_min) + ...
                  (((1.75.*L.*rho)./Dp).*((1-e)./e.^3).*vs_min);
              
%% Define the parameters for Bayesian model updating:
Nsamp = 100;                                 % No. of samples to obtain from the Normal distribution Function
Nsamples = 1000;                             % No. of samples to obtain from the posterior

%% Bayesian model update to infer the hyper-parameters of the Normal distribution for Dp:

bounds1 = [0.001, 0.005]; bounds2 = [0.0001, 0.0008]; 
prior_pdf_mu = @(x) unifpdf(x, bounds1(1), bounds1(2)); prior_pdf_sigma = @(x) unifpdf(x, bounds2(1), bounds2(2)); 
prior_pdf = @(x) prior_pdf_mu(x(:,1)) .* prior_pdf_sigma(x(:,2));
prior_rnd = @(N) [unifrnd(bounds1(1), bounds1(2), N, 1), unifrnd(bounds2(1), bounds2(2), N, 1)];

rnd = @(x, N) normrnd(x(:,1), x(:,2), N, 1); % Normal random number generator
width = 5e-5;       % Width parameter of the loglikelihood function
logL = @(x) loglikelihood(x, Dp_data, width, Nsamp, rnd);

tic;
TEMCMC1 = TEMCMCsampler('nsamples',Nsamples,'loglikelihood',logL,...
                        'priorpdf',prior_pdf,'priorrnd',prior_rnd);
timeTEMCMC1 = toc;
TEMCMC1_samples = TEMCMC1.samples;
fprintf('Time elapsed is for the TEMCMC sampler: %f \n',timeTEMCMC1)

%% Bayesian model update to infer the hyper-parameters of the Normal distribution for Epsilon:

bounds1 = [0.01, 0.50]; bounds2 = [0.001, 0.08]; 
prior_pdf_mu = @(x) unifpdf(x, bounds1(1), bounds1(2)); prior_pdf_sigma = @(x) unifpdf(x, bounds2(1), bounds2(2)); 
prior_pdf = @(x) prior_pdf_mu(x(:,1)) .* prior_pdf_sigma(x(:,2));
prior_rnd = @(N) [unifrnd(bounds1(1), bounds1(2), N, 1), unifrnd(bounds2(1), bounds2(2), N, 1)];

rnd = @(x, N) normrnd(x(:,1), x(:,2), N, 1); % Normal random number generator
width = 5e-3;       % Width parameter of the loglikelihood function
logL = @(x) loglikelihood(x, epsilon_data, width, Nsamp, rnd);

tic;
TEMCMC2 = TEMCMCsampler('nsamples',Nsamples,'loglikelihood',logL,...
                        'priorpdf',prior_pdf,'priorrnd',prior_rnd);
timeTEMCMC2 = toc;
TEMCMC2_samples = TEMCMC2.samples;
fprintf('Time elapsed is for the TEMCMC sampler: %f \n',timeTEMCMC2)

%% Bayesian model update to infer the hyper-parameters of the SDF for L:

bounds1 = [0.01, 200]; bounds2 = [0.01, 100]; 
prior_pdf_1 = @(x) unifpdf(x, bounds1(1), bounds1(2)); prior_pdf_2 = @(x) unifpdf(x, bounds2(1), bounds2(2)); 
prior_pdf = @(x) prior_pdf_1(x(:,1)) .* prior_pdf_2(x(:,2));
prior_rnd = @(N) [unifrnd(bounds1(1), bounds1(2), N, 1), unifrnd(bounds2(1), bounds2(2), N, 1)];

rnd = @(x, N) 5.000.*betarnd(x(:,1), x(:,2), N, 1);
width = 5e-2;       % Width parameter of the loglikelihood function
logL = @(x) loglikelihood(x, L_data, width, Nsamp, rnd);

tic;
TEMCMC3 = TEMCMCsampler('nsamples',Nsamples,'loglikelihood',logL,...
                        'priorpdf',prior_pdf,'priorrnd',prior_rnd);
timeTEMCMC3 = toc;
TEMCMC3_samples = TEMCMC3.samples;
fprintf('Time elapsed is for the TEMCMC sampler: %f \n',timeTEMCMC3)

%% Save the simulation data:
save('NAFEMS_Problem_2', 'TEMCMC1', 'timeTEMCMC1', 'TEMCMC2', 'timeTEMCMC2', 'TEMCMC3', 'timeTEMCMC3')

%% Plot histograms:

samp1 = TEMCMC1.samples; samp2 = TEMCMC2.samples; samp3 = TEMCMC3.samples; 
nbin = 50; % No. of bins for histogram

figure; 
subplot(2,3,1)
hold on; box on; grid on;
histogram(samp1(:,1), nbin)
xlabel('$\mu_{D_p}$ $[m]$', 'Interpreter', 'latex'); ylabel('Count'); set(gca, 'Fontsize', 18)
subplot(2,3,2)
hold on; box on; grid on;
histogram(samp2(:,1), nbin)
xlabel('$\mu_{\varepsilon}$', 'Interpreter', 'latex'); ylabel('Count'); set(gca, 'Fontsize', 18)
subplot(2,3,3)
hold on; box on; grid on;
histogram(samp3(:,1), nbin)
xlabel('$\alpha_{L}$ $[m]$', 'Interpreter', 'latex'); ylabel('Count'); set(gca, 'Fontsize', 18)
subplot(2,3,4)
hold on; box on; grid on;
histogram(samp1(:,2), nbin)
xlabel('$\sigma_{D_p}$ $[m]$', 'Interpreter', 'latex'); ylabel('Count'); set(gca, 'Fontsize', 18)
subplot(2,3,5)
hold on; box on; grid on;
histogram(samp2(:,2), nbin)
xlabel('$\sigma_{\varepsilon}$', 'Interpreter', 'latex'); ylabel('Count'); set(gca, 'Fontsize', 18)
subplot(2,3,6)
hold on; box on; grid on;
histogram(samp3(:,2), nbin)
xlabel('$\beta_{L}$ $[m]$', 'Interpreter', 'latex'); ylabel('Count'); set(gca, 'Fontsize', 18)

%% Obtain Alpha-cut and verify the P-boxes against the ECDF of the data:

samp1 = TEMCMC1.samples; samp2 = TEMCMC2.samples; samp3 = TEMCMC3.samples;

% Define Alpha-cut levels for Dp, epsilon, and L:
al_Dp = 0.0; al_e = 0.5; al_L = 41.0;

Dp_mu_int = prctile(samp1(:,1), [al_Dp, 100-al_Dp]);  % Mean interval of Normal dist. of Dp
Dp_std_int = prctile(samp1(:,2), [al_Dp, 100-al_Dp]); % Std interval of Normal dist. of Dp
 
E_mu_int = prctile(samp2(:,1), [al_e, 100-al_e]);  % Mean interval of Normal dist. of Epsilon
E_std_int = prctile(samp2(:,2), [al_e, 100-al_e]); % Std interval of Normal dist. of Epsilon

L_a_int = prctile(samp3(:,1), [al_L, 100-al_L]); % Shape parameter 1 interval of Beta dist. of L
L_b_int = prctile(samp3(:,2), [al_L, 100-al_L]); % Shape parameter 2 interval of Beta dist. of L
