%% Challenge Problem 1:
clc; clear; 
%% Define the Data set:

R_data = [503.252, 460.005, 485.503, 466.061, 475.449]; % [MPa]
S_data = [376.594, 278.222, 331.535, 330.774, 395.173, 394.203, 387.309, 361.754, 300.191, 381.09]; % [MPa]

figure;
subplot(1,2,1)
hold on; box on; grid on;
[yp, xp] = ecdf(R_data); stairs(xp, yp, 'r', 'linewidth', 2)
xlabel('$R$ $[MPa]$', 'Interpreter', 'latex'); ylabel('CDF value'); title('$R$ Data', 'Interpreter', 'latex'); set(gca, 'Fontsize', 18)
subplot(1,2,2)
hold on; box on; grid on;
[yp, xp] = ecdf(S_data); stairs(xp, yp, 'r', 'linewidth', 2)
xlabel('$S$ $[MPa]$', 'Interpreter', 'latex'); ylabel('CDF value'); title('$S$ Data', 'Interpreter', 'latex'); set(gca, 'Fontsize', 18)

%% Define the parameters for Bayesian model updating:
Nsamp = 500;     % No. of samples to obtain from the Normal distribution
Nsamples = 1000; % No. of samples to obtain from the posterior
rnd = @(x, N) normrnd(x(:,1), x(:,2), N, 1); % Normal random number generator

%% Bayesian model update to infer mu and std of the Normal distribution for R:
bounds1 = [300, 600]; bounds2 = [5, 100];

prior_pdf_mu = @(x) unifpdf(x, bounds1(1), bounds1(2)); 
prior_pdf_std = @(x) unifpdf(x, bounds2(1), bounds2(2)); 
prior_pdf = @(x) prior_pdf_mu(x(:,1)) .* prior_pdf_std(x(:,2));
prior_rnd = @(N) [unifrnd(bounds1(1), bounds1(2), N, 1), unifrnd(bounds2(1), bounds2(2), N, 1)];

width = 5; % Width parameter of the logikelihood function
logL = @(x) loglikelihood(x, R_data, width, Nsamp, rnd);

tic;
TEMCMC1 = TEMCMCsampler('nsamples',Nsamples,'loglikelihood', logL,...
                        'priorpdf',prior_pdf,'priorrnd',prior_rnd);
timeTEMCMC1 = toc;
fprintf('Time elapsed is for the TEMCMC sampler: %f \n',timeTEMCMC1)

%% Bayesian model update to infer mu and std of the Normal distribution for S:
bounds1 = [200, 500]; bounds2 = [5, 100];

prior_pdf_mu = @(x) unifpdf(x, bounds1(1), bounds1(2)); 
prior_pdf_std = @(x) unifpdf(x, bounds2(1), bounds2(2)); 
prior_pdf = @(x) prior_pdf_mu(x(:,1)) .* prior_pdf_std(x(:,2));
prior_rnd = @(N) [unifrnd(bounds1(1), bounds1(2), N, 1), unifrnd(bounds2(1), bounds2(2), N, 1)];

width = 5; % Width parameter of the logikelihood function
logL = @(x) loglikelihood(x, S_data, width, Nsamp, rnd);

tic;
TEMCMC2 = TEMCMCsampler('nsamples',Nsamples,'loglikelihood', logL,...
                        'priorpdf',prior_pdf,'priorrnd',prior_rnd);
timeTEMCMC2 = toc;
fprintf('Time elapsed is for the TEMCMC sampler: %f \n',timeTEMCMC2)

%% Save the simulation data:
save('NAFEMS_Problem_1', 'TEMCMC1', 'TEMCMC2')

%% Plot histograms:

samp1 = TEMCMC1.samples; samp2 = TEMCMC2.samples; 
nbin = 50; % No. of bins for histogram

figure; 
subplot(2,2,1)
hold on; box on; grid on;
histogram(samp1(:,1), nbin)
xlabel('$\mu_{R}$ $[MPa]$', 'Interpreter', 'latex'); ylabel('Count'); set(gca, 'Fontsize', 18)
subplot(2,2,2)
hold on; box on; grid on;
histogram(samp2(:,1), nbin)
xlabel('$\mu_{S}$ $[MPa]$', 'Interpreter', 'latex'); ylabel('Count'); set(gca, 'Fontsize', 18)
subplot(2,2,3)
hold on; box on; grid on;
histogram(samp1(:,2), nbin)
xlabel('$\sigma_{R}$ $[MPa]$', 'Interpreter', 'latex'); ylabel('Count'); set(gca, 'Fontsize', 18)
subplot(2,2,4)
hold on; box on; grid on;
histogram(samp2(:,2), nbin)
xlabel('$\sigma_{S}$ $[MPa]$', 'Interpreter', 'latex'); ylabel('Count'); set(gca, 'Fontsize', 18)

%% Obtain Alpha-cut and verify the P-boxes against the ECDF of the data:

% Define Alpha-cut levels for R and S:
al_R = 5.0; al_S = 0.0; 

R_mu_int = prctile(samp1(:,1), [al_R, 100-al_R]);  % Mean interval of the Normal dist. of R
R_std_int = prctile(samp1(:,2), [al_R, 100-al_R]); % Std interval of the Normal dist. of R

S_mu_int = prctile(samp2(:,1), [al_S, 100-al_S]);  % Mean interval of the Normal dist. of S
S_std_int = prctile(samp2(:,2), [al_S, 100-al_S]); % Std interval of the Normal dist. of S