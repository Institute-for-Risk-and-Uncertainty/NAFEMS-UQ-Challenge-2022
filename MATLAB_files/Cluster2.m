%% Define the Data set and key parameters:

Dp_data = [0.0032, 0.0039, 0.0037, 0.0035, 0.0031, 0.0040, 0.0038, 0.0038, 0.0040, 0.0037]; % [m]
epsilon_data = [0.375, 0.347, 0.329, 0.352, 0.388, 0.419, 0.404, 0.394, 0.352, 0.370]; % [~] dimensionless
L_data = [2.86, 3.13, 3.08, 3.12, 2.94, 2.90, 2.80, 3.05, 3.02, 3.04]; % [m]
rho = 1.225;         % [kg/m^3] Density of fluid air
mu_0 = 1.81e-05;     % [kg/(m.s)] Dynamic viscosity of fluid
vs_min = 0.35;       % [m/s] Minimum fluid velocity
delta_p_lim = 15250; % [Pa] Pressure drop limit

%% Define the parameters for Bayesian model updating:
Nsamp = 100;                                 % No. of samples to obtain from the Normal distribution Function
Nsamples = 1000;                             % No. of samples to obtain from the posterior
rnd = @(x, N) normrnd(x(:,1), x(:,2), N, 1); % Normal random number generator

Nbatch = 100;
TEMCMC1_logevi = zeros(Nbatch,1); TEMCMC2_logevi = zeros(Nbatch,1); TEMCMC3_logevi = zeros(Nbatch,1);

parpool(15)

tic;
parfor idx = 1:Nbatch
%% Bayesian model update to infer the hyper-parameters of the SDF for Dp:

bounds1 = [0.001, 0.005]; bounds2 = [0.0001, 0.0008]; 
prior_pdf_mu = @(x) unifpdf(x, bounds1(1), bounds1(2)); prior_pdf_sigma = @(x) unifpdf(x, bounds2(1), bounds2(2)); 
prior_pdf = @(x) prior_pdf_mu(x(:,1)) .* prior_pdf_sigma(x(:,2));
prior_rnd = @(N) [unifrnd(bounds1(1), bounds1(2), N, 1), unifrnd(bounds2(1), bounds2(2), N, 1)];
width = 5e-5;       % Width parameter of the loglikelihood function
logL = @(x) loglikelihood(x, Dp_data, width, Nsamp, rnd);

tic;
TEMCMC1 = TEMCMCsampler('nsamples',Nsamples,'loglikelihood',logL,...
                        'priorpdf',prior_pdf,'priorrnd',prior_rnd);
timeTEMCMC1 = toc;
TEMCMC1_logevi(idx) = TEMCMC1.log_evidence;
fprintf('Time elapsed is for the TEMCMC sampler: %f \n',timeTEMCMC1)

%% Bayesian model update to infer the hyper-parameters of the SDF for Epsilon:

bounds1 = [0.01, 0.50]; bounds2 = [0.001, 0.08]; 
prior_pdf_mu = @(x) unifpdf(x, bounds1(1), bounds1(2)); prior_pdf_sigma = @(x) unifpdf(x, bounds2(1), bounds2(2)); 
prior_pdf = @(x) prior_pdf_mu(x(:,1)) .* prior_pdf_sigma(x(:,2));
prior_rnd = @(N) [unifrnd(bounds1(1), bounds1(2), N, 1), unifrnd(bounds2(1), bounds2(2), N, 1)];

width = 5e-3;       % Width parameter of the loglikelihood function
logL = @(x) loglikelihood(x, epsilon_data, width, Nsamp, rnd);

tic;
TEMCMC2 = TEMCMCsampler('nsamples',Nsamples,'loglikelihood',logL,...
                        'priorpdf',prior_pdf,'priorrnd',prior_rnd);
timeTEMCMC2 = toc;
TEMCMC2_logevi(idx) = TEMCMC2.log_evidence;
fprintf('Time elapsed is for the TEMCMC sampler: %f \n',timeTEMCMC2)

%% Bayesian model update to infer the hyper-parameters of the SDF for L:

bounds1 = [1.00, 5.00]; bounds2 = [0.01, 0.500]; 
prior_pdf_mu = @(x) unifpdf(x, bounds1(1), bounds1(2)); prior_pdf_sigma = @(x) unifpdf(x, bounds2(1), bounds2(2)); 
prior_pdf = @(x) prior_pdf_mu(x(:,1)) .* prior_pdf_sigma(x(:,2));
prior_rnd = @(N) [unifrnd(bounds1(1), bounds1(2), N, 1), unifrnd(bounds2(1), bounds2(2), N, 1)];

width = 5e-2;       % Width parameter of the loglikelihood function
logL = @(x) loglikelihood(x, L_data, width, Nsamp, rnd);

tic;
TEMCMC3 = TEMCMCsampler('nsamples',Nsamples,'loglikelihood',logL,...
                        'priorpdf',prior_pdf,'priorrnd',prior_rnd);
timeTEMCMC3 = toc;
TEMCMC3_logevi(idx) = TEMCMC3.log_evidence;
fprintf('Time elapsed is for the TEMCMC sampler: %f \n',timeTEMCMC3)
end
timeCluster = toc;

%% Obtain key results
mean_norm = mean([TEMCMC1_logevi, TEMCMC2_logevi, TEMCMC3_logevi]);
std_norm = std([TEMCMC1_logevi, TEMCMC2_logevi, TEMCMC3_logevi]);

%% Save the simulation data:
save('Cluster_NAFEMS_Problem_2b', 'TEMCMC1_logevi', 'TEMCMC2_logevi', 'TEMCMC3_logevi', 'timeCluster', 'mean_norm', 'std_norm')