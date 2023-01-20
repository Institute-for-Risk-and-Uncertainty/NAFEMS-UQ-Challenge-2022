function [logL] = loglikelihood(theta, data, width, Nsamp, rnd)
% This is the function handle of the loglikelihood function:

% Inputs:
% theta: N x dim vector of epistemic parameters (dim = 2);
% data:  The Ndata x 1 vector of input data;
% width: The width parameter of the loglikelihood function;
% Nsamp: The no. of aleatory samples to generate from the distribution; 
% rnd:   The random number generator;
%
% Note: theta(:,1) is the hyper-parameter 1 of the dist., theta(:,2) is the hyper-parameter 2 of the dist.;

% Output:
% logL:           The N x 1 vector of loglikelihood values;

%% Define the function:
logL = zeros(size(theta,1),1);

for i = 1:size(theta,1)
model_out = rnd(theta(i,:), Nsamp);
area_met = areaMe(model_out, data);
logL(i) = sum(- (1./width).^2 .* ((area_met).*(area_met)));

if isnan(logL(i)) || isinf(logL(i))
logL(i) = 0;
end
end
end

function [outputArg1] = areaMe(D1,D2)
%AREAME Computes the area between two ECDFs
%   It does not work with a single datum.
%   
% . 
% . by The Liverpool Git Pushers
if length(D1)>length(D2)
    d1 = D2(:);
    d2 = D1(:);
else
    d1 = D1(:);
    d2 = D2(:);
end
[Pxs,xs] = ecdf_Lpool(d1);            % Compute the ecdf of the data sets
[Pys,ys] = ecdf_Lpool(d2);            
Pys_eqx = Pxs;
Pys_pure = Pys(2:end-1); % this does not work with a single datum
Pall = sort([Pys_eqx;Pys_pure]);
ys_eq_all = zeros(length(Pall),1);
ys_eq_all(1)=ys(1);
ys_eq_all(end)=ys(end);
for k=2:length(Pall)-1
    ys_eq_all(k,1) = interpCDF_2(ys,Pys,Pall(k));
end
xs_eq_all = zeros(length(Pall),1);
xs_eq_all(1)=xs(1);
xs_eq_all(end)=xs(end);
for k=2:length(Pall)-1
    xs_eq_all(k,1) = interpCDF_2(xs,Pxs,Pall(k));
end
diff_all_s = abs(ys_eq_all-xs_eq_all);
diff_all_s = diff_all_s(2:end);
diff_all_p = diff(Pall);
area = diff_all_s' * diff_all_p;
outputArg1 = area;
end

function [outputArg1] = interpCDF_2(xd,yd,pvalue)
%INTERPCDF Summary of this function goes here
%   Detailed explanation goes here
%   
% . 
% . by The Liverpool Git Pushers

% [yd,xd]=ecdf_Lpool(data);
beforr = diff(pvalue <= yd) == 1; % && diff(0.5>pv) == -1;
beforrr = [0;beforr(:)];
if pvalue==0
    xvalue = xd(1);
else
    xvalue = xd(beforrr==1);
end
outputArg1 = xvalue;
end

function [ps,xs] = ecdf_Lpool(x)

    xs = sort(x);
    xs = [xs(1);xs(:)];
    ps = linspace(0,1,length(xs))';
    
end

function [samples] = SDF_rnd(theta, bounds, objective, Nsamp)
% This is the function handle of the Staircase Density Function Random Number Generator:

% Inputs:
% theta:     N x dim vector of epistemic Staricase Density Function parameters;
% bounds:    Bounds of the aleatory parameters;
% objective: Numerical objective function flag for the optimization problem;
% Nsamp:     Numerical value of the number of samples to obtain from the joint distribution defined by the Staircase Density Function;

% Output:
% samples:   The Nsamp x dim vector of sample output;

%% Error check:
assert(size(theta,1)==1);

%% Define the function:
objective_func = objective;
N = Nsamp; 
dim = size(theta,2)./4;

theta_a = cell(1);
samples = zeros(N, dim);
for ia = 1:dim
theta_a{ia} = theta(1 + 4*(ia - 1):4*ia);
    
% Fit a staircase density:
[z_i, l, ~] = staircasefit(bounds, theta_a{ia}, objective_func);
l(l < 0) = 0;
    
% Generate samples from the staircase density:
samples(:, ia) = staircasernd(N, z_i, l, bounds);
end

end

function [z_i, l, c_i] = staircasefit(bounds, theta, objective, varargin)
% Calculation of the staircase random variables
%
%     INPUT : bounds    -- prior distribution of aleatory parameters
%             theta     -- samples of the epistemic parameters
%             objective -- objective function flag for the optimization problem
%
%     OUTPUT : z_i -- partitioning points
%              l   -- staircase heights
%              c_i -- centers of the bins
%

n_b = 50;   % n. of bins of staircase RVs
if nargin >= 4
    n_b = varargin{1};
end

z_i = linspace(bounds(1), bounds(2), n_b + 1);   % partitioning points
kappa = diff(bounds)/n_b;                        % subintervals
c_i = z_i(1:end - 1) + kappa/2;                  % centers of the bins

[feasible, ~] = isfeasible(bounds, theta);

theta(3) = theta(3)*theta(2)^(3/2);
theta(4) = theta(4)*theta(2)^2;

if feasible
    Aeq = [kappa*ones(size(c_i));
        kappa*c_i;
        kappa*c_i.^2 + kappa^3/12;
        kappa*c_i.^3 + kappa^3*c_i/4;
        kappa*c_i.^4 + kappa^3*c_i.^2/2 + kappa^5/80];
    beq = [1;
        theta(1);
        theta(1)^2 + theta(2);
        theta(3)+3*theta(1)*theta(2) + theta(1)^3;
        theta(4) + 4*theta(3)*theta(1) + 6*theta(2)*theta(1)^2 + theta(1)^4];
    
    options = optimoptions('fmincon', 'Display', 'off', 'Algorithm', 'sqp');
    options.MaxFunctionEvaluations = 10000*n_b;
    options.MaxIterations = 1000;
    options.ConstraintTolerance = 1e-6;
    options.StepTolerance = 1e-12;
    switch objective
        case 1
            J = @(l) kappa*log(l)*l';
        case 2
            J = @(l) l*l';
        case 3
            J = @(l) -omega*log(l)';
    end
    
    % Do fmincon starting from a uniform distribution over bounds
    l = fmincon(J, 1/(diff(bounds))*ones(size(c_i)), [], [], Aeq, beq, zeros(size(c_i)), [], [], options);
else
    error('unfeasible set of parameters')
end

l(l < 0) = 0;   % ignore negative values

end

function [Lfeasible, constraints] = isfeasible(bounds, theta)
% Return a column of the prior pdf.
%
%     INPUT : bouds -- prior distribution of aleatory parameters
%             theta -- theta(:, 1): mean
%                      theta(:, 2): variance
%                      theta(:, 3): the third-order central moment
%                      theta(:, 4): the fourth-order central moment
%
%     OUTPUT : Lfeasible   -- the prior pdf
%              constraints -- the moment constraints
%

Nsample = size(theta, 1);
Lfeasible = zeros(Nsample, 1);

theta(:, 3) = theta(:, 3).*theta(:, 2).^(3/2);
theta(:, 4) = theta(:, 4).*theta(:, 2).^2;

for isample = 1:Nsample
    u = bounds(1) + bounds(2) - 2*(theta(isample, 1));
    v = (theta(isample, 1) - bounds(1))*(bounds(2) - theta(isample, 1));
    
    constraints = [bounds(1) - theta(isample, 1);   % g2
        theta(isample, 1) - bounds(2);   % g3
        -theta(isample, 2);   % g4
        theta(isample, 2) - v;   % g5
        theta(isample, 2)^2 - theta(isample, 2)*(theta(isample, 1) - bounds(1))^2 - theta(isample, 3)*...
        (theta(isample, 1) - bounds(1));   % g6
        theta(isample, 3)*(bounds(2) - theta(isample, 1)) - theta(isample, 2)*...
        (bounds(2) - theta(isample, 1))^2 + theta(isample, 2)^2;   % g7
        4*theta(isample, 2)^3 + theta(isample, 3)^2 - theta(isample, 2)^2*diff(bounds)^2;   % g8
        6*sqrt(3)*theta(isample, 3) - diff(bounds)^3;   % g9
        -6*sqrt(3)*theta(isample, 3) - diff(bounds)^3;   % g10
        -theta(isample, 4);   % g11
        12*theta(isample, 4) - diff(bounds)^4;   % g12
        (theta(isample, 4) - v*theta(isample, 2) - u*theta(isample, 3))*...
        (v - theta(isample, 2)) + (theta(isample, 3)- u*theta(isample, 2))^2;   % g13
        theta(isample, 3)^2 + theta(isample, 2)^3 - theta(isample, 4)*theta(isample, 2)];   % g14
    
    Lfeasible(isample) = all(constraints <= 0);
end
end

function x = staircasernd(N, z_i, l, bounds)
% Return a column of parameters sampled from the prior pdf
%
%     INPUT : N       -- n. of samples
%             z_i     -- partitioning points
%             l       -- staircase heights
%             bounds  -- prior distribution of aleatory parameters
%
%     OUTPUT : x -- matrix of samples from x_pdf
%

n_b = length(l);   % n. of bins of staircase RVs
idx = (randsample(length(l), N, true, diff(bounds)/n_b*l));   % select random stair
x = unifrnd(z_i(idx), z_i(idx + 1))';   % generate uniform in each stair
end
