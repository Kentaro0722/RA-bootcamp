

tic

% Parameters
alpha = 0.2;  % Share of Capital
nu = 0.64;  % Share of Labour
phi = 0.0;  % working capital constraint partial switch, phi=0: no wc constraint, phi=1: full wc constraint
delta = 0.089;  % capital depreciation rate
betas = 0.99;  % subjective discount factor
qrf = betas;  % risk free discount factor
rexp = 1.0 - betas;  % interest rate expense parameter
rrf = 1 / betas - 1.0;  % risk free interest rate
tauc = 0.232;  % corporate tax level (Katagiri, 2014)
taud = 0.0;  % dividend tax level (Katagiri, 2014)
lambda = 0.08;  % equity funding cost (Katagiri, 2014)
eta = 2.29;  % preference parameter on leisure
pid = 0.02;  % exogenous exit rate for firms (need total exit to come to 8.69 %)
xiparam = 0.06;  % Fixed operating cost: Old value from TS code 0.0615
k0common = 0.35;  % COMMON CAPITAL OF NEWBORNS
dcost = 0.59;
p0 = 2.5;
mass_entrants = 0.02;
initial_xval = 5;
roecutoff = 0.08;
minimum_payout_ratio = 0.0;


precisionxdfp = 1.0e-5;
precisionv = 1.0e-5;
precisionxd = 1.0e-5;
maxIterations = 200;



fprintf('\n..............................................................\n\n');
fprintf(' Parameter values\n\n');
fprintf('     tauc     %7.4f\n', tauc);
fprintf('     taud     %7.4f\n', taud);
fprintf('     lambda   %7.4f\n', lambda);
fprintf('     phi      %7.4f\n', phi);
fprintf('     rrf      %7.4f\n', rrf);
fprintf('     xiparam  %7.4f\n', xiparam);
fprintf('\n..............................................................\n\n');

% PARAMETERS FOR COMBO PERMANENT BASE & PERSISTENT PRODUCTIVITY PROCESS (11/2016)
% PERSISTENT (poision-arrival/new iid draw **OR** standard AR(1) ) productivities
rhoe0 = 0.9;  % 0.83 persistence of idiosyncratic shock process (targeting Syverson's 0.757 as a result; or just setting rhoe0=0.757)
stde0 = 0.0335;  % stde0 = 0.032
neps = 5;  % size of the support for the idiosyncratic shock

% DECISION GRIDS
klow = 0.0;
khigh = 3.0;
nk = 50;

llow = 0.0;
lhigh = 1.2;
nl = 50;

eqlow = 0.0;
eqhigh = 2.5;
neq = 50;

% DISTRIBUTION GRIDS for capital, labour, equity, debt, cash of all firms
% To make grids fine around default thresholds: Inversed log-grid
% from lower bd to mid point, log-grid from from mid point to upper bd.

% bounds and number of grid points for the distribution kagrid [STORES RAW k]
kglow = klow; %0.0;
kghigh = khigh; %10.0;
nkg = nk; %150;  % AVOID 0 ON GRID WHEN GRID IS COARSE.

lglow = llow; %0.0;
lghigh = lhigh; %10.0;
nlg = nl; %150;

eqglow = eqlow; %0.0;
eqghigh = eqhigh; %10.0;
neqg = neq; %150;

% bounds and numbers of grid points for net cash-on-hand
xlow = -0.2;
xhigh = 3.0; %1.0; % x lower bound / mid point / upper bound
nx = 65;

% Create the grids
kgrid = linspace(klow, khigh, nk);
lgrid = linspace(llow, lhigh, nl);
eqgrid = linspace(eqlow, eqhigh, neq);

kagrid = linspace(kglow, kghigh, nkg);
lagrid = linspace(lglow, lghigh, nlg);
eqagrid = linspace(eqglow, eqghigh, neqg);
xgrid = linspace(xlow, xhigh, nx);

[evec, pie] = rouwenhorst(rhoe0, stde0, neps);
evec = exp(evec);


% Initialize vinitial
vinitial = zeros(neps, nx);

%% low wage
plow = 0.953*p0;
wage = eta/plow;

% Step 1: Call the decisions function
[v, vc, vc0, xd, qarray, kmax, lmax, eqmax, roemax, roamax, leveragemax, divmax, exit] = decisions(vinitial, neps, nx, nk, nl, neq, 1, evec, pie, xgrid, kgrid, lgrid, eqgrid, ...
    alpha, nu, xiparam, phi, delta, wage, qrf, rexp, tauc, taud, lambda, betas, pid, precisionxdfp, precisionv, precisionxd, roecutoff, dcost, minimum_payout_ratio);

% Step 2: Call the distribution function 
[mu, firmnum, kagg, kfagg, lagg] = distribution(neps, nkg, nlg, neqg, evec, kagrid, lagrid, eqagrid, alpha, nu, tauc, rexp, phi, pid, wage, delta, xiparam, nx, xgrid, kmax, lmax, eqmax, nk, kgrid, nl, lgrid, neq, eqgrid, pie, precisionv, maxIterations, exit, mass_entrants, initial_xval);

% Step 3: Obtain additional aggregate variables
[yagg, agg_roe, agg_roa, avg_roe, avg_roa, agg_payout_ratio, avg_payout_ratio] = aggregation(mu, neps, nkg, nlg, neqg, evec, kagrid, lagrid, eqagrid, alpha, nu, tauc, rexp, phi, wage, delta, xiparam, nx, xgrid, kmax, lmax, eqmax, nk, kgrid, nl, lgrid, neq, eqgrid, pie, precisionv, maxIterations, qrf, taud, lambda);

iagg = kfagg - (1.0 - delta)*kagg;
cagg = yagg - iagg;
tfp = yagg / (kagg^alpha * lagg^(1-alpha));

ps = 1.0/cagg;
pgap = plow - ps;
pgaplow = pgap;

% Display results
fprintf('Aggregate variables:\n');
fprintf('yagg = %f\n', yagg);
fprintf('cagg = %f\n', cagg);
fprintf('iagg = %f\n', iagg);
fprintf('kagg = %f\n', kagg);
fprintf('lagg = %f\n', lagg);
fprintf('capital-output ratio = %f\n', kagg/yagg);
fprintf('investment rate = %f\n', iagg/yagg);
fprintf('tfp = %f\n', tfp);
fprintf('\nAggregate and Average Returns:\n');
fprintf('Aggregate ROE = %f%%\n', agg_roe * 100);
fprintf('Average ROE = %f%%\n', avg_roe * 100);
fprintf('Aggregate ROA = %f%%\n', agg_roa * 100);
fprintf('Average ROA = %f%%\n', avg_roa * 100);

fprintf('Excess demand = %f%%\n', pgaplow);

%% high wage 
phigh = 0.957*p0;
wage = eta/phigh;


% Step 1: Call the decisions function
[v, vc, vc0, xd, qarray, kmax, lmax, eqmax, roemax, roamax, leveragemax, divmax, exit] = decisions(vinitial, neps, nx, nk, nl, neq, 1, evec, pie, xgrid, kgrid, lgrid, eqgrid, ...
    alpha, nu, xiparam, phi, delta, wage, qrf, rexp, tauc, taud, lambda, betas, pid, precisionxdfp, precisionv, precisionxd, roecutoff, dcost, minimum_payout_ratio);

% Step 2: Call the distribution function 
[mu, firmnum, kagg, kfagg, lagg] = distribution(neps, nkg, nlg, neqg, evec, kagrid, lagrid, eqagrid, alpha, nu, tauc, rexp, phi, pid, wage, delta, xiparam, nx, xgrid, kmax, lmax, eqmax, nk, kgrid, nl, lgrid, neq, eqgrid, pie, precisionv, maxIterations, exit, mass_entrants, initial_xval);

% Step 3: Obtain additional aggregate variables
[yagg, agg_roe, agg_roa, avg_roe, avg_roa, agg_payout_ratio, avg_payout_ratio] = aggregation(mu, neps, nkg, nlg, neqg, evec, kagrid, lagrid, eqagrid, alpha, nu, tauc, rexp, phi, wage, delta, xiparam, nx, xgrid, kmax, lmax, eqmax, nk, kgrid, nl, lgrid, neq, eqgrid, pie, precisionv, maxIterations, qrf, taud, lambda);

iagg = kfagg - (1.0 - delta)*kagg;
cagg = yagg - iagg;
tfp = yagg / (kagg^alpha * lagg^(1-alpha));

ps = 1.0/cagg;
pgap = phigh - ps;
pgaphigh = pgap;


% Display results
fprintf('Aggregate variables:\n');
fprintf('yagg = %f\n', yagg);
fprintf('cagg = %f\n', cagg);
fprintf('iagg = %f\n', iagg);
fprintf('kagg = %f\n', kagg);
fprintf('lagg = %f\n', lagg);
fprintf('capital-output ratio = %f\n', kagg/yagg);
fprintf('investment rate = %f\n', iagg/yagg);
fprintf('tfp = %f\n', tfp);
fprintf('\nAggregate and Average Returns:\n');
fprintf('Aggregate ROE = %f%%\n', agg_roe * 100);
fprintf('Average ROE = %f%%\n', avg_roe * 100);
fprintf('Aggregate ROA = %f%%\n', agg_roa * 100);
fprintf('Average ROA = %f%%\n', avg_roa * 100);
fprintf('Excess demand = %f%%\n', pgaphigh);


%% bisecinting wage 
fprintf('Excess demand (plow) = %f%%\n', pgaplow);
fprintf('Excess demand (phigh) = %f%%\n', pgaphigh);

pause;

pgap = 2.0*precisionv;
iterationv = 0;

while true

    if abs(pgap) <= precisionv || iterationv >= 5
        if pgap > precisionv
            fprintf('leaving price bisection do loop; max iterations reached.\n');
        end
        break;
    end

p = 0.5*(plow + phigh);
    wage = eta/p;
        
% Step 1: Call the decisions function
[v, vc, vc0, xd, qarray, kmax, lmax, eqmax, roemax, roamax, leveragemax, divmax, exit] = decisions(vinitial, neps, nx, nk, nl, neq, 1, evec, pie, xgrid, kgrid, lgrid, eqgrid, ...
    alpha, nu, xiparam, phi, delta, wage, qrf, rexp, tauc, taud, lambda, betas, pid, precisionxdfp, precisionv, precisionxd, roecutoff, dcost, min_payout_ratio);

% Step 2: Call the distribution function 
[mu, firmnum, kagg, kfagg, lagg] = distribution(neps, nkg, nlg, neqg, evec, kagrid, lagrid, eqagrid, alpha, nu, tauc, rexp, phi, pid, wage, delta, xiparam, nx, xgrid, kmax, lmax, eqmax, nk, kgrid, nl, lgrid, neq, eqgrid, pie, precisionv, maxIterations, exit, mass_entrants, initial_xval);

% Step 3: Obtain additional aggregate variables
[yagg, agg_roe, agg_roa, avg_roe, avg_roa, agg_payout_ratio, avg_payout_ratio] = aggregation(mu, neps, nkg, nlg, neqg, evec, kagrid, lagrid, eqagrid, alpha, nu, tauc, rexp, phi, wage, delta, xiparam, nx, xgrid, kmax, lmax, eqmax, nk, kgrid, nl, lgrid, neq, eqgrid, pie, precisionv, maxIterations, qrf, taud, lambda);

iagg = kfagg - (1.0 - delta)*kagg;
cagg = yagg - iagg;
tfp = yagg / (kagg^alpha * lagg^(1-alpha));

ps = 1.0/cagg;
pgap = p - ps;

% Display results
% Display results
fprintf('Aggregate variables:\n');
fprintf('yagg = %f\n', yagg);
fprintf('cagg = %f\n', cagg);
fprintf('iagg = %f\n', iagg);
fprintf('kagg = %f\n', kagg);
fprintf('lagg = %f\n', lagg);
fprintf('capital-output ratio = %f\n', kagg/yagg);
fprintf('investment rate = %f\n', iagg/yagg);
fprintf('tfp = %f\n', tfp);
fprintf('\nAggregate and Average Returns:\n');
fprintf('Aggregate ROE = %f%%\n', agg_roe * 100);
fprintf('Average ROE = %f%%\n', avg_roe * 100);
fprintf('Aggregate ROA = %f%%\n', agg_roa * 100);
fprintf('Average ROA = %f%%\n', avg_roa * 100);
fprintf('\nPayout Ratios:\n');
fprintf('Aggregate Payout Ratio = %f%%\n', agg_payout_ratio * 100);
fprintf('Average Payout Ratio = %f%%\n', avg_payout_ratio * 100);
fprintf('Excess demand = %f%%\n', pgap);
fprintf('Final marginal utility = %f\n', p);
fprintf('Final wage = %f\n', wage)
% check how to update plow or phigh

if (pgap*pgaphigh>0.0)
    phigh = p;
else
    plow = p;
end

iterationv = iterationv + 1;

fprintf('Iteration %d: p = %f, pgap = %f%%\n', iterationv, p, pgap);

end

total_value = 0;
total_firms = 0;

% Loop through all dimensions
for h = 1:neqg
    for i = 1:nlg
        for j = 1:nkg
            for k = 1:neps
                muval = mu(k,j,i,h);
                if muval > 0
                    % Calculate cash position for this state
                    tfpval = evec(k);
                    kval = kagrid(j);
                    lval = lagrid(i);
                    eqval = eqagrid(h);
                    
                    % Calculate cash position
                    revenue = tfpval * kval^alpha * lval^nu;
                    costs = wage * lval + xiparam;
                    profit = revenue - costs;
                    xval = profit + eqval;
                    
                    % Find corresponding value
                    [xloc, weight] = gridlookup(nx, xgrid, xval);
                    xloc = min(max(xloc, 1), nx-1);
                    firm_value = weight * v(k,xloc) + (1-weight) * v(k,xloc+1);
                    
                    % Accumulate weighted values
                    total_value = total_value + firm_value * muval;
                    total_firms = total_firms + muval;
                end
            end
        end
    end
end

% Calculate average value
average_value = total_value / total_firms;

% Display results
fprintf('\nValue Statistics:\n');
fprintf('Aggregate Firm Value: %f\n', total_value);
fprintf('Average Firm Value: %f\n', average_value);
fprintf('Total Number of Firms: %f\n', total_firms);

% Visualize value function
figure('Position', [100, 100, 1200, 400]);

% Plot 1: 3D surface of value function
subplot(1, 2, 1);
[X, Y] = meshgrid(xgrid, evec);
surf(X, Y, v);
colormap('jet');
colorbar;

title('Firm Value Function');
xlabel('Cash Position (x)');
ylabel('Productivity (ε)');
zlabel('Value');
view(45, 30);
grid on;

% Plot 2: Value function for different productivity levels
subplot(1, 2, 2);
hold on;
plot(xgrid, v(1,:), 'r--', 'LineWidth', 2, 'DisplayName', 'Low ε');
plot(xgrid, v(3,:), 'g-', 'LineWidth', 2, 'DisplayName', 'Medium ε');
plot(xgrid, v(5,:), 'b-', 'LineWidth', 2, 'DisplayName', 'High ε');
title('Value Function by Productivity Level');
xlabel('Cash Position (x)');
ylabel('Value');
legend('Location', 'northwest');
grid on;

set(gcf, 'Color', 'w')

toc




%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%% plotting the distribution
for h = 1:nkg
    for i = 1:neps
        mu1(i,h) = sum(sum(mu(i,h,:,:)));
    end
end   

figure;
mesh(mu1)
ylabel('Productivity');
xlabel('Capital');
title('Distribution of firms');

% Improve the appearance of the plot
colorbar; % Add a color bar
view(45, 30); % Adjust the view angle for better visibility


%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%% plotting the decisions
bmax = kmax - eqmax;
% Plot decisions
cash = 1:50;

figure;
subplot(2, 4, 1);
hold on;
plot(xgrid, kmax(1,:), 'r--', 'LineWidth', 2);
plot(xgrid, kmax(3,:), 'g-', 'LineWidth', 2);
plot(xgrid, kmax(5,:), 'b-', 'LineWidth', 2);
title('Capital');
    
subplot(2, 4, 2);
hold on;
plot(xgrid, lmax(1,:), 'r--', 'LineWidth', 2);
plot(xgrid, lmax(3,:), 'g-', 'LineWidth', 2);
plot(xgrid, lmax(5,:), 'b-', 'LineWidth', 2);
title('Labour');

subplot(2, 4, 3);
hold on;
plot(xgrid, eqmax(1,:), 'r--', 'LineWidth', 2);
plot(xgrid, eqmax(3,:), 'g-', 'LineWidth', 2);
plot(xgrid, eqmax(5,:), 'b-', 'LineWidth', 2);
title('Equity');

subplot(2, 4, 4);
hold on;
plot(xgrid, roemax(1,:), 'r--', 'LineWidth', 2);
plot(xgrid, roemax(3,:), 'g-', 'LineWidth', 2);
plot(xgrid, roemax(5,:), 'b-', 'LineWidth', 2);
title('Return on Equity');

subplot(2, 4, 5);
hold on;
plot(xgrid, divmax(1,:), 'r--', 'LineWidth', 2);
plot(xgrid, divmax(3,:), 'g-', 'LineWidth', 2);
plot(xgrid, divmax(5,:), 'b-', 'LineWidth', 2);
title('Dividend');

subplot(2, 4, 6);
hold on;
plot(xgrid, bmax(1,:), 'r--', 'LineWidth', 2);
plot(xgrid, bmax(3,:), 'g-', 'LineWidth', 2);
plot(xgrid, bmax(5,:), 'b-', 'LineWidth', 2);
title('Debt');

subplot(2, 4, 7);
hold on;
plot(xgrid, leveragemax(1,:), 'r--', 'LineWidth', 2);
plot(xgrid, leveragemax(3,:), 'g-', 'LineWidth', 2);
plot(xgrid, leveragemax(5,:), 'b-', 'LineWidth', 2);
title('Leverage');

subplot(2, 4, 8);
hold on;
plot(xgrid, roamax(1,:), 'r--', 'LineWidth', 2);
plot(xgrid, roamax(3,:), 'g-', 'LineWidth', 2);
plot(xgrid, roamax(5,:), 'b-', 'LineWidth', 2);
title('Return on Assets');

plot_qarray_relationships(qarray, neps, nk, nl, neq, evec, kgrid, lgrid, eqgrid);

%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%
 

% Function definitions

function [v, vc, vc0, xd, qarray, kmax, lmax, eqmax, roemax, roamax, leveragemax, divmax, exit] = decisions(vin, neps, nx, nk, nl, neq, ixdhigh, evec, pie, xgrid, kgrid, lgrid, eqgrid, ...
    alpha, nu, xiparam, phi, delta, wage, qrf, rexp, tauc, taud, lambda, beta, pid, precisionxdfp, precisionv, precisionxd, roecutoff, dcost, minimum_payout_ratio)
    
    iterationr = 1;
    distanced = 2.0 * precisionxdfp;
    vguess = vin;
    qarray = repmat(qrf, [neps, nk, nl, neq]);
    xd = zeros(1, neps);

    %while true
    %    if distanced < precisionxdfp || iterationr >= 100
    %        fprintf('.........................................................................\n\n');
    %        fprintf('VFI solved such that the FI zero profit condition holds\n\n');
    %        fprintf('.........................................................................\n');
    %        fprintf('Default Thresholds: ');
    %        fprintf('%12.3e ', xd);
    %        fprintf('\n.........................................................................\n\n');
    %        break;
    %    end

        fprintf('-------------------------------------------------------------------------\n');
        fprintf(' Default threshold iteration %d distanced = %12.4e\n\n', iterationr, distanced);

        qarray = loanschedule(xd, neps, nx, nk, nl, neq, evec, pie, xgrid, kgrid, lgrid, eqgrid, ...
            alpha, nu, xiparam, phi, delta, wage, qrf, rexp, tauc, taud, pid, qarray, dcost);

        fprintf('\n     Min interest rate: %9.5f     Max interest rate: %9.5f\n\n', min(qarray(:)), max(qarray(:)));

        [v, vc, vc0, kmax, lmax, eqmax, roemax, roamax, leveragemax, divmax, exit] = contraction(vguess, neps, nx, nk, nl, neq, evec, pie, xgrid, kgrid, lgrid, eqgrid, ...
            xd, alpha, nu, xiparam, phi, delta, wage, qarray, qrf, rexp, tauc, taud, lambda, beta, pid, precisionv, roecutoff, minimum_payout_ratio);


        fprintf('Default thresholds: ');
        txd = zeros(1, neps);
        for ie = 1:neps
            h = vc0(ie, 1:nx);
            txd(ie) = default(nx, xgrid, ixdhigh, h, precisionxd);
            fprintf('%7.4f ', txd(ie));
        end
        fprintf('\n');

        distanced = max(abs(txd - xd));
        iterationr = iterationr + 1;
        vguess = v;
        xd = txd;
    %end
end

function qarray = loanschedule(xd, neps, nx, nk, nl, neq, evec, pie, xgrid, kgrid, lgrid, eqgrid, ...
    alpha, nu, xiparam, phi, delta, wage, qrf, rexp, tauc, taud, pid, qarray, dcost)
    
    for ie = 1:neps
        pievec = pie(ie, :);
        for ikv = 1:nk
            kfval = kgrid(ikv);
            for ilv = 1:nl
                lfval = lgrid(ilv);
                for ieq = 1:neq
                    eqfval = eqgrid(ieq);
                    if eqfval >= kfval
                        qarray(ie, ikv, ilv, ieq) = qrf;
                    else
                        xfval = equityvec(neps, evec, alpha, nu, xiparam, phi, delta, wage, rexp, tauc, kfval, lfval, eqfval);
                        cashval = equityvec(neps, evec, alpha, nu, xiparam, phi, delta, wage, rexp, tauc, kfval, lfval, kfval);
                        payout = 0.0;
                        for ief = 1:neps
                            if xfval(ief) <= xd(ief)
                                payout = payout + pievec(ief) * (1-delta)*(dcost * kfval);
                            else
                                payout = payout + pievec(ief) * (1.0 - pid) * (kfval - eqfval) + pid * ((1-delta)*(dcost * kfval));
                            end
                        end
                        qnew = max(0, payout / (kfval - eqfval));
                        qarray(ie, ikv, ilv, ieq) = min(qnew, qrf);
                    end
                end
            end
        end
    end
end

function [v, vc, vc0, kmax, lmax, eqmax, roemax, roamax, leveragemax, divmax, exit] = contraction(vin, neps, nx, nk, nl, neq, evec, pie, xgrid, kgrid, lgrid, eqgrid, ...
    xd, alpha, nu, xiparam, phi, delta, wage, qarray, qrf, rexp, tauc, taud, lambda, beta, pid, precisionv, roecutoff, minimum_payout_ratio)
    
    distance = 100.0 * precisionv;
    iterationv = 0;
    v = vin;
    vc = zeros(neps, nx);
    vc0 = zeros(neps, nx);
    exit = zeros(neps, nx);

    while true
        if distance <= precisionv || iterationv >= 30
            if distance > precisionv
                fprintf('leaving value fcn do loop; max iterations reached.\n');
            end
            break;
        end

        iterationv = iterationv + 1;
        
        fprintf('Iteration %d\n', iterationv);
        
        for ie = 1:neps
            fprintf('Epsilon %d\n', ie);
            
            for ix = 1:nx
                xval = xgrid(ix);
                [vc(ie, ix), kmax(ie, ix), lmax(ie, ix), eqmax(ie, ix), roemax(ie, ix),roamax(ie, ix), leveragemax(ie, ix), divmax(ie, ix)] = operator(xval, ie, v, neps, nx, nk, nl, neq, evec, pie, xgrid, kgrid, lgrid, eqgrid, ...
                    xd(ie), alpha, nu, xiparam, phi, delta, wage, qarray, qrf, rexp, tauc, taud, lambda, beta, roecutoff, minimum_payout_ratio);
                vc0(ie, ix) = pid * (1.0 - taud) * max(xval, 0.0) + (1.0 - pid) * vc(ie, ix);
                v(ie, ix) = max(vc0(ie, ix), 0.0);
                if v(ie, ix) < 0
                    exit(ie, ix) = 1;
                end
            end
        end

        distance = max(abs(v(:) - vin(:)));
        if iterationv <= 10 || mod(iterationv, 10) == 0
            fprintf(' v(epsilon, x) iteration %d normv = %f\n', iterationv, distance);
        end
        vin = v;
    end
end

function [xfval, profitfval] = equityvec(neps, evec, alpha, nu, xiparam, phi, delta, wage, rexp, tauc, kfval, lfval, eqfval)
    capterm = kfval ^ alpha;
    labterm = lfval ^ nu;
    debterm = tauc * rexp * (kfval - eqfval);
    wageterm = (1.0 - phi) * wage * lfval;
    deprterm = delta * kfval;
    xfval = zeros(1, neps);
    profitfval = zeros(1, neps);
    
    for indf = 1:neps
        tfpval = evec(indf);
        term1 = tfpval * capterm * labterm;
        taxterm = term1 - wageterm - deprterm - xiparam;
        
        % Modified tax calculation
        if taxterm > 0
            profitfval(indf) = (1.0 - tauc) * taxterm;
        else
            profitfval(indf) = taxterm;
        end
        
        xfval(indf) = profitfval(indf) + debterm + eqfval;
    end
end

function [vcmax, kmax, lmax, eqmax, roemax, roamax, leveragemax, divmax] = operator(xval, ie, v, neps, nx, nk, nl, neq, evec, pie, xgrid, kgrid, lgrid, eqgrid, ...
 xd, alpha, nu, xiparam, phi, delta, wage, qarray, qrf, rexp, tauc, taud, lambda, beta, roecutoff, minimum_payout_ratio)

% Initialize output variables
vcmax = -1000.0;
kmax = 0;
lmax = 0;
eqmax = 0;
roemax = 0;
roamax = 0;
leveragemax = 0;
divmax = 0;

% Set parameters
objective = zeros(nk, nl, neq);

% Main loop over capital, labor, and equity
for ikv = 1:nk
    kfval = kgrid(ikv);
    temp_objective = zeros(nl, neq);

    for ilv = 1:nl
        lfval = lgrid(ilv);

        for ieq = 1:neq
            eqfval = eqgrid(ieq);

            % Calculate profit and ROE
            [xfval, profitfmaxvec] = equityvec(neps, evec, alpha, nu, xiparam, phi, delta, wage, rexp, tauc, kfval, lfval, eqfval);
            profitfmax = pie(ie, 1:neps) * profitfmaxvec(1:neps)';
            current_roe = profitfmax / eqfval;
            
            % New dividend policy based only on ROE condition
            if current_roe < roecutoff
                % Calculate minimum dividend based on equity
                minimum_dividend = minimum_payout_ratio * eqfval;
                
                % Calculate raw dividend from remaining cash after minimum dividend
                raw_dividend = dividend(xval - minimum_dividend, xiparam, phi, wage, qarray(ie, ikv, ilv, ieq), tauc, kfval, lfval, eqfval);
                
                % Calculate final dividend value
                if raw_dividend >= 0
                    dval = (1 - taud) * (minimum_dividend + raw_dividend);
                else
                    dval = (1 - taud) * minimum_dividend + (1 + lambda) * raw_dividend;
                end
            else
                % High ROE - original dividend policy
                raw_dividend = dividend(xval, xiparam, phi, wage, qarray(ie, ikv, ilv, ieq), tauc, kfval, lfval, eqfval);
                if raw_dividend >= 0
                    dval = (1 - taud) * raw_dividend;
                else
                    dval = (1 + lambda) * raw_dividend;
                end
            end
            
            % Calculate expected future value
            evalue = zeros(1, neps);
            evalue1 = zeros(1, neps);
            for iie = 1:neps
                [xloc, weight] = gridlookup(nx, xgrid, xfval(iie));
                xloc = min(max(xloc, 1), nx - 1);
                weight = min(max(weight, 0.0), 1.0);
                if xloc >= nx - 1
                    evalue(iie) = v(iie, nx);
                elseif xloc < 1
                    evalue(iie) = v(iie, 1);
                else
                    evalue(iie) = weight * v(iie, xloc) + (1.0 - weight) * v(iie, xloc + 1);
                end
                evalue1(iie) = pie(ie, iie) * evalue(iie);
            end
            
            % Calculate objective function
            vmax = sum(evalue1);
            temp_objective(ilv, ieq) = dval + beta * vmax;
        end
    end
    objective(ikv, :, :) = temp_objective;
end

% Find maximum value and corresponding indices
[temp_max, idx] = max(objective(:));
if temp_max > vcmax
    vcmax = temp_max;
    [idx_k, idx_l, idx_eq] = ind2sub([nk, nl, neq], idx);
    kmax = kgrid(idx_k);
    lmax = lgrid(idx_l);
    eqmax = eqgrid(idx_eq);
    
    % Calculate additional maximum values
    [~, profitfmaxvec] = equityvec(neps, evec, alpha, nu, xiparam, phi, delta, wage, rexp, tauc, kmax, lmax, eqmax);
    profitfmax = pie(ie, 1:neps) * profitfmaxvec(1:neps)';
    roemax = profitfmax / eqmax;
    roamax = profitfmax / kmax;
    leveragemax = kmax / eqmax;
    
    % Calculate divmax with same logic as main calculation
    if (roemax < roecutoff)
        % Calculate minimum dividend based on equity
        minimum_dividend = minimum_payout_ratio * eqmax;
        
        % Calculate raw dividend from remaining cash after minimum dividend
        raw_dividend = dividend(xval - minimum_dividend, xiparam, phi, wage, qarray(ie, idx_k, idx_l, idx_eq), tauc, kmax, lmax, eqmax);
        
        % Set divmax based on raw_dividend
        if raw_dividend >= 0
            divmax = (1 - taud) * (minimum_dividend + raw_dividend);
        else
            divmax = (1 - taud) * minimum_dividend + (1 + lambda) * raw_dividend;
        end
    else
        % High ROE case - use original dividend calculation
        raw_dividend = dividend(xval, xiparam, phi, wage, qarray(ie, idx_k, idx_l, idx_eq), tauc, kmax, lmax, eqmax);
        if raw_dividend >= 0
            divmax = (1 - taud) * raw_dividend;
        else
            divmax = (1 + lambda) * raw_dividend;
        end
    end
end
end

function divval = dividend(xval, xiparam, phi, wage, qval, tauc, kfval, lfval, eqfval)
    divval = xval - (1.0 - tauc) * phi * wage * lfval + qval * (kfval - eqfval) - kfval;
end

% ... [Previous code remains the same] ...

function xdeval = default(nx, xgrid, ixdhigh, vc0vec, precisionxd)
    xdlow = xgrid(1);
    xdhigh = xgrid(ixdhigh);

    flow = vc0vec(1);
    fhigh = vc0vec(ixdhigh);

    if fhigh < 0.0
        fprintf(' Default threshold above highest x considered: %f; fhigh: %f.\n', xgrid(ixdhigh), fhigh);
        xdeval = xgrid(ixdhigh);
    elseif flow > 0.0
        fprintf(' Default threshold lower than lowest x considered: %f; flow: %f.\n', xgrid(1), flow);
        xdeval = xgrid(1);
    else
        distance = 0.0;
        s1 = 0;
        vc0val = 10.0 * precisionxd;

        while true
            distance = xdhigh - xdlow;
            s1 = s1 + 1;

            if abs(xdhigh - xdlow) <= precisionxd
                break;
            end

            xdeval = (xdlow + xdhigh) / 2.0;
            [xloc, weight] = gridlookup(nx, xgrid, xdeval);
            vc0val = weight * vc0vec(xloc) + (1.0 - weight) * vc0vec(xloc + 1);

            if vc0val < 0.0
                xdlow = xdeval;
            else
                xdhigh = xdeval;
            end
        end
    end
end

function [ixlow, iweight] = gridlookup(gridnum, xdist, xval)
    ixhigh = gridnum;
    ixlow = 1;

    while ixhigh - ixlow > 1
        ixplace = floor((ixhigh + ixlow) / 2);
        if xdist(ixplace) >= xval
            ixhigh = ixplace;
        else
            ixlow = ixplace;
        end
    end

    iweight = (xdist(ixlow + 1) - xval) / (xdist(ixlow + 1) - xdist(ixlow));
end

function [z, pi] = rouwenhorst(rho, sigmas, n)
    p = (rho + 1.0) / 2.0;
    q = p;
    hlag = zeros(n, n);
    hlag(1, 1) = 1.0;

    for i = 2:n
        h = zeros(i, i);
        h(1:i-1, 1:i-1) = p * hlag(1:i-1, 1:i-1);
        h(1:i-1, 2:i) = h(1:i-1, 2:i) + (1.0 - p) * hlag(1:i-1, 1:i-1);
        h(2:i, 1:i-1) = h(2:i, 1:i-1) + (1.0 - q) * hlag(1:i-1, 1:i-1);
        h(2:i, 2:i) = h(2:i, 2:i) + q * hlag(1:i-1, 1:i-1);
        h(2:i-1, :) = h(2:i-1, :) / 2.0;
        hlag(1:i, 1:i) = h;
    end

    pi = hlag;

    zvar = (sigmas ^ 2.0) / (1.0 - rho ^ 2.0);
    epsilonz = sqrt((n - 1.0) * zvar);
    y = 1:(n-2);
    y = ((2.0 * epsilonz) / (n - 1)) * y - epsilonz;

    z = zeros(1, n);
    z(1) = -1.0 * epsilonz;
    z(2:n-1) = y;
    z(n) = epsilonz;
end

function [mu, firmnum, kagg, kfagg, lagg] = distribution(neps, nkg, nlg, neqg, evec, kagrid, lagrid, eqagrid, alpha, nu, tauc, rexp, phi, pid, wage, delta, xiparam, nx, xgrid, kmax, lmax, eqmax, nk, kgrid, nl, lgrid, neq, eqgrid, pie, precisionv, maxIterations, exit, mass_entrants, initial_xval) % Initialize distribution arrays

    mu = zeros(neps, nkg, nlg, neqg);
    munext = zeros(neps, nkg, nlg, neqg);
    
    % Initial condition: place some mass in the middle of the grids
    mu(5, 35, 35, 45) = 1;
    
    % Calculate entry profitability and distribution
    [profitable_entry, entry_eps_dist, expected_profits] = check_entry_profitability(neps, evec, xgrid, initial_xval, alpha, nu, wage, xiparam, kmax, lmax);

    % Main iteration loop
    for it = 1:maxIterations
        % Initialize aggregates for this iteration
        firmnum = 0.0;
        kagg = 0.0;
        kfagg = 0.0;
        lagg = 0.0;
        exitnum = 0.0;
        
        % Loop over all states to update distribution
        for h = 1:neqg
            eqval = eqagrid(h);
            for i = 1:nlg
                lval = lagrid(i);
                for j = 1:nkg
                    kval = kagrid(j);
                    for k = 1:neps
                        tfpval = evec(k);
                        muval = mu(k,j,i,h);
                        
                        if muval > 0
                            % Calculate cash position
                            capterm = kval ^ alpha;
                            labterm = lval ^ nu;
                            debterm = tauc * rexp * (kval - eqval);
                            wageterm = (1.0 - phi) * wage * lval;
                            deprterm = delta * kval;
                            
                            term1 = tfpval * capterm * labterm;
                            taxterm = term1 - wageterm - deprterm - xiparam;
                            
                            % Modified tax calculation
                            if taxterm > 0
                                profitval = (1.0 - tauc) * taxterm;
                            else
                                profitval = taxterm;
                            end
                            
                            xval = profitval + debterm + eqval;
                            
                            % Grid lookups
                            [xloc, xweight] = gridlookup(nx, xgrid, xval);
                            xweight = min(max(xweight, 0.0), 1.0);
                            
                            % Calculate optimal choices
                            kfmax = xweight*kmax(k, xloc) + (1.0-xweight)*kmax(k,xloc+1);
                            [kfloc, kfweight] = gridlookup(nk, kgrid, kfmax);
                            
                            lfmax = xweight*lmax(k, xloc) + (1.0-xweight)*lmax(k,xloc+1);
                            [lfloc, lfweight] = gridlookup(nl, lgrid, lfmax);
                            
                            eqfmax = xweight*eqmax(k, xloc) + (1.0-xweight)*eqmax(k,xloc+1);
                            [eqfloc, eqfweight] = gridlookup(neq, eqgrid, eqfmax);
                            
                            % Check for exit
                            exitindex = exit(k, xloc);
                            
                            % Apply exogenous exit rate
                            muval = muval * (1.0 - pid);
                            
                            % Update distribution for surviving firms
                            if exitindex == 0
                                % Loop over future productivity states
                                for ief = 1:neps
                                    % Calculate transition probability
                                    prob = pie(k,ief);
                                    
                                    % Update munext with bilinear interpolation
                                    % Bottom layer
                                    munext(ief,kfloc,lfloc,eqfloc) = munext(ief,kfloc,lfloc,eqfloc) + ...
                                        kfweight*lfweight*eqfweight*prob*muval;
                                    
                                    munext(ief,kfloc,lfloc,eqfloc+1) = munext(ief,kfloc,lfloc,eqfloc+1) + ...
                                        kfweight*lfweight*(1.0-eqfweight)*prob*muval;
                                    
                                    munext(ief,kfloc,lfloc+1,eqfloc) = munext(ief,kfloc,lfloc+1,eqfloc) + ...
                                        kfweight*(1.0-lfweight)*eqfweight*prob*muval;
                                    
                                    munext(ief,kfloc,lfloc+1,eqfloc+1) = munext(ief,kfloc,lfloc+1,eqfloc+1) + ...
                                        kfweight*(1.0-lfweight)*(1.0-eqfweight)*prob*muval;
                                    
                                    % Top layer
                                    munext(ief,kfloc+1,lfloc,eqfloc) = munext(ief,kfloc+1,lfloc,eqfloc) + ...
                                        (1.0-kfweight)*lfweight*eqfweight*prob*muval;
                                    
                                    munext(ief,kfloc+1,lfloc,eqfloc+1) = munext(ief,kfloc+1,lfloc,eqfloc+1) + ...
                                        (1.0-kfweight)*lfweight*(1.0-eqfweight)*prob*muval;
                                    
                                    munext(ief,kfloc+1,lfloc+1,eqfloc) = munext(ief,kfloc+1,lfloc+1,eqfloc) + ...
                                        (1.0-kfweight)*(1.0-lfweight)*eqfweight*prob*muval;
                                    
                                    munext(ief,kfloc+1,lfloc+1,eqfloc+1) = munext(ief,kfloc+1,lfloc+1,eqfloc+1) + ...
                                        (1.0-kfweight)*(1.0-lfweight)*(1.0-eqfweight)*prob*muval;
                                end
                            end
                            
                            % Update aggregates
                            firmnum = firmnum + (1.0-exitindex)*muval;
                            kagg = kagg + kval*muval;
                            kfagg = kfagg + kfmax*muval;
                            lagg = lagg + lval*muval;
                            exitnum = exitnum + exitindex*muval;
                        end
                    end
                end
            end
        end
        
        % Add new entrants based on profitability
        for ie = 1:neps
            if profitable_entry(ie)
                % Calculate mass of entrants for this productivity level
                entry_mass_ie = mass_entrants * entry_eps_dist(ie);
                
                % Calculate initial choices for entrants
                entry_kfval = kmax(ie, initial_xval);
                [entry_kfloc, entry_kfweight] = gridlookup(nk, kgrid, entry_kfval);
                
                entry_lfval = lmax(ie, initial_xval);
                [entry_lfloc, entry_lfweight] = gridlookup(nl, lgrid, entry_lfval);
                
                entry_eqval = eqmax(ie, initial_xval);
                [entry_eqloc, entry_eqweight] = gridlookup(neq, eqgrid, entry_eqval);
                
                % Add entrants to distribution with bilinear interpolation
                munext(ie,entry_kfloc,entry_lfloc,entry_eqloc) = munext(ie,entry_kfloc,entry_lfloc,entry_eqloc) + ...
                    entry_kfweight*entry_lfweight*entry_eqweight*entry_mass_ie;
                
                munext(ie,entry_kfloc,entry_lfloc,entry_eqloc+1) = munext(ie,entry_kfloc,entry_lfloc,entry_eqloc+1) + ...
                    entry_kfweight*entry_lfweight*(1.0-entry_eqweight)*entry_mass_ie;
                
                munext(ie,entry_kfloc,entry_lfloc+1,entry_eqloc) = munext(ie,entry_kfloc,entry_lfloc+1,entry_eqloc) + ...
                    entry_kfweight*(1.0-entry_lfweight)*entry_eqweight*entry_mass_ie;
                
                munext(ie,entry_kfloc,entry_lfloc+1,entry_eqloc+1) = munext(ie,entry_kfloc,entry_lfloc+1,entry_eqloc+1) + ...
                    entry_kfweight*(1.0-entry_lfweight)*(1.0-entry_eqweight)*entry_mass_ie;
                
                munext(ie,entry_kfloc+1,entry_lfloc,entry_eqloc) = munext(ie,entry_kfloc+1,entry_lfloc,entry_eqloc) + ...
                    (1.0-entry_kfweight)*entry_lfweight*entry_eqweight*entry_mass_ie;
                
                munext(ie,entry_kfloc+1,entry_lfloc,entry_eqloc+1) = munext(ie,entry_kfloc+1,entry_lfloc,entry_eqloc+1) + ...
                    (1.0-entry_kfweight)*entry_lfweight*(1.0-entry_eqweight)*entry_mass_ie;
                
                munext(ie,entry_kfloc+1,entry_lfloc+1,entry_eqloc) = munext(ie,entry_kfloc+1,entry_lfloc+1,entry_eqloc) + ...
                    (1.0-entry_kfweight)*(1.0-entry_lfweight)*entry_eqweight*entry_mass_ie;
                
                munext(ie,entry_kfloc+1,entry_lfloc+1,entry_eqloc+1) = munext(ie,entry_kfloc+1,entry_lfloc+1,entry_eqloc+1) + ...
                    (1.0-entry_kfweight)*(1.0-entry_lfweight)*(1.0-entry_eqweight)*entry_mass_ie;
            end
        end
        
        % Check convergence
        error = max(abs(munext(:) - mu(:)));
        
        % Print iteration information
        fprintf('Iteration %4i: error %9.8f, firms %9.8f, exits %9.8f\n', ...
            it, error, firmnum, exitnum);
        
        if error < precisionv
            fprintf('\nDistribution converged after %i iterations\n', it);
            break
        elseif it == maxIterations
            fprintf('\nWarning: Maximum iterations reached without convergence\n');
        end
        
        % Update for next iteration
        mu = munext;
        munext = zeros(neps, nkg, nlg, neqg);
    end
    
    function updateMunext(ief, exitindex, kfloc, lfloc, eqfloc, kfweight, lfweight, eqfweight, pieVal, muval)
        weights = [
            kfweight*lfweight*eqfweight,
            kfweight*lfweight*(1.0-eqfweight),
            kfweight*(1.0-lfweight)*eqfweight,
            kfweight*(1.0-lfweight)*(1.0-eqfweight),
            (1.0-kfweight)*lfweight*eqfweight,
            (1.0-kfweight)*lfweight*(1.0-eqfweight),
            (1.0-kfweight)*(1.0-lfweight)*eqfweight,
            (1.0-kfweight)*(1.0-lfweight)*(1.0-eqfweight)
        ];
        
        indices = [
            kfloc, lfloc, eqfloc;
            kfloc, lfloc, eqfloc+1;
            kfloc, lfloc+1, eqfloc;
            kfloc, lfloc+1, eqfloc+1;
            kfloc+1, lfloc, eqfloc;
            kfloc+1, lfloc, eqfloc+1;
            kfloc+1, lfloc+1, eqfloc;
            kfloc+1, lfloc+1, eqfloc+1
        ];
        
        for idx = 1:8
            munext(ief, indices(idx,1), indices(idx,2), indices(idx,3)) = ...
            munext(ief, indices(idx,1), indices(idx,2), indices(idx,3)) + weights(idx) * pieVal * muval * (1.0-exitindex);
        end
    end
    
end


function [yagg, agg_roe, agg_roa, avg_roe, avg_roa, agg_payout_ratio, avg_payout_ratio] = aggregation(mu, neps, nkg, nlg, neqg, evec, kagrid, lagrid, eqagrid, alpha, nu, tauc, rexp, phi, wage, delta, xiparam, nx, xgrid, kmax, lmax, eqmax, nk, kgrid, nl, lgrid, neq, eqgrid, pie, precisionv, maxIterations, qrf, taud, lambda)
% Initialize aggregate variables
    yagg = 0.0;
    total_profit = 0.0;
    total_equity = 0.0;
    total_capital = 0.0;
    total_dividends = 0.0;
    
    % Initialize variables for averages
    sum_roe = 0.0;
    sum_roa = 0.0;
    sum_payout_ratio = 0.0;
    total_firms = 0.0;
    total_profitable_firms = 0.0;  % Only count firms with positive profits for payout ratio

    for h = 1:neqg
        eqval = eqagrid(h);
        for i = 1:nlg
            lval = lagrid(i);
            for j = 1:nkg
                kval = kagrid(j);
                for k = 1:neps
                    muval = mu(k,j,i,h);
                    if muval > 0
                        % Calculate production
                        tfpval = evec(k);
                        yval = tfpval * kval^alpha * lval^nu;
                        yagg = yagg + yval * muval;

                        % Calculate profit components
                        revenue = yval;
                        labor_cost = (1.0 - phi) * wage * lval;
                        depreciation = delta * kval;
                        operating_cost = xiparam;
                        
                        % Calculate pre-tax profit
                        taxable_profit = revenue - labor_cost - depreciation - operating_cost;
                        
                        % Calculate profit after tax
                        if taxable_profit > 0
                            profit = (1.0 - tauc) * taxable_profit;
                        else
                            profit = taxable_profit;
                        end

                        % Calculate dividend for this state
                        xval = profit + tauc * rexp * (kval - eqval) + eqval;
                        additional_dividend = dividend(xval, xiparam, phi, wage, qrf, tauc, kval, lval, eqval);
                        
                        if additional_dividend >= 0
                            div_amount = (1 - taud) * additional_dividend;
                        else
                            div_amount = (1 + lambda) * additional_dividend;
                        end

                        % Accumulate weighted values
                        total_profit = total_profit + profit * muval;
                        total_equity = total_equity + eqval * muval;
                        total_capital = total_capital + kval * muval;
                        total_dividends = total_dividends + div_amount * muval;
                        
                        % Calculate firm-level ratios
                        if eqval > 0
                            firm_roe = profit / eqval;
                        else
                            firm_roe = 0;
                        end
                        
                        if kval > 0
                            firm_roa = profit / kval;
                        else
                            firm_roa = 0;
                        end
                        
                        % Calculate firm-level payout ratio only for profitable firms
                        if profit > 0
                            firm_payout_ratio = div_amount / profit;
                            sum_payout_ratio = sum_payout_ratio + firm_payout_ratio * muval;
                            total_profitable_firms = total_profitable_firms + muval;
                        end
                        
                        % Accumulate for averages
                        sum_roe = sum_roe + firm_roe * muval;
                        sum_roa = sum_roa + firm_roa * muval;
                        total_firms = total_firms + muval;
                    end
                end
            end
        end
    end

    % Calculate aggregate ratios
    if total_equity > 0
        agg_roe = total_profit / total_equity;
    else
        agg_roe = 0;
    end
    
    if total_capital > 0
        agg_roa = total_profit / total_capital;
    else
        agg_roa = 0;
    end
    
    % Calculate aggregate payout ratio
    if total_profit > 0
        agg_payout_ratio = total_dividends / total_profit;
    else
        agg_payout_ratio = 0;
    end
    
    % Calculate averages
    avg_roe = sum_roe / total_firms;
    avg_roa = sum_roa / total_firms;
    
    % Calculate average payout ratio (only for profitable firms)
    if total_profitable_firms > 0
        avg_payout_ratio = sum_payout_ratio / total_profitable_firms;
    else
        avg_payout_ratio = 0;
    end
end

function plot_qarray_relationships(qarray, neps, nk, nl, neq, evec, kgrid, lgrid, eqgrid)
    % Check dimensions of qarray
    [dim1, dim2, dim3, dim4] = size(qarray);
    
    % Calculate middle indices (considering odd numbers)
    mid1 = ceil(dim1/2);
    mid2 = ceil(dim2/2);
    mid3 = ceil(dim3/2);
    mid4 = ceil(dim4/2);

    % Create a new figure with 2x2 subplots
    figure('Position', [100, 100, 1000, 800]);

    % 1. Relationship between productivity shock (eps) and qarray
    subplot(2, 2, 1);
    plot(evec, qarray(:, mid2, mid3, mid4));
    title('Productivity Shock vs q');
    xlabel('Productivity Shock');
    ylabel('q');

    % 2. Relationship between capital (k) and qarray
    subplot(2, 2, 2);
    plot(kgrid, squeeze(qarray(mid1, :, mid3, mid4)));
    title('Capital vs q');
    xlabel('Capital');
    ylabel('q');

    % 3. Relationship between labor (l) and qarray
    subplot(2, 2, 3);
    plot(lgrid, squeeze(qarray(mid1, mid2, :, mid4)));
    title('Labor vs q');
    xlabel('Labor');
    ylabel('q');

    % 4. Relationship between equity (eq) and qarray for different productivity levels
    subplot(2, 2, 4);
    hold on;
    
    % Low productivity (first element)
    plot(eqgrid, squeeze(qarray(1, mid2, mid3, :)), 'r-', 'LineWidth', 2);
    
    % Medium productivity (middle element)
    plot(eqgrid, squeeze(qarray(mid1, mid2, mid3, :)), 'g-', 'LineWidth', 2);
    
    % High productivity (last element)
    plot(eqgrid, squeeze(qarray(end, mid2, mid3, :)), 'b-', 'LineWidth', 2);
    
    title('Equity vs q for Different Productivity Levels');
    xlabel('Equity');
    ylabel('q');
    legend('Low', 'Medium', 'High', 'Location', 'best');
    hold off;

    % Adjust the layout and add a main title
    sgtitle('Relationships between Various Factors and q');
    set(gcf, 'Color', 'w');
end

function [profitable_entry, entry_eps_dist, expected_profits] = check_entry_profitability(neps, evec, xgrid, initial_xval, alpha, nu, wage, xiparam, kmax, lmax)
    % AR(1)パラメータ 
    rho_s = 0.9;
    sigma_eps = 0.0335;
    
    % 対数正規分布のパラメータ
    sigma_s = sigma_eps / sqrt(1 - rho_s^2);
    mu_s = -0.5 * sigma_s^2;
    
    % evecはそのまま使用し、確率分布のみ対数正規で計算
    entry_eps_dist = lognpdf(evec, mu_s, sigma_s);
    entry_eps_dist = entry_eps_dist / sum(entry_eps_dist);  % 正規化
    
    % 参入判断の初期化
    profitable_entry = zeros(1, neps);
    expected_profits = zeros(1, neps);
    
    % 各生産性レベルでの参入判断
    for ie = 1:neps
        s = evec(ie);
        
        k_entry = kmax(ie, initial_xval);
        l_entry = lmax(ie, initial_xval);
        
        revenue = s * (k_entry^alpha) * (l_entry^nu);
        costs = wage * l_entry + xiparam;
        profit = revenue - costs;
        
        expected_profits(ie) = profit;
        profitable_entry(ie) = profit >= 0;
    end
end