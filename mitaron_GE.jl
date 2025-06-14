# Import necessary packages
using Plots
using LinearAlgebra
using Statistics
using Printf
using Distributions

# Parameters structure to keep code organized
struct ModelParameters
    alpha::Float64  # Share of Capital
    nu::Float64    # Share of Labour
    phi::Float64   # working capital constraint partial switch
    delta::Float64 # capital depreciation rate
    beta::Float64  # subjective discount factor
    qrf::Float64   # risk free discount factor
    rexp::Float64  # interest rate expense parameter
    rrf::Float64   # risk free interest rate
    tauc::Float64  # corporate tax level
    taud::Float64  # dividend tax level
    lambda::Float64 # equity funding cost
    eta::Float64   # preference parameter on leisure
    pid::Float64   # exogenous exit rate for firms
    xi::Float64    # Fixed operating cost
    k0common::Float64  # COMMON CAPITAL OF NEWBORNS
    dcost::Float64
    p0::Float64
    wage::Float64
    mass_entrants::Float64
    initial_xval::Int64
    roecutoff::Float64
    min_payout_ratio::Float64
end

# Grid structure to keep grid parameters organized
struct ModelGrids
    # Grid sizes
    nk::Int64
    nl::Int64
    neq::Int64
    nx::Int64
    neps::Int64
    
    # Grid bounds
    klow::Float64
    khigh::Float64
    llow::Float64
    lhigh::Float64
    eqlow::Float64
    eqhigh::Float64
    xlow::Float64
    xhigh::Float64
    
    # Actual grids
    kgrid::Vector{Float64}
    lgrid::Vector{Float64}
    eqgrid::Vector{Float64}
    xgrid::Vector{Float64}
end

"""
Initialize model parameters
"""
function init_parameters()
    return ModelParameters(
        0.2,    # alpha
        0.64,   # nu
        0.0,    # phi
        0.089,  # delta
        0.99,   # beta
        0.99,   # qrf
        0.01,   # rexp
        0.0101, # rrf
        0.232,  # tauc
        0.203,    # taud
        0.08,   # lambda
        2.4,    # eta
        0.02,   # pid
        0.0715,   # xi
        0.35,   # k0common
        0.59,   # dcost
        2.5,    # p0
        0.95,  # wage
        0.02,   # mass_entrants
        5,      # initial_xval
        0.2,    # roecutoff
        0.0     # min_payout_ratio
    )
end

"""
Initialize model grids
"""
function init_grids()
    # Grid sizes
    nk = 60
    nl = 60
    neq = 50
    nx = 65
    neps = 5
    
    # Grid bounds
    klow, khigh = 0.0, 3.0
    llow, lhigh = 0.0, 1.2
    eqlow, eqhigh = 0.0, 2.5
    xlow, xhigh = -0.2, 3.0
    
    # Create grids
    kgrid = range(klow, khigh, length=nk)
    lgrid = range(llow, lhigh, length=nl)
    eqgrid = range(eqlow, eqhigh, length=neq)
    xgrid = range(xlow, xhigh, length=nx)
    
    return ModelGrids(
        nk, nl, neq, nx, neps,
        klow, khigh, llow, lhigh, eqlow, eqhigh, xlow, xhigh,
        collect(kgrid), collect(lgrid), collect(eqgrid), collect(xgrid)
    )
end

"""
Rouwenhorst method for discretizing AR(1) process
"""
function rouwenhorst(rho::Float64, sigma::Float64, n::Int64)
    p = (rho + 1.0) / 2.0
    q = p
    
    # Initialize matrices
    hlag = zeros(n, n)
    hlag[1, 1] = 1.0
    
    for i in 2:n
        h = zeros(i, i)
        h[1:i-1, 1:i-1] = p * hlag[1:i-1, 1:i-1]
        h[1:i-1, 2:i] = h[1:i-1, 2:i] .+ (1.0 - p) * hlag[1:i-1, 1:i-1]
        h[2:i, 1:i-1] = h[2:i, 1:i-1] .+ (1.0 - q) * hlag[1:i-1, 1:i-1]
        h[2:i, 2:i] = h[2:i, 2:i] .+ q * hlag[1:i-1, 1:i-1]
        h[2:i-1, :] = h[2:i-1, :] ./ 2.0
        hlag[1:i, 1:i] = h
    end
    
    pi = hlag
    
    # Calculate z values
    zvar = (sigma^2.0) / (1.0 - rho^2.0)
    eps_z = sqrt((n - 1.0) * zvar)
    
    z = zeros(n)
    z[1] = -eps_z
    z[n] = eps_z
    
    if n > 2
        y = range(-eps_z, eps_z, length=n)[2:end-1]
        z[2:end-1] = y
    end
    
    return exp.(z), pi
end



"""
Grid lookup function
"""
function gridlookup(xdist::Vector{Float64}, xval::Float64)
    n = length(xdist)
    if xval <= xdist[1]
        return 1, 1.0
    elseif xval >= xdist[end]
        return n-1, 0.0
    end
    
    ixlow = 1
    ixhigh = n
    
    while ixhigh - ixlow > 1
        ixmid = ((ixhigh + ixlow) ÷ 2)
        if xdist[ixmid] >= xval
            ixhigh = ixmid
        else
            ixlow = ixmid
        end
    end
    
    weight = (xdist[ixlow + 1] - xval) / (xdist[ixlow + 1] - xdist[ixlow])
    return ixlow, weight
end

"""
Calculate dividend
"""
function dividend(xval::Float64, p::ModelParameters, wage::Float64, qval::Float64, kfval::Float64, lfval::Float64, eqfval::Float64)
    return xval - (1.0 - p.tauc) * p.phi * wage * lfval + qval * (kfval - eqfval) - kfval
end




"""
Calculate default threshold
"""
function default_threshold(xgrid::Vector{Float64}, ixdhigh::Int64, vc0vec::Vector{Float64}, precision::Float64)
    xdlow = xgrid[1]
    xdhigh = xgrid[ixdhigh]
    
    flow = vc0vec[1]
    fhigh = vc0vec[ixdhigh]
    
    if fhigh < 0.0
        @printf(" Default threshold above highest x considered: %f; fhigh: %f.\n", xgrid[ixdhigh], fhigh)
        return xgrid[ixdhigh]
    elseif flow > 0.0
        @printf(" Default threshold lower than lowest x considered: %f; flow: %f.\n", xgrid[1], flow)
        return xgrid[1]
    else
        while abs(xdhigh - xdlow) > precision
            xdeval = (xdlow + xdhigh) / 2.0
            idx, weight = gridlookup(xgrid, xdeval)
            vc0val = weight * vc0vec[idx] + (1.0 - weight) * vc0vec[idx + 1]
            
            if vc0val < 0.0
                xdlow = xdeval
            else
                xdhigh = xdeval
            end
        end
        return (xdlow + xdhigh) / 2.0
    end
end

"""
Calculate equity values for different productivity levels
"""
function equityvec(evec::Vector{Float64}, p::ModelParameters, kfval::Float64, lfval::Float64, eqfval::Float64)
    capterm = kfval ^ p.alpha
    labterm = lfval ^ p.nu
    debterm = p.tauc * p.rexp * (kfval - eqfval)
    wageterm = (1.0 - p.phi) * p.wage * lfval
    deprterm = p.delta * kfval
    
    xfval = zeros(length(evec))
    profitfval = zeros(length(evec))
    
    for (i, tfpval) in enumerate(evec)
        term1 = tfpval * capterm * labterm
        taxterm = term1 - wageterm - deprterm - p.xi
        
        # Modified tax calculation
        profitfval[i] = taxterm > 0 ? (1.0 - p.tauc) * taxterm : taxterm
        xfval[i] = profitfval[i] + debterm + eqfval
    end
    
    return xfval, profitfval
end

"""
Main operator for value function iteration
"""
function operator(xval::Float64, ie::Int, v::Matrix{Float64}, grids::ModelGrids, p::ModelParameters,
                evec::Vector{Float64}, pie::Matrix{Float64}, xd::Float64, qarray::Array{Float64,4},
                roecutoff::Float64, min_payout_ratio::Float64)
    
    # Initialize output values
    vcmax = -1000.0
    kmax_val = lmax_val = eqmax_val = roemax_val = roamax_val = leveragemax_val = divmax_val = 0.0
    
    # Create arrays for objectives
    objective = zeros(grids.nk, grids.nl, grids.neq)
    
    # Main loop over capital, labor, and equity
    for ikv in 1:grids.nk
        kfval = grids.kgrid[ikv]
        temp_objective = zeros(grids.nl, grids.neq)
        
        for ilv in 1:grids.nl
            lfval = grids.lgrid[ilv]
            
            for ieq in 1:grids.neq
                eqfval = grids.eqgrid[ieq]
                
                # Calculate profit components
                xfval, profitfmaxvec = equityvec(evec, p, kfval, lfval, eqfval)
                profitfmax = sum(pie[ie, :] .* profitfmaxvec)
                current_roe = profitfmax / eqfval
                
                # New dividend policy based on ROE condition
                if current_roe < roecutoff
                    min_dividend = min_payout_ratio * eqfval
                    additional_dividend = dividend(xval - min_dividend, p, p.wage, 
                                                qarray[ie, ikv, ilv, ieq], kfval, lfval, eqfval)
                    
                    dval = if additional_dividend >= 0
                        (1 - p.taud) * (min_dividend + additional_dividend)
                    else
                        (1 - p.taud) * min_dividend + (1 + p.lambda) * additional_dividend
                    end
                else
                    additional_dividend = dividend(xval, p, p.wage, 
                                                qarray[ie, ikv, ilv, ieq], kfval, lfval, eqfval)
                    dval = if additional_dividend >= 0
                        (1 - p.taud) * additional_dividend
                    else
                        (1 + p.lambda) * additional_dividend
                    end
                end
                
                # Calculate expected future value
                evalue = zeros(grids.neps)
                for iie in 1:grids.neps
                    xloc, weight = gridlookup(grids.xgrid, xfval[iie])
                    xloc = clamp(xloc, 1, grids.nx-1)
                    weight = clamp(weight, 0.0, 1.0)
                    
                    if xloc >= grids.nx - 1
                        evalue[iie] = v[iie, grids.nx]
                    elseif xloc < 1
                        evalue[iie] = v[iie, 1]
                    else
                        evalue[iie] = weight * v[iie, xloc] + (1.0 - weight) * v[iie, xloc + 1]
                    end
                end
                
                # Calculate objective function
                vmax = sum(pie[ie, :] .* evalue)
                temp_objective[ilv, ieq] = dval + p.beta * vmax
            end
        end
        objective[ikv, :, :] = temp_objective
    end
    
    # Find maximum value and corresponding indices
    max_val, idx = findmax(objective)
    if max_val > vcmax
        vcmax = max_val
        idx_k, idx_l, idx_eq = Tuple(CartesianIndices(size(objective))[idx])
        
        kmax_val = grids.kgrid[idx_k]
        lmax_val = grids.lgrid[idx_l]
        eqmax_val = grids.eqgrid[idx_eq]
        
        # Calculate additional maximum values
        _, profitfmaxvec = equityvec(evec, p, kmax_val, lmax_val, eqmax_val)
        profitfmax = sum(pie[ie, :] .* profitfmaxvec)
        roemax_val = profitfmax / eqmax_val
        roamax_val = profitfmax / kmax_val
        leveragemax_val = kmax_val / eqmax_val
        
        # Calculate divmax with same logic as main calculation
        if roemax_val < roecutoff
            min_dividend = min_payout_ratio * eqmax_val
            additional_dividend = dividend(xval - min_dividend, p, p.wage,
                                        qarray[ie, idx_k, idx_l, idx_eq], kmax_val, lmax_val, eqmax_val)
            
            divmax_val = if additional_dividend >= 0
                (1 - p.taud) * (min_dividend + additional_dividend)
            else
                (1 - p.taud) * min_dividend + (1 + p.lambda) * additional_dividend
            end
        else
            additional_dividend = dividend(xval, p, p.wage,
                                        qarray[ie, idx_k, idx_l, idx_eq], kmax_val, lmax_val, eqmax_val)
            divmax_val = if additional_dividend >= 0
                (1 - p.taud) * additional_dividend
            else
                (1 + p.lambda) * additional_dividend
            end
        end
    end
    
    return vcmax, kmax_val, lmax_val, eqmax_val, roemax_val, roamax_val, leveragemax_val, divmax_val
end

"""
Value function iteration with loan pricing
"""
function contraction(vin::Matrix{Float64}, grids::ModelGrids, p::ModelParameters,
                    evec::Vector{Float64}, pie::Matrix{Float64}, xd::Vector{Float64},
                    qarray::Array{Float64,4}, precision_v::Float64)
    
    distance = 100.0 * precision_v
    iteration_v = 0
    v = copy(vin)
    vc = zeros(grids.neps, grids.nx)
    vc0 = zeros(grids.neps, grids.nx)
    exit = zeros(grids.neps, grids.nx)
    
    # Arrays to store optimal decisions
    kmax = zeros(grids.neps, grids.nx)
    lmax = zeros(grids.neps, grids.nx)
    eqmax = zeros(grids.neps, grids.nx)
    roemax = zeros(grids.neps, grids.nx)
    roamax = zeros(grids.neps, grids.nx)
    leveragemax = zeros(grids.neps, grids.nx)
    divmax = zeros(grids.neps, grids.nx)
    
    while true
        if distance <= precision_v || iteration_v >= 30
            if distance > precision_v
                println("leaving value fcn do loop; max iterations reached.")
            end
            break
        end
        
        iteration_v += 1
        println("Iteration ", iteration_v)
        
        for ie in 1:grids.neps
            println("Epsilon ", ie)
            
            for ix in 1:grids.nx
                xval = grids.xgrid[ix]
                vc[ie, ix], kmax[ie, ix], lmax[ie, ix], eqmax[ie, ix],
                roemax[ie, ix], roamax[ie, ix], leveragemax[ie, ix], divmax[ie, ix] = 
                    operator(xval, ie, v, grids, p, evec, pie, xd[ie], qarray,
                            p.roecutoff, p.min_payout_ratio)
                
                vc0[ie, ix] = p.pid * (1.0 - p.taud) * max(xval, 0.0) + (1.0 - p.pid) * vc[ie, ix]
                v[ie, ix] = max(vc0[ie, ix], 0.0)
                
                if v[ie, ix] < 0
                    exit[ie, ix] = 1
                end
            end
        end
        
        distance = maximum(abs.(v - vin))
        if iteration_v <= 10 || iteration_v % 10 == 0
            @printf(" v(epsilon, x) iteration %d normv = %f\n", iteration_v, distance)
        end
        vin = copy(v)
    end
    
    return v, vc, vc0, kmax, lmax, eqmax, roemax, roamax, leveragemax, divmax, exit
end

"""
Calculate loan schedule
"""
function loanschedule(xd::Vector{Float64}, grids::ModelGrids, p::ModelParameters, evec::Vector{Float64}, pie::Matrix{Float64}, qarray::Array{Float64,4})
    
    for ie in 1:grids.neps
        pievec = pie[ie, :]
        
        for ikv in 1:grids.nk
            kfval = grids.kgrid[ikv]
            
            for ilv in 1:grids.nl
                lfval = grids.lgrid[ilv]
                
                for ieq in 1:grids.neq
                    eqfval = grids.eqgrid[ieq]
                    
                    if eqfval >= kfval
                        qarray[ie, ikv, ilv, ieq] = p.qrf
                    else
                        xfval, _ = equityvec(evec, p, kfval, lfval, eqfval)
                        payout = 0.0
                        
                        for ief in 1:grids.neps
                            if xfval[ief] <= xd[ief]
                                payout += pievec[ief] * (1-p.delta) * (p.dcost * kfval)
                            else
                                payout += pievec[ief] * ((1.0 - p.pid) * (kfval - eqfval) + 
                                         p.pid * min((1-p.delta)*(p.dcost * kfval), (kfval - eqfval)))
                            end
                        end
                        
                        qnew = max(0, payout / (kfval - eqfval))
                        qarray[ie, ikv, ilv, ieq] = min(qnew, p.qrf)
                    end
                end
            end
        end
    end
    
    return qarray
end

"""
Check entry profitability
"""
function check_entry_profitability(evec::Vector{Float64}, grids::ModelGrids, p::ModelParameters,
                                kmax::Matrix{Float64}, lmax::Matrix{Float64})
    # AR(1) parameters
    rho_s = 0.9
    sigma_eps = 0.0335
    
    # Log-normal distribution parameters
    sigma_s = sigma_eps / sqrt(1 - rho_s^2)
    mu_s = -0.5 * sigma_s^2
    
    # Calculate entry probability distribution
    d = LogNormal(mu_s, sigma_s)
    entry_eps_dist = pdf.(d, evec)
    entry_eps_dist = entry_eps_dist ./ sum(entry_eps_dist)  # Normalize
    
    # Initialize arrays
    profitable_entry = zeros(Bool, grids.neps)
    expected_profits = zeros(grids.neps)
    
    # Check profitability for each productivity level
    for ie in 1:grids.neps
        s = evec[ie]
        
        k_entry = kmax[ie, p.initial_xval]
        l_entry = lmax[ie, p.initial_xval]
        
        revenue = s * (k_entry^p.alpha) * (l_entry^p.nu)
        costs = p.wage * l_entry + p.xi
        profit = revenue - costs
        
        expected_profits[ie] = profit
        profitable_entry[ie] = profit >= 0
    end
    
    return profitable_entry, entry_eps_dist, expected_profits
end

"""
Calculate firm distribution
"""
function distribution(grids::ModelGrids, p::ModelParameters, evec::Vector{Float64}, pie::Matrix{Float64},
    kmax::Matrix{Float64}, lmax::Matrix{Float64}, eqmax::Matrix{Float64},
    exit::Matrix{Float64}, precision_v::Float64, max_iterations::Int64)

# Initialize distribution arrays
mu = zeros(grids.neps, grids.nk, grids.nl, grids.neq)
munext = zeros(grids.neps, grids.nk, grids.nl, grids.neq)

# Initialize aggregate variables
firmnum = 0.0
exitnum = 0.0
kagg = 0.0
kfagg = 0.0
lagg = 0.0

# Initial condition
mu[5, 35, 35, 45] = 1

# Calculate entry profitability
profitable_entry, entry_eps_dist, _ = check_entry_profitability(evec, grids, p, kmax, lmax)

# Main iteration loop
for it in 1:max_iterations
# Reset aggregate variables for each iteration
firmnum = 0.0
exitnum = 0.0
kagg = 0.0
kfagg = 0.0
lagg = 0.0

# Loop over all states
for h in 1:grids.neq
eqval = grids.eqgrid[h]
for i in 1:grids.nl
lval = grids.lgrid[i]
for j in 1:grids.nk
    kval = grids.kgrid[j]
    for k in 1:grids.neps
        tfpval = evec[k]
        muval = mu[k,j,i,h]
        
        if muval > 0
            # Calculate cash position
            xfval, _ = equityvec(evec, p, kval, lval, eqval)
            
            # Grid lookups for optimal choices
            xloc, xweight = gridlookup(grids.xgrid, xfval[k])
            xweight = clamp(xweight, 0.0, 1.0)
            
            # Calculate optimal choices
            kfmax = xweight*kmax[k,xloc] + (1.0-xweight)*kmax[k,min(xloc+1,grids.nx)]
            kfloc, kfweight = gridlookup(grids.kgrid, kfmax)
            
            lfmax = xweight*lmax[k,xloc] + (1.0-xweight)*lmax[k,min(xloc+1,grids.nx)]
            lfloc, lfweight = gridlookup(grids.lgrid, lfmax)
            
            eqfmax = xweight*eqmax[k,xloc] + (1.0-xweight)*eqmax[k,min(xloc+1,grids.nx)]
            eqfloc, eqfweight = gridlookup(grids.eqgrid, eqfmax)
            
            # Apply exogenous exit rate
            muval *= (1.0 - p.pid)
            
            # Update distribution for surviving firms
            if exit[k,xloc] == 0
                update_distribution!(munext, k, kfloc, lfloc, eqfloc, 
                                    kfweight, lfweight, eqfweight, 
                                    pie[k,:], muval, grids.neps)
            end
            
            # Update aggregates
            firmnum += (1.0-exit[k,xloc])*muval
            kagg += kval*muval
            kfagg += kfmax*muval
            lagg += lval*muval
            exitnum += exit[k,xloc]*muval
        end
    end
end
end
end

# Add new entrants
for ie in 1:grids.neps
if profitable_entry[ie]
entry_mass_ie = p.mass_entrants * entry_eps_dist[ie]

# Calculate initial choices for entrants
entry_kfval = kmax[ie, round(Int, p.initial_xval)]
entry_kfloc, entry_kfweight = gridlookup(grids.kgrid, entry_kfval)

entry_lfval = lmax[ie, round(Int, p.initial_xval)]
entry_lfloc, entry_lfweight = gridlookup(grids.lgrid, entry_lfval)

entry_eqval = eqmax[ie, round(Int, p.initial_xval)]
entry_eqloc, entry_eqweight = gridlookup(grids.eqgrid, entry_eqval)

# Add entrants to distribution
update_distribution!(munext, ie, entry_kfloc, entry_lfloc, entry_eqloc,
                    entry_kfweight, entry_lfweight, entry_eqweight,
                    [ie == j ? 1.0 : 0.0 for j in 1:grids.neps],
                    entry_mass_ie, grids.neps)
end
end

# Check convergence
error = maximum(abs.(munext - mu))
@printf("Iteration %4i: error %9.8f, firms %9.8f, exits %9.8f\n",
it, error, firmnum, exitnum)

if error < precision_v
println("\nDistribution converged after $it iterations")
break
elseif it == max_iterations
println("\nWarning: Maximum iterations reached without convergence")
end

# Update for next iteration
mu = copy(munext)
fill!(munext, 0.0)  # Reset munext for next iteration
end

return mu, firmnum, kagg, kfagg, lagg
end

"""
Helper function to update distribution
"""
function update_distribution!(munext::Array{Float64,4}, k::Int64, kfloc::Int64, lfloc::Int64, 
                            eqfloc::Int64, kfweight::Float64, lfweight::Float64, 
                            eqfweight::Float64, pievec::Vector{Float64}, muval::Float64, neps::Int64)
    
    weights = [
        kfweight*lfweight*eqfweight,
        kfweight*lfweight*(1.0-eqfweight),
        kfweight*(1.0-lfweight)*eqfweight,
        kfweight*(1.0-lfweight)*(1.0-eqfweight),
        (1.0-kfweight)*lfweight*eqfweight,
        (1.0-kfweight)*lfweight*(1.0-eqfweight),
        (1.0-kfweight)*(1.0-lfweight)*eqfweight,
        (1.0-kfweight)*(1.0-lfweight)*(1.0-eqfweight)
    ]
    
    indices = [
        (kfloc, lfloc, eqfloc),
        (kfloc, lfloc, eqfloc+1),
        (kfloc, lfloc+1, eqfloc),
        (kfloc, lfloc+1, eqfloc+1),
        (kfloc+1, lfloc, eqfloc),
        (kfloc+1, lfloc, eqfloc+1),
        (kfloc+1, lfloc+1, eqfloc),
        (kfloc+1, lfloc+1, eqfloc+1)
    ]
    
    for ief in 1:neps
        for (idx, w) in zip(indices, weights)
            munext[ief, idx...] += w * pievec[ief] * muval
        end
    end
end

"""
Calculate aggregation
"""
function aggregation(mu::Array{Float64,4}, grids::ModelGrids, p::ModelParameters, 
                    evec::Vector{Float64}, kmax::Matrix{Float64}, lmax::Matrix{Float64},
                    eqmax::Matrix{Float64})
    
    # Initialize aggregate variables
    yagg = total_profit = total_equity = total_capital = total_dividends = 0.0
    sum_roe = sum_roa = sum_payout_ratio = 0.0
    total_firms = total_profitable_firms = 0.0
    
    for h in 1:grids.neq
        eqval = grids.eqgrid[h]
        for i in 1:grids.nl
            lval = grids.lgrid[i]
            for j in 1:grids.nk
                kval = grids.kgrid[j]
                for k in 1:grids.neps
                    muval = mu[k,j,i,h]
                    if muval > 0
                        # Production and profits
                        tfpval = evec[k]
                        yval = tfpval * kval^p.alpha * lval^p.nu
                        
                        # Calculate components
                        revenue = yval
                        labor_cost = (1.0 - p.phi) * p.wage * lval
                        depreciation = p.delta * kval
                        operating_cost = p.xi
                        
                        # Calculate profits
                        taxable_profit = revenue - labor_cost - depreciation - operating_cost
                        profit = taxable_profit > 0 ? (1.0 - p.tauc) * taxable_profit : taxable_profit
                        
                        # Calculate dividends
                        xval = profit + p.tauc * p.rexp * (kval - eqval) + eqval
                        additional_dividend = dividend(xval, p, p.wage, p.qrf, kval, lval, eqval)
                        div_amount = additional_dividend >= 0 ? 
                                   (1 - p.taud) * additional_dividend :
                                   (1 + p.lambda) * additional_dividend
                        
                        # Accumulate weighted values
                        yagg += yval * muval
                        total_profit += profit * muval
                        total_equity += eqval * muval
                        total_capital += kval * muval
                        total_dividends += div_amount * muval
                        
                        # Calculate firm-level ratios
                        firm_roe = eqval > 0 ? profit / eqval : 0.0
                        firm_roa = kval > 0 ? profit / kval : 0.0
                        
                        if profit > 0
                            firm_payout_ratio = div_amount / profit
                            sum_payout_ratio += firm_payout_ratio * muval
                            total_profitable_firms += muval
                        end
                        
                        sum_roe += firm_roe * muval
                        sum_roa += firm_roa * muval
                        total_firms += muval
                    end
                end
            end
        end
    end
    
    # Calculate aggregate ratios
    agg_roe = total_equity > 0 ? total_profit / total_equity : 0.0
    agg_roa = total_capital > 0 ? total_profit / total_capital : 0.0
    agg_payout_ratio = total_profit > 0 ? total_dividends / total_profit : 0.0
    
    # Calculate averages
    avg_roe = total_firms > 0 ? sum_roe / total_firms : 0.0
    avg_roa = total_firms > 0 ? sum_roa / total_firms : 0.0
    avg_payout_ratio = total_profitable_firms > 0 ? sum_payout_ratio / total_profitable_firms : 0.0
    
    return yagg, agg_roe, agg_roa, avg_roe, avg_roa, agg_payout_ratio, avg_payout_ratio
end

using Plots
using ColorSchemes

"""
Plot the value function
"""
function plot_value_function(v::Matrix{Float64}, xgrid::Vector{Float64}, evec::Vector{Float64})
    # Create a figure with two subplots
    p = plot(layout=(1,2), size=(1200,400), margin=10Plots.px)
    
    # 3D surface plot
    X = repeat(xgrid', length(evec), 1)
    Y = repeat(evec, 1, length(xgrid))
    
    # Surface plot
    surface!(p[1], xgrid, evec, v,
            xlabel="Cash Position (x)",
            ylabel="Productivity (ε)",
            zlabel="Value",
            title="Firm Value Function",
            camera=(45,30),
            colorbar=true,
            c=:jet)
    
    # Line plot for different productivity levels
    plot!(p[2], xgrid, v[1,:], 
        label="Low ε", 
        linestyle=:dash, 
        linewidth=2, 
        color=:red)
    plot!(p[2], xgrid, v[3,:], 
        label="Medium ε", 
        linewidth=2, 
        color=:green)
    plot!(p[2], xgrid, v[5,:], 
        label="High ε", 
        linewidth=2, 
        color=:blue,
        xlabel="Cash Position (x)",
        ylabel="Value",
        title="Value Function by Productivity Level")
    
    return p
end

"""
Plot the distribution of firms
"""
function plot_distribution(mu::Array{Float64,4})
    # Aggregate over labor and equity dimensions
    mu1 = sum(sum(mu, dims=4), dims=3)[:,:,1,1]
    
    # Create surface plot
    p = surface(mu1,
                xlabel="Capital",
                ylabel="Productivity",
                title="Distribution of firms",
                camera=(45,30),
                colorbar=true,
                c=:viridis)
    
    return p
end

"""
Plot decisions
"""
function plot_decisions(xgrid::Vector{Float64}, kmax::Matrix{Float64}, lmax::Matrix{Float64}, 
                        eqmax::Matrix{Float64}, roemax::Matrix{Float64}, roamax::Matrix{Float64},
                        leveragemax::Matrix{Float64}, divmax::Matrix{Float64})
    
    # Calculate bmax
    bmax = kmax .- eqmax
    
    # Create figure with subplots
    p = plot(layout=(2,4), size=(1600,800), margin=10Plots.px)
    
    # Plot each decision variable
    # Capital
    plot!(p[1], xgrid, kmax[1,:], label="Low ε", linestyle=:dash, color=:red, linewidth=2)
    plot!(p[1], xgrid, kmax[3,:], label="Medium ε", color=:green, linewidth=2)
    plot!(p[1], xgrid, kmax[5,:], label="High ε", color=:blue, linewidth=2)
    title!(p[1], "Capital")
    
    # Labor
    plot!(p[2], xgrid, lmax[1,:], label="Low ε", linestyle=:dash, color=:red, linewidth=2)
    plot!(p[2], xgrid, lmax[3,:], label="Medium ε", color=:green, linewidth=2)
    plot!(p[2], xgrid, lmax[5,:], label="High ε", color=:blue, linewidth=2)
    title!(p[2], "Labour")
    
    # Equity
    plot!(p[3], xgrid, eqmax[1,:], label="Low ε", linestyle=:dash, color=:red, linewidth=2)
    plot!(p[3], xgrid, eqmax[3,:], label="Medium ε", color=:green, linewidth=2)
    plot!(p[3], xgrid, eqmax[5,:], label="High ε", color=:blue, linewidth=2)
    title!(p[3], "Equity")
    
    # Return on Equity
    plot!(p[4], xgrid, roemax[1,:], label="Low ε", linestyle=:dash, color=:red, linewidth=2)
    plot!(p[4], xgrid, roemax[3,:], label="Medium ε", color=:green, linewidth=2)
    plot!(p[4], xgrid, roemax[5,:], label="High ε", color=:blue, linewidth=2)
    title!(p[4], "Return on Equity")
    
    # Dividend
    plot!(p[5], xgrid, divmax[1,:], label="Low ε", linestyle=:dash, color=:red, linewidth=2)
    plot!(p[5], xgrid, divmax[3,:], label="Medium ε", color=:green, linewidth=2)
    plot!(p[5], xgrid, divmax[5,:], label="High ε", color=:blue, linewidth=2)
    title!(p[5], "Dividend")
    
    # Debt
    plot!(p[6], xgrid, bmax[1,:], label="Low ε", linestyle=:dash, color=:red, linewidth=2)
    plot!(p[6], xgrid, bmax[3,:], label="Medium ε", color=:green, linewidth=2)
    plot!(p[6], xgrid, bmax[5,:], label="High ε", color=:blue, linewidth=2)
    title!(p[6], "Debt")
    
    # Leverage
    plot!(p[7], xgrid, leveragemax[1,:], label="Low ε", linestyle=:dash, color=:red, linewidth=2)
    plot!(p[7], xgrid, leveragemax[3,:], label="Medium ε", color=:green, linewidth=2)
    plot!(p[7], xgrid, leveragemax[5,:], label="High ε", color=:blue, linewidth=2)
    title!(p[7], "Leverage")
    
    # Return on Assets
    plot!(p[8], xgrid, roamax[1,:], label="Low ε", linestyle=:dash, color=:red, linewidth=2)
    plot!(p[8], xgrid, roamax[3,:], label="Medium ε", color=:green, linewidth=2)
    plot!(p[8], xgrid, roamax[5,:], label="High ε", color=:blue, linewidth=2)
    title!(p[8], "Return on Assets")
    
    return p
end

"""
Plot loan pricing relationships
"""
function plot_qarray_relationships(qarray::Array{Float64,4}, evec::Vector{Float64}, 
                                kgrid::Vector{Float64}, lgrid::Vector{Float64}, 
                                eqgrid::Vector{Float64})
    
    # Get dimensions
    neps, nk, nl, neq = size(qarray)
    
    # Calculate middle indices
    mid_k = div(nk, 2)
    mid_l = div(nl, 2)
    mid_eq = div(neq, 2)
    
    # Create figure with subplots
    p = plot(layout=(2,2), size=(1000,800), margin=10Plots.px)
    
    # 1. Productivity shock vs q
    plot!(p[1], evec, qarray[:,mid_k,mid_l,mid_eq],
        xlabel="Productivity Shock",
        ylabel="q",
        title="Productivity Shock vs q",
        label="")
    
    # 2. Capital vs q
    plot!(p[2], kgrid, qarray[div(neps,2),:,mid_l,mid_eq],
        xlabel="Capital",
        ylabel="q",
        title="Capital vs q",
        label="")
    
    # 3. Labor vs q
    plot!(p[3], lgrid, qarray[div(neps,2),mid_k,:,mid_eq],
        xlabel="Labor",
        ylabel="q",
        title="Labor vs q",
        label="")
    
    # 4. Equity vs q for different productivity levels
    plot!(p[4], eqgrid, qarray[1,mid_k,mid_l,:],
        label="Low ε",
        linestyle=:dash,
        color=:red,
        linewidth=2)
    plot!(p[4], eqgrid, qarray[div(neps,2),mid_k,mid_l,:],
        label="Medium ε",
        color=:green,
        linewidth=2)
    plot!(p[4], eqgrid, qarray[end,mid_k,mid_l,:],
        label="High ε",
        color=:blue,
        linewidth=2,
        xlabel="Equity",
        ylabel="q",
        title="Equity vs q for Different Productivity Levels")
    
    return p
end

"""
Main execution function
"""
function main()
    # Initialize parameters and grids
    params = init_parameters()
    grids = init_grids()
    
    # Set precision parameters
    precision_xdfp = 1.0e-5
    precision_v = 1.0e-5
    precision_xd = 1.0e-5
    max_iterations = 200
    
    # Print parameters
    println("\n..............................................................\n")
    println(" Parameter values\n")
    @printf("     tauc     %7.4f\n", params.tauc)
    @printf("     taud     %7.4f\n", params.taud)
    @printf("     lambda   %7.4f\n", params.lambda)
    @printf("     phi      %7.4f\n", params.phi)
    @printf("     rrf      %7.4f\n", params.rrf)
    @printf("     xi       %7.4f\n", params.xi)
    println("\n..............................................................\n")
    
    # Initialize productivity process
    rhoe0 = 0.9
    stde0 = 0.0335
    evec, pie = rouwenhorst(rhoe0, stde0, grids.neps)
    
    # Initialize value function and loan pricing
    vinitial = zeros(grids.neps, grids.nx)
    qarray = fill(params.qrf, (grids.neps, grids.nk, grids.nl, grids.neq))
    xd = zeros(grids.neps)
    
    # Main computation
    v, vc, vc0, kmax, lmax, eqmax, roemax, roamax, leveragemax, divmax, exit = 
        contraction(vinitial, grids, params, evec, pie, xd, qarray, precision_v)
    
    # Calculate distribution
    mu, firmnum, kagg, kfagg, lagg = distribution(grids, params, evec, pie,
                                                kmax, lmax, eqmax, exit,
                                                precision_v, max_iterations)
    
    # Calculate aggregates
    yagg, agg_roe, agg_roa, avg_roe, avg_roa, agg_payout_ratio, avg_payout_ratio = 
        aggregation(mu, grids, params, evec, kmax, lmax, eqmax)
    
    # Calculate additional aggregates
    iagg = kfagg - (1.0 - params.delta)*kagg
    cagg = yagg - iagg
    tfp = yagg / (kagg^params.alpha * lagg^(1-params.alpha))
    
    # Print results
    println("Aggregate variables:")
    @printf("yagg = %f\n", yagg)
    @printf("cagg = %f\n", cagg)
    @printf("iagg = %f\n", iagg)
    @printf("kagg = %f\n", kagg)
    @printf("lagg = %f\n", lagg)
    @printf("capital-output ratio = %f\n", kagg/yagg)
    @printf("investment rate = %f\n", iagg/yagg)
    @printf("tfp = %f\n", tfp)
    println("\nAggregate and Average Returns:")
    @printf("Aggregate ROE = %f%%\n", agg_roe * 100)
    @printf("Average ROE = %f%%\n", avg_roe * 100)
    @printf("Aggregate ROA = %f%%\n", agg_roa * 100)
    @printf("Average ROA = %f%%\n", avg_roa * 100)
    println("\nPayout Ratios:")
    @printf("Aggregate Payout Ratio = %f%%\n", agg_payout_ratio * 100)
    @printf("Average Payout Ratio = %f%%\n", avg_payout_ratio * 100)
    
    # Create and save plots
    value_plot = plot_value_function(v, grids.xgrid, evec)
    savefig(value_plot, "value_function.png")
    
    dist_plot = plot_distribution(mu)
    savefig(dist_plot, "distribution.png")
    
    decisions_plot = plot_decisions(grids.xgrid, kmax, lmax, eqmax, roemax, 
                                    roamax, leveragemax, divmax)
    savefig(decisions_plot, "decisions.png")
    
    qarray_plot = plot_qarray_relationships(qarray, evec, grids.kgrid, 
                                            grids.lgrid, grids.eqgrid)
    savefig(qarray_plot, "qarray_relationships.png")
    
    return v, mu, kmax, lmax, eqmax, qarray
end

# Run the model
@time v, mu, kmax, lmax, eqmax, qarray = main()

# still working on ...