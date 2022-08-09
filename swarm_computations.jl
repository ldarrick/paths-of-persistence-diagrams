include("PathSignatures.jl")
include("orsogna.jl")
using MAT
using Random
# using Eirene
using ScikitLearn
@sk_import svm:SVR
using ScikitLearn.CrossValidation: cross_val_score, KFold
using Statistics
using DifferentialEquations
using Distributions

## Folders for storing intermediate data at each step
SW_folder = "SW_data/" # Folder for swarm simulation data
PD_folder = "PD_data/" # Folder for persistence data
FT_folder = "FT_data/" # Folder for features
KE_folder = "KE_data/" # Folder for kernels
RG_folder = "RG_data/" # Folder for regression

# Make directories
all_folders = [SW_folder, PD_folder, FT_folder, KE_folder, RG_folder]
for folder in all_folders
    if !isdir(folder)
        mkdir(folder)
    end
end

#################################################################
### HELPER FUNCTIONS ############################################
#################################################################

# Generates temporal subsamples with seed given by index
# st_type = 1 : Random time (permuted)
# st_type = 2 : initial time
function subsampled_times(st, st_type)

    numT = 200
    SUBTP = Array{Array{Int64, 1}, 1}(undef, 500)

    if st_type == 1
        for i = 1:500
            SUBTP[i] = sort(randperm(MersenneTwister(i),numT)[1:st])
        end
    elseif st_type == 2
        for i = 1:500
            SUBTP[i] = 1:st
        end
    end

    return SUBTP
end

# Creates a string to specify the kernel
# ktype - kernel type (1: moment, 2: perspath, 3: crocker)
# ss - subsample
# st - subtime 
# st_type - subtime type (0: none (FT=Full Time), 1: random (RT = Random Time), 2: initial (IT = Initial Time))
# normalized - normalization for agent subsampling
## only for moments and Perspath
# lag - lag for signature
# inner_level - inner truncation level (moment or persistence path)
# outer_level - outer truncation level (of signature)
function kernel_name_string(ktype, ss, st, st_type, normalized, lag, in_lvl, out_lvl, mixed, combined, initial="kernel")

    KT_STR = ["MO", "PP", "CR"]
    STT_STR = ["FT", "RT", "IT"]

    kname = initial
    if normalized
        kname = string(kname, "Norm")
    end

    if mixed
        kname = string(kname, "Mix")
    end

    if combined
        kname = string(kname, "Comb")
    end

    k_str = ""

    if ktype < 3
        if st_type == 0
            k_str = string(kname, "_SS", ss, "_", STT_STR[st_type+1], "_", KT_STR[ktype], lag, "_", in_lvl, "_", out_lvl)
        else
            k_str = string(kname, "_SS", ss, "_", STT_STR[st_type+1], st, "_", KT_STR[ktype], lag, "_", in_lvl, "_", out_lvl)
        end
    else
        if st_type == 0
            k_str = string(kname, "_SS", ss, "_", STT_STR[st_type+1], "_", KT_STR[ktype])
        else
            k_str = string(kname, "_SS", ss, "_", STT_STR[st_type+1], st, "_", KT_STR[ktype])
        end
    end
    
    return k_str
end

# Function that specifies D'Orsogna 3D ODE model.
function dorsogna3d(du,u,p,t)

	N = Int(p[8])

	v_mag = zeros(N)
	r = zeros(N, N)

	for i = 1:N
		v_mag[i] = u[6*(i-1)+4]^2 + u[6*(i-1)+5]^2 + u[6*(i-1)+6]^2
	end

	for i = 1:N
		for j = 1:N
			r[i,j] = sqrt((u[6*(i-1)+1] - u[6*(j-1)+1])^2 + (u[6*(i-1)+2] - u[6*(j-1)+2])^2 + (u[6*(i-1)+3] - u[6*(j-1)+3])^2)
		end
	end
	
	for i = 1:N
		du[6*(i-1)+1] = u[6*(i-1)+4]
		du[6*(i-1)+2] = u[6*(i-1)+5]
		du[6*(i-1)+3] = u[6*(i-1)+6]

		cur_dvx = (p[1] - p[2]*v_mag[i])* u[6*(i-1)+4] + randn()*sqrt(6*p[9])
		cur_dvy = (p[1] - p[2]*v_mag[i])* u[6*(i-1)+5] + randn()*sqrt(6*p[9])
		cur_dvz = (p[1] - p[2]*v_mag[i])* u[6*(i-1)+6] + randn()*sqrt(6*p[9])

		for j = 1:N
			if (i==j)
				continue
			end
			cur_dvx -= (-p[3]/p[5])*(u[6*(i-1)+1] - u[6*(j-1)+1])/r[i,j]*exp(-r[i,j]/p[5]) + (p[4]/p[6])*(u[6*(i-1)+1] - u[6*(j-1)+1])/r[i,j]*exp(-r[i,j]/p[6])
			cur_dvy -= (-p[3]/p[5])*(u[6*(i-1)+2] - u[6*(j-1)+2])/r[i,j]*exp(-r[i,j]/p[5]) + (p[4]/p[6])*(u[6*(i-1)+2] - u[6*(j-1)+2])/r[i,j]*exp(-r[i,j]/p[6])
			cur_dvz -= (-p[3]/p[5])*(u[6*(i-1)+3] - u[6*(j-1)+3])/r[i,j]*exp(-r[i,j]/p[5]) + (p[4]/p[6])*(u[6*(i-1)+3] - u[6*(j-1)+3])/r[i,j]*exp(-r[i,j]/p[6])

		end

		du[6*(i-1)+4] = (1/p[7])*cur_dvx
		du[6*(i-1)+5] = (1/p[7])*cur_dvy
		du[6*(i-1)+6] = (1/p[7])*cur_dvz
	end
end

# Moment feature map
function moment_map(BC, max_level, H0=false)

	numT = length(BC)

    # For H0, just compute lifetime moments
    if H0
        numMoments = max_level
        mu = zeros(numT, numMoments)
        for t = 1:numT
            y = BC[t][:,2] - BC[t][:,1]

            for i = 1:max_level
                mu[t,i] = sum((y.^i))/sqrt(factorial(i))
            end
        end
        return mu
    else
        numMoments = Int(max_level*(max_level+1)/2)
        mu = zeros(numT, numMoments)

        for t = 1:numT
            mcount = 1
            x = BC[t][:,1]
            y = BC[t][:,2] - BC[t][:,1]

            for i = 1:max_level
                for j = 1:i
                    mu[t,mcount] = sum((x.^(i-j)).*(y.^j))*sqrt(binomial(i,j)/factorial(i))
                    mcount += 1
                end
            end

        end

        return mu
    end
end

#################################################################
### MAIN FUNCTIONS ##############################################
#################################################################

function swarm_simulate_3d()
    # Common parameters
    alpha = 1.0 # This is alpha - gamma in the nguyen paper
    gamma = 1.0
    beta = 0.5
    Ca = 1
    la = 1
    m = 1
    N = 200
    temp = 0.0
    
    numT = 200
    numRun = 500
    endT = 400.0
    tspan = (0,endT)
    trange = range(0, stop=endT, length=numT)
    
    boundedlimit = 40.0
    
    # Store all parameters in a dictionary
    paramDict = Dict{String, Float64}()
    paramDict["alpha"] = alpha
    paramDict["beta"] = beta
    paramDict["Ca"] = Ca
    paramDict["la"] = la
    paramDict["m"] = m
    paramDict["N"] = N
    paramDict["temp"] = temp
    paramDict["numT"] = numT
    paramDict["numRun"] = numRun
    paramDict["endT"] = endT
    paramDict["boundedlimit"] = boundedlimit
    
    # Compute everything ######################################
    PP = Array{Array{Float64, 3},1}(undef, numRun)
    
    CL = zeros(numRun,2)
    
    for j = 1:numRun
        u0 = rand(Uniform(-1,1),6*N)
    
        for i = 1:N
            u0[6*(i-1)+4:6*(i-1)+6] = randn(3).+1
        end
        
        cur_maxpos = 100
        curP = zeros(3,N,numT)
        curV = zeros(3,N,numT)
        C=0
        l=0
        
        # Run new simulations until we get a bounded phenotype
        while cur_maxpos > boundedlimit
            C = rand(Uniform(0.1,2))
            l = rand(Uniform(0.1,2))
            cur_params = [alpha; beta; C; Ca; l; la; m; N; temp]
            prob = ODEProblem(dorsogna3d, u0, tspan, cur_params)
            sol = solve(prob)
            sol_interp = hcat(sol(trange).u...)
    
            for n = 1:N
                curP[1,n,:] = sol_interp[6*(n-1)+1,:]
                curP[2,n,:] = sol_interp[6*(n-1)+2,:]
                curP[3,n,:] = sol_interp[6*(n-1)+3,:]
                curV[1,n,:] = sol_interp[6*(n-1)+4,:]
                curV[2,n,:] = sol_interp[6*(n-1)+5,:]
                curV[3,n,:] = sol_interp[6*(n-1)+6,:]
            end
            
            cur_maxpos = maximum(abs.(curP[:,:,1:100]))
        end
        
        CL[j,1] = C
        CL[j,2] = l
        PP[j] = curP
        println(string("Completed simulation ", j, "/", numRun))
        sleep(0.1)
    end

    # Save swarm simulation data
    fname = string(SW_folder,"swarm3d_data.mat")
    ofile = matopen(fname, "w")
    write(ofile, "PP", PP)
    write(ofile, "paramDict", paramDict)
    close(ofile)

    ofile = matopen("CL_data.mat", "w")
    write(ofile, "CL", CL)
    close(ofile)
end

### Compute Persistence Diagrams
# REQUIRES: swarm position data computed
function swarm_compute_persistence(subsample=200; num_agents=200, num_timesteps=200, num_runs=500)

    # Parameters
    N = num_agents
    numT = num_timesteps
    numRun = num_runs

    # Log-discretization for scale parameter (for betti curves)
    tp = 10 .^(range(-4, stop=0, length=100))

    # Scaling factor (relative to total = 200 points)
    sf = subsample/float(N)

    # Load swarm data
    fname = string(SW_folder,"swarm3d_data.mat")
    file = matopen(fname,"r")
    PP = read(file, "PP")
    close(file)

    # Initialize arrays for (unnormalized) persistence data
    # B0, B1, B2: persistence diagrams (birth, death)
    # BE: Betti curve
    B0 = Array{Array{Array{Float64, 2}, 1},1}(undef, numRun) 
    B1 = Array{Array{Array{Float64, 2}, 1},1}(undef, numRun)
    B2 = Array{Array{Array{Float64, 2}, 1},1}(undef, numRun)
    BE = Array{Array{Array{Float64, 2}, 1},1}(undef, numRun)
    NBE = Array{Array{Array{Float64, 2}, 1},1}(undef, numRun)

    # Compute unnormalized persistence
    for j = 1:numRun

        # Get current swarm simulation run
        curP = PP[j]

        # Initialize current arrays for persistence data
        curB0 = Array{Array{Float64, 2},1}(undef, numT)
        curB1 = Array{Array{Float64, 2},1}(undef, numT)
        curB2 = Array{Array{Float64, 2},1}(undef, numT)
        curBE = Array{Array{Float64, 2},1}(undef, numT)
        curNBE = Array{Array{Float64, 2},1}(undef, numT)
        
        for t = 1:numT

            # Subsample points if necessary
            if subsample < 200
                keepInd = randperm(N)[1:subsample]
                keepInd = sort!(keepInd)
                curP_ss = curP[:,keepInd,t]
            else
                curP_ss = curP[:,:,t]
            end

            # Compute persistence
            C = eirene(curP_ss, model="pc", maxdim=2)
            curB0[t] = barcode(C, dim=0)
            curB0[t] = curB0[t][1:end-1,:]
            curB1[t] = barcode(C, dim=1)
            curB2[t] = barcode(C, dim=2)
            
            # Compute Betti curves
            curB = [curB0[t], curB1[t], curB2[t]]
            curBE[t] = betti_embedding(curB, tp)

            # Compute normalized Betti curve
            curNBE[t] = betti_embedding(curB.*(sf^(1/3)), tp)/sf
        end

        B0[j] = curB0
        B1[j] = curB1
        B2[j] = curB2
        BE[j] = curBE
        NBE[j] = curNBE
    end

    # Write data
    ofname = string(PD_folder, "PD_ss", subsample, ".mat")
    ofile = matopen(ofname, "w")
    write(ofile, "B0", B0)
    write(ofile, "B1", B1)
    write(ofile, "B2", B2)
    write(ofile, "BE", BE)
    write(ofile, "NBE", NBE)
    close(ofile)
end

### Compute Features
# REQUIRES: persistence data computed
function swarm_compute_features(subsample=200)

    # Folders
    PD_folder = "PD_data/" # Folder for persistence data
    FT_folder = "FT_data/" # Folder for features

    # Parameters
    N = 200
    numT = 200
    numRun = 500

    moment_level = 6
    numMoments = (moment_level+2)*moment_level

    signature_level = 5
    numSig = sum(3 .^(1:signature_level))

    # Scaling factor (relative to total = 200 points)
    sf = subsample/float(N)

    # Load persistence data
    ifname = string(PD_folder, "PD_ss", subsample, ".mat")
    ifile = matopen(ifname,"r")
    B0 = read(ifile, "B0")
    B1 = read(ifile, "B1")
    B2 = read(ifile, "B2")
    BE = read(ifile, "BE")
    NBE = read(ifile, "NBE")
    close(ifile)

    # Initialize matrices for features
    MO_F = Array{Array{Float64, 2}, 1}(undef, numRun) 
    PP_F = Array{Array{Float64, 2}, 1}(undef, numRun) 

    # Initialize matrices for normalized features
    NMO_F = Array{Array{Float64, 2}, 1}(undef, numRun) 
    NPP_F = Array{Array{Float64, 2}, 1}(undef, numRun) 

    ## MOMENT COMPUTATION
    for i = 1:numRun
        M0 = moment_map(B0[i], moment_level, true)
        M1 = moment_map(B1[i], moment_level, false)
        M2 = moment_map(B2[i], moment_level, false)

        NM0 = moment_map(B0[i].*(sf^(1/3)), moment_level, true)/sf
        NM1 = moment_map(B1[i].*(sf^(1/3)), moment_level, false)/sf
        NM2 = moment_map(B2[i].*(sf^(1/3)), moment_level, false)/sf

        MO_F[i] = hcat(M0, M1, M2)
        NMO_F[i] = hcat(NM0, NM1, NM2)
    end

    ## SIGNATURE COMPUTATION
    for i = 1:numRun
        curPP = zeros(numT, numSig)
        curNPP = zeros(numT, numSig)

        for t = 1:numT
            curBE = BE[i][t]
            curNBE = NBE[i][t]

            curS = dsignature(curBE, signature_level, "R")
            curNS = dsignature(curNBE, signature_level, "R")

            curS_vector = curS[1]
            curNS_vector = curNS[1]

            for l = 2:signature_level
                append!(curS_vector, curS[l][:])
                append!(curNS_vector, curNS[l][:])
            end

            curPP[t,:] = curS_vector
            curNPP[t,:] = curNS_vector
        end

        PP_F[i] = curPP
        NPP_F[i] = curNPP
    end

    # Write data
    ofname = string(FT_folder, "FT_ss", subsample, ".mat")
    ofile = matopen(ofname, "w")
    write(ofile, "MO_F", MO_F)
    write(ofile, "PP_F", PP_F)
    write(ofile, "NMO_F", NMO_F)
    write(ofile, "NPP_F", NPP_F)
    close(ofile)

end

### Compute Kernel
# REQUIRES: features computed
# subtime_type = 0 : random
# subtime_type = 1 : initial
function swarm_compute_kernel(ktype, ss, st, st_type, normalized, lag, in_lvl, out_lvl, mixed)

    TIMER_START = time()

    # Parameters
    numT = 200
    numRun = 500
    numE = 100 # number of epsilon points in betti curve
    numT_CRKR = 20 # number of time points in crocker plot

    K = zeros(numRun,numRun)

    # Subsampled times
    numST = numT
    if st_type > 0
        SUBTP = subsampled_times(st, st_type)
        numST = st
    end

    ## MOMENT KERNEL ###############################
    if ktype == 1

        numMoments = (in_lvl+2)*in_lvl
        lagp = lag+1
        
        # Load data
        ifname = string(FT_folder, "FT_ss", ss, ".mat")
        ifile = matopen(ifname, "r")
        if normalized
            MO_F = read(ifile, "NMO_F")
        else
            MO_F = read(ifile, "MO_F")
        end
        close(ifile)

        # Perform time subsampling
        if st_type > 0
            for i = 1:numRun
                MO_F[i] = MO_F[i][SUBTP[i],:]
            end
        end

        # Subsample moments properly
        if in_lvl < 6
            newMO_F = Array{Array{Float64, 2}, 1}(undef, numRun)
            numMomentsPD = Int(in_lvl*(in_lvl+1)/2)

            for i = 1:numRun
                newMO_F[i] = hcat(MO_F[i][:,1:in_lvl], MO_F[i][:,7:6+numMomentsPD], MO_F[i][:, 28:27+numMomentsPD])
            end
            MO_F = newMO_F
        end

        # Add lags
        MO_lag = Array{Array{Float64, 2}, 1}(undef, numRun)
        for i = 1:numRun
            curMO_lag = zeros(numST, numMoments*lagp)

            for l = 1:lagp
                curMO_lag[l:end, (l-1)*numMoments+1:l*numMoments] = MO_F[i][1:end-(l-1),:]
            end

            MO_lag[i] = curMO_lag
        end

        # Load and process mixed data if necessary
        if mixed
            full_fname = string(FT_folder, "FT_ss200.mat")
            full_file = matopen(full_fname, "r")
            fullMO_F = read(full_file, "MO_F")
            close(full_file)

            # Subsample moments properly
            if in_lvl < 6
                newMO_F = Array{Array{Float64, 2}, 1}(undef, numRun)
                numMomentsPD = Int(in_lvl*(in_lvl+1)/2)

                for i = 1:numRun
                    newMO_F[i] = hcat(fullMO_F[i][:,1:in_lvl], fullMO_F[i][:,7:6+numMomentsPD], fullMO_F[i][:, 28:27+numMomentsPD])
                end
                fullMO_F = newMO_F
            end

            fullMO_lag = Array{Array{Float64, 2}, 1}(undef, numRun)
            for i = 1:numRun
                curMO_lag = zeros(numT, numMoments*lagp)
    
                for l = 1:lagp
                    curMO_lag[l:end, (l-1)*numMoments+1:l*numMoments] = fullMO_F[i][1:end-(l-1),:]
                end

                fullMO_lag[i] = curMO_lag
            end

            K = dsignature_kernel_matrix(fullMO_lag, MO_lag, out_lvl, "R")
        else
            K = dsignature_kernel_matrix(MO_lag, [], out_lvl, "R")
        end

    ## PERSPATH KERNEL ###############################    
    elseif ktype == 2
        numSig = sum(3 .^(1:in_lvl))
        lagp = lag+1

        # Load data
        ifname = string(FT_folder, "FT_ss", ss, ".mat")
        ifile = matopen(ifname, "r")
        if normalized
            PP_F = read(ifile, "NPP_F")
        else
            PP_F = read(ifile, "PP_F")
        end
        close(ifile)

        # Perform time subsampling
        if st_type > 0
            for i = 1:numRun
                PP_F[i] = PP_F[i][SUBTP[i],:]
            end
        end

        # Add lags
        PP_lag = Array{Array{Float64, 2}, 1}(undef, numRun)
        for i = 1:numRun
            curPP_lag = zeros(numST, numSig*lagp)

            for l = 1:lagp
                curPP_lag[l:end, (l-1)*numSig+1:l*numSig] = PP_F[i][1:end-(l-1),1:numSig]
            end

            PP_lag[i] = curPP_lag
        end

        # Load and process mixed data if necessary
        if mixed
            full_fname = string(FT_folder, "FT_ss200.mat")
            full_file = matopen(full_fname, "r")
            fullPP_F = read(full_file, "PP_F")
            close(full_file)

            fullPP_lag = Array{Array{Float64, 2}, 1}(undef, numRun)
            for i = 1:numRun
                curPP_lag = zeros(numT, numSig*lagp)
    
                for l = 1:lagp
                    curPP_lag[l:end, (l-1)*numSig+1:l*numSig] = fullPP_F[i][1:end-(l-1),1:numSig]
                end

                fullPP_lag[i] = curPP_lag
            end

            K = dsignature_kernel_matrix(fullPP_lag, PP_lag, out_lvl, "R")
        else
            K = dsignature_kernel_matrix(PP_lag, [], out_lvl, "R")
        end


    ## CROCKER KERNEL ###############################
    elseif ktype == 3 

        # Load data
        ifname = string(PD_folder, "PD_ss", ss, ".mat")
        ifile = matopen(ifname,"r")
        if normalized
            BE = read(ifile, "NBE")
        else
            BE = read(ifile, "BE")
        end
        close(ifile)

        # Perform time subsampling
        if st_type > 0 
            for i = 1:numRun
                BE[i] = BE[i][SUBTP[i]]
            end
        end

        numdim_CRKR = numT_CRKR*numE*3
        ndim_block = numE*3

        # Further subsample the time axis to reduce dimensionality
        tp_CRKR = Int.(round.(collect(range(numST/numT_CRKR, stop=numST ,length=numT_CRKR))))

        CRKR = zeros(numRun, numdim_CRKR)

        for i = 1:numRun
		    for t = 1:numT_CRKR
		        CRKR[i,(t-1)*ndim_block+1:t*ndim_block] = BE[i][tp_CRKR[t]][:]
            end
        end

        K = zeros(numRun, numRun)

        # Compute kernel
        if mixed
            full_fname = string(PD_folder, "PD_ss200.mat")
            full_file = matopen(full_fname,"r")
            BE = read(full_file, "BE")
            close(full_file)

            tp_full_CRKR = 10:10:numT
            full_CRKR = zeros(numRun, numdim_CRKR)

            for i = 1:numRun
                for t = 1:numT_CRKR
                    full_CRKR[i,(t-1)*ndim_block+1:t*ndim_block] = BE[i][tp_full_CRKR[t]][:]
                end
            end

            for i = 1:numRun
                for j = 1:numRun
                    K[i,j] = dot(full_CRKR[i,:], CRKR[j,:])
                end
            end

        else
            for i = 1:numRun
                for j = i:numRun
                    K[i,j] = dot(CRKR[i,:], CRKR[j,:])
                    
                    if i!=j
                        K[j,i] = K[i,j]
                    end
                end
            end
        end

    end

    kname = kernel_name_string(ktype, ss, st, st_type, normalized, lag, in_lvl, out_lvl, mixed, false)
    println(string(kname, ": ", TIMER_ELAPSED))
    kfname = string(KE_folder, kname, ".mat")
    ofile = matopen(kfname, "w")
    write(ofile, "K", K)
    write(ofile, "timer", TIMER_ELAPSED)
    close(ofile)

end


### Perform Regression
# REQUIRES: kernel data computed
# spec_IT is a specific iteration and only runs one regression; doesn't save to file, but returns results
#   Set spec_IT == -1 to run full regression
function swarm_regression(ktype, ss, st, st_type, normalized, lag, in_lvl, out_lvl, mixed=false, spec_IT=-1)

    TIMER_START = time()
    
    # Load training kernel
    # If mixed, this is the kernel using full dataset with all time steps
    # If not mixed, this is the specified kernel
    if mixed
        kname = kernel_name_string(ktype, 200, st, 0, false, lag, in_lvl, out_lvl, false, false)
    else
        kname = kernel_name_string(ktype, ss, st, st_type, normalized, lag, in_lvl, out_lvl, mixed, false)
    end
    kfname = string(KE_folder, kname, ".mat")
    ifile = matopen(kfname, "r")
    K_TRAIN = read(ifile, "K")
    close(ifile)

    # Load test kernel
    K_TEST = 0
    if mixed
        kname = kernel_name_string(ktype, ss, st, st_type, normalized, lag, in_lvl, out_lvl, mixed, false)
        kfname = string(KE_folder, kname, ".mat")
        ifile = matopen(kfname, "r")
        K_TEST = read(ifile, "K")
        close(ifile)
    else
        K_TEST = K_TRAIN
    end

    # Load CL data
    file = matopen("CL_data.mat", "r")
    CL = read(file, "CL")
    close(file)

    # Parameters
    numRun = 500
    all_numRun = 1000
    numIterations = 100
    hyp_cv = 4
    numTr = 400
    numTe = 100

    # If a specific iteration is specified, there is only one iteration
    # Also initialize the matrix which carries all loss information
    if spec_IT > -1
        numIterations = 1
        saved_SVR_predict = zeros(100, 2)
        cl_te = zeros(100, 2)
    end

    # SVR Hyperparameters
    SVR_C = 10. .^(-3:0.5:3)
    SVR_eps = 10. .^(-5:0.5:1)
    num_SVRC = length(SVR_C)
    num_SVReps = length(SVR_eps)

    # Save the best SVM params (numIterations, CL, SVM_c, SVM_eps)
    SVM_params = zeros(numIterations,2,2)

    K_error = zeros(numIterations,2)

    for it = 1:numIterations
        if spec_IT == -1
            curperm = randperm(MersenneTwister(1000+it), numRun)
        else
            curperm = randperm(MersenneTwister(1000+spec_IT), numRun)
        end

        # Shuffle data
        curK_TRAIN = K_TRAIN[curperm, curperm]
        curK_TEST = K_TEST[curperm, curperm]

        curCL = CL[curperm,:]
        cl_tr = curCL[1:400,:]
        cl_te = curCL[401:end,:]

        K_tr = curK_TRAIN[1:400,1:400]
        K_te = curK_TEST[1:400,401:end]

        # Cross-validation for hyperparameter tuning
        # Further split the training set into train/validation sets for hyperparameter tuning
        kf2 = KFold(400, n_folds=hyp_cv, shuffle=false);
        scores1_hyp = zeros(num_SVRC, num_SVReps);
        scores2_hyp = zeros(num_SVRC, num_SVReps);

        for k = 1:hyp_cv
            K_tr2 = K_tr[kf2[k][1], kf2[k][1]]
            K_val = K_tr[kf2[k][2], kf2[k][1]]
            cl_tr2 = cl_tr[kf2[k][1],:]
            cl_val = cl_tr[kf2[k][2],:]
            numVal = length(cl_val[:,1])

            for j1 = 1:num_SVRC
                for j2 = 1:num_SVReps
                    # C hyperparameter tuning
                    cls = SVR(kernel="precomputed", C=SVR_C[j1], epsilon=SVR_eps[j2])
                    ScikitLearn.fit!(cls, K_tr2, cl_tr2[:,1])
                    SVR_predict = ScikitLearn.predict(cls, K_val)
                    scores1_hyp[j1,j2] += sum(abs.(SVR_predict - cl_val[:,1]).^2)/numVal

                    # eps hyperparameter tuning
                    cls = SVR(kernel="precomputed", C=SVR_C[j1], epsilon=SVR_eps[j2])
                    ScikitLearn.fit!(cls, K_tr2, cl_tr2[:,2])
                    SVR_predict = ScikitLearn.predict(cls, K_val)
                    scores2_hyp[j1,j2] += sum(abs.(SVR_predict - cl_val[:,2]).^2)/numVal
                end
            end
        end

        # Find best hyperparameters
        best1 = findfirst(scores1_hyp .== minimum(scores1_hyp))
        C1 = SVR_C[best1[1]]
        eps1 = SVR_eps[best1[2]]
        
        best2 = findfirst(scores2_hyp .== minimum(scores2_hyp))
        C2 = SVR_C[best2[1]]
        eps2 = SVR_eps[best2[2]]
        
        # Save these hyperparameters
        SVM_params[it,1,1] = C1
        SVM_params[it,1,2] = eps1
        SVM_params[it,2,1] = C2
        SVM_params[it,2,2] = eps2

        # Perform regression on full dataset
        cls = SVR(kernel="precomputed", C=C1, epsilon=eps1)
        ScikitLearn.fit!(cls, K_tr, cl_tr[:,1])
        SVR_predict = ScikitLearn.predict(cls, K_te')
        K_error[it, 1] = sum(abs.(SVR_predict - cl_te[:,1]).^2)/numTe

        if spec_IT > -1
            saved_SVR_predict[:,1] = SVR_predict
        end

        cls = SVR(kernel="precomputed", C=C2, epsilon=eps2)
        ScikitLearn.fit!(cls, K_tr, cl_tr[:,2])
        SVR_predict = ScikitLearn.predict(cls, K_te')
        K_error[it, 2] = sum(abs.(SVR_predict - cl_te[:,2]).^2)/numTe

        if spec_IT > -1
            saved_SVR_predict[:,2] = SVR_predict
        end
    end

    TIMER_ELAPSED = time() - TIMER_START

    if spec_IT == -1
        rname = kernel_name_string(ktype, ss, st, st_type, normalized, lag, in_lvl, out_lvl, mixed, false, "regression")
        println(string(rname, ": ", TIMER_ELAPSED))

        rfname = string(RG_folder, rname, ".mat")
        ofile = matopen(rfname, "w")
        write(ofile, "K_error", K_error)
        write(ofile, "SVM_params", SVM_params)
        write(ofile, "timer", TIMER_ELAPSED)
        close(ofile)
    else
        cur_perm = randperm(MersenneTwister(1000+spec_IT), numRun)
        return K_error, SVM_params, saved_SVR_predict, cl_te, cur_perm
    end

end
