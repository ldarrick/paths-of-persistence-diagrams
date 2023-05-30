# Compute features of persistence diagrams
using MAT
using ThreadPools
include("PathSignatures.jl")

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


function compute_PD_moments(B0, B1, B2, moment_level, fpath, save=true)

    numRun = length(B0)
    FT = Array{Array{Float64, 2}, 1}(undef, numRun) 

    for i = 1:numRun
        M0 = moment_map(B0[i], moment_level, true)
        M1 = moment_map(B1[i], moment_level, false)
        M2 = moment_map(B2[i], moment_level, false)
        FT[i] = hcat(M0, M1, M2)
    end

    if save == true
        fname = string(fpath, "MO_", moment_level, ".mat")
        file = matopen(fname, "w")
        write(file, "FT", FT)
        close(file)
    else
        return FT
    end

end

function compute_PD_perspath(BE, level, fpath)

    numRun = length(BE)
    numSig = sum(3 .^(1:level))
    FT = Array{Array{Float64, 2}, 1}(undef, numRun) 

    for i = 1:numRun
        curPA = zeros(numT, numSig)

        for t = 1:numT
            curBE = BE[i][t]

            curS = dsignature(curBE, level, "R")
            curS_vector = curS[1]

            for l = 2:level
                append!(curS_vector, curS[l][:])
            end

            curPA[t,:] = curS_vector
        end

        FT[i] = curPA
    end

    fname = string(fpath, "PA_", level, ".mat")
    file = matopen(fname, "w")
    write(file, "FT", FT)
    close(file)

end

## SLICED WASSERSTEIN ##

# Computes SWD for a single persistence diagram
function SW_distance_single(D1, D2, M)

	if ndims(D1) > 1
		numPdims = 1
	else
		numPdims = length(D1)
	end

	SW = zeros(numPdims)

	for d = 1:numPdims
		# Convert to birth/lifetime coordingates

		if numPdims == 1
			BL1 = deepcopy(D1)
			BL2 = deepcopy(D2)
		else	
			BL1 = deepcopy(D1[d])
			BL2 = deepcopy(D2[d])
		end

		BL1[:,2] = BL1[:,2] - BL1[:,1]
		BL2[:,2] = BL2[:,2] - BL2[:,1]


		if(d==1)
			V1 = sort(vcat(BL1[:,2], zeros(length(BL2[:,2]))))
			V2 = sort(vcat(BL2[:,2], zeros(length(BL1[:,2]))))
			SW[d] = sum(abs.(V1-V2))
			continue
		end

		# Add each diagram to the l=0 line
		BL1_proj = deepcopy(BL1)
		BL1_proj[:,2] .= 0
		BL2_proj = deepcopy(BL2)
		BL2_proj[:,2] .= 0

		ABL1 = vcat(BL1, BL2_proj)
		ABL2 = vcat(BL2, BL1_proj)

		theta = -pi/2
		s = pi/M

		for i = 1:M
			v_theta = [cos(theta); sin(theta)]
			V1 = ABL1*v_theta
			V2 = ABL2*v_theta

			V1 = sort(V1)
			V2 = sort(V2)

			SW[d] += s*sum(abs.(V1-V2))
		end
	end

	SW_tot = sum(SW)/pi

	return SW_tot
end


# Computes sliced wasserstein distance for a single path of persistence diagrams
function SW_distance(B0, B1, B2, M)
    
    n = length(B0)
    D = zeros(n, n)

    for i = 1:n
        for j = i:n
            D1 = [B0[i], B1[i], B2[i]]
            D2 = [B0[j], B1[j], B2[j]]

            SWD = SW_distance_single(D1, D2, M)

            D[i,j] = SWD

            if i != j
                D[j,i] = D[i,j]
            end
        end
    end

    return D
end


function flt_lt(A)
    return A[1] <= A[2]
end

# Computes sliced wasserstein distance for a batch of paths of persistence diagrams
function batch_SWD(B0, B1, B2, tss=2, numRun=500, maxT=50, M=5)

    numT = length(1:tss:maxT)
    D_all = zeros(numRun, numRun, 2*numT, 2*numT)

    # ThreadPools.@qthreads for (i,j) = Iterators.filter(flt_lt, Iterators.product(1:numRun, 1:numRun))
    for (i,j) = Iterators.filter(flt_lt, Iterators.product(1:numRun, 1:numRun))
        # Compute kernel
        X0 = B0[i][1:tss:maxT]
        X1 = B1[i][1:tss:maxT]
        X2 = B2[i][1:tss:maxT]

        Y0 = B0[j][1:tss:maxT]
        Y1 = B1[j][1:tss:maxT]
        Y2 = B2[j][1:tss:maxT]

        Z0 = [X0; Y0]
        Z1 = [X1; Y1]
        Z2 = [X2; Y2]

        D= SW_distance(Z0, Z1, Z2, M)
        D_all[i,j,:,:] = D
    end

    return D_all
end



function SW_MMD_kernel_from_SWD(D, sigma=0.1, sigma_out=0.1)

    numRun = size(D)[1]
    numT = size(D)[3]

    MMDK = zeros(numRun, numRun)
    
    for i = 1:numRun
        for j = i:numRun
            cur_SWD = D[i,j,:,:]

            K = exp.(-cur_SWD./(2*sigma))
            # K = SW_kernel_from_SWD(D[i,j,:,:], sigma)

            MMD = (1/numT)^2 * (sum(K[1:numT, 1:numT]) + sum(K[numT+1:end, numT+1:end])) - 2*(1/numT)^2 *sum(K[1:numT, numT+1:end])
            MMDK[i,j] = exp(-MMD/(2*sigma_out))

            if i != j
                MMDK[j,i] = MMDK[i,j]
            end
        end
    end

    return MMDK
end