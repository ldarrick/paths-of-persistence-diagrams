using MAT
using Statistics
include("slicedwasserstein.jl")


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

# Reformats persistence diagrams into giotto-tda format
# Giotto-tda requires a batch of persistence diagrams contain multiple dimensions
# and all be the same length -- we pad persistence diagrams with 0's, as is
# done in giotto-tda.
function pd_to_giotto_mat(ifname, ofname, tss, numT, sf=0)

    numRun = 500
    time_axis = collect(1:tss:numT)

    ifile = matopen(ifname,"r")

    B0 = read(ifile, "B0")
    B1 = read(ifile, "B1")
    B2 = read(ifile, "B2")
    close(ifile)

    # Find max number of B0, B1 and B2
    maxB0 = 0
    maxB1 = 0
    maxB2 = 0
    for i = 1:numRun
        for j =1:numT
            t = time_axis[j]
            cur0 = size(B0[i][t])[1]
            if cur0 > maxB0
                maxB0 = cur0
            end

            cur1 = size(B1[i][t])[1]
            if cur1 > maxB1
                maxB1 = cur1
            end

            cur2 = size(B2[i][t])[1]
            if cur2 > maxB2
                maxB2 = cur2
            end
        end
    end

    numpt = maxB0 + maxB1 + maxB2
    PD_all = zeros(numRun, numT, numpt, 3)
    PD_all[:,:,maxB0+1:maxB0+maxB1,3] .= 1
    PD_all[:,:,maxB0+maxB1+1:end,3] .= 2

    for i = 1:numRun
        for j = 1:numT
            t = time_axis[j]
            cur0 = size(B0[i][t])[1]
            cur1 = size(B1[i][t])[1]
            cur2 = size(B2[i][t])[1]
            
            PD_all[i,j,1:cur0,1:2] = B0[i][t]
            PD_all[i,j,maxB0+1:maxB0+cur1,1:2] = B1[i][t]
            PD_all[i,j,maxB0+maxB1+1:maxB0+maxB1+cur2,1:2] = B2[i][t]
        end
    end

    if sf != 0
        PD_all[:,:,:,1:2] = PD_all[:,:,:,1:2].*(sf^(1/3))
    end

    ofile = matopen(ofname,"w")
    write(ofile, "PD", PD_all)
    close(ofile)
end

# Generates the betti curve given a persistence diagram D, and T is either
# - the number of time points (uniformly spaced)
# - a 1D array consisting of the time points
function betti_embedding(D, T)

	# Find number of persistence dimensions
	if ndims(D) > 1
		numPdims = 1
	else
		numPdims = length(D)
	end

	# If T is a number, use this as the number of time points between min and maximum
	# If T is an array, simply use this as the array of time points
	if ndims(T) == 0
		numT = T

		# Find maximum time
		maxT = maximum(maximum.(D))

		# Generate time points
		tp = collect(range(0, stop=maxT, length=numT))
	elseif ndims(T) == 1
		tp = T
		numT = length(tp)
	else
		error("Invalid T: must be an integer or a 1D array of timepoints")
	end

	# Generate Betti embedding
	if numPdims==1
		BC = zeros(numT)

		for t = 1:numT
			BC[t] = sum((D[:,1] .<= tp[t]) .& (D[:,2] .>= tp[t]))
		end
	else
		BC = zeros(numT, numPdims)

		for t = 1:numT
			for d = 1:numPdims
				BC[t,d] = sum((D[d][:,1] .<= tp[t]) .& (D[d][:,2] .>= tp[t]))
			end
		end
	end

	return BC
end
