

###############################################################################
## D'ORSOGNA MODEL FUNCTIONS ##################################################
###############################################################################

### Order of parameters in p
# 1: alpha - self propulsion
# 2: beta - drag
# 3: Cr - repulsive force strength
# 4: Ca - attractive force strength
# 5: lr - repulsive force characteristic length
# 6: la - attractive force characteristic length
# 7: m - mass of the agents
# 8: N - number of agents

###############################################################################
# Function: dorsogna2d
# Description: Function used to model the differential equations for the 2D
#	D'Orsogna model.
# Input 
# 	du: the derivative of u at time t
#		- a length 4N vector, where N is the number of agents; each agent has 4 coordinates:
#			(1) x-position
#			(2) y-position
#			(3) x-velocity
#			(4) y-velocity
#	p: parameters (see the parameter list at the beginning)
#	t: time
# Output 
# 	Updates the du vector to the current derivative
###############################################################################
function dorsogna2d(du,u,p,t)

	N = p[8]

	v_mag = zeros(N)
	r = zeros(N, N)

	for i = 1:N
		v_mag[i] = u[4*(i-1)+3]^2 + u[4*(i-1)+4]^2
	end

	for i = 1:N
		for j = 1:N
			r[i,j] = sqrt((u[4*(i-1)+1] - u[4*(j-1)+1])^2 + (u[4*(i-1)+2] - u[4*(j-1)+2])^2)
		end
	end


	for i = 1:N
		du[4*(i-1)+1] = u[4*(i-1)+3]
		du[4*(i-1)+2] = u[4*(i-1)+4]

		cur_dvx = (p[1] - p[2]*v_mag[i])* u[4*(i-1)+3]
		cur_dvy = (p[1] - p[2]*v_mag[i])* u[4*(i-1)+4]

		for j = 1:N
			if (i==j)
				continue
			end
			cur_dvx -= (-p[3]/p[5])*(u[4*(i-1)+1] - u[4*(j-1)+1])/r[i,j]*exp(-r[i,j]/p[5]) + (p[4]/p[6])*(u[4*(i-1)+1] - u[4*(j-1)+1])/r[i,j]*exp(-r[i,j]/p[6])
			cur_dvy -= (-p[3]/p[5])*(u[4*(i-1)+2] - u[4*(j-1)+2])/r[i,j]*exp(-r[i,j]/p[5]) + (p[4]/p[6])*(u[4*(i-1)+2] - u[4*(j-1)+2])/r[i,j]*exp(-r[i,j]/p[6])
		end

		du[4*(i-1)+3] = (1/p[7])*cur_dvx
		du[4*(i-1)+4] = (1/p[7])*cur_dvy
	end
end

###############################################################################
# Function: dorsogna3d
# Description: Function used to model the differential equations for the 3D
#	D'Orsogna model.
# Input 
# 	du: the derivative of u at time t
#		- a length 6N vector, where N is the number of agents; each agent has 6 coordinates:
#			(1) x-position
#			(2) y-position
#			(3) z-position
#			(4) x-velocity
#			(5) y-velocity
#			(6) z-velocity
#	p: parameters (see the parameter list at the beginning)
#	t: time
# Output 
# 	Updates the du vector to the current derivative
###############################################################################
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

		cur_dvx = (p[1] - p[2]*v_mag[i])* u[6*(i-1)+4]
		cur_dvy = (p[1] - p[2]*v_mag[i])* u[6*(i-1)+5]
		cur_dvz = (p[1] - p[2]*v_mag[i])* u[6*(i-1)+6]

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


###############################################################################
## MOMENT MAP FOR PERSISTENCE DIAGRAMS ########################################
###############################################################################

###############################################################################
# Function: moment_map
# Description: Computes the moment map for a path of persistence diagram using
#	birth/lifetime coordinates.
#*** *NOTE*: Input is expected to be in birth/death coordinates; can directly use the
#***	output from the Eirene package. However, the output computes the moments using
#***	birth/lifetime coordinates
# Input 
# 	PD: a path of persistence diagram as a 1D array of 2D arrays
#		- each 2D array is a persistence diagram at a given time point as an (M, 2) array)
#		- input arrays are 
#	max_level: truncation level for the moments
#	H0: flag for whether the persistence diagram is an H0 persistence diagram
#		-true: only computes 1D moments of the lifetimes since all birth times are 0
#		-false: computes the full set of 2D moments (without the pure x moments for continuity)
# Output 
# 	mu: path of moments as a 2D (numT, numMoments) array
###############################################################################
function moment_map(PD, max_level, H0=false)

	numT = length(PD)

	if H0
		numMoments = max_level
		mu = zeros(numT, numMoments)

		for t = 1:numT
			mcount = 1
			y = PD[t][:,2] - PD[t][:,1]
			numP = length(y)

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
			x = PD[t][:,1]
			y = PD[t][:,2] - PD[t][:,1]
			numP = length(y)

			# Compute order statistics
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

###############################################################################
# Function: moment_kernel
# Description: Efficient algorithm for moment kernel computation.
#*** *NOTE*: Input is expected to be in birth/death coordinates; can directly use the
#***	output from the Eirene package. However, the output computes the moments using
#***	birth/lifetime coordinates
# Input 
# 	D1, D2: 1D arrays of 2D arrays, which stores persistence diagrams up to dimension k
#		- D1[i] and D2[i] are dimension i-1 persistence diagrams
# Output 
# 	K: kernel of D1 and D2
###############################################################################
function moment_kernel(D1, D2)

	if ndims(D1) > 1
		numPdims = 1
	else
		numPdims = length(D1)
	end

	K = 0.0

	for d = 1:numPdims

		if d == 1
			K += sum(exp.(D1[d][:,2] * D2[d][:,2]'))
			continue
		end

		X = D1[d][:,1] * D2[d][:,1]'
		XY = (D1[d][:,2]-D1[d][:,1]) * (D2[d][:,2]-D2[d][:,1])' + X

		K += sum(exp.(XY) .- exp.(X))	
	end

	return K
end


###############################################################################
## BETTI CURVES ###############################################################
###############################################################################

###############################################################################
# Function: betti_embedding
# Description: Computes Betti curves from persistence diagrams
# Input 
# 	D: a 1D array of 2D arrays, which stores persistence diagrams up to dimension k
#		- D[i] is a dimension i-1 persistence diagram
#	EPS: used to specify the epsilon discretization, can be of two forms:
#		(1) EPS is a float - use this as the number of time points between the min and max
#			epsilon values in D
#		(2) EPS is a 1D array - use this as the array of discretized epsilon points
# Output 
# 	BC: Betti curve as a 2D (numEPS, numPdims) array, where numPdims is the number of dimensions
#		in the collection of persistence diagrams.
###############################################################################
function betti_embedding(D, EPS)

	# Find number of persistence dimensions
	if ndims(D) > 1
		numPdims = 1
	else
		numPdims = length(D)
	end

	# If EPS is a number, use this as the number of time points between min and maximum
	# If EPS is an array, simply use this as the array of time points
	if ndims(EPS) == 0
		numEPS = EPS

		# Find maximum time
		maxT = maximum(maximum.(D))

		# Generate time points
		tp = collect(range(0, stop=maxT, length=numEPS))
	elseif ndims(EPS) == 1
		tp = EPS
		numEPS = length(tp)
	else
		error("Invalid EPS: must be an integer or a 1D array of timepoints")
	end

	# Generate Betti embedding
	if numPdims==1
		BC = zeros(numEPS)

		for t = 1:numEPS
			BC[t] = sum((D[:,1] .<= tp[t]) .& (D[:,2] .>= tp[t]))
		end
	else
		BC = zeros(numEPS, numPdims)

		for t = 1:numEPS
			for d = 1:numPdims
				BC[t,d] = sum((D[d][:,1] .<= tp[t]) .& (D[d][:,2] .>= tp[t]))
			end
		end
	end

	return BC
end



