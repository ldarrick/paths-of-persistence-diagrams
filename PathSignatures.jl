
using Combinatorics
using LinearAlgebra
using TensorOperations
using Roots
using Random

###############################################################################
#=
COMMON PARAMETER NAMES
T+1 = number of timepoints (length of time series), so that derivative is length T
N = total dimension of space
n = dimension of a single copy of a space
b = parameter for Lie group (ie. SO(b))
d = number of copies of a space (is. SO(b)^d)
M = truncation level

COMMON VARIABLE NAMES
P = time series (path)
BP = batch of time series (array of arrays)
p = discrete derivative of time series
bp = batch of discrete derivatives

TIME SERIES FORMATTING
Ordering:
	1) Time parameter
	2) Lie group parametrization
	3) Copies of space
R^N : (T, N) array
SO(b)^d : (T, b, b, d) array (here, N = dim G = db(b-1)/2)
=#
###############################################################################


###############################################################################
## HELPER FUNCTIONS ###########################################################
###############################################################################


###############################################################################
# Function: initialize_TA 
# Description: Initializes array for truncated tensor algebra element.
# Input 
# 	N: dimension of underlying vector space
#	M: truncation level
# Output 
# 	A: length M array of arrays, the j^th array is a j dimensional array
#	   of length N
###############################################################################
function initialize_TA(N, M)

	A = Array{Array{Float64},1}(undef, M)
	for i = 1:M
	    # A[i] = Array{Float64, i}(undef, (ones(Int,i)*N)...)
	    A[i] = zeros((ones(Int,i)*N)...)
	end
	return A

end


###############################################################################
# Function: multiply_TA!
# Description: Multiplies two truncated tensor algebra elements, returning
#	S \otimes T.
# Input 
# 	A: variable to hold output of multiplication
#	S, T: truncated tensor algebra elements to multiply
#	M: truncation level
###############################################################################
function multiply_TA!(A, S, T, M)

    # Check if input tensors have required length
    if length(S) < M || length(T) < M
        error("Input tensors not defined up to specified level.")
    end

    N = length(S[1])

    # Check if tensors are the correct size
    for i = 1:M
        if any(size(S[i]).!=N) || any(size(T[i]).!=N)
            error("Input tensors are not the correct size.")
        end
    end

    TUP1 = Tuple(collect(1:M))
    TUP2 = Tuple(collect(M+1:2M))

    # Initialize output tensor
    # A = initialize_TA(N,level)
    for l = 1:M
        # M[l] = zeros((ones(Int,l)*N)...)
        A[l] = S[l] + T[l]
    end

    for l1 = 1:M
        for l2 = 1:M-l1
            A[l1+l2] += tensorproduct(S[l1], TUP1[1:l1], T[l2], TUP2[1:l2])
        end
    end

end


###############################################################################
# Function: tensor_exp!
# Description: Computes truncated tensor exponential
# Input 
# 	A: variable to hold output of multiplication
#	v: vector to exponentiate
#	M: truncation level
###############################################################################
function tensor_exp!(A, v, M)

	# Check if T is the right size
	if length(A) != M
		error("A is the wrong size")
	end

	N = length(v)

	for i = 1:M
		if any(size(A[i]).!=N)
			println(i)
			error("A is the wrong size")
		end
	end

	TUP = Tuple(collect(1:M))

	A[1] = v
	for m = 2:M
        A[m] = tensorproduct(A[m-1], TUP[1:m-1], v, TUP[m:m])/(factorial(m))
	end

end


###############################################################################
# Function: PHI
# Description: Function used in tensor normalization
# Input 
# 	x: argument for function
#	M, a: parameters of function
###############################################################################
function PHI(x, M::Float64, a::Float64)

    if x^2 <= M
        return x^2
    else
        return M + (M^(1+a))*(M^(-a) - x^(-2a))/a
    end
end


###############################################################################
# Function: kernel_submatrix
# Description: Computes the kernel matrix for a batch of data using the specified
#   kernel function. If only a submatrix is required, the row indices and column
#   indices can be specified.
# Input 
#   X: a batch of data in one of two forms:
#       (1) (T+1, N)-array, where columns represent the point at time t
#       (2) (T+1)-array of objects, where each column represents the object at time t
#   kernelFunc: a kernel function taking two inputs of the same type as the data in X
#   rowInd: the row of the kernel matrix to be computed
#   colInd: the columns of the kernel matrix to be computed
# Output 
#   K: kernel matrix
###############################################################################
function kernel_submatrix(X, kernelFunc, rowInd, colInd)

    if ndims(X) > 1
        nestedfeatures = false
    else
        nestedfeatures = true
    end

    nr = length(rowInd)
    nc = length(colInd)

    K = zeros(nr, nc)

    rc_intersect = intersect(rowInd, colInd)

    for i = 1:nr
        for j = 1:nc

            RI = rowInd[i]
            CI = colInd[j]


            fRI = findfirst(RI .== colInd)
            fCI = findfirst(CI .== rowInd)


            if !isnothing(fRI) && !isnothing(fCI) && (RI > CI)
                continue
            end

            if nestedfeatures
                K[i,j] = kernelFunc(X[RI], X[CI])
            else
                K[i,j] = kernelFunc(X[RI,:], X[CI,:])
            end

            if !isnothing(fRI) && !isnothing(fCI) && (RI < CI)
                K[fCI,fRI] = K[i,j]
            end

        end
    end

    return K

end


###############################################################################
## DISCRETE DERIVATIVE FUNCTIONS ##############################################
###############################################################################

###############################################################################
# Function: discrete_derivative 
# Description: Computes the discrete derivative of time series.
# Input 
# 	P: time series
# 		(T+1, N) array if dtype = "R"
# 		(T+1, b, b) or T(b, b, d) array if dtype = "SO"
# 	dtype: type of time series, currently implemented:
# 		"R": real valued time series
# 		"SO": SO(b)^d valued time series (possibly with d = 1)
# Output 
# 	p: discrete derivative of time series (T, N) array for all dtype
###############################################################################
function discrete_derivative(P, dtype)

	# Real time series
	if dtype == "R"
		return P[2:end, :] - P[1:end-1, :]

	# SO(b)^d valued time series
	elseif dtype == "SO"
		# If d=1, then the input is a size (T, b, b) array
		if ndims(P) == 3
		    T, b, ~ = size(P)
		    n = div((b^2-b),2)
		    N = n

		    Pderiv = zeros(T-1, b, b)
		    p = zeros(T-1, N)

		    for t = 1:T-1
		        Pderiv[t,:,:] = real.(log(transpose(P[t,:,:])*P[t+1,:,:]))
		    end

		    dcount = 1
		    for i = 1:b-1
		    	for j = i+1:b
		    		p[:,dcount] = Pderiv[:,i,j]
		    		dcount += 1
		    	end
		    end
		# If d > 1, then the input is a size (T, b, b, d) array
		elseif ndims(P) == 4
		    T, b, ~, d = size(P)
		    n = div((b^2-b),2)
		    N = n*d

		    Pderiv = zeros(T-1, b, b, d)
		    p = zeros(T-1, N)

		    for t = 1:T-1
		        for i = 1:d
		            Pderiv[t,:,:,i] = real.(log(transpose(P[t,:,:,i])*P[t+1,:,:,i]))
		        end
		    end

		    dcount = 1
		    for k = 1:d
			    for i = 1:b-1
			    	for j = i+1:b
			    		p[:,dcount] = Pderiv[:,i,j,k]
			    		dcount += 1
			    	end
			    end
			end

		else
		    error("Incorrect input. For SO(b)^d valued time series, input must be size 
		    	(T, b, b) or (T, b, b, d).")
		end
	end

	return p
end



###############################################################################
# Function: kernel_derivative
# Description: Computes Gram matrices for the discrete derivatives of two paths
#   using a kernel function, used in the signature kernel computation when
#   composed with an initial kernel
# Input 
#   P1, P2: time series, either in the form of:
#       (1) (T+1, N)-array, where columns represent the point at time t
#       (2) (T+1)-array of objects, where each column represents the object at time t
#           - each object can be of any type
#           - used when objects don't have a fixed-size representation such as persistence diagrams
#   kernel: a function kernel(a,b) with two inputs
#       - the types of the inputs are the same
#       - the types must correspond with the form of the time series:
#           (1) type must be a length N vector
#           (2) type must be the type of the object in P
# Output 
#   DK1: the Gram matrix for the discrete derivative of P1
#   DK2: the Gram matrix for the discrete derivative of P1
#   DK12: the Gram matrix between the discrete derivative of P1 and the discrete derivative of P2
###############################################################################
function kernel_derivative(P1, P2, kernel::Function)

    T1 = size(P1,1)
    T2 = size(P2,1)

    Kfull = kernel_submatrix(vcat(P1, P2), kernel, 1:(T1+T2), 1:(T1+T2))

    K1 = Kfull[1:T1, 1:T1]
    K2 = Kfull[T1+1:end,T1+1:end]
    K12 = Kfull[1:T1, T1+1:end]

    DK1 = zeros(T1-1, T1-1)
    DK2 = zeros(T2-1, T2-1)
    DK12 = zeros(T1-1, T2-1)

    for i = 1:T1-1
        for j = 1:T1-1
            if i==j
                DK1[i,j] = K1[i+1, j+1] - K1[i+1, j] - K1[i, j+1] + K1[i,j]
            else
                DK1[i,j] = K1[i+1, j+1] - K1[i+1, j] - K1[i, j+1] + K1[i,j]
                DK1[j,i] = DK1[i,j]
            end
        end
    end

    for i = 1:T2-1
        for j = 1:T2-1
            if i==j
                DK2[i,j] = K2[i+1, j+1] - K2[i+1, j] - K2[i, j+1] + K2[i,j]
            else
                DK2[i,j] = K2[i+1, j+1] - K2[i+1, j] - K2[i, j+1] + K2[i,j]
                DK2[j,i] = DK2[i,j]
            end
        end
    end

    for i = 1:T1-1
        for j = 1:T2-1
            DK12[i,j] = K12[i+1, j+1] - K12[i+1,j] - K12[i,j+1] + K12[i,j]
        end
    end

    return DK1, DK2, DK12
end


#############################################################################
# Function: batch_discrete_derivative 
# Description: Batch discrete derivative computation
# Input 
# 	BP: batch of time series (array of arrays of time series)
# 	dtype: type of time series
# Output 
# 	bp: batch discrete derivative
#############################################################################
function batch_discrete_derivative(BP, dtype)

	S = size(BP)[1] # size of batch
	CC = Array{Colon,1}(undef, ndims(BP)-1)

	# Initialize the array for the collection of disrete derivatives
	bp = Array{Array{Float64, 2}, 1}(undef, S)

	for i = 1:S
		bp[i] = discrete_derivative(BP[i,CC...], dtype)
	end

	return bp
end

###############################################################################
## SIGNATURE FUNCTIONS ########################################################
###############################################################################

###############################################################################
# Function: signature 
# Description: Computes the continuous signature of interpolated time series.
# Input 
#	P: time series
#		(T+1, N) array if dtype = "R"
#		(T+1, b, b) or T(b, b, d) array if dtype = "SO"
#	M: truncation level 
#	dtype: type of time series, currently implemented:
#		"R": real valued time series
#		"SO": SO(b)^d valued time series (possibly with d = 1)
# Output 
#	S: truncated continuous signature of time series
###############################################################################
function signature(P, M, dtype)

	# Compute discrete derivative
	p = discrete_derivative(P, dtype)

	# Get size of path signature
	T, N = size(p)

	# Initialize tensor algebra
	sig1 = initialize_TA(N, M)
	sig2 = initialize_TA(N, M)
	# lastsig = initialize_TA(N,M)
	cur_exp = initialize_TA(N, M) # variable for the current tensor exponent

	# Initialize first time segment
	tensor_exp!(sig1,p[1,:], M)

	for t = 2:T
		tensor_exp!(cur_exp, p[t,:], M)
		if mod(t, 2) == 0
			multiply_TA!(sig2, sig1, cur_exp, M)
		else
			multiply_TA!(sig1, sig2, cur_exp, M)
		end
	end

	if mod(T,2) == 0
		return sig2
	else
		return sig1
	end

end


###############################################################################
# Function: dsignature 
# Description: Computes the discrete approximation of the path signature.
# Input 
#	P: time series
#		(T+1, N) array if dtype = "R"
#		(T+1, b, b) or T(b, b, d) array if dtype = "SO"
#	M: truncation level 
#	dtype: type of time series, currently implemented:
#		"R": real valued time series
#		"SO": SO(b)^d valued time series (possibly with d = 1)
# Output 
#	S: discrete signature truncated at level m
###############################################################################
function dsignature(P, M, dtype)

	# Compute discrete derivative
	p = discrete_derivative(P, dtype)

	# Get size of path signature
	T, N = size(p)

	# Initialize the tensor algebra element as an array of arrays
	S = initialize_TA(N, M)

	for i = 1:N
	    cur_ind = zeros(Int, M)
	    cur_ind[1] = i
	    Q = cumsum(view(p,:,i))
	    S[1][i] = Q[end]

	    dsig_forward(S, Q, p, cur_ind, 2, M, N)
	end

	return S
end


###############################################################################
# Function: dsig_forward 
# Description: The forward recursion step in the discrete signature function.
# Input 
#	S: current signature 
#	lastQ: last signature path
#	p: discrete derivative
#	cur_ind: current signature index
#	cur_level: current level
#	last_level: truncation level
#	N: dimension of Lie group
###############################################################################
function dsig_forward(sigl, lastQ, p, cur_ind, cur_level, last_level, N)

    if cur_level < last_level

        for i = 1:N-1
            cur_ind[cur_level] = i
            Q = cumsum(lastQ .* view(p,:,i))
            sigl[cur_level][cur_ind[1:cur_level]...] = Q[end]

            dsig_forward(sigl, Q, p, cur_ind, cur_level+1, last_level, N)
        end

        # On the last run through, we no longer need the information from
        # lastQ, so just use that variable instead of allocating more memory
        cur_ind[cur_level] = N
        cumsum!(lastQ, lastQ .* view(p,:,N))
        sigl[cur_level][cur_ind[1:cur_level]...] = lastQ[end]

        dsig_forward(sigl, lastQ, p, cur_ind, cur_level+1, last_level, N)
    else

        for i = 1:N
            cur_ind[cur_level] = i
            sigl[cur_level][cur_ind...] = sum(lastQ .* view(p,:,i))
        end
    end
end


###############################################################################
## SIGNATURE KERNEL FUNCTIONS #################################################
###############################################################################


###############################################################################
# Function: dsignature_kernel (dtype is string)
# Description: Computes the normalized discrete signature kernel for two paths. 
# Input 
#	P1, P2: two time series
#	M: truncation level
#	dtype: type of time series
# Output 
#	K: kernel value
###############################################################################
function dsignature_kernel(P1, P2, M, dtype::String)

    p1 = discrete_derivative(P1, dtype)
    p2 = discrete_derivative(P2, dtype)

    K1 = p1*p1'
    K2 = p2*p2'
    K12 = p1*p2'

    return dsignature_kernel_gram(K1, K2, K12, M)
end

###############################################################################
# Function: dsignature_kernel (dtype is function)
# Description: Computes the normalized discrete signature kernel for two paths,
#   with respect to the specified initial kernel.
# Input 
#   P1, P2: two time series
#   M: truncation level
#   dtype: initial kernel for data
# Output 
#   K: kernel value
###############################################################################
function dsignature_kernel(P1, P2, M, dtype::Function)

    DK1, DK2, DK12 = kernel_derivative(P1, P2, dtype)

    return dsignature_kernel_gram(DK1, DK2, DK12, M)
end

###############################################################################
# Function: dsignature_kernel_gram
# Description: Computes the normalized discrete signature kernel for two paths,
#   where all derivative gram matrices are precomputed
# Input
#   K1, K2: derivative Gram matrix for the two paths
#   K12: derivative Gram matrix between paths 1 and 2 
#   M: truncation level
# Output 
#   K: signature kernel
###############################################################################
function dsignature_kernel_gram(DK1, DK2, DK12, M)

    lambda1 = tensor_normalization_computation(DK1, M)
    lambda2 = tensor_normalization_computation(DK2, M)

    K = dsignature_kernel_computation(DK12, lambda1, lambda2, M)

    return K
end


###############################################################################
# Function: tensor_normalization_computation
# Description: Computes the normalization constant for the signature using the
#   derivative Gram matrix.
# Input 
#   dK: derivative Gram matrix for a path
#   M: truncation level
# Output 
#   lambda: normalization constant for the underlying path
###############################################################################
function tensor_normalization_computation(dK, M)
    # Compute the tensor normalization
    cumcoeff = zeros(M+1)

    # Compute normalization for P1
    K = dK
    A = deepcopy(K)
    cumcoeff[2] = sum(view(A,:))

    for i = 2:M   
        cumsum!(A,A,dims=1)
        cumsum!(A,A,dims=2)
        A =  @. K*(1 + A)
        cumcoeff[i+1] = sum(view(A,:))
    end

    tnorm = cumcoeff[end] + 1
    coeff = cumcoeff[2:end] - cumcoeff[1:end-1]
    f(x) = sum(coeff.*(x.^((1:M)*2))) + 1 - PHI(sqrt(tnorm),4.,1.)
    lambda = find_zero(f, (0., max(2.,tnorm)))

    return lambda
end

###############################################################################
# Function: dsignature_kernel_computation
# Description: Computes the discrete signature kernel given the derivative Gram
#   matrix between two paths, and the normalizatin constants for each path.
# Input 
#   dK: derivative Gram matrix for a path
#   lambda1, lambda2: normalization constants for the two underlying paths
#   M: truncation level
# Output 
#   K: signature kernel
###############################################################################
function dsignature_kernel_computation(dK, lambda1, lambda2, M)
    
    ll = lambda1*lambda2

    K = dK*ll
    A = deepcopy(K)
    for i = 2:M   
        cumsum!(A,A,dims=1)
        cumsum!(A,A,dims=2)
        A =  @. K*(1 + A)
    end
    K = 1 + sum(view(A,:))

    return K
end


###############################################################################
# Function: dsignature_kernel_matrix
# Description: Computes the signature kernel matrix for two batches of time series, using
#   the normalized discrete signature kernel. 
# Input 
#   BP1, BP2: two batches of time series, where each is a 1D array of arrays
#       - each element of an array is a time series, which can be of two forms:
#          (1) (T+1, N)-array, where columns represent the point at time t
#          (2) (T+1)-array of objects, where each column represents the object at time t
#       - we use an array of arrays to allow for different lengths for the time series
#   M: truncation level
#   dtype: either the type of the time series, or a kernel function
# Output 
#   K: signature kernel matrix
###############################################################################
function dsignature_kernel_matrix(BP1, BP2, M, dtype)

    isgram = false

    if isempty(BP2)
        BP2 = BP1
        isgram = true
    end

    S1 = length(BP1)
    S2 = length(BP2)

    K = zeros(S1, S2)

    for i = 1:S1
        for j = 1:S2
            if isgram && i < j
                continue
            end

            K[i,j] = dsignature_kernel(BP1[i], BP2[j], M, dtype)

            if isgram && i != j
                K[j,i] = K[i,j]
            end
        end
    end

    return K
end


###############################################################################
# Function: dsignature_MMDu
# Description: Computes the unbiased maximum mean discrepancy (MMD).
# Input 
#	BP1, BP2: two batches of time series
#	M: truncation level
#	dtype: type of time series
# Output 
#	MMD_val: MMD value
###############################################################################
function dsignature_MMDu(BP1, BP2, M, dtype)

	# Compute all discrete derivatives
    bp1 = batch_discrete_derivative(BP1, dtype)
    bp2 = batch_discrete_derivative(BP2, dtype)

    S1 = length(bp1)
    S2 = length(bp2)

    # Compute MMD
    MMD_val1 = 0.0
    MMD_val2 = 0.0
    MMD_val3 = 0.0

    for i = 1:S1-1	
        for j = i+1:S1
        	MMD_val1 += dsignature_kernel_preprocessed(bp1[i], bp1[j], M)
        end
    end

    for i = 1:S2-1
        for j = i+1:S2
        	MMD_val2 += dsignature_kernel_preprocessed(bp2[i], bp2[j], M)
        end
    end

    for i = 1:S1
        for j = 1:S2
        	MMD_val3 += dsignature_kernel_preprocessed(bp1[i], bp2[j], M)
        end
    end

    MMD_val = MMD_val1*2/(S1*(S1-1)) + MMD_val2*2/(S2*(S2-1)) - MMD_val3*2/(S1*S2)

    return MMD_val
end





