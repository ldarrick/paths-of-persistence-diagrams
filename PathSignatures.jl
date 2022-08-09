
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
## MAIN FUNCTIONS #############################################################
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
# Description: Computes the normalized discrete signature kernel for two paths.
# The inputted function is assumed to be a kernel function for the corresponding
# input data. 
# Input 
#   P1, P2: two time series
#   M: truncation level
#   dtype: type of time series
# Output 
#   K: kernel value
###############################################################################
function dsignature_kernel(P1, P2, M, dtype::Function, lr=-1)

    DK1, DK2, DK12 = kernel_derivative(P1, P2, dtype, lr)

    return dsignature_kernel_gram(DK1, DK2, DK12, M)
end

###############################################################################
###############################################################################
###############################################################################

function kernel_derivative(P1, P2, kernel::Function, lr=-1)

    T1 = size(P1,1)
    T2 = size(P2,1)

    if lr > 0
        Kfull = nystrom(vcat(P1,P2), lr, kernel)
        # Kfull = U*V'
    else
        Kfull = kernel_submatrix(vcat(P1, P2), kernel, 1:(T1+T2), 1:(T1+T2))
    end

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

###############################################################################
###############################################################################
###############################################################################

# function dsignature_kernel_lr(DK1, DK2, DK12, M, r)
function dsignature_kernel_lr(P1, P2, M, kernelFunc, r)

    n1 = size(P1,1)-1
    n2 = size(P2,1)-1

    # Compute full low rank approximation
    P = vcat(P1, P2)
    U, V = nystrom(P, r, kernelFunc)

    dU = U[2:end,:] - U[1:end-1,:]
    dV = V[2:end,:] - V[1:end-1,:]

    # Separate approximation into parts
    dU1 = dU[1:n1,:]
    dV1 = dV[1:n1,:]

    dU2 = dU[n1+2:end,:]
    dV2 = dV[n1+2:end,:]

    dU12 = dU1
    dV12 = dV2

    lambda1 = lr_tensor_normalization_computation(dU1, dV1, M)
    lambda2 = lr_tensor_normalization_computation(dU2, dV2, M)

    K = dsignature_kernel_lr_computation(dU12, dV12, lambda1, lambda2, M)

    return K

end

###############################################################################
###############################################################################
###############################################################################

function lr_tensor_normalization_computation(dU, dV, M)
    n1 = size(dU, 1)
    n2 = size(dV, 1)

    cumcoeff = zeros(M+1)

    B = dU
    C = dV

    P = deepcopy(B)
    Q = deepcopy(C)

    R = sum(P,dims=1)
    S = sum(Q,dims=1)

    cumcoeff[2] = sum(R.*S)

    for d = 2:M
        cumsum!(P,P,dims=1)
        cumsum!(Q,Q,dims=1)

        P = hcat(P,ones(n1))
        Q = hcat(Q,ones(n2))

        P = starprod(dU, P)
        Q = starprod(dV, Q)

        R = sum(P,dims=1)
        S = sum(Q,dims=1)

        cumcoeff[d+1] = sum(R.*S)
    end

    tnorm = cumcoeff[end] + 1
    coeff = cumcoeff[2:end] - cumcoeff[1:end-1]
    f(x) = sum(coeff.*(x.^((1:M)*2))) + 1 - PHI(sqrt(tnorm),4.,1.)
    lambda = find_zero(f, (0., max(2.,tnorm)))

    return lambda
end

###############################################################################
###############################################################################
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
###############################################################################
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
###############################################################################
###############################################################################

function dsignature_kernel_lr_computation(dU, dV, lambda1, lambda2, M)
    n1 = size(dU, 1)
    n2 = size(dV, 1)

    ll = lambda1*lambda2

    # B = dU*lambda1
    # C = dV*lambda2

    B = dU*sqrt(ll)
    C = dV*sqrt(ll)

    P = deepcopy(B)
    Q = deepcopy(C)

    for d = 2:M
        cumsum!(P,P,dims=1)
        cumsum!(Q,Q,dims=1)

        P = hcat(P,ones(n1))
        Q = hcat(Q,ones(n2))

        P = starprod(B, P)
        Q = starprod(C, Q)
    end
    R = sum(P,dims=1)
    S = sum(Q,dims=1)

    return 1 + sum(R.*S)
end

###############################################################################
###############################################################################
###############################################################################

function starprod(A, B)

    na, ma = size(A)
    nb, mb = size(B)

    if na != nb
        error("Matrix sizes are incompatible")
    end

    M = ma*mb

    C = zeros(na, M)

    for i = 1:na
        for j1 = 1:ma
            for j2 = 1:mb

                C[i, (j1-1)*mb + j2] = A[i,j1]*B[i,j2]
            end
        end
    end

    return C

end

###############################################################################
# Function: dsignature_kernel_gram
# Description: Computes the normalized discrete signature kernel for two paths,
#	where all gram matrices are precomputed
# Input
#   K1: Gram matrix for path 1
#   K2: Gram matrix for path 2
#   K12: Mixed Gram matrix for path 1 and 2 
#	p1, p2: discrete derivatives for two time series
#	M: truncation level
# Output 
#	K: kernel value
###############################################################################
function dsignature_kernel_gram(DK1, DK2, DK12, M)

    lambda1 = tensor_normalization_computation(DK1, M)
    lambda2 = tensor_normalization_computation(DK2, M)

    K = dsignature_kernel_computation(DK12, lambda1, lambda2, M)

    return K
end


###############################################################################
# Function: dsignature_gram_matrix
# Description: Computes the gram matrix for two batches of time series, using
#	the normalized discrete signature kernel. 
# Input 
#	BP1, BP2: two batches of time series
#	M: truncation level
#	dtype: type of time series
# Output 
#	K: gram matrix
###############################################################################
function dsignature_kernel_matrix(BP1, BP2, M, dtype::String, mt=false)

    # If BP2 is empty, then we are computing the gram matrix for BP1

	# Compute all discrete derivatives
    bp1 = batch_discrete_derivative(BP1, dtype)
    S1 = length(bp1)

    if !isempty(BP2)
        bp2 = batch_discrete_derivative(BP2, dtype)
        S2 = length(bp2)
        K = zeros(S1, S2)
    else
        K = zeros(S1, S1)
        S2 = S1
        bp2 = bp1
    end

    if mt 
        Threads.@threads for i = 1:S1
            for j = 1:S2
                if isempty(BP2) && i < j
                    continue
                end
    
                DK1 = bp1[i]*bp1[i]'
                DK2 = bp2[j]*bp2[j]'
                DK12 = bp1[i]*bp2[j]'
    
                K[i,j] = dsignature_kernel_gram(DK1, DK2, DK12, M)
    
                if isempty(BP2)
                    K[j,i] = K[i,j]
                end
            end
        end
    else
        for i = 1:S1
            for j = 1:S2
                if isempty(BP2) && i < j
                    continue
                end

                DK1 = bp1[i]*bp1[i]'
                DK2 = bp2[j]*bp2[j]'
                DK12 = bp1[i]*bp2[j]'

                K[i,j] = dsignature_kernel_gram(DK1, DK2, DK12, M)

                if isempty(BP2)
                    K[j,i] = K[i,j]
                end
            end
        end
    end

    return K
end

###############################################################################
# Function: dsignature_gram_matrix
# Description: Computes the gram matrix for two batches of time series, using
#   the normalized discrete signature kernel. 
# Input 
#   BP1, BP2: two batches of time series
#   M: truncation level
#   dtype: type of time series
# Output 
#   K: gram matrix
###############################################################################
function dsignature_kernel_matrix(BP1, BP2, M, dtype::Function)

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



###############################################################################
# Description: Diagonalizes a skew-symmetric matrix into 2x2 blocks of paired
# eigenvalues, along with its corresponding matrix of eigenvectors. Eigenvalues 
# are ordered from largest to smallest in magnitude.
#
function skew_diagonalize(A)
    tol = 1e-10 # tolerance for sanity check at the end
    d, ~ = size(A)

    # Check skew-symmetric
    if A != -transpose(A)
        error("Input matrix is not skew-symmetric.")
    end

    # Transformation to take a conjugate pair of eigenvalues into a 2x2 block
    T = [1 -im; 1 im]/sqrt(2)

    # Compute eigenvalues and sort the pairs from largest to smallest
    # Also, put 0 eigenvalues at the end.
    F = eigen(A)
    evals = F.values

    pp = sortperm(evals,by=abs, rev=true)
    evals = evals[pp]

    zeroeig = findall(imag(evals).==0)
    pp2 = collect(1:d)

    for i = reverse(1:length(zeroeig))
        splice!(pp2,zeroeig[i])
        push!(pp2,zeroeig[i])
    end

    evals = evals[pp2]

    # for i = 1:d
    #     println(evals[i])
    # end

    D = Diagonal(evals)
    Evecs = F.vectors
    Evecs = Evecs[:,pp]
    Evecs = Evecs[:,pp2]

    # If its odd dimensional, we will definitely have a 0 eigenvalue at the end
    if d%2==0
        id = Matrix{Float64}(I,Int(d/2),Int(d/2))
        TT = kron(id, T)
    else
        TT = zeros(Complex{Float64},d,d)
        id = Matrix{Float64}(I,Int(floor(d/2)),Int(floor(d/2)))
        TT[1:end-1, 1:end-1] = kron(id, T)
        TT[end,end] = 1
    end

    V = Evecs*TT
    V = real(inv(V)) # Matrix of eigenvectors
    SD = real(TT'*D*TT) # Skew diagonal matrix of eigenvalues

    # Sanity check
    if maximum(abs.(V*A*V' - SD)) > tol
        error("Something went wrong with the decomposition")
    end

    return V, SD
end


# Input is the kernel
function kernel_SA(K)

    T, ~ = size(K)

    M = zeros(T,T)

    CK = cumsum(K, dims=2)
    M = CK[:,1:end-1]*K[2:end,:]
    M = M - M'

    # Transformation to take a conjugate pair of eigenvalues into a 2x2 block
    TX = [1 -im; 1 im]/sqrt(2)

    F = eigen(M, K)
    evals = F.values

    pp = sortperm(evals,by=abs, rev=true)
    evals = evals[pp]

    zeroeig = findall(imag(evals).==0)
    pp2 = collect(1:T)

    for i = reverse(1:length(zeroeig))
        splice!(pp2,zeroeig[i])
        push!(pp2,zeroeig[i])
    end

    evals = evals[pp2]

    D = Diagonal(evals)
    Evecs = F.vectors
    Evecs = Evecs[:,pp]
    Evecs = Evecs[:,pp2]

    # If its odd dimensional, we will definitely have a 0 eigenvalue at the end
    if T%2==0
        id = Matrix{Float64}(I,Int(T/2),Int(T/2))
        TT = kron(id, TX)
    else
        TT = zeros(Complex{Float64},T,T)
        id = Matrix{Float64}(I,Int(floor(T/2)),Int(floor(T/2)))
        TT[1:end-1, 1:end-1] = kron(id, TX)
        TT[end,end] = 1
    end

    V = Evecs*TT
    V = real(inv(V)) # Matrix of eigenvectors
    SD = real(TT'*D*TT) # Skew diagonal matrix of eigenvalues

    V1 = V[1,:]*sqrt(abs(evals[1]))
    V2 = V[2,:]*sqrt(abs(evals[1]))

    C1 = K*V1
    C2 = K*V2

    return V, SD, C1, C2

end



##############################################################

function recursive_nystroem(X, s, kernelFunc)

    # m = length(X)

    # if m <= s
    #     return Diagonal(ones(m))
    # end

    # ss_ind = findall(rand(Bernoulli(), m).==1)
    # Xbar = X[ss_ind]

    # mbar = length(ss_ind)

    # Sbar = zeros(m, mbar)
    # for i = 1:mbar
    #     Sbar[ss_ind[i],i] = 1
    # end

    # Stilde = recursive_nystroem(Xbar, kernel, s, delta/3)

    # Shat = Sbar*Stilde

    sLevel = s
    n, d = size(X)

    # Start of algorithm
    oversamp = log(sLevel)
    k = ceil(Int,sLevel/(4*oversamp))
    nLevels = ceil(Int,log(n/sLevel)/log(2))

    # Create random permutation for successful uniform samples
    perm = randperm(n)

    # Set up sizes for recursive levels
    lSize = zeros(Int, nLevels+1)
    lSize[1] = n
    for i = 2:nLevels+1
        lSize[i] = ceil(Int,lSize[i-1]/2)
    end

    # rInd: indices of points selected at previous level of recursion
    # at the base level, it's just a uniform sample of ~sLevel points
    samp = 1:lSize[end]
    rInd = perm[samp]
    weights = ones(length(rInd))

    # We need the diagonal of the whole kernel matrix, so compute upfront
    kDiag = kernelFunc(X, 1:n, [])

    # main recursion
    for l = nLevels:-1:1
        # Indices of current uniform sample
        rIndCurr = perm[1:lSize[l]]

        # Build sampled kernel
        KS = kernelFunc(X, rIndCurr, rInd)
        SKS = KS[samp, :]
        SKSn = size(SKS,1)

        # Optimal lambda for taking O(klogk) samples
        if k >= SKSn
            lambda = 10e-6
        else
            F = eigen(SKS)
            lambda = sum(F.values[1:k])/k
        end

        # Compute and sample by lambda ridge leverage scores
        if l != 1
            # On intermediate levels, we independently sample each column
            # by its leverage score. the sample size is sLevel in expectation
            R = inv(SKS + Diagonal(lambda*weights.^(-2)))
            # max(0,.) helps avoid numerical issues, unnecessary in theory
            levs = min.(1, oversamp*(1/lambda)*max.(0, (kDiag[rIndCurr] - sum((KS*R).*KS,dims=2))))
            samp = findall(rand(1, lSize[l]) .< levs)

            # With very low probability, we could accidentally sample no
            # columns. In this case, just take a fixed size uniform sample
            if(isempty(samp))
                levs[:] = sLevel/lSize[l]
                samp = randperm(lSize[l])[1:sLevel]
            end

            weights = sqrt.(1 ./(levs[samp]))
        else
            # On the top level, we sample exactly x landmark points without replacement
            R = inv(SKS + Diagonal(lambda*weights.^(-2)))
            levs = min.(1, (1/lambda)*max.(0, (kDiag[rIndCurr] - sum((KS*R).*KS,dims=2))))
            samp = sample(collect(1:n), pweights(levs), s; replace=false)
        end
        rInd = perm(samp)
    end

    # Build final Nystrom approximation
    # pinv or inversion with slight regularization helps stability
    C = kernelFunc(X, 1:n, rInd);
    SKS = C(rInd, :)
    W = inv(SKS + (10e-6)*I)

    return C, W


end



function gaussianKernel(X, rowInd, colInd)
    gamma = 20.

    if isempty(colInd)
        Ksub = ones(length(rowInd))
    else
        nsqRows = sum(X[rowInd,:].^2, dims=2)
        nsqCols = sum(X[colInd,:].^2, dims=2)
        Ksub = nsqRows .- (X[rowInd,:]*(2*X[colInd,:])')
        Ksub = nsqCols' .+ Ksub
        Ksub = exp.(-gamma*Ksub)
    end
    return Ksub

end


# This is just the basic nystrom approximation by uniformly sampling
# s points
function nystrom(X, s, kernelFunc)

    if ndims(X) > 1
        n = size(X,1)
    else
        n = length(X)
    end

    # sampInd = randperm(n)[1:s]
    # sampInd = sort!(sampInd)

    stepsize = floor(Int, n/s)
    sampInd = 1:stepsize:n

    kernelSubFunc = (X, rowInd, colInd) -> kernel_submatrix(X, kernelFunc, rowInd, colInd)

    KS = kernelSubFunc(X, 1:n, sampInd)

    SKS = kernelSubFunc(X, sampInd, sampInd)

    # U = KS
    # V = pinv(SKS)*KS'

    # return U, V'

    return KS *pinv(SKS)*KS'

end


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

function sGaussianKernel(x, y)
    gamma = 20


    return exp(-gamma*norm(x-y))
end





