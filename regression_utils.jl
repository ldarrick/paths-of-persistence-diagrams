using ScikitLearn
@sk_import svm:SVR
using ScikitLearn.CrossValidation: cross_val_score, KFold
# using ThreadPools

function run_regression(K_train, K_test, CL, numIterations, SVR_C, SVR_eps, n_hyp_cv=4, tr_te_split = 0.8)

    # Check whether we need to perform cross-validation for hyperparameters
    hyp_cv = true
    num_SVRC = 0
    num_SVReps = 0
    if ndims(SVR_C) == 0 && ndims(SVR_eps) == 0
        hyp_cv = false
    else
        num_SVRC = length(SVR_C)
        num_SVReps = length(SVR_eps)
    end

    # Numbers of train and test set
    N = size(K_train)[1]
    n_tr = round(Int, N*tr_te_split)
    n_te = N - n_tr

    # Save the best SVM params (numIterations, CL, SVM_c/eps)
    SVR_params = zeros(numIterations,2,2)   

    # Regression error
    reg_error = zeros(numIterations, 2)

    for it = 1:numIterations
        
        # Generate current permutation and split
        curperm = randperm(MersenneTwister(it), N)

        pK_train = K_train[curperm, curperm]
        pK_test = K_test[curperm, curperm]

        curCL = CL[curperm, :]
        cl_tr = curCL[1:n_tr, :]
        cl_te = curCL[n_tr+1:end, :]

        K_tr = pK_train[1:n_tr, 1:n_tr]
        K_te = pK_test[n_tr+1:end, 1:n_tr]

        # Cross-validation for hyperparameter tuning
        if hyp_cv
            # Further split the training set into train/validation sets for hyperparameter tuning
            kf2 = KFold(n_tr, n_folds=n_hyp_cv, shuffle=false);
            scores1_hyp = zeros(num_SVRC, num_SVReps);
            scores2_hyp = zeros(num_SVRC, num_SVReps);

            for k = 1:n_hyp_cv
                K_tr2 = K_tr[kf2[k][1], kf2[k][1]]
                K_val = K_tr[kf2[k][2], kf2[k][1]]
                cl_tr2 = cl_tr[kf2[k][1],:]
                cl_val = cl_tr[kf2[k][2],:]
                n_val = length(cl_val[:,1])

                for j1 = 1:num_SVRC
                    for j2 = 1:num_SVReps
                        # C hyperparameter tuning
                        cls = SVR(kernel="precomputed", C=SVR_C[j1], epsilon=SVR_eps[j2])
                        ScikitLearn.fit!(cls, K_tr2, cl_tr2[:,1])
                        SVR_predict = ScikitLearn.predict(cls, K_val)
                        scores1_hyp[j1,j2] += sum(abs.(SVR_predict - cl_val[:,1]).^2)/n_val

                        # eps hyperparameter tuning
                        cls = SVR(kernel="precomputed", C=SVR_C[j1], epsilon=SVR_eps[j2])
                        ScikitLearn.fit!(cls, K_tr2, cl_tr2[:,2])
                        SVR_predict = ScikitLearn.predict(cls, K_val)
                        scores2_hyp[j1,j2] += sum(abs.(SVR_predict - cl_val[:,2]).^2)/n_val
                    end
                end
            end

            # Find best hyperparameters
            best1 = findlast(scores1_hyp .== minimum(scores1_hyp))
            C1 = SVR_C[best1[1]]
            eps1 = SVR_eps[best1[2]]
            
            best2 = findlast(scores2_hyp .== minimum(scores2_hyp))
            C2 = SVR_C[best2[1]]
            eps2 = SVR_eps[best2[2]]
        else
            C1 = SVR_C
            eps1 = SVR_eps

            C2 = SVR_C
            eps2 = SVR_eps
        end

        # Save these hyperparameters
        SVR_params[it,1,1] = C1
        SVR_params[it,1,2] = eps1
        SVR_params[it,2,1] = C2
        SVR_params[it,2,2] = eps2

        # Perform regression on full dataset
        cls = SVR(kernel="precomputed", C=C1, epsilon=eps1)
        ScikitLearn.fit!(cls, K_tr, cl_tr[:,1])
        SVR_predict = ScikitLearn.predict(cls, K_te)
        reg_error[it, 1] = sum(abs.(SVR_predict - cl_te[:,1]).^2)/n_te

        cls = SVR(kernel="precomputed", C=C2, epsilon=eps2)
        ScikitLearn.fit!(cls, K_tr, cl_tr[:,2])
        SVR_predict = ScikitLearn.predict(cls, K_te)
        reg_error[it, 2] = sum(abs.(SVR_predict - cl_te[:,2]).^2)/n_te

    end

    return reg_error, SVR_params

end
###########################################################################
# Note: Here, we assume that we will do hyperparameter tuning over kernels, C, and eps
function run_regression_multikernel(K_train, K_test, CL, numIterations, SVR_C, SVR_eps, n_hyp_cv=4, tr_te_split = 0.8)

    # Check whether we need to perform cross-validation for hyperparameters
    num_SVRC = length(SVR_C)
    num_SVReps = length(SVR_eps)
    num_kernel = length(K_train)

    # Numbers of train and test set
    N = size(K_train[1])[1]
    n_tr = round(Int, N*tr_te_split)
    n_te = N - n_tr

    # Save the best SVM params (numIterations, CL, SVM_c/eps/kernel)
    SVR_params = zeros(numIterations,2,3)   

    # Regression error
    reg_error = zeros(numIterations, 2)

    for it = 1:numIterations
        
        # Generate current permutation
        curperm = randperm(MersenneTwister(it), N)

        # Permute CL and split
        curCL = CL[curperm, :]
        cl_tr = curCL[1:n_tr, :]
        cl_te = curCL[n_tr+1:end, :]

        # Further split the training set into train/validation sets for hyperparameter tuning
        kf2 = KFold(n_tr, n_folds=n_hyp_cv, shuffle=false);
        scores1_hyp = zeros(num_SVRC, num_SVReps, num_kernel);
        scores2_hyp = zeros(num_SVRC, num_SVReps, num_kernel);

        for j3 = 1:num_kernel
            # Cross-validation for hyperparameter tuning
            pK_train = K_train[j3][curperm, curperm]
            # pK_test = K_test[j3][curperm, curperm]

            K_tr = pK_train[1:n_tr, 1:n_tr]
            # K_te = pK_test[n_tr+1:end, 1:n_tr]

            for k = 1:n_hyp_cv
                K_tr2 = K_tr[kf2[k][1], kf2[k][1]]
                K_val = K_tr[kf2[k][2], kf2[k][1]]
                cl_tr2 = cl_tr[kf2[k][1],:]
                cl_val = cl_tr[kf2[k][2],:]
                n_val = length(cl_val[:,1])

                for j1 = 1:num_SVRC
                    for j2 = 1:num_SVReps
                        # C hyperparameter tuning
                        cls = SVR(kernel="precomputed", C=SVR_C[j1], epsilon=SVR_eps[j2])
                        ScikitLearn.fit!(cls, K_tr2, cl_tr2[:,1])
                        SVR_predict = ScikitLearn.predict(cls, K_val)
                        scores1_hyp[j1,j2,j3] += sum(abs.(SVR_predict - cl_val[:,1]).^2)/n_val

                        # eps hyperparameter tuning
                        cls = SVR(kernel="precomputed", C=SVR_C[j1], epsilon=SVR_eps[j2])
                        ScikitLearn.fit!(cls, K_tr2, cl_tr2[:,2])
                        SVR_predict = ScikitLearn.predict(cls, K_val)
                        scores2_hyp[j1,j2,j3] += sum(abs.(SVR_predict - cl_val[:,2]).^2)/n_val
                    end
                end
            end
        end

        # Find best hyperparameters
        best1 = findlast(scores1_hyp .== minimum(scores1_hyp))
        C1 = SVR_C[best1[1]]
        eps1 = SVR_eps[best1[2]]
        ker1 = best1[3]
        
        best2 = findlast(scores2_hyp .== minimum(scores2_hyp))
        C2 = SVR_C[best2[1]]
        eps2 = SVR_eps[best2[2]]
        ker2 = best2[3]
 
        # Save these hyperparameters
        SVR_params[it,1,1] = C1
        SVR_params[it,1,2] = eps1
        SVR_params[it,1,3] = ker1
        SVR_params[it,2,1] = C2
        SVR_params[it,2,2] = eps2
        SVR_params[it,2,3] = ker2

        # Use best kernel for C regression
        pK_train = K_train[ker1][curperm, curperm]
        pK_test = K_test[ker1][curperm, curperm]
        K_tr = pK_train[1:n_tr, 1:n_tr]
        K_te = pK_test[n_tr+1:end, 1:n_tr]

        # Perform regression on full dataset
        cls = SVR(kernel="precomputed", C=C1, epsilon=eps1)
        ScikitLearn.fit!(cls, K_tr, cl_tr[:,1])
        SVR_predict = ScikitLearn.predict(cls, K_te)
        reg_error[it, 1] = sum(abs.(SVR_predict - cl_te[:,1]).^2)/n_te

        # Use best kernel for l regression
        pK_train = K_train[ker2][curperm, curperm]
        pK_test = K_test[ker2][curperm, curperm]
        K_tr = pK_train[1:n_tr, 1:n_tr]
        K_te = pK_test[n_tr+1:end, 1:n_tr]

        cls = SVR(kernel="precomputed", C=C2, epsilon=eps2)
        ScikitLearn.fit!(cls, K_tr, cl_tr[:,2])
        SVR_predict = ScikitLearn.predict(cls, K_te)
        reg_error[it, 2] = sum(abs.(SVR_predict - cl_te[:,2]).^2)/n_te
    end

    return reg_error, SVR_params

end