from gtda.diagrams import PersistenceLandscape, PersistenceImage
import mat73
import numpy as np
from scipy.io import savemat

def compute_PD_landscape(PDfname, n_layers, n_bins, logsc, fpath):
    file = mat73.loadmat(PDfname)
    PD = file['PD']

    numRun = PD.shape[0]
    numT = PD.shape[1]
    numP = PD.shape[2]
    numD = PD.shape[3]

    PDR = np.reshape(PD, (numRun*numT, numP, numD))

    if logsc:
        B1s = np.where(PDR[1,:,2] == 1.0)[0][0]
        B2s = np.where(PDR[1,:,2] == 2.0)[0][0]

        # Log scale everything
        ind = np.where(PDR[:,:,:2] != 0.0)
        PDR[ind] = np.log10(PDR[ind])

        # Find minimum H0 death time
        min_pd = 1000
        for i in range(numRun*numT):
            nz_ind = np.where(PDR[i,:B1s,1] != 0.0)[0]
            cur_min = np.min(PDR[i,nz_ind,1])
            if cur_min < min_pd:
                min_pd = cur_min

        # Set H0 bith time to min_pd - 0.5
        LBT = min_pd - 0.5
        N = PDR.shape[0]

        for i in range(N):
            nz_ind = np.where(PDR[i,:B1s,1] != 0.0)[0]
            PDR[i,nz_ind,0] = LBT

    PL = PersistenceLandscape(n_layers=n_layers, n_bins=n_bins)
    PL.fit(PDR)
    LS = PL.fit_transform(PDR)

    LSR = np.reshape(LS, (numRun, numT, n_layers*n_bins*numD))

    mdic = {"FT": LSR}
    if logsc:
        fname = fpath + "LPL_" + str(n_layers) + "_" + str(n_bins) + ".mat"
    else:
        fname = fpath + "PL_" + str(n_layers) + "_" + str(n_bins) + ".mat"

    savemat(fname, mdic)
    
    return

def compute_PD_image(PDfname, sigma, n_bins, fpath):
    file = mat73.loadmat(PDfname)
    PD = file['PD']

    numRun = PD.shape[0]
    numT = PD.shape[1]
    numP = PD.shape[2]
    numD = PD.shape[3]

    PDR = np.reshape(PD, (numRun*numT, numP, numD))


    PI = PersistenceImage(sigma=sigma, n_bins=n_bins)
    PI.fit(PDR)
    IM = PI.fit_transform(PDR)

    IMR = np.reshape(IM, (numRun, numT, n_bins*n_bins*numD))

    mdic = {"FT": IMR}
    fname = fpath + "PI_" + '{:02d}'.format(int(sigma*10)) + "_" + str(n_bins) + ".mat"

    savemat(fname, mdic)
    
    return
