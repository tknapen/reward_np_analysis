def PHmodel(trial_array, alpha, kappa, pos, which = 'V'):

    Vs = np.zeros(trial_array.shape[0])
    PEs = np.zeros(trial_array.shape[0])
    alphas = np.zeros(trial_array.shape[0])

    # start at
    V = 1

    for i in range(trial_array.shape[0]):
    
        Vs[i] = V
        alphas[i] = alpha

        # are we expecting a reward because there's a stim?
        # for now, these are the only trials we update on.
        if trial_array[i][0] == 'S': 
            expectation = V
        else:
            expectation = 0

        if trial_array[i][1] == 'R':
            feedback = pos
        else:
            feedback = 0

        # calculate prediction error
        PEs[i] = feedback - expectation

        # calculate present alpha
        alpha = kappa * np.abs(PEs[i])

        # update V
        V = V + alpha * PEs[i]

    if which == 'V':
        return Vs
    elif which == 'PE':
        return PEs
    elif which == 'alphas':
        return alphas

    return Vs, PEs, alphas



