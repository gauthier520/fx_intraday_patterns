import pandas as pd
from arch import arch_model

def garch_1_1_test(sessions):
    # We will restrict ourselves to a lag of 3 trading sessions
    # effectively including all the most recent different trading sessions
    # We will fit the GARCH(1,1) model and will look at the constant mean of the mean equation across trading sessions
    ccy_list = sessions['Domestic'].columns
    session_list = list(sessions.keys())
    means_ccy = pd.DataFrame(index=ccy_list, columns=session_list)
    pvalues_ccy = means_ccy.copy()
    for ccy in ccy_list:
        ccy_all_sessions = pd.concat(
            [sessions[session_name][ccy].rename(session_name) for session_name in session_list], axis=1).dropna()
        ccy_all_sessions['intercept'] = 1
        for session_name in session_list:
            df = ccy_all_sessions.copy()
            df = df.dropna()
            # Fit GARCH Model
            y = df.loc[:, session_name]
            # We rescale our returns by 1000 to facilitate convergence of the optimizer
            # otherwise the optimizer will throw a marning that y is too small in magnitude
            model = arch_model(y * 1000, vol='Garch',p=1,q=1,mean='constant',lags=2)
            # Fit model and disable verbose
            fitted_model = model.fit(disp='off')
            # Access summary statistics
            mu = fitted_model.params['mu']/1000
            p_val = fitted_model.pvalues['mu']
            means_ccy.loc[ccy, session_name] = mu
            pvalues_ccy.loc[ccy, session_name] = p_val
    return means_ccy, pvalues_ccy