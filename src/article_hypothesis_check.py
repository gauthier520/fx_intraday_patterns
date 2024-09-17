import numpy as np
import pandas as pd
from scipy.stats import ttest_ind, ttest_1samp, f_oneway

def do_and_visualize_analysis(data):
    # Transform into log returns for normality assumption required in usual statistical tests
    # Returns are often approximated to be log normals
    returns = np.log(data.pct_change() + 1)
    sessions = partition_returns_into_session(returns)

    mean_ret, mean_pval = one_sample_t_test(sessions)
    bivariate_test, bivariate_pval = two_sample_t_test(sessions)
    welch_test, welch_pval = simultaneous_welch_f_test(sessions)

    # Format Results for Display
    # We display cell background as green for values significant at 5% level otherwise red
    styled_df_mean = style_df_based_on_pval(mean_ret, mean_pval)
    styled_df_bivariate = style_df_based_on_pval(bivariate_test, bivariate_pval)
    styled_welch = style_df_based_on_pval(welch_test.to_frame(), welch_pval.to_frame())


    # Aggregate all results into a dictionary for easy access
    results = dict()
    results['styled_mean_by_session'] = styled_df_mean
    results['styled_two_sample_ttest'] = styled_df_bivariate
    results['styled_simultaneous_welch_test'] = styled_welch
    results['mean_by_session'] = mean_ret
    results['two_sample_ttest'] = bivariate_test
    results['simultaneous_welch_test'] = welch_test
    results['mean_by_session_pval'] = mean_pval
    results['two_sample_ttest_pval'] = bivariate_pval
    results['simultaneous_welch_test_pval'] = welch_pval

    return results


def partition_returns_into_session(returns):
    # We will partition the returns into the different trading sessions identified
    # based on the trading hours

    domestic_session = returns.loc[returns.index.hour.isin(range(8, 15))]
    domestic_session = domestic_session.groupby(domestic_session.index.date).sum()

    ldn_ny_overlap = returns.loc[returns.index.hour.isin(range(15, 19))]
    ldn_ny_overlap = ldn_ny_overlap.groupby(ldn_ny_overlap.index.date).sum()

    us_session = returns.loc[returns.index.hour.isin(range(19, 23))]
    us_session = us_session.groupby(us_session.index.date).sum()

    non_us_session = returns.loc[returns.index.hour <= 7]
    non_us_session = non_us_session.groupby(non_us_session.index.date).sum()

    sessions = {
        'Domestic': domestic_session,
        'LDN-NY': ldn_ny_overlap,
        'US': us_session,
        'Non US': non_us_session
    }
    return sessions


def get_session_length_config():
    sessions_length = {
        'Domestic': 7,
        'LDN-NY': 4,
        'US': 4,
        'Non US': 7
    }
    return sessions_length


def one_sample_t_test(sessions):
    # Perform one sample t-tests that sessions means are significantly different from 0 for each ccy
    ccy_list = sessions['Domestic'].columns
    session_list = list(sessions.keys())
    means = pd.DataFrame(index=ccy_list, columns=session_list)
    p_values = means.copy()
    for ccy in ccy_list:
        for session_name in session_list:
            ret_session_ccy = sessions[session_name][ccy]
            _, p_val = ttest_1samp(ret_session_ccy, 0)
            means.loc[ccy, session_name] = ret_session_ccy.mean()
            p_values.loc[ccy, session_name] = p_val
    return means, p_values


def two_sample_t_test(sessions):
    # Perform two-sample t-tests that domestic returns are significantly different from foreign returns
    # for each ccy
    domestic_session = sessions['Domestic']
    ccy_list = domestic_session.columns
    session_list = ['LDN-NY', 'US', 'Non US']
    sessions_length = get_session_length_config()
    t_stats = pd.DataFrame(index=ccy_list, columns=session_list)
    p_values = t_stats.copy()
    # Compare domestic session to foreign sessions (LDN-NY, U.S., non-U.S.)
    for ccy in ccy_list:
        for session_name in session_list:
            comparison_session = sessions[session_name]
            # Perform two-sample t-test (domestic vs foreign session)
            # Normalize returns by length of period to make them comparable, otherwise the test will lose its significance
            domestic_session_ccy = domestic_session[ccy] / sessions_length['Domestic']
            comparison_session_ccy = comparison_session[ccy] / sessions_length[session_name]
            t_st, p_val = ttest_ind(domestic_session_ccy, comparison_session_ccy, equal_var=False)
            p_values.loc[ccy, session_name] = p_val
            t_stats.loc[ccy, session_name] = t_st
    return t_stats, p_values


def simultaneous_welch_f_test(sessions):
    # Perform Welch's F-test to test if the mean returns are equal across all sessions.
    sessions_length = get_session_length_config()
    domestic_session = sessions['Domestic']
    ccy_list = domestic_session.columns
    f_stats = pd.Series(index=ccy_list)
    p_values = f_stats.copy()
    for ccy in ccy_list:
        # Normalize returns by length of period to make them comparable, otherwise the test will lose its significance
        session_returns_ccy = [sessions[session_name][ccy] / sessions_length[session_name] for session_name in
                               sessions]
        f_st, p_val = f_oneway(*session_returns_ccy)
        p_values.loc[ccy] = p_val
        f_stats.loc[ccy] = f_st
    return f_stats, p_values

def style_df_based_on_pval(df_val, pvals):
    # Function to apply custom background color with transparency based on cell values
    # P-values above 0.05 will be highlighted in red otherwise in green
    styled_df = df_val.style.apply(
        lambda row: [color_with_transparency(pvals.loc[row.name, col]) for col in row.index], axis=1)
    return styled_df


def color_with_transparency(val):
    if val > 0.05:
        # Red with 70% transparency for negative values
        color = 'rgba(255, 0, 0, 0.70)'
    else:
        # Green with 50% transparency for positive values
        color = 'rgba(0, 255, 0, 0.70)'
    return f'background-color: {color}'
