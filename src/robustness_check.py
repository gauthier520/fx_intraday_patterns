import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from tqdm import tqdm



def bootstrap_analysis(sessions, n_iterations=1000, sample_size=1000):
    # We will perform a bootstrap analysis to check the robustness of our results
    # instead of using the full dataset, we will sample random days for each session
    # For each sampling we will check if we do see negative mean returns across local session
    # and positive returns across foreign sessions
    # We will plot the distribution of Sharpe Ratio of the returns by bucket across each ccy
    # We will record the % of negative SRs/positive SRs over our sampling experiments

    # Reset the seed here as it was set to 1 for generating the data
    np.random.seed(None)

    session_list = list(sessions.keys())
    ccy_list = sessions['Domestic'].columns
    neg_sr_pct = pd.DataFrame(0, index=ccy_list, columns=session_list)
    session_results = {session_name: [] for session_name in session_list}
    for i in tqdm(range(n_iterations)):
        new_sessions = {session_name: sessions[session_name].sample(sample_size) for session_name in sessions}
        for session_name in session_list:
            sr = new_sessions[session_name].mean() / new_sessions[session_name].std()
            neg_sr_pct[session_name] += (sr < 0).astype(int)
            session_results[session_name].append(sr.rename(i))
    # Format and concatenate simulation results
    for session_name in session_list:
        session_results[session_name] = pd.concat(session_results[session_name], axis=1).T
    neg_sr_pct = neg_sr_pct / n_iterations

    # Do a barplot for each session over all ccy to show the frequency of sampling events with negative
    # with negative SR across the session
    make_barplot(neg_sr_pct)

    # Do a distribution plot for each ccy of the session SR over our sampling events
    distribution_plot(session_results)
    return


def make_barplot(df):
    # Create bar plots for each session
    for session in df.columns:
        plt.figure(figsize=(8, 5))
        plt.bar(df.index, df[session])
        plt.title(f'Simulated Probability of Negative Sharpe Ratios for {session}')
        plt.xlabel('Currencies')
        plt.ylabel('Probability of Negative Sharpe Ratio')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()


def distribution_plot(session_results):
    # Loop through each currency and create a subplot
    currencies = session_results['Domestic'].columns
    for i, ccy in enumerate(currencies):
        fig, axes = plt.subplots(1, 4, sharey=True,figsize=(8, 5))
        # Loop through each session and create a subplot for the current currency
        for i, (session, df) in enumerate(session_results.items()):
            ax = axes[i]
            sns.histplot(df[ccy], kde=True, ax=ax, color="skyblue")
            # Add a vertical line at 0
            ax.axvline(0, color='black')
            # Set title and labels
            ax.set_title(f'{session} - {ccy}')
            ax.set_xlabel('Sharpe Ratio')

        # Set ylabel only on the first subplot
        axes[0].set_ylabel('Frequency')

        # Adjust the layout
        fig.suptitle(f'Simulated Distribution of the Sharpe Ratio of each trading sesion returns - {ccy}', fontsize=14)
        plt.tight_layout()
        plt.show()
