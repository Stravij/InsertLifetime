##############################
# 2. Funzioni per grafici 
##############################

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np


def plot_cumVSuse_groups(triplet_df):
    # function to plot the history of a an utensil on different machines

    triplet_df = triplet_df.sort_values(by=['mmdb_name','timestamp'])#.head(30)
    # print(triplet_df.shape)
    # print(triplet_df.head(20))
    if len(triplet_df['mtdb_gno'].unique())>1:
        print('dataset can contain only on single utensil code. Instead following are present')
        print(triplet_df['mtdb_gno'].unique())
    else:
        gno_name = triplet_df['mtdb_gno'].unique()[0]
        n_subplots = len(triplet_df['mmdb_name'].unique())
        mach_plot = triplet_df['mmdb_name'].unique()
        # print(mach_plot[0])
        print('N. of machine: ', n_subplots)

        fig, ax = plt.subplots(n_subplots,figsize=(12,4*n_subplots),sharex=False)
        # Normalize ax to always be iterable
        if n_subplots == 1:
            ax = [ax]
        # Loop over machines
        for i in range(n_subplots):
            # Create subset of utensil and machine for plotting
            plot_df = triplet_df[triplet_df['mmdb_name']==mach_plot[i]]
            if plot_df['mtdb_tid'].nunique()>1:
                ax[i].text(0.5, 0.5, f'Warning: chart not created!\nMultiple mtdb_tid present on the same machine {mach_plot[i]}!', horizontalalignment='center', verticalalignment='center')
                continue
            # Use index positions as x instead of real datetimes
            x = range(len(plot_df))
            # Bar plot of use
            ax[i].bar(x, plot_df['mtdb_use'], label='Use', alpha=0.4, width=0.8)
            # Line plot of cumulative use
            ax[i].plot(x, plot_df['cum_use'], color='red', marker='o', markersize=2, label='Cumulated Use')
            # Line plot of max use
            ax[i].plot(x, plot_df['mtdb_lif'], color='black', linestyle='dashed', label='Max use')
            # Adjust xlabel tick at most N tick labels
            if len(x)>=50:
                N = 50  # adjustable
            else:
                N = len(x)
            idx = np.linspace(0, len(x)-1, min(N, len(x)), dtype=int)
            ax[i].set_xticks(idx)
            ax[i].set_xticklabels(
                # plot_df['timestamp'].iloc[idx].dt.strftime('%Y-%m-%d %H:%M'),
                plot_df['timestamp'].iloc[idx].dt.strftime('%Y-%m-%d'),
                rotation=60, ha='right'
            )
            # ax[i].set_title(f'Overview utensil {gno_name} with id {tid_lab} on machine {mach_plot}')
            ax[i].set_title(f'Overview utensil {gno_name} on machine {mach_plot[i]}')
            # ax[i].set_xlabel("Timestamp")
            ax[i].set_ylabel("Use / Cumulated Use / Max use [s]")
            ax[i].legend()
            ax[i].grid(True, alpha=0.3)
        fig.suptitle(f'Detailed Overview of insert {gno_name} - cumulated use vs real use')
        plt.tight_layout()
        plt.show()

def plot_cumVSuse_triplet(triplet_df, ax = None):
    # print('Columns: ', triplet_df.columns)
    standalone = ax is None
    if ax is None:  # if no axis passed, fall back to global plt
        fig, ax = plt.subplots(figsize=(12,6))

    # function to plot the history of a unique triplet

    # check combinations of triplets
    check_df = triplet_df[['mtdb_gno','mtdb_tid','mmdb_name']].copy()
    check_df.drop_duplicates(inplace=True)
    
    if check_df.shape[0] > 1:
        print('dataset can contain only on single triplet (mtdb_gno + mtdb_tid + mmdb_name). Instead following cases are present')
        print(check_df)

    ###################
    # Perfectly working
    ###################

    gno_lab = check_df['mtdb_gno'].unique()[0]
    tid_lab = check_df['mtdb_tid'].unique()[0]
    mach_lab = check_df['mmdb_name'].unique()[0]

    # Use index positions as x instead of real datetimes
    x = range(len(triplet_df))

    # Bar plot of use
    ax.bar(x, triplet_df['mtdb_use'], label='Use', alpha=0.4)

    # Line plot of cumulative use
    ax.plot(x, triplet_df['cum_use'], color='red', marker='o', markersize=2, label='Cumulated Use')
    # ax.plot(x, triplet_df['cum_use'], color='red', label='Cumulated Use')

    # Line plot of max use
    ax.plot(x, triplet_df['mtdb_lif'], color='black', linestyle='dashed', label='Max use')

    # Adjust xlabel tick at most N tick labels
    if len(x)>=50:
        N = 50  # adjustable
    else:
        N = len(x)
    idx = np.linspace(0, len(x)-1, min(N, len(x)), dtype=int)
    ax.set_xticks(idx)
    ax.set_xticklabels(
        # triplet_df['timestamp'].iloc[idx].dt.strftime('%Y-%m-%d %H:%M'),
        triplet_df['timestamp'].iloc[idx].dt.strftime('%Y-%m-%d'),
        rotation=90, ha='right'
    )
    # ax.set_title(f'Overview utensil {gno_lab} with id {tid_lab} on machine {mach_lab}')
    ax.set_title(f'Detailed Overview')
    ax.set_xlabel("Timestamp")
    ax.set_ylabel("Use / Cumulated Use / Max use [s]")
    ax.legend()
    ax.grid(True, alpha=0.3)
    if standalone:
        plt.show()
        plt.tight_layout()