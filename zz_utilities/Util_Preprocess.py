##############################
# 2. Funzioni preprocessing 
##############################
import pandas as pd

def input_prep(data):
    # Function to manipulate the input raw dataframe
    df = data.copy()
    print('Total n. of records:', df.shape[0], ' | and columns:', df.shape[1])
    # remove not relevant columns
    df.drop(columns=['Extra_ToolData.location_subid','mtdb_pno','Extra_ToolData.location_id',
                    'mtdb_rot', 'mtdb_tedg','mtdb_ang', 'mtdb_twid', 'mtdb_mtang', 'mtdb_dpw',
                    'mtdb_clif', 'mtdb_cuse','mtdb_mamox', 'mtdb_mamoz'],inplace=True)
    # rename columns
    df = df.rename(columns={'Extra_ToolData.Mazak_MachineData.mmdb_name': 'mmdb_name'})
    # fix time date in the correct format
    df['timestamp'] = pd.to_datetime(df['timestamp'], format='%d/%m/%Y %H:%M', errors='coerce')
    # modify columns with the correct unit
    df[['mtdb_len']]=df[['mtdb_len']]/1000000
    df[['mtdb_dia']]=df[['mtdb_dia']]/1000000
    # filtrare righe in cui è presente il limite vita utensile
    df = df[df['mtdb_lif']>0]
    # rimuovere utensili tornio e con 6 cifre
    df = df[~df['mtdb_gno'].isin([0, 36,111111, 111112])]
    # rimuovere tid = 0 
    df = df[~df['mtdb_tid'].isin([0])]
    
    # drop additional columns not relevant
    df.drop(columns=['mtdb_dia','mtdb_len','mtdb_hpw','mtdb_tpr','mtdb_lenb','mtdb_nomDia','mtdb_tno','id','mtdb_id'],inplace=True)
    # Sorting dataset
    df = df.sort_values(['mtdb_gno','mtdb_tid','mmdb_name','timestamp'])
    
    # Stampare dimensioni dataset
    print('Total n. of records:', df.shape[0], ' | and columns:', df.shape[1])
    return df


def process_group(group, threshold_pct):
    # function to detect insert change or manual drop over a specific unique combination (group)

    # set list of values
    cum_use = []
    reset_counter = []
    drop_counter = []
    drop_increase = []
    delta_increase = []

    # set counters
    running_total = 0
    reset_count = 1 # start from 1
    drop_count = 0
    drop_increase_total = 0
    delta = 0
    prev_use = None

    # read max use allowed
    lif = group['mtdb_lif'].iloc[0]
    
    # iter through each group of triplets
    for use in group['mtdb_use']:
        if prev_use is None:
            running_total = use
            delta = use
        else:
            # a decrease happened
            if use < prev_use:  
                if use < threshold_pct * lif:
                    # in case of change of the insert
                    reset_count += 1 # increase
                    running_total = use  # restart from current use
                    drop_count = 0 # reset drop count
                    drop_increase_total = 0 # reset increase after a drop
                    # print('current use',use, '| current lif', lif)
                    delta = use
                else:
                    # Drop detected
                    drop_count += 1 # increase drop count
                    drop_increase_total = 0 # reset increase after a drop 
                    running_total = running_total  # keep summing values
                    delta = use
            # an increase happened
            else:
                running_total += (use - prev_use)
                drop_increase_total += (use - prev_use)
                delta = (use - prev_use)
        cum_use.append(running_total)
        reset_counter.append(reset_count)
        drop_counter.append(drop_count)
        drop_increase.append(drop_increase_total)
        delta_increase.append(delta)
        prev_use = use
    # return new columns with previous lists
    group['cum_use'] = cum_use
    group['reset_counter'] = reset_counter
    group['drop_counter'] = drop_counter
    group['drop_increase'] = drop_increase
    group['delta_increase'] = delta_increase
    
    # add event_state: default 1 (event)
    group["event_state"] = 1
    # mark last row as censored (0)
    group.loc[group.index[-1], "event_state"] = 0 
    return group


def process_delta(group):
    # function to detect anonamlies computing delta increments

    # set list of values
    delta_increase = []
    prev_use = None

    # read max use allowed
    lif = group['mtdb_lif'].iloc[0]
    
    # iter through each group of triplets
    for use in group['mtdb_use']:
        if prev_use is None:
            delta = use
        else:
            # a decrease happened
            if use < prev_use:  
                delta = use
            # an increase happened
            else:
                delta = (use - prev_use)
        delta_increase.append(delta)
        prev_use = use
    # return new columns with previous lists
    group['delta_increase'] = delta_increase
  
    return group


def process_grouping(df,threshold_rec,min_ratio=0,max_ratio=8000,min_reset=1):
    # 1st level of aggragation
    df_gr = df.groupby(['mtdb_gno','mtdb_tid','mmdb_name','reset_counter']).agg({'mtdb_lif':'max', 
                                                                        'cum_use':'max',
                                                                        'timestamp':'count',
                                                                        'drop_counter':'max',
                                                                        'drop_increase':'max',
                                                                        'event_state':'min'}).reset_index()
    # 2nd level of aggragataton adding metrix (min, max, mean, std, count)
    df_gr2 = df_gr.groupby(['mtdb_gno','mtdb_tid','mmdb_name']).agg({'mtdb_lif':'max',
                                                                    'timestamp':'sum',
                                                                'reset_counter':'max',
                                                                'cum_use':['min','max','mean','std'],
                                                                'drop_counter':['min','max','mean'],
                                                                'drop_increase':['min','max','mean'],}).copy().reset_index()
    print('Dimensions',df.shape[0])
    # 3rd level of filter
    df3 = df_gr2[df_gr2[('timestamp',  'sum')]>threshold_rec].sort_values(by=('timestamp','sum'),ascending=False).copy()
    # Flatten multiindex columns
    if isinstance(df3.columns, pd.MultiIndex):
        df3.columns = ['_'.join(map(str, col)).strip() for col in df3.columns.values]
    # create new column with ratio cum_use_mean
    df3['ratio_cumuse_lif']=df3['cum_use_mean']/df3['mtdb_lif_max']
    df3 = df3.sort_values(by='ratio_cumuse_lif',ascending=False)

    # Round all numeric results to 2 decimals
    float_cols = df3.select_dtypes(include="float")
    df3[float_cols.columns] = float_cols.round(2)

    # apply other filters
    df3 = df3[(df3['ratio_cumuse_lif']>min_ratio)&(df3['ratio_cumuse_lif']<max_ratio)&(df3['reset_counter_max']>=min_reset)]

    # Filter
    # Create a set of triplets from df3
    main_triplets = set(df3[['mtdb_gno_','mtdb_tid_','mmdb_name_']].itertuples(index=False, name=None))

    # Keep only rows that match most important triplets
    df_detailed = df[df[['mtdb_gno','mtdb_tid','mmdb_name']].apply(tuple, axis=1).isin(main_triplets)]
    print(' --> shape:',df_detailed.shape)

    # Keep only rows that match most important triplets
    df_gr = df_gr[df_gr[['mtdb_gno','mtdb_tid','mmdb_name']].apply(tuple, axis=1).isin(main_triplets)]
    print(' --> shape:',df_gr.shape)

    # # rename columns
    # df = df.rename(columns={'mtdb_gno_':'mtdb_gno','mtdb_tid_':'mtdb_tid','mmdb_name_': 'mmdb_name'})


    return df_gr, df_detailed, df3