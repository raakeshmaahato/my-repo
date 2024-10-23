import pandas as pd
import numpy as np

def calculate_distance_matrix(df)->pd.DataFrame():
    """
    Calculate a distance matrix based on the dataframe, df.

    Args:
        df (pandas.DataFrame)

    Returns:
        pandas.DataFrame: Distance matrix
    """
    # Write your logic here
    df_pivot = df.pivot(index='ID1', columns='ID2', values='Distance').fillna(0)
    distance_matrix = df_pivot + df_pivot.T
    np.fill_diagonal(distance_matrix.values, 0)
    for i in range(len(distance_matrix)):
        for j in range(i + 1, len(distance_matrix)):
            if distance_matrix.iloc[i, j] == 0:  
                distance_matrix.iloc[i, j] = distance_matrix.iloc[i].add(distance_matrix.iloc[:, j]).min()
                distance_matrix.iloc[j, i] = distance_matrix.iloc[i, j]

    return df


def unroll_distance_matrix(df)->pd.DataFrame():
    """
    Unroll a distance matrix to a DataFrame in the style of the initial dataset.

    Args:
        df (pandas.DataFrame)

    Returns:
        pandas.DataFrame: Unrolled DataFrame containing columns 'id_start', 'id_end', and 'distance'.
    """
    # Write your logic here
    unrolled = []
    for id_start in df.index:
        for id_end in df.columns:
            distance = df.loc[id_start, id_end]
            if id_start != id_end:  
                unrolled.append({'id_start': id_start, 'id_end': id_end, 'distance': distance})

    return df


def find_ids_within_ten_percentage_threshold(df, reference_id)->pd.DataFrame():
    """
    Find all IDs whose average distance lies within 10% of the average distance of the reference ID.

    Args:
        df (pandas.DataFrame)
        reference_id (int)

    Returns:
        pandas.DataFrame: DataFrame with IDs whose average distance is within the specified percentage threshold
                          of the reference ID's average distance.
    """
    # Write your logic here
    reference_distance = df[df['id_start'] == reference_id]['distance'].mean()
    lower_bound = reference_distance * 0.9
    upper_bound = reference_distance * 1.1

    within_threshold = df[(df['distance'] >= lower_bound) & (df['distance'] <= upper_bound)]

    return df


def calculate_toll_rate(df)->pd.DataFrame():
    """
    Calculate toll rates for each vehicle type based on the unrolled DataFrame.

    Args:
        df (pandas.DataFrame)

    Returns:
        pandas.DataFrame
    """
    # Wrie your logic here
    toll_rates = {
        'moto': 0.8,  
        'car': 1.2,   
        'rv': 1.5,    
        'bus': 2.2,   
        'truck': 3.6  
    }

    for vehicle, rate in toll_rates.items():
        df[vehicle] = df['distance'] * rate

    return df


def calculate_time_based_toll_rates(df)->pd.DataFrame():
    """
    Calculate time-based toll rates for different time intervals within a day.

    Args:
        df (pandas.DataFrame)

    Returns:
        pandas.DataFrame
    """
    # Write your logic here
    discounts = {
        'weekday': 0.9,  
        'weekend': 1.0   
    }
    df['toll_rate'] = df[['moto', 'car', 'rv', 'bus', 'truck']].mean(axis=1)  
    df['final_toll'] = np.where(
        df['start_day'].isin(['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']),
        df['toll_rate'] * discounts['weekday'],
        df['toll_rate'] * discounts['weekend']
    )

    return df
