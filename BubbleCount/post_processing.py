def get_normalized_mse(df1, df2):
    # reset the indices to align
    df1 = df1.reset_index(drop=True)
    df2 = df2.reset_index(drop=True)

    # Calculate the mean of each dataframe
    mean_df1 = df1.mean()
    mean_df2 = df2.mean()

    # Normalize the dataframes by dividing each by its mean
    normalized_df1 = df1 / mean_df1
    normalized_df2 = df2 / mean_df2

    # Calculate the MSE between the normalized dataframes
    nmse = ((normalized_df1 - normalized_df2) ** 2).mean()

    return nmse