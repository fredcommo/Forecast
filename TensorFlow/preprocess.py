def train_test_set(df):
    column_indices = {name: i for i, name in enumerate(df.columns)}

    n = len(df)
    train_df = df[0:int(n*0.7)]
    val_df = df[int(n*0.7):int(n*0.9)]
    test_df = df[int(n*0.9):]

    num_features = df.shape[1]

    print(f"train size: {train_df.shape[0]}")
    print(f"val size: {val_df.shape[0]}")
    print(f"test size: {test_df.shape[0]}")
    print(f"n features: {num_features}")

    return train_df, val_df, test_df


def normalize(df, train_df, val_df, test_df):
    train_mean = train_df.mean()
    train_std = train_df.std()

    norm_train_df = (train_df - train_mean) / train_std
    norm_val_df = (val_df - train_mean) / train_std
    norm_test_df = (test_df - train_mean) / train_std
    df_std = (df - train_mean) / train_std

    # df_std is returned for plotting only
    return norm_train_df, norm_val_df, norm_test_df, df_std
