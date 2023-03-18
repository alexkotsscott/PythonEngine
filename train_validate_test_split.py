def train_validate_test_split(df, train_percent=.6, validate_percent=.2, seed=None):
    
    """
        Args:
            df:                 data frame to split into train, validate, split segments
            train_percent:      split for the train split of the df
            validate_percent:   split for the validation data
            seed:               set seed for replication
        Returns:    
            Train Data
            Validation Data
            Test Data
    """
    
    np.random.seed(seed)
    perm = np.random.permutation(df.index)
    m = len(df.index)
    train_end = int(train_percent * m)
    validate_end = int(validate_percent * m) + train_end
    train = df.iloc[perm[:train_end]]
    validate = df.iloc[perm[train_end:validate_end]]
    test = df.iloc[perm[validate_end:]]
    return train, validate, test
