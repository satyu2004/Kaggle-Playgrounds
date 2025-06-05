def outlier_iqr(data, column):
    q1 = data[column].quantile(0.25)
    q3 = data[column].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    return lower_bound, upper_bound

def outlier_removal(data, columns):
    for column in columns:
        lower_bound, upper_bound = outlier_iqr(data, column)
        data = data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]
    return data