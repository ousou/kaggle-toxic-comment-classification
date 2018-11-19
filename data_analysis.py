from data_reader import read_data

def print_data_stats():
    X_train, y_train, X_test, y_test = read_data()
    print("Train data stats:")
    for column in y_train.columns:
        value_counts = y_train[column].value_counts()
        print("%s: Zeroes: %i, Ones: %i, zero perc: %.5f" %
              (column, value_counts[0], value_counts[1], value_counts[0] / (value_counts[1] + value_counts[0])))

    print("Validation data stats:")
    for column in y_test.columns:
        value_counts = y_test[column].value_counts()
        print("%s: Zeroes: %i, Ones: %i, zero perc: %.5f" %
              (column, value_counts[0], value_counts[1], value_counts[0] / (value_counts[1] + value_counts[0])))


if __name__ == '__main__':
    print_data_stats()