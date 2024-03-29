import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


def basic_visualizations():

    data_dir = 'data/'
    rpc_file = 'implanter_rtc.csv'
    apc_file = 'implanter_apc.csv'

    rpc = pd.read_csv(data_dir + rpc_file)
    apc = pd.read_csv(data_dir + apc_file)

    # remove any columns that only have nans
    rpc.dropna(axis=1, how='all', inplace=True)
    apc.dropna(axis=1, how='all', inplace=True)

    # Turn time columns into datetime.
    # There might be more data cleaning we want to do.
    rpc['time_stamp'] = pd.to_datetime(rpc['time_stamp'])
    apc['runstart'] = pd.to_datetime(apc['runstart'])

    # Sort by timestamp.
    rpc.sort_values('time_stamp', inplace=True)
    apc.sort_values('runstart', inplace=True)

    # todo convert main status codes to integers
    # ...

    # do some basic stats on the main status codes

    # make histogram on number of occurrences of status
    ax = rpc['main_status'].value_counts().plot.bar()
    ax.set_xlabel('status code')
    ax.set_ylabel('occurrences')
    plt.tight_layout()
    plt.savefig('status_code_bar_chart.png')
    plt.clf()

    # make histogram of amount of time spent in each status
    # todo make sure that duration is working the way I think it is (that it is for the correct status)
    status_values = rpc['main_status'].unique()
    status_to_time_spent = []
    for status_val in status_values:
        time_spent_in_status = rpc[rpc['main_status'] == status_val]['duration'].sum()
        # decimal in number of days
        time_spent_in_status *= 24
        status_to_time_spent.append((status_val, time_spent_in_status))
    status_to_time_spent = sorted(status_to_time_spent, key=lambda x: x[1], reverse=True)
    status_to_time_spent = pd.DataFrame(status_to_time_spent, columns=['main_status', 'duration'])
    status_to_time_spent.set_index('main_status', inplace=True)
    ax = status_to_time_spent.plot.bar()
    ax.set_xlabel('Status')
    ax.set_ylabel('Duration')
    plt.tight_layout()
    plt.savefig('time_spent_in_status.png')
    plt.clf()

def get_feature_columns():
    features_to_include_file = 'data/features_to_include.txt'
    features = []
    with open(features_to_include_file) as file:
        for line in file:
            line = line.strip()
            features.append(line)
    return features


def split_data_into_recipe_train_split():
    data_dir = 'data/'
    labeled_data_file = 'implanter_with_data.csv'
    data = pd.read_csv(data_dir + labeled_data_file)
    # remove any columns that only have nans and
    features = get_feature_columns()
    data = data[['label'] + features]
    data.dropna(axis=1, how='all', inplace=True)
    # Turn time columns into datetime.
    # There might be more data cleaning we want to do.
    data['runstart'] = pd.to_datetime(data['runstart'])
    # Sort by timestamp.
    data.sort_values('runstart', inplace=True)
    recipe_data = []
    for recipe in data['origrecipename'].unique():
        local_data = data[data['origrecipename'] == recipe]
        number_of_fails = len(local_data[local_data['label'] == -1])
        recipe_data.append((number_of_fails, local_data))
    recipe_data = sorted(recipe_data, key=lambda x: x[0], reverse=True)
    recipe_data = [x[1] for x in recipe_data]
    recipe_data = recipe_data[:4]
    for i in range(len(recipe_data)):
        local_data = recipe_data[i]
        negative_set = local_data[local_data['label'] == -1]
        train_negative, test_negative = train_test_split(negative_set, test_size=0.25)
        positive_set = local_data[local_data['label'] == 1]
        train_positive, test_positive = train_test_split(positive_set, test_size=0.25)

        train = pd.concat((train_negative, train_positive))
        test = pd.concat((test_positive, test_negative))

        train = train.sample(frac=1)
        test = test.sample(frac=1)

        recipe_folder_test = 'data/recipe/test/'
        recipe_folder_train = 'data/recipe/train/'

        train.drop('origrecipename', axis=1, inplace=True)
        train.drop('runstart', axis=1, inplace=True)
        test.drop('origrecipename', axis=1, inplace=True)
        test.drop('runstart', axis=1, inplace=True)

        train.to_csv(recipe_folder_train + f'train_data_{i + 1}.csv')
        test.to_csv(recipe_folder_test + f'test_data_{i + 1}.csv')


def discretize_std(data):
    std = data.std()
    avg = data - data.mean()
    avg = avg.abs()
    for i in reversed(range(1, 120)):
        mask = avg <= i * std
        data[mask] = i

    return data


def process_training_data():
    training_folder = 'data/recipe/train/'
    training_data = []
    labels = []
    for i in range(1, 5):
        file_name = f'train_data_{i}.csv'
        data = pd.read_csv(training_folder + file_name)
        # fixme: not sure if this is the right thing to do. Need to check.
        # remove this stuff
        data.drop('Unnamed: 0', axis=1, inplace=True)
        labels.append(data['label'])
        data.drop('label', axis=1, inplace=True)
        training_data.append(data)

    return training_data, labels


def create_all_data():
    split_data_into_recipe_train_split()
    data_sets, labels = process_training_data()
    data_sets = list(map(discretize_std, data_sets))
    data_sets = [pd.concat((l, d), axis=1) for (l, d) in zip(labels, data_sets)]

    for i, data in enumerate(data_sets):
        recipe_folder_test = 'data/recipe/test/'
        recipe_folder_train = 'data/recipe/train/'
        data.to_csv(recipe_folder_train + f'train_data_{i + 1}_binned.csv')
        data.to_csv(recipe_folder_test + f'test_data_{i + 1}_binned.csv')


if __name__ == '__main__':
    from sklearn import decomposition, model_selection, naive_bayes, metrics
    from sklearn.svm import SVC

    data = pd.read_csv('data/recipe/train/train_data_1.csv')
    y = data['label']
    X = data.drop('label', axis=1)
    X.drop('origrecipename', axis=1, inplace=True)
    X.drop('runstart', axis=1, inplace=True)

    # nb = naive_bayes.GaussianNB()
    # nb = SVC()

    # pca = decomposition.LatentDirichletAllocation(2)
    # thing = pca.fit_transform(X, y)
    # thing = pd.concat((y, pd.DataFrame(thing)), axis=1)
    #
    # import matplotlib.pyplot as plt
    # ax = thing.plot.scatter(x=0, y=1, c='label', colormap='viridis')
    # plt.show()


    # print()





