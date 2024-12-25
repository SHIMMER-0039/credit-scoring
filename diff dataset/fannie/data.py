data = pd.read_csv(r'D:\study\credit_scroing_datasets\FannieMae/2008q1.csv', low_memory=True)
features = data.drop(['DEFAULT', 'LOAN IDENTIFIER'], axis=1).replace([-np.inf, np.inf, np.nan], 0)
labels = data['DEFAULT']

train_size = int(features.shape[0] * 0.7)
valid_size = int(features.shape[0] * 0.15)
test_size = features.shape[0] - train_size - valid_size

with open(shuffle_path + 'fannie/shuffle_index.pickle', 'rb') as f:
    shuffle_index = pickle.load(f)
train_index = shuffle_index[:train_size]
valid_index = shuffle_index[train_size:(train_size + valid_size)]
test_index = shuffle_index[(train_size + valid_size):(train_size + valid_size + test_size)]

train_x, train_y = features.iloc[train_index, :], labels.iloc[train_index]
valid_x, valid_y = features.iloc[valid_index, :], labels.iloc[valid_index]
test_x, test_y = features.iloc[test_index, :], labels.iloc[test_index]

full_train_x = pd.concat([train_x, valid_x], axis=0)
full_train_y = pd.concat([train_y, valid_y], axis=0)