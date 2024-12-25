data = pd.read_csv(r'D:\study\credit_scroing_datasets\bankfear.csv', low_memory=True)
features = data.drop(['loan_status', 'member_id'], axis=1).replace([-np.inf, np.inf, -np.nan, np.nan], 0)
labels = data['loan_status']

train_size = int(features.shape[0] * 0.8)
valid_size = int(features.shape[0] * 0.1)
test_size = valid_size

with open(shuffle_path + 'bankfear/shuffle_index.pickle', 'rb') as f:
    shuffle_index = pickle.load(f)

train_index = shuffle_index[:train_size]
valid_index = shuffle_index[train_size:(train_size + valid_size)]
test_index = shuffle_index[(train_size + valid_size):(train_size + valid_size + test_size)]
remaining_index = shuffle_index[(valid_size + test_size):]

train_x, train_y = features.iloc[train_index, :], labels.iloc[train_index]
valid_x, valid_y = features.iloc[valid_index], labels.iloc[valid_index]
test_x, test_y = features.iloc[test_index], labels.iloc[test_index]
remaining_x, remaining_y = features.iloc[remaining_index], labels.iloc[remaining_index]

full_train_x = pd.concat([train_x, valid_x], axis=0)
full_train_y = pd.concat([train_y, valid_y], axis=0)