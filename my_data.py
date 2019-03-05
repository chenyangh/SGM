import pickle as pkl
import pandas as pd
MAX_LEN_DATA = 50
MIN_LEN_DATA = 3
def load_data():
    # data_pata = 'data/EmoSet_RemoveDup_GloveProcess_OneEmo.csv'
    data_pata = '/home/chenyang/git_repos/UniversalTextClassification/tasks/EmoSet/data/EmoSetProcessedEkmanNoDupMulti.csv'
    df_data = pd.read_csv(data_pata)

    # extract the subset which only contains the full sentences.
    source = []
    target = []
    max_len = -1
    for index, row in df_data.iterrows():
        next_token = str(row['tweet']).strip().split()
        if len(next_token) > max_len:
            max_len = len(next_token)
        if MAX_LEN_DATA > len(next_token) > MIN_LEN_DATA:
            source.append(' '.join(next_token))

            target.append(str(row['emo']).strip().split(','))
    print('Loading data finished, MIN_LEN set to ' + str(MIN_LEN_DATA) + ' found ' + str(len(source)))
    print('max len is', max_len)
    with open('emo_set_data_ekman_decreased_freq.pkl', 'bw') as f:
        pkl.dump([source, target], f)

    return source, target


X, y = load_data()

# with open('emo_set_data_ekman_decreased_freq.pkl', 'br') as f:
#     X, y = pkl.load(f)

from sklearn.model_selection import ShuffleSplit, KFold

ss = ShuffleSplit(n_splits=1, test_size=0.1, random_state=0)
ss.get_n_splits(X, y)

kf = KFold(n_splits=5, random_state=0)
kf.get_n_splits(X)

train_index, test_index = next(ss.split(y))
X_train_dev, X_test = [X[i] for i in train_index], [X[i] for i in test_index]
y_train_dev, y_test = [y[i] for i in train_index], [y[i] for i in test_index]

train_index, dev_index = next(kf.split(y_train_dev))
X_train, X_dev = [X_train_dev[i] for i in train_index], [X_train_dev[i] for i in dev_index]
y_train, y_dev = [y_train_dev[i] for i in train_index], [y_train_dev[i] for i in dev_index]


def gen_data_files(X, y, post_fix):
    f = open('text_' + post_fix, 'w')
    for line in X:
        f.write(line + '\n')
    f.close()

    f = open('label_' + post_fix, 'w')
    for label in y:
        f.write(' '.join(label) + '\n')
    f.close()

gen_data_files(X_train, y_train, 'train')
gen_data_files(X_dev, y_dev, 'val')
gen_data_files(X_test, y_test, 'test')