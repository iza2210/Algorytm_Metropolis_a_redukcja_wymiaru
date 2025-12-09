import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tqdm import tqdm

import argparse

def ParseArguments():
    parser = argparse.ArgumentParser(description="Opis")
    parser.add_argument('--r', default="1", help="Start (default: %(default)s)")
    parser.add_argument('--iter', default="200", help="Iter  (default: %(default)s)")
    parser.add_argument('--tau', default="500", help="tau")
    parser.add_argument('--max_fraction', default="1.0", help="max fraction (default: %(default)s)")
    parser.add_argument('--seed', default="3141",
                        help="Seed for the PRNG; use 'None' for no fixed seed (default: %(default)s)")
    return parser.parse_args()


args = ParseArguments()

np.random.seed(int(args.seed))


X = pd.read_parquet('./dane/X_data_without_nulls.parquet', engine = 'fastparquet')
y = pd.read_parquet('./dane/y_data.parquet', engine = 'fastparquet')


def model(X_train, X_test, y_train, y_test):
    clf = XGBClassifier()
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    test_acc = accuracy_score(y_test, y_pred)
    test_gini = 2*roc_auc_score(y_test, y_pred)-1

    y_train_pred = clf.predict(X_train)
    train_acc = accuracy_score(y_train, y_train_pred)
    train_gini = 2*roc_auc_score(y_train, y_train_pred)-1

    df = pd.DataFrame({'ACCURACY TEST' : [test_acc],
                        'ACCURACY TRAIN' : [train_acc],
                        'GINI TEST' : [test_gini],
                        'GINI TRAIN' : [train_gini],
                        'PRZEUCZENIE' : [train_gini - test_gini]})
    return(df)


X_train, X_test, y_train, y_test = train_test_split(
        X, y['NY.GDP.PCAP.CD'], test_size=0.3, stratify=y['NY.GDP.PCAP.CD'], random_state=42
    )
 

print(y_train.value_counts())
print(y_test.value_counts())
# NY.GDP.PCAP.CD
# 0    1750
# 1    1563
# Name: count, dtype: int64

# NY.GDP.PCAP.CD
# 0    750
# 1    671
# Name: count, dtype: int64


# 0) benchmark - ocena modelu na calym zbiorze
full = model(X_train, X_test, y_train, y_test)

full_acc = full['ACCURACY TEST'][0]

 

def metropolis(tau, liczba_iteracji, r=1, d=100, max_frac=0.5):
    indices_current = np.zeros(d, dtype=int)
    indices_current[np.random.choice(100, r, replace=False)] = 1
    df = model(X_train.iloc[:, np.where(indices_current == 1)[0]], X_test.iloc[:, np.where(indices_current == 1)[0]], y_train, y_test)
    df['Numer iteracji'] = 0
    df['Wybrane indeksy'] = str(indices_current)
    df['Liczba wybranych'] = np.sum(indices_current)
    df['ind_used'] = 1

    for nr in tqdm(np.arange(liczba_iteracji)):
        ind_chosen = np.random.choice(100, 1, replace = False)
        indices_new = indices_current.copy()
        indices_new[ind_chosen] = 1 - indices_new[ind_chosen]

        tmp = model(X_train.iloc[:, np.where(indices_new == 1)[0]], X_test.iloc[:, np.where(indices_new == 1)[0]], y_train, y_test)
        tmp['Numer iteracji'] = nr + 1
        tmp['Wybrane indeksy'] = str(indices_current)
        tmp['Liczba wybranych'] = np.sum(indices_current)

        pi_ratio = np.exp(tau*(tmp['ACCURACY TEST'].iloc[-1] - df[df['ind_used'] == 1]['ACCURACY TEST'].iloc[-1]))

        if (np.sum(indices_current) == 0) | (np.sum(indices_current) >= max_frac*d):
            tmp['ind_used'] = 0
        elif pi_ratio >= 1:
            indices_current = indices_new
            tmp['ind_used'] = 1
        else:
            los = np.random.uniform(0, 1)
            if los <= pi_ratio:
                indices_current = indices_new
                tmp['ind_used'] = 1
            else:
                tmp['ind_used'] = 0

        df = pd.concat([df, tmp])
    return df  # df[df['ACCURACY TEST'] == np.max(df['ACCURACY TEST'])]  -- zwroci tylko najlepsze wyniki


df = metropolis(tau = float(args.tau), liczba_iteracji = int(args.iter), r=1, d=100, max_frac = float(args.max_fraction))
print(df[df['ACCURACY TEST'] == np.max(df['ACCURACY TEST'])])
print(df['Liczba wybranych'].max())

ile_wsp = df["Liczba wybranych"].to_numpy()

plt.plot(np.arange(len(df)), df['ACCURACY TEST'])
plt.plot([0,len(df)-1],[full_acc,full_acc])

plt.savefig("./wykresy/accuracy_vs_liczba_iteracji_r_zmiennych_WDI_2000_2017.png", dpi=300)

plt.figure()
plt.plot(np.arange(len(ile_wsp)),ile_wsp)
plt.show()