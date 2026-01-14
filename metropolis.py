import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse
from sklearn.linear_model import Lasso, Ridge


def ParseArguments():
    parser = argparse.ArgumentParser(description="Algorytm Metropolis")
    parser.add_argument('--k', default=1, help="Start with k variables (default: %(default))")
    parser.add_argument('--iter', default=200, help="Number of iterations  (default: %(default)s)")
    parser.add_argument('--tau', default=500, help="Learning rate - tau parameter (default: %(default)s)")
    parser.add_argument('--r', default=0,
                        help="Fixed number of variables to use. If r > 0, then parameters max_fraction and k will be ommited (default: %(default)s)")
    parser.add_argument('--max_fraction', default=1.0,
                        help="Max acceptable fraction of number of all variables (default: %(default)s)")
    parser.add_argument('--seed', default=3141,
                        help="Seed for the PRNG; use 'None' for no fixed seed (default: %(default)s)")
    return parser.parse_args()


args = ParseArguments()
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
        X, y['NY.GDP.PCAP.CD'], test_size=0.3, stratify=y['NY.GDP.PCAP.CD'], random_state=int(args.seed)
    )


# 0) benchmark - ocena modelu na calym zbiorze
full = model(X_train, X_test, y_train, y_test)

full_acc = full['ACCURACY TEST'][0]
print(f'Accuracy modelu na zbiorze testowym przy użyciu wszystkich charakterystek to: {full_acc}')
 

def metropolis(k, iter, tau, r, max_fraction, seed, X_train = X_train, X_test = X_test, y_train = y_train, y_test = y_test):
    rng = np.random.default_rng(seed)
    d = X_train.shape[1]
    indices_current = np.zeros(d, dtype=int)
    if r > 0:
        indices_current[rng.choice(np.arange(0, d), size = r, replace=False)] = 1
    else:
        indices_current[rng.choice(np.arange(0, d), size = k, replace=False)] = 1

    df = model(X_train.iloc[:, np.where(indices_current == 1)[0]],
               X_test.iloc[:, np.where(indices_current == 1)[0]],
               y_train,
               y_test)


    df['Numer iteracji'] = 0
    df['Wybrane indeksy'] = str(indices_current)
    df['Liczba wybranych'] = np.sum(indices_current)
    df['ind_used'] = 1

    for nr in tqdm(np.arange(iter)):
        indices_new = indices_current.copy()
        if r>0:
            idx0 = rng.choice(np.where(indices_current == 0)[0])
            idx1 = rng.choice(np.where(indices_current == 1)[0])
            indices_new[idx0] = 1
            indices_new[idx1] = 0
        else:
            ind_chosen = rng.choice(np.arange(0, d), size=1, replace = False)
            indices_new[ind_chosen] = 1 - indices_new[ind_chosen]
        tmp = model(X_train.iloc[:, np.where(indices_new == 1)[0]],
                    X_test.iloc[:, np.where(indices_new == 1)[0]],
                    y_train,
                    y_test)
        tmp['Numer iteracji'] = nr + 1
        tmp['Wybrane indeksy'] = str(indices_new)
        tmp['Liczba wybranych'] = np.sum(indices_new)

        pi_ratio = np.exp(tau*(tmp['ACCURACY TEST'].iloc[-1] - df[df['ind_used'] == 1]['ACCURACY TEST'].iloc[-1]))

        if (r == 0) & ((np.sum(indices_current) == 0) | (np.sum(indices_current) >= max_fraction*d)):
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


df = metropolis(k = int(args.k), iter = int(args.iter), tau = float(args.tau), r = int(args.r),
                max_fraction = float(args.max_fraction), seed = int(args.seed))
print('Najlepsze wyniki: ')
best = df[df['ACCURACY TEST'] == np.max(df['ACCURACY TEST'])]
print(best)
best_one = best[best['Numer iteracji'] == np.min(best['Numer iteracji'])]
print('\n Najlepszy wynik: ')
print(best_one)
print(f"Wybrano {best_one['Liczba wybranych']} cech")


fig, ax1 = plt.subplots()
ax1.plot(df["Numer iteracji"], df["Liczba wybranych"], color = "orange")
ax1.set_xlabel("Numer iteracji")
ax1.set_ylabel("liczba wybranych")
ax1.set_ylim(0, X_train.shape[1])

ax2 = ax1.twinx()
ax2.plot(df["Numer iteracji"], df["ACCURACY TEST"])
ax2.set_ylabel("Accuracy test")
ax2.axhline(full_acc, linestyle = '--', color = 'green')
ax2.scatter(best["Numer iteracji"], best["ACCURACY TEST"], color = 'red')

plt.tight_layout()
plt.savefig("wykresy/Liczba zmiennych i accuracy vs numer iteracji.png", dpi=30)
#plt.show()


def lasso(max_iter, X_train = X_train, X_test = X_test, y_train = y_train, y_test = y_test):
    layoutLasso = Lasso(max_iter, fit_intercept=True).fit(X_train, y_train)
    coefficients = layoutLasso.coef_
    selected_features = [i > 0 for i in coefficients]
    df = model(X_train.iloc[:, selected_features],
                X_test.iloc[:, selected_features],
                y_train,
                y_test)
    df['Wybrane indeksy'] = str(selected_features)
    df['Liczba wybranych'] = np.sum(selected_features)
    return df


df_lasso = lasso(max_iter = int(args.iter))
print('Najlepsze wyniki Lasso: ')
best_lasso = df_lasso[df_lasso['ACCURACY TEST'] == np.max(df_lasso['ACCURACY TEST'])]
print(best_lasso)

# best_one_lasso = best_lasso[best_lasso['Numer iteracji'] == np.min(best_lasso['Numer iteracji'])]
# print('\n Najlepszy wynik Lasso: ')
# print(best_one_lasso)
print(f"Wybrano {best_lasso['Liczba wybranych']} cech")

df_lasso_met = metropolis(k = int(args.k), iter = int(args.iter), tau = float(args.tau), r = df_lasso['Liczba wybranych'].max(),
                max_fraction = float(args.max_fraction), seed = int(args.seed))
print('Najlepsze wyniki: ')
best_lasso_met = df_lasso_met[df_lasso_met['ACCURACY TEST'] == np.max(df_lasso_met['ACCURACY TEST'])]
print(best_lasso_met)
best_one_lasso_met = best_lasso_met[best_lasso_met['Numer iteracji'] == np.min(best_lasso_met['Numer iteracji'])]
print('\n Najlepszy wynik: ')
print(best_one_lasso_met)
print(f"Wybrano {best_one_lasso_met['Liczba wybranych']} cech")


def ridge(max_iter, r, X_train = X_train, X_test = X_test, y_train = y_train, y_test = y_test):
    layoutRidge = Ridge(max_iter, fit_intercept=True).fit(X_train, y_train)
    coefficients = layoutRidge.coef_
    selected_features = np.argsort(coefficients)[-r:]
    df = model(X_train.iloc[:, selected_features],
                X_test.iloc[:, selected_features],
                y_train,
                y_test)
    df['Wybrane indeksy'] = str(selected_features)
    df['Liczba wybranych'] = len(selected_features)
    return df

print(best_one['Liczba wybranych'][0])
df_ridge = ridge(max_iter = int(args.iter), r = best_one['Liczba wybranych'][0])
print('Najlepsze wyniki Ridge: ')
best_ridge = df_ridge[df_ridge['ACCURACY TEST'] == np.max(df_ridge['ACCURACY TEST'])]
print(best_ridge)
# best_one_ridge = best_ridge[best_ridge['Numer iteracji'] == np.min(best_ridge['Numer iteracji'])]
# print('\n Najlepszy wynik Ridge: ')
# print(best_one_ridge)
print(f"Wybrano {best_ridge['Liczba wybranych']} cech")

df_ridge_met = metropolis(k = int(args.k), iter = int(args.iter), tau = float(args.tau), r = best_ridge['Liczba wybranych'][0],
                max_fraction = float(args.max_fraction), seed = int(args.seed))
print('Najlepsze wyniki: ')
best_ridge_met = df_ridge_met[df_ridge_met['ACCURACY TEST'] == np.max(df_ridge_met['ACCURACY TEST'])]
print(best_ridge_met)
best_one_ridge_met = best_ridge_met[best_ridge_met['Numer iteracji'] == np.min(best_ridge_met['Numer iteracji'])]
print('\n Najlepszy wynik: ')
print(best_one_ridge_met)
print(f"Wybrano {best_one_ridge_met['Liczba wybranych']} cech")