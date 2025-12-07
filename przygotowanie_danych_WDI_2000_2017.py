import pandas as pd
import numpy as np
import matplotlib.pyplot as plt



# Wczytanie danych
df = pd.read_csv("./dane/WDIData.csv")
print(df.head())
print(df.columns[:10])   # zobaczymy, ktore kolumny maja lata
df.info()


# Ograniczenie danych do lat 2000-2017
columns_to_keep = ['Country Code', 'Indicator Code'] + [str(year) for year in range(2000, 2018)]
df_filtered = df[columns_to_keep]
print(df_filtered)

 

# Przeksztalcenie danych z szerokiego formatu na dlugi
df_melted = df_filtered.melt(
    id_vars=['Country Code', 'Indicator Code'],
    var_name='Rok',
    value_name='Wartosc'
)
print(df_melted)


# Przeksztalcenie na szeroki format z kolumnami dla wskaznikow (Indicator Code)
df_pivot = df_melted.pivot_table(
    index=['Country Code', 'Rok'],
    columns='Indicator Code',
    values='Wartosc',
    aggfunc='first'
).reset_index()

df_pivot.columns.name = None
print(df_pivot)


# Usuwamy wskazniki, ktore maja zbyt duzo brakow
print("Przed usunieciem rzadkich cech:", df_pivot.shape)
mask = df_pivot.isna().mean() < 0.4
X = df_pivot.loc[:, mask]
print("Po usunieciu rzadkich cech:", X.shape)
# Przed usunieciem rzadkich cech: (4734, 1595)
# Po usunieciu rzadkich cech: (4734, 644)


corr_cols = [x for x in X.columns if x not in ['Country Code', 'Rok']]
korelacje = X[corr_cols].corr()[['NY.GDP.PCAP.CD']].sort_values(by='NY.GDP.PCAP.CD', ascending=False)
print(korelacje)


korelacje = X[corr_cols].corr()[['NY.GDP.PCAP.CD']]
kolumny_do_usuniecia = korelacje[(korelacje['NY.GDP.PCAP.CD'] > 0.7) | (korelacje['NY.GDP.PCAP.CD'] < -0.7)].index
kolumny_do_usuniecia = [col for col in kolumny_do_usuniecia if col != 'NY.GDP.PCAP.CD']
X = X.drop(columns=kolumny_do_usuniecia)


corr_cols = [x for x in X.columns if x not in ['Country Code', 'Rok']]
X[corr_cols].corr()[['NY.GDP.PCAP.CD']].sort_values(by='NY.GDP.PCAP.CD', ascending=False)
korelacje = X[corr_cols].corr()
# Ustaw prog korelacji
prog = 0.7
# Znajdz pary zmiennych o korelacji powyzej progu
kolumny_do_usuniecia = set()
for kolumna1 in korelacje.columns:
    for kolumna2 in korelacje.columns:
        if kolumna1 != kolumna2 and np.abs(korelacje.loc[kolumna1, kolumna2]) > prog:
            kolumny_do_usuniecia.add(kolumna2)
# Usun kolumny z DataFrame
X = X.drop(columns=kolumny_do_usuniecia)


# Tworzymy etykiete do klasyfikacji
# Kod wskaznika GDP per capita: "NY.GDP.PCAP.CD"
if "NY.GDP.PCAP.CD" in X.columns:
    y = (X["NY.GDP.PCAP.CD"] > X["NY.GDP.PCAP.CD"].median()).astype(int)
    X = X.drop(columns=["NY.GDP.PCAP.CD"])
else:
    raise ValueError("Twoj WDI nie zawiera wskaznika GDP per capita (NY.GDP.PCAP.CD).")
print("Rozklad klas:", y.value_counts())


# Uzupelnienie brakow mediana
X_filled = X.copy()
cols_to_fill = [x for x in X.columns if x not in ['Country Code', 'Rok']]
X_filled = X_filled[cols_to_fill].apply(lambda col: col.fillna(col.median()), axis=0)
 

# wybieramy 100 cech o najwiekszej wariancji
print(f"Before truncating:", X.shape)
variances = X_filled.var().sort_values(ascending=False)
selected_features = variances.head(100).index
X_filled = X_filled[selected_features]
X = X[selected_features]
print("Final X with filled nulls:", X_filled.shape)
print("Final X with nulls:", X.shape)
# Before truncating: (4734, 103)
# Final X with filled nulls: (4734, 100)
# Final X with nulls: (4734, 100)


X.to_parquet("./dane/X_data_with_nulls.parquet")
X_filled.to_parquet('./dane/X_data_without_nulls.parquet')
pd.DataFrame(y).to_parquet('./dane/y_data.parquet')


indicators_dict_full = df.groupby(['Indicator Code', 'Indicator Name']).count()['Country Name'].reset_index()[['Indicator Code', 'Indicator Name']]
indicators_dict = df[df['Indicator Code'].isin(X.columns)].groupby(['Indicator Code', 'Indicator Name']).count()['Country Name'].reset_index()[['Indicator Code', 'Indicator Name']]
countries_dict = df.groupby(['Country Code', 'Country Name']).count()['Indicator Name'].reset_index()[['Country Code', 'Country Name']]

indicators_dict_full.to_parquet('./dane/indicators_dict_full.parquet')
indicators_dict.to_parquet('./dane/indicators_dict.parquet')
countries_dict.to_parquet('./dane/countries_dict.parquet')