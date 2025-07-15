import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder

def get_data(path, x):
    excel_file = pd.ExcelFile(path)

    X = []
    y = []

    if x =='c':
        sheet_names = ['S24','S23', 'S22', 'S21', 'S20', 'S19', 'S18']
    elif x== 'n':
        sheet_names = ['S1','S2', 'S23', 'S4', 'S5', 'S6', 'S7',
                   'S8','S9', 'S10', 'S11', 'S12', 'S13', 'S14',
                   'S15', 'S16', 'S17']
    else:
        sheet_names = ['S1', 'S2', 'S23', 'S4', 'S5', 'S6', 'S7',
        'S8', 'S9', 'S10', 'S11', 'S12', 'S13', 'S14',
        'S15', 'S16', 'S17','S24','S23', 'S22', 'S21', 'S20', 'S19', 'S18']

    for s in excel_file.sheet_names:
        df = pd.read_excel(path, sheet_name=s, header=None)

        for i in range(1, df.shape[1]):
            spectrum = df[i].values
            X.append(spectrum)
            y.append(s)

    # Konwersja do NumPy
    X = np.array(X)
    y = np.array(y)
    print (X.shape)
    # Zamiana etykiet tekstowych na liczby
    encoder = LabelEncoder()
    y = encoder.fit_transform(y)

    # Normalizacja danych
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Dodanie wymiaru kanału dla Conv1D
    X = X[..., np.newaxis]  # Kształt (samples, 601, 1)

    return X, y