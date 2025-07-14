import pandas as pd
import os
import scipy
from scipy.stats import wilcoxon
from scipy.stats import friedmanchisquare

def statistic(folder_path, grupped):
            df = pd.read_excel(filename)

            df["Czas treningu pojedynczej epoki (s)"] = df["Czas treningu (s)"]/50

            # Kolumny, dla których obliczamy statystyki
            kolumny_docelowe = ["Dokładność walidacji", "Dokładność treningowa","Overfitting",	"Czas treningu pojedynczej epoki (s)",
            "Czas treningu (s)", "Liczba epok", ]

            if grupped==1:# Grupowanie po 'Stride (druga warstwa)' i liczenie średnich i odchyleń std
                wyniki = df.groupby(['Pool size', 'Stride'])[kolumny_docelowe].agg(['mean', 'std'])
            else:
                wyniki = df.groupby(df.columns[0])[kolumny_docelowe].agg(['mean', 'std'])


            # Spłaszczenie MultiIndex w nazwach kolumn
            wyniki.columns = [' '.join(col).strip() for col in wyniki.columns.values]
            wyniki.reset_index(inplace=True)

            # Zapis do nowego arkusza w tym samym pliku
            with pd.ExcelWriter(filename, engine='openpyxl', mode='a', if_sheet_exists='replace') as writer:
                wyniki.to_excel(writer, sheet_name='Statystyki', index=False)

            print("✅ Zapisano statystyki do arkusza 'Statystyki'")


def wilcoxon_test(file_path):
    df = pd.read_excel(file_path, sheet_name="Wyniki")
    df["OverfitAbs"] = df["Overfitting"].abs()

    metrics = [
        "Dokładność walidacji",
        "Dokładność treningowa",
        "Czas treningu pojedynczej epoki (s)",
        "Liczba epok",
        "OverfitAbs"
    ]

    group_col = df.columns[0]
    groups = df[group_col].unique()

    if len(groups) != 2:
        print("❌ Test Wilcoxona wymaga dokładnie dwóch grup.")
        return

    results = []

    for metric in metrics:
        pivot = df.pivot(index="Fold", columns=group_col, values=metric)

        if pivot.isnull().values.any():
            print(f"⚠️ Brak danych dla metryki: {metric}")
            continue

        try:
            stat, p_value = wilcoxon(pivot[groups[0]], pivot[groups[1]])
        except ValueError as e:
            print(f"❌ Błąd w metryce {metric}: {e}")
            continue

        interpretation = (
            "✅ Istnieją istotne różnice między grupami."
            if p_value < 0.05 else
            "ℹ️ Brak statystycznie istotnych różnic między grupami."
        )

        results.append({
            "Metryka": metric,
            "Grupy porównywane": f"{groups[0]} vs {groups[1]}",
            "Statystyka Wilcoxona": round(stat, 4),
            "Wartość p": round(p_value, 4),
            "Interpretacja": interpretation
        })

    if results:
        results_df = pd.DataFrame(results)
        with pd.ExcelWriter(file_path, engine="openpyxl", mode="a", if_sheet_exists="replace") as writer:
            results_df.to_excel(writer, sheet_name="Wilcoxon", index=False)

columns= ['Dokładność walidacji', 'Dokładność treningowa'	,'Czas treningu (s)'	,'Liczba epok'	,'Overfitting']
folder_path = "."

def friedman_test(file_path, fixed_col=None):
    df = pd.read_excel(file_path, sheet_name="Wyniki")
    results = []

    df["OverfitAbs"] = df["Overfitting"].abs()

    metrics = [
        "Dokładność walidacji",
        "Dokładność treningowa",
        "Czas treningu pojedynczej epoki (s)",
        "Liczba epok",
        "OverfitAbs"
    ]
    if fixed_col is not None:
        df["Group"] = df["Pool size"].astype(str) + "_" + df["Stride"].astype(str)
        group_col = "Group"
    else:
        group_col = df.columns[0]

    for metric in metrics:
            pivot = df.pivot(index="Fold", columns=group_col, values=metric)

            if pivot.shape[1] < 3:
                print(f"⚠️ Za mało grup do testu Friedmana dla metryki: {metric}")
                continue

            if pivot.isnull().values.any():
                print(f"⚠️ Braki danych dla metryki: {metric}")
                continue

            stat, p_value = friedmanchisquare(*[pivot[col] for col in pivot.columns])
            interpretation = (
                "✅ Różnice między grupami są statystycznie istotne."
                if p_value < 0.05 else
                "ℹ️ Brak statystycznie istotnych różnic między grupami.")

            results.append({
                "Metryka": metric,
                "Porównywana zmienna": group_col,
                "Typ testu": "pojedynczy",
                "Statystyka Friedmana": round(stat, 4),
                "Wartość p": round(p_value, 4),
                "Interpretacja": interpretation
            })

    combined = pd.DataFrame(results)

    with pd.ExcelWriter(file_path, engine="openpyxl", mode="a", if_sheet_exists="replace") as writer:
        combined.to_excel(writer, sheet_name="Test Friedmana", index=False)

for filename in os.listdir(folder_path):
   if "wyniki" in filename.lower() and filename.endswith(".xlsx"):
        filepath = os.path.join(folder_path, filename)
        print(f"\n📄 Przetwarzanie: {filename}")

        wilcoxon_test(filename)


