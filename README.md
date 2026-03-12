# Loan Training Dataset

To repozytorium zawiera **zbiór danych kredytowych przygotowany w celach edukacyjnych i do trenowania modeli uczenia maszynowego**.  

Dane przedstawiają 2000 rekordów klientów banku z podstawowymi cechami finansowymi oraz informacją, czy kredyt został przyznany. **Nie są to dane prawdziwych klientów – zostały wygenerowane sztucznie w celach treningowych.**

---

## 📊 Kolumny w zbiorze danych

| Kolumna             | Typ       | Opis                                                                 |
|--------------------|-----------|----------------------------------------------------------------------|
| `age`               | int       | Wiek klienta (lata)                                                 |
| `income`            | float     | Roczny dochód klienta w USD                                          |
| `credit_score`      | float     | Scoring kredytowy (300–850)                                         |
| `debt_ratio`        | float     | Debt-to-Income Ratio – procent dochodu przeznaczony na spłatę długów |
| `employment_years`  | int       | Lata zatrudnienia                                                    |
| `loan_approved`     | int       | Target – 1 = kredyt przyznany, 0 = kredyt odrzucony                 |

---

## 🔍 Przykładowe wiersze

| age | income   | credit_score | debt_ratio | employment_years | loan_approved |
|-----|---------|--------------|------------|-----------------|---------------|
| 56  | 37000.5 | 783          | 0.422      | 3               | 0             |
| 69  | 68744.5 | 665          | 0.094      | 7               | 1             |
| 46  | 46767.5 | 747          | 0.126      | 23              | 1             |

---

## 💻 Jak używać danych w Pythonie

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Ładowanie danych
url = "https://raw.githubusercontent.com/kszontek/test/main/loan_training_data.parquet"
df = pd.read_parquet(url)

# Podział na cechy i target
X = df.drop("loan_approved", axis=1)
y = df["loan_approved"]

# Standaryzacja cech
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Podział na trening i test
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)
