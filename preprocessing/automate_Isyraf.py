# automate_Isyraf.py
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

def preprocess(input_csv="../Customer_Churn_Dataset.csv",
               output_csv="Customer-Churn_processed.csv"):
    """
    Fungsi ini melakukan preprocessing otomatis:
    1. Drop kolom customerID
    2. Tangani TotalCharges bermasalah
    3. Encode target variable Churn
    4. Encode semua kolom kategori
    5. Standardize kolom numerik
    6. Simpan hasil preprocessing ke CSV
    """
    df = pd.read_csv(input_csv)
    df = df.drop(columns=["customerID"])
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df['TotalCharges'].fillna(df['TotalCharges'].median(), inplace=True)
    df['Churn'] = df['Churn'].map({'Yes':1, 'No':0})
    categorical_cols = df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        df[col] = LabelEncoder().fit_transform(df[col])

    numerical_cols = ['tenure', 'MonthlyCharges', 'TotalCharges', 'SeniorCitizen']
    scaler = StandardScaler()
    df[numerical_cols] = scaler.fit_transform(df[numerical_cols])

    df.to_csv(output_csv, index=False)
    print(f"Preprocessing selesai! File tersimpan di {output_csv}")

# Jika file ini dijalankan langsung
if __name__ == "__main__":
    preprocess()
