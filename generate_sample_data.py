import pandas as pd
import numpy as np

np.random.seed(42)

# 1. Buat tanggal mingguan sepanjang 2024
dates = pd.date_range("2024-01-01", "2024-12-31", freq="W-MON")

regions = ["North", "South", "East", "West"]
channels = ["Online", "Retail", "Wholesale"]
categories = ["Electronics", "Fashion", "Home", "Grocery"]
segments = ["Consumer", "Corporate", "Small Business"]

rows = []

for d in dates:
    # tiap minggu buat 3 transaksi acak
    for _ in range(3):
        region = np.random.choice(regions)
        channel = np.random.choice(channels)
        cat = np.random.choice(categories)
        segment = np.random.choice(segments)

        # pilih produk berdasarkan kategori
        product = {
            "Electronics": np.random.choice(["Phone", "Laptop", "Headphones"]),
            "Fashion": np.random.choice(["T-Shirt", "Jeans", "Sneakers"]),
            "Home": np.random.choice(["Chair", "Table", "Lamp"]),
            "Grocery": np.random.choice(["Coffee", "Cereal", "Snacks"]),
        }[cat]

        base_price = {
            "Phone": 500, "Laptop": 900, "Headphones": 80,
            "T-Shirt": 15, "Jeans": 40, "Sneakers": 70,
            "Chair": 60, "Table": 150, "Lamp": 35,
            "Coffee": 8, "Cereal": 5, "Snacks": 3,
        }[product]

        units = np.random.randint(5, 80)
        discount = float(np.round(np.random.choice([0, 0.05, 0.10, 0.15]), 2))
        revenue = units * base_price * (1 - discount)
        cost = revenue * np.random.uniform(0.6, 0.85)
        profit = revenue - cost
        returned = np.random.choice(["Yes", "No"], p=[0.1, 0.9])

        rows.append([
            d.date(),          # date
            region,            # region
            channel,           # channel
            cat,               # category
            product,           # product
            segment,           # customer_segment
            units,             # units_sold
            base_price,        # unit_price
            discount,          # discount_pct
            round(revenue, 2), # revenue
            round(cost, 2),    # cost
            round(profit, 2),  # profit
            returned           # returned
        ])

# buat DataFrame
df = pd.DataFrame(rows, columns=[
    "date", "region", "channel", "category", "product", "customer_segment",
    "units_sold", "unit_price", "discount_pct",
    "revenue", "cost", "profit", "returned"
])

# Tambahkan sedikit missing value supaya fitur 'Missing Value' kepakai
for col in ["units_sold", "discount_pct", "customer_segment"]:
    idx = np.random.choice(df.index, size=8, replace=False)
    df.loc[idx, col] = np.nan

print("Jumlah baris:", len(df))
print("Jumlah kolom:", len(df.columns))

# Simpan ke Excel dan CSV
df.to_excel("sample_sales_data.xlsx", index=False)
df.to_csv("sample_sales_data.csv", index=False)

print("File berhasil dibuat:")
print(" - sample_sales_data.xlsx")
print(" - sample_sales_data.csv")
