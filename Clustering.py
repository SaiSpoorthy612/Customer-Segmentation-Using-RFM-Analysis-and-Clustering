import tkinter as tk
from tkinter import messagebox
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

def cluster_and_rfm():
    # Get input values from the user
    avg_credit_limit = float(avg_credit_limit_entry.get())
    total_credit_cards = int(total_credit_cards_entry.get())
    total_visits_bank = int(total_visits_bank_entry.get())
    Limit_used = int(Limit_used_entry.get())

    # Load the dataset
    data = pd.read_csv("C:/Users/spoor/Downloads/Dataset/creditcard.csv")

    # Perform K-means clustering with the input values
    scaler = StandardScaler()
    input_data = scaler.fit_transform([[avg_credit_limit, total_credit_cards, total_visits_bank, Limit_used]])
    kmeans = KMeans(n_clusters=3, random_state=42)
    kmeans.fit(data[['Avg_Credit_Limit', 'Total_Credit_Cards', 'Total_visits_bank', 'Limit_used']])
    cluster_label = kmeans.predict(input_data)[0]

    # Perform RFM analysis
    rfm_range = "High Value Customer"
    credit_limit_usage = Limit_used / avg_credit_limit * 100
    if total_credit_cards > 4 or credit_limit_usage > 70:
        rfm_range = "Low Value Customer"
    elif total_credit_cards > 2 or credit_limit_usage > 50:
        rfm_range = "Medium Value Customer"

    # Display the result
    messagebox.showinfo("Result", f"The customer belongs to Cluster {cluster_label} and RFM Range: {rfm_range}")

# Create the main window
root = tk.Tk()
root.title("Customer Clustering and RFM Analysis")

# Create input fields
tk.Label(root, text="Avg_Credit_Limit").grid(row=0, column=0)
avg_credit_limit_entry = tk.Entry(root)
avg_credit_limit_entry.grid(row=0, column=1)

tk.Label(root, text="Total_Credit_Cards").grid(row=1, column=0)
total_credit_cards_entry = tk.Entry(root)
total_credit_cards_entry.grid(row=1, column=1)

tk.Label(root, text="Total_visits_bank").grid(row=2, column=0)
total_visits_bank_entry = tk.Entry(root)
total_visits_bank_entry.grid(row=2, column=1)

tk.Label(root, text="Limit_used").grid(row=3, column=0)
Limit_used_entry = tk.Entry(root)
Limit_used_entry.grid(row=3, column=1)

# Create a button to perform clustering and RFM analysis
tk.Button(root, text="Perform Clustering and RFM Analysis", command=cluster_and_rfm).grid(row=4, columnspan=2)

root.mainloop()
