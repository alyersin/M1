# Interfață grafică pentru selectarea algoritmului și parametrilor
import tkinter as tk
from tkinter import ttk


# Interfata minimalista cu dropdown pentru selectarea algoritmului si parametrilor
# Args: algoritm_dict - Dicționar cu algoritmii disponibili, norma_dict - Dicționar cu normele disponibile
# Returns: Dict cu selecțiile utilizatorului
def INTERFATA_SELECTARE(algoritm_dict, norma_dict):
    root = tk.Tk()
    root.title("Selectare Algoritm")
    root.geometry("400x400")
    
    rezultat = {}
    
    # Algoritm
    tk.Label(root, text="Algoritm:", font=("Arial", 10)).pack(pady=5)
    algoritm_var = tk.StringVar()
    algoritm_combo = ttk.Combobox(root, textvariable=algoritm_var, state="readonly", width=30)
    algoritm_combo['values'] = [f"{k} - {v}" for k, v in algoritm_dict.items()]
    algoritm_combo.pack(pady=5)
    
    # Norma
    tk.Label(root, text="Norma:", font=("Arial", 10)).pack(pady=5)
    norma_var = tk.StringVar()
    norma_combo = ttk.Combobox(root, textvariable=norma_var, state="readonly", width=30)
    norma_combo['values'] = [f"{k} - {v}" for k, v in norma_dict.items()]
    norma_combo.pack(pady=5)
    
    # k pentru KNN
    tk.Label(root, text="k (pentru KNN, impar si > 1):", font=("Arial", 10)).pack(pady=5)
    k_var = tk.StringVar()
    k_entry = tk.Entry(root, textvariable=k_var, width=30)
    k_entry.pack(pady=5)
    
    # k pentru Eigenfaces
    tk.Label(root, text="k Eigenfaces (20, 40, 60, 80, 100):", font=("Arial", 10)).pack(pady=5)
    k_eigenfaces_var = tk.StringVar()
    k_eigenfaces_combo = ttk.Combobox(root, textvariable=k_eigenfaces_var, state="readonly", width=30)
    k_eigenfaces_combo['values'] = ['20', '40', '60', '80', '100']
    k_eigenfaces_combo.pack(pady=5)
    
    def on_submit():
        # Extragem valorile
        alg_str = algoritm_var.get()
        if alg_str:
            rezultat['algoritm'] = alg_str.split(' - ')[0]
        
        norm_str = norma_var.get()
        if norm_str:
            rezultat['norma'] = norm_str.split(' - ')[0]
        
        k_val = k_var.get()
        if k_val:
            try:
                rezultat['k'] = int(k_val)
            except ValueError:
                rezultat['k'] = None
        else:
            rezultat['k'] = None
        
        k_eig_val = k_eigenfaces_var.get()
        if k_eig_val:
            rezultat['k_eigenfaces'] = int(k_eig_val)
        else:
            rezultat['k_eigenfaces'] = None
        
        root.quit()
        root.destroy()
    
    tk.Button(root, text="Start", command=on_submit, width=25, height=3, font=("Arial", 12)).pack(pady=30)
    
    root.mainloop()
    
    return rezultat

