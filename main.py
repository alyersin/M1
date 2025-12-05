import time
import numpy as np
from datetime import datetime

# Utils
from utils.date import LOAD_IMGS, MATRICE_ANTRENARE

# Algoritmi
from algoritmi.nn import ALG_NN
from algoritmi.knn import ALG_KNN
from algoritmi.eigenfaces import (
    PREPROCESARE_EIGENFACES_CLASIC,
    PREPROCESARE_EIGENFACES_REPREZENTANTI,
    ALG_EIGENFACES_CLASIC,
    ALG_EIGENFACES_REPREZENTANTI
)

# UI
from ui.interfata import INTERFATA_SELECTARE

# Grafic
from grafic import GENEREAZA_GRAFICE_NORME, AFISEAZA_COMPARATIE_IMAGINI

# Configurare - dicționare pentru algoritmi și norme
norma_dict = {'1': "Manhattan", '2': "Euclidian", '3': "Infinit", '4': "Cosinus"}
algoritm_dict = {
    '1': "NN",
    '2': "KNN",
    '3': "Eigenfaces Clasic",
    '4': "Eigenfaces Reprezentanti"
}

output_file = "rezultate_recunoastere.txt"


def TEST_ALGORITM(A, A_test, etichete_antrenare, etichete_test, tip_algoritm, norm, 
                  k=None, k_eigenfaces=None, preprocesare_data=None, nr_persoane=None):
    num_test_images = A_test.shape[1]
    nr_predictii_corecte = 0
    exec_times = []
    
    for test_idx in range(num_test_images):
        poza_test = A_test[:, test_idx]
        eticheta_reala = etichete_test[test_idx]
        
        start = time.perf_counter()
        
        norma_selectata = norma_dict[norm]
        
        if tip_algoritm == '1':
            # NN
            _, eticheta_gasita = ALG_NN(A, poza_test, norma=norma_selectata, etichete_antrenare=etichete_antrenare)
        elif tip_algoritm == '2':
            # KNN
            if k is None:
                raise ValueError("k trebuie specificat pentru alg KNN")
            _, eticheta_gasita = ALG_KNN(A, poza_test, k=k, norma=norma_selectata, etichete_antrenare=etichete_antrenare)
        elif tip_algoritm == '3':
            # Eigenfaces Clasic
            if preprocesare_data is None:
                raise ValueError("Preprocesare data lipseste pentru Eigenfaces")
            media, HQPB, proiectii = preprocesare_data
            _, eticheta_gasita = ALG_EIGENFACES_CLASIC(
                poza_test, media, HQPB, proiectii, 
                norma=norma_selectata, etichete_antrenare=etichete_antrenare
            )
        elif tip_algoritm == '4':
            # Eigenfaces Reprezentanti
            if preprocesare_data is None or nr_persoane is None:
                raise ValueError("Preprocesare data sau nr_persoane lipseste pentru Eigenfaces Reprezentanti")
            media, HQPB, proiectii_rc = preprocesare_data
            _, eticheta_gasita = ALG_EIGENFACES_REPREZENTANTI(
                poza_test, media, HQPB, proiectii_rc,
                norma=norma_selectata, etichete_antrenare=etichete_antrenare, nr_persoane=nr_persoane
            )
        else:
            raise ValueError(f"Algoritm necunoscut: {tip_algoritm}")
        
        elapsed = time.perf_counter() - start
        exec_times.append(elapsed)
        
        if eticheta_reala == eticheta_gasita:
            nr_predictii_corecte += 1
    
    rata_recunoastere = nr_predictii_corecte / num_test_images * 100
    avg_time = np.mean(exec_times)
    return rata_recunoastere, avg_time


def main():
    # Folosim interfata cu dropdown
    selectie = INTERFATA_SELECTARE(algoritm_dict, norma_dict)
    
    tip_algoritm = selectie.get('algoritm', '')
    norm = selectie.get('norma', '')
    k = selectie.get('k', None)
    k_eigenfaces = selectie.get('k_eigenfaces', None)
    metoda_eigenfaces = selectie.get('metoda_eigenfaces', 'SVD')  # Default SVD
    
    # Validare
    if not tip_algoritm or not norm:
        print("Eroare: Trebuie sa selectati algoritm si norma!")
        return
    
    if tip_algoritm == '2' and (k is None or k <= 1 or k % 2 == 0):
        print("Eroare: Pentru KNN, k trebuie sa fie impar si > 1!")
        return
    
    if tip_algoritm in ['3', '4'] and k_eigenfaces is None:
        print("Eroare: Pentru Eigenfaces, trebuie sa selectati k!")
        return
    
    print(f"Algoritm selectat: {algoritm_dict[tip_algoritm]}")
    print(f"Norma selectata: {norma_dict[norm]}")
    if k:
        print(f"k (KNN): {k}")
    if k_eigenfaces:
        print(f"k (Eigenfaces): {k_eigenfaces}")
    if tip_algoritm in ['3', '4']:
        print(f"Metoda Eigenfaces: {metoda_eigenfaces}")
    print("\nIncarc datele...")
    
    baza_date = LOAD_IMGS('att_faces', nr_persoane=40, poze_per_persoana=10)
    A, etichete_antrenare, A_test, etichete_test = MATRICE_ANTRENARE(baza_date, poze_antrenare=8)
    nr_persoane = baza_date['nr_persoane']
    
    preprocesare_data = None
    timp_preprocesare = None
    
    # Preprocesare pentru Eigenfaces
    if tip_algoritm == '3':
        print(f"Preprocesare Eigenfaces Clasic (metoda: {metoda_eigenfaces})...")
        media, HQPB, proiectii, timp_preprocesare = PREPROCESARE_EIGENFACES_CLASIC(A, k_eigenfaces, metoda=metoda_eigenfaces)
        preprocesare_data = (media, HQPB, proiectii)
        print(f"Timp preprocesare: {timp_preprocesare:.5f} sec")
    elif tip_algoritm == '4':
        print(f"Preprocesare Eigenfaces Reprezentanti (metoda: {metoda_eigenfaces})...")
        media, HQPB, proiectii_rc, timp_preprocesare = PREPROCESARE_EIGENFACES_REPREZENTANTI(
            A, etichete_antrenare, nr_persoane, k_eigenfaces,
            metoda_hqpb='clasic', metoda_reprezentant='media', metoda=metoda_eigenfaces
        )
        preprocesare_data = (media, HQPB, proiectii_rc)
        print(f"Timp preprocesare: {timp_preprocesare:.5f} sec")
    
    print("Testez algoritmul...")
    
    # Testare
    if tip_algoritm in ['1', '2']:
        rata_recunoastere, avg_time = TEST_ALGORITM(
            A, A_test, etichete_antrenare, etichete_test, tip_algoritm, norm, k=k
        )
    elif tip_algoritm == '3':
        rata_recunoastere, avg_time = TEST_ALGORITM(
            A, A_test, etichete_antrenare, etichete_test, tip_algoritm, norm,
            k_eigenfaces=k_eigenfaces, preprocesare_data=preprocesare_data
        )
    elif tip_algoritm == '4':
        rata_recunoastere, avg_time = TEST_ALGORITM(
            A, A_test, etichete_antrenare, etichete_test, tip_algoritm, norm,
            k_eigenfaces=k_eigenfaces, preprocesare_data=preprocesare_data, nr_persoane=nr_persoane
        )
    
    # Salvare rezultate
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(output_file, 'a', encoding='utf-8') as f:
        f.write(f"{timestamp}\n")
        f.write(f"Algoritm: {algoritm_dict[tip_algoritm]}\n")
        f.write(f"Norma: {norma_dict[norm]}\n")
        if tip_algoritm == '2':
            f.write(f"Input k (KNN): {k}\n")
        if tip_algoritm in ['3', '4']:
            f.write(f"k Eigenfaces: {k_eigenfaces}\n")
            if timp_preprocesare:
                f.write(f"Timp preprocesare: {timp_preprocesare:.5f} sec\n")
        f.write(f"Rata de recunoastere: {rata_recunoastere:.2f}%\n")
        f.write(f"Timp mediu interogare: {avg_time:.5f} sec / imagine\n")
        f.write(" \n \n")
    
    print(f"\nRata de recunoastere: {rata_recunoastere:.2f}%")
    print(f"Timp mediu interogare: {avg_time:.5f} sec / imagine")
    if timp_preprocesare:
        print(f"Timp preprocesare: {timp_preprocesare:.5f} sec")
    
    # Afisare comparatie imagini
    AFISEAZA_COMPARATIE_IMAGINI(
        A,
        A_test,
        etichete_antrenare,
        etichete_test,
        tip_algoritm,
        norm,
        norma_dict,
        k=k,
        preprocesare_data=preprocesare_data,
        nr_persoane=nr_persoane if tip_algoritm == '4' else None,
    )
    
    # Grafic comparativ
    if A_test.shape[1] > 0 and A.shape[1] > 0:
        print("\nGenerez grafic comparativ pentru toate normele...\n")
        GENEREAZA_GRAFICE_NORME(
            TEST_ALGORITM,
            A,
            A_test,
            etichete_antrenare,
            etichete_test,
            tip_algoritm,
            norma_dict,
            algoritm_dict,
            k=k,
            k_eigenfaces=k_eigenfaces,
            preprocesare_data=preprocesare_data,
            nr_persoane=nr_persoane if tip_algoritm == '4' else None,
        )


if __name__ == "__main__":
    main()

