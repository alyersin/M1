import time
import numpy as np
from datetime import datetime
from functii import LOAD_IMGS, MATRICE_ANTRENARE, ALG_NN, ALG_KNN
from grafic import GENEREAZA_GRAFICE_NORME, AFISEAZA_COMPARATIE_IMAGINI

norma_dict = {'1': "Manhattan", '2': "Euclidian", '3': "Infinit", '4': "Cosinus"}
algoritm_dict = {'1': "NN", '2': "KNN"}

norm_mapping = {
    '1': 'Manhattan',
    '2': 'Euclidian',
    '3': 'Infinit',
    '4': 'Cosinus'
}

output_file = "rezultate_recunoastere.txt"


def TEST_ALGORITM(A, A_test, etichete_antrenare, etichete_test, tip_algoritm, norm, k=None):
    num_test_images = A_test.shape[1]
    nr_predictii_corecte = 0
    exec_times = []
    
    for test_idx in range(num_test_images):
        poza_test = A_test[:, test_idx]
        eticheta_reala = etichete_test[test_idx]
        
        start = time.perf_counter()
        if tip_algoritm == '1':
            _, eticheta_gasita = ALG_NN(A, poza_test, norma=norm_mapping[norm], etichete_antrenare=etichete_antrenare)
        else:
            if k is None:
                raise ValueError("k trebuie specificat pentru alg KNN")
            _, eticheta_gasita = ALG_KNN(A, poza_test, k=k, norma=norm_mapping[norm], etichete_antrenare=etichete_antrenare)
        elapsed = time.perf_counter() - start
        exec_times.append(elapsed)
        
        if eticheta_reala == eticheta_gasita:
            nr_predictii_corecte += 1
    
    rata_recunoastere = nr_predictii_corecte / num_test_images * 100
    avg_time = np.mean(exec_times)
    return rata_recunoastere, avg_time


def main():
    norm = ''
    tip_algoritm = ''
    k = None
    
    while norm not in norma_dict.keys():
        norm = input("Norma: (1=Manhattan, 2=Euclidian, 3=Infinit, 4=Cosinus): ")
    
    while tip_algoritm not in algoritm_dict.keys():
        tip_algoritm = input("Algoritmul: (1=NN, 2=KNN): ")
    
    if tip_algoritm == '2':
        while k is None or k <= 1 or k % 2 == 0:
            try:
                k = int(input("Input k (impar si > 1, ): "))
            except ValueError:
                print("Introdu un numar intreg valid pentru k.")
    
    baza_date = LOAD_IMGS('att_faces', nr_persoane=40, poze_per_persoana=10)
    A, etichete_antrenare, A_test, etichete_test = MATRICE_ANTRENARE(baza_date, poze_antrenare=8)
    
    if tip_algoritm == '1':
        rata_recunoastere, avg_time = TEST_ALGORITM(A, A_test, etichete_antrenare, etichete_test, tip_algoritm, norm)
    else:
        rata_recunoastere, avg_time = TEST_ALGORITM(A, A_test, etichete_antrenare, etichete_test, tip_algoritm, norm, k)
    
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(output_file, 'a', encoding='utf-8') as f:
        f.write(f"{timestamp}\n")
        f.write(f"Algoritm: {algoritm_dict[tip_algoritm]}\n")
        f.write(f"Norma: {norma_dict[norm]}\n")
        if tip_algoritm == '2':
            f.write(f"Input k: {k}\n")
        f.write(f"Rata de recunoastere: {rata_recunoastere:.2f}%\n")
        f.write(f"Timp mediu interogare: {avg_time:.5f} sec / imagine\n")
        f.write(" \n \n")
    
    print(f"Rata de recunoastere: {rata_recunoastere:.2f}%")
    print(f"Timp mediu interogare: {avg_time:.5f} sec / imagine")

    AFISEAZA_COMPARATIE_IMAGINI(
        A,
        A_test,
        etichete_antrenare,
        etichete_test,
        tip_algoritm,
        norm,
        norm_mapping,
        k=k,
    )

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
        )


if __name__ == "__main__":
    main()

