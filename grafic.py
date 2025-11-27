import matplotlib.pyplot as plt

from functii import ALG_NN, ALG_KNN


def AFISEAZA_COMPARATIE_IMAGINI(
    A,
    A_test,
    etichete_antrenare,
    etichete_test,
    tip_algoritm,
    norm_cheie,
    norm_mapare,
    k=None,
    persoana=29,
    numar_poza_test=1,
    dimensiune_imagine=(112, 92),
):

    if A_test.shape[1] == 0 or A.shape[1] == 0:
        return

    norma_selectata = norm_mapare.get(norm_cheie)
    if norma_selectata is None:
        return

    index_test_selectat = (persoana - 1) * 2 + (numar_poza_test - 1)
    total_imagini_test = A_test.shape[1]

    if index_test_selectat >= total_imagini_test:
        index_test_selectat = total_imagini_test - 1

    if index_test_selectat < 0:
        index_test_selectat = 0

    poza_test_curenta = A_test[:, index_test_selectat]

    if tip_algoritm == '1':
        pozitie_identificata, eticheta_identificata = ALG_NN(
            A,
            poza_test_curenta,
            norma=norma_selectata,
            etichete_antrenare=etichete_antrenare,
        )
    else:
        pozitie_identificata, eticheta_identificata = ALG_KNN(
            A,
            poza_test_curenta,
            k=k,
            norma=norma_selectata,
            etichete_antrenare=etichete_antrenare,
        )

    plt.figure(figsize=(8, 4))
    plt.subplot(1, 2, 1)
    plt.title(f"Test s{etichete_test[index_test_selectat] + 1}")
    plt.imshow(poza_test_curenta.reshape(dimensiune_imagine), cmap='gray')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.title(f"Identificat s{(eticheta_identificata or 0) + 1}")
    poza_referinta = A[:, pozitie_identificata]
    plt.imshow(poza_referinta.reshape(dimensiune_imagine), cmap='gray')
    plt.axis('off')

    plt.tight_layout()
    plt.show()


def GENEREAZA_GRAFICE_NORME(
    functie_test,
    A,
    A_test,
    etichete_antrenare,
    etichete_test,
    tip_algoritm,
    dictionar_norme,
    dictionar_algoritmi,
    k=None,
):

    if A_test.shape[1] == 0 or A.shape[1] == 0:
        return

    etichete_norme_plot = []
    valori_rata_recunoastere = []
    valori_timp_executie = []

    for norm_cheie, nume_norma in dictionar_norme.items():
        acuratete, timp_mediu = functie_test(
            A,
            A_test,
            etichete_antrenare,
            etichete_test,
            tip_algoritm,
            norm_cheie,
            k,
        )
        etichete_norme_plot.append(nume_norma)
        valori_rata_recunoastere.append(acuratete)
        valori_timp_executie.append(timp_mediu)

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    bare_rata_recunoastere = plt.bar(etichete_norme_plot, valori_rata_recunoastere, color='skyblue')
    plt.title(f"Rata de recunoastere ({dictionar_algoritmi[tip_algoritm]}) pentru fiecare norma")
    plt.ylabel("Rata de recunoastere (%)")
    plt.ylim(0, 100)
    plt.grid(axis='y', linestyle='--', alpha=0.6)

    for bara, valoare in zip(bare_rata_recunoastere, valori_rata_recunoastere):
        plt.text(
            bara.get_x() + bara.get_width() / 2,
            bara.get_height() + 1.5,
            f"{valoare:.2f}%",
            ha='center',
            va='bottom',
            fontsize=10,
            fontweight='bold',
        )

    plt.subplot(1, 2, 2)
    bare_timp_executie = plt.bar(etichete_norme_plot, valori_timp_executie, color='orange')
    plt.title(f"Timp mediu de executie ({dictionar_algoritmi[tip_algoritm]})")
    plt.ylabel("Timp (secunde / imagine)")
    plt.grid(axis='y', linestyle='--', alpha=0.6)

    if valori_timp_executie:
        timp_maxim_executie = max(valori_timp_executie)
        decalaj_text_timp = timp_maxim_executie * 0.03 if timp_maxim_executie > 0 else 0.01
        for bara, valoare in zip(bare_timp_executie, valori_timp_executie):
            plt.text(
                bara.get_x() + bara.get_width() / 2,
                bara.get_height() + decalaj_text_timp,
                f"{valoare:.4f}s",
                ha='center',
                va='bottom',
                fontsize=10,
                fontweight='bold',
            )

    plt.tight_layout()
    plt.show()

