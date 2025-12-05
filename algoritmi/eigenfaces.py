# Algoritmul Eigenfaces - recunoaștere facială bazată pe PCA/SVD/Lanczos
import numpy as np
import time
import random
from numpy.linalg import svd, norm
from utils.distante import CALC_DISTANTA_NORMA


# Preprocesare pentru algoritmul Eigenfaces clasic folosind algoritmul Lanczos
# Args: A - Matricea de antrenare (nrPixeli x nrPoze), k - Nivel de trunchiere (20, 40, 60, 80, 100)
# Returns: media, HQPB, proiectii, timp_preprocesare
def PREPROCESARE_EIGENFACES_LANCZOS(A, k):
    t0 = time.perf_counter()
    
    # Pas 1: Calculam poza medie
    media = np.mean(A, axis=1, keepdims=True)  # (nrPixeli x 1)
    
    # Pas 2: Centram toate pozele de antrenare
    A_centrat = A - media  # (nrPixeli x nrPoze)
    
    m, n = A_centrat.shape  # m = nrPixeli, n = nrPoze
    
    # Initializare pentru algoritmul Lanczos
    q = np.zeros((m, k + 2))  # Matricea de vectori ortogonali
    
    beta = 0
    q[:, 0] = np.zeros(m)  # q_0 = 0
    q[:, 1] = np.ones(m)  # q_1 = vector de 1
    q[:, 1] = q[:, 1] / norm(q[:, 1])  # Normalizare
    
    # Algoritmul Lanczos
    for i in range(1, k + 1):
        # omega_i = A * (A^T * q_i) - beta_i * q_{i-1}
        v = A_centrat.T @ q[:, i]  # A^T * q_i
        term1 = A_centrat @ v  # A * (A^T * q_i)
        
        omega = term1 - beta * q[:, i-1]
        
        # alpha_i = <omega_i, q_i>
        alpha = omega.T @ q[:, i]
        
        # omega_i = omega_i - alpha_i * q_i (Gram-Schmidt)
        omega = omega - alpha * q[:, i]
        
        # beta_{i+1} = ||omega_i||_2
        beta = norm(omega)
        
        # q_{i+1} = omega_i / beta_{i+1}
        if beta != 0:
            q[:, i+1] = omega / beta
        else:
            break
    
    # Eigenfaces = q[:, 2:] (sărim peste q_0 și q_1)
    HQPB = q[:, 2:]
    
    # Trunchiere la k (dacă e necesar)
    if HQPB.shape[1] > k:
        HQPB = HQPB[:, :k]
    
    # Pas 3: Proiectam pozele pe HQPB
    # Proiecția: HQPB.T @ A_centrat = (k x m) @ (m x n) = (k x n)
    # Dar pentru consistență cu restul codului, returnăm (n x k)
    proiectii = (HQPB.T @ A_centrat).T  # (n x k)
    
    # Convertim media la vector (nu keepdims)
    media = media.flatten()
    
    t1 = time.perf_counter()
    timp_preprocesare = t1 - t0
    
    return media, HQPB, proiectii, timp_preprocesare


# Preprocesare pentru algoritmul Eigenfaces clasic
# Varianta SVD (truncated SVD) sau PCA (optimizata/neoptimizata) sau Lanczos
# Args: A - Matricea de antrenare (nrPixeli x nrPoze), k - Nivel de trunchiere (20, 40, 60, 80, 100)
#       metoda - 'SVD', 'PCA' (neoptimizata cu C), 'PCA_optimizata' (optimizata cu L), 'Lanczos'
# Returns: media, HQPB, proiectii, timp_preprocesare
def PREPROCESARE_EIGENFACES_CLASIC(A, k, metoda='SVD'):
    t0 = time.perf_counter()
    
    # Pas 1: Calculam poza medie
    media = np.mean(A, axis=1)  # media pe axa orizontala (pe coloane)
    
    # Pas 2: Centram toate pozele de antrenare
    B = A.copy()  # backup
    A_centrat = A - media.reshape(-1, 1)  # broadcasting
    
    if metoda == 'Lanczos':
        # Varianta Lanczos: folosim algoritmul Lanczos
        return PREPROCESARE_EIGENFACES_LANCZOS(A, k)
    elif metoda == 'SVD':
        # Varianta SVD: aplicam SVD pe A_centrat.T
        # A_centrat.T are dimensiunea (nrPoze x nrPixeli)
        U, s, Vt = svd(A_centrat.T, full_matrices=False)
        
        # Trunchierea la k
        U = U[:, :k]
        s = s[:k]
        Vt = Vt[:k, :]
        
        # HQPB = V.T (trunchiat) = Vt.T
        HQPB = Vt.T
        
        # Proiectiile: U (trunchiat) x S (diagonala din s)
        S = np.diag(s)
        proiectii = U @ S
    elif metoda == 'PCA':
        # Varianta PCA neoptimizata
        # Pas 3: Calculam matricea de covarianta C = A * A^T
        # C are dimensiunea (nrPixeli x nrPixeli) = (10304 x 10304) - FOARTE MARE!
        C = A_centrat @ A_centrat.T
        
        # Pas 4: Calculam vectorii proprii ai matricei C
        d, v = np.linalg.eig(C)
        
        # Convertim la real (in cazul in care sunt complexe din cauza erorilor numerice)
        d = np.real(d)
        v = np.real(v)
        
        # Sortam dupa marimea valorilor proprii (descrescator)
        indici_sortati = np.argsort(d)[::-1]
        indici_k = indici_sortati[:k]
        
        # HQPB = cei k vectori proprii
        HQPB = v[:, indici_k]
        
        # Pas 5: Proiectam pozele pe HQPB
        proiectii = A_centrat.T @ HQPB
    elif metoda == 'PCA_optimizata':
        # Varianta PCA optimizata
        # Pas 3: Calculam matricea L = A^T * A (in loc de C = A * A^T)
        # L are dimensiunea (nrPoze x nrPoze) = (320 x 320) - MULT MAI MICA decat C!
        L = A_centrat.T @ A_centrat
        
        # Pas 4: Calculam vectorii proprii ai matricei L
        d, v = np.linalg.eig(L)
        
        # Convertim la real
        d = np.real(d)
        v = np.real(v)
        
        # Sortam dupa marimea valorilor proprii (descrescator)
        indici_sortati = np.argsort(d)[::-1]
        indici_k = indici_sortati[:k]
        
        # Vectorii proprii ai L sunt pe coloane in v
        # Trebuie sa ii inmultim la stanga cu A pentru a obtine vectorii proprii ai C
        # HQPB = A * v (vectorii proprii ai L, trunchiati la k)
        v_k = v[:, indici_k]  # vectorii proprii ai L (320 x k)
        HQPB = A_centrat @ v_k  # vectorii proprii ai C (10304 x k)
        
        # Normalizam vectorii proprii
        for i in range(k):
            HQPB[:, i] = HQPB[:, i] / (np.linalg.norm(HQPB[:, i]) + 1e-10)
        
        # Pas 5: Proiectam pozele pe HQPB
        proiectii = A_centrat.T @ HQPB
    
    A = B  # restore
    
    t1 = time.perf_counter()
    timp_preprocesare = t1 - t0
    
    return media, HQPB, proiectii, timp_preprocesare


# Creeaza matricea reprezentantilor de clasa RC
# Args: A - Matricea de antrenare, etichete_antrenare - Etichetele pozelor, nr_persoane - Numarul de persoane, metoda - 'media' sau 'aleatorie'
# Returns: RC - Matricea reprezentantilor (nrPixeli x nrPersoane)
def MATRICE_REPREZENTANTI(A, etichete_antrenare, nr_persoane, metoda='media'):
    nr_pixeli = A.shape[0]
    RC = np.zeros((nr_pixeli, nr_persoane))
    
    for persoana in range(nr_persoane):
        # Gasim indicii pozelor persoanei i
        indici_persoana = [i for i, et in enumerate(etichete_antrenare) if et == persoana]
        
        if len(indici_persoana) == 0:
            continue
        
        if metoda == 'media':
            # Varianta a) Media pozelor
            poze_persoana = A[:, indici_persoana]
            RC[:, persoana] = np.mean(poze_persoana, axis=1)
        else:
            # Varianta b) Poza aleatorie
            idx_aleator = random.choice(indici_persoana)
            RC[:, persoana] = A[:, idx_aleator]
    
    return RC


# Preprocesare pentru algoritmul Eigenfaces cu reprezentanti de clasa
# Args: A - Matricea de antrenare, etichete_antrenare - Etichetele, nr_persoane - Numarul de persoane, k - Nivel de trunchiere
#       metoda_hqpb - 'clasic' (HQPB din A) sau 'direct' (HQPB din RC), metoda_reprezentant - 'media' sau 'aleatorie'
#       metoda - 'SVD', 'PCA', 'PCA_optimizata', 'Lanczos' - metoda pentru calculul HQPB
# Returns: media, HQPB, proiectii_rc, timp_preprocesare
def PREPROCESARE_EIGENFACES_REPREZENTANTI(A, etichete_antrenare, nr_persoane, k, 
                                         metoda_hqpb='clasic', metoda_reprezentant='media', metoda='SVD'):
    t0 = time.perf_counter()
    
    # Creem matricea reprezentantilor
    RC = MATRICE_REPREZENTANTI(A, etichete_antrenare, nr_persoane, metoda_reprezentant)
    
    if metoda_hqpb == 'clasic':
        # Varianta a): HQPB exact ca la Eigenfaces clasic, dar proiectam doar RC
        media, HQPB, _, _ = PREPROCESARE_EIGENFACES_CLASIC(A, k, metoda=metoda)
        
        # Centram reprezentantii
        RC_centrat = RC - media.reshape(-1, 1)
        
        # Proiectam reprezentantii pe HQPB
        proiectii_rc = RC_centrat.T @ HQPB
    else:
        # Varianta b): Folosim RC in loc de A in preprocesare
        media, HQPB, _, _ = PREPROCESARE_EIGENFACES_CLASIC(RC, k, metoda=metoda)
        
        # Centram reprezentantii
        RC_centrat = RC - media.reshape(-1, 1)
        
        # Proiectam reprezentantii pe HQPB
        proiectii_rc = RC_centrat.T @ HQPB
    
    t1 = time.perf_counter()
    timp_preprocesare = t1 - t0
    
    return media, HQPB, proiectii_rc, timp_preprocesare


# Interogare pentru algoritmul Eigenfaces clasic
# Args: poza_test - Poza de test (vectorizata), media - Poza medie, HQPB - High-Quality Pseudo-Basis
#       proiectii - Proiectiile pozelor de antrenare (nrPoze x k), norma - Norma de distanta, etichete_antrenare - Etichetele
# Returns: pozitia, eticheta_gasita
def ALG_EIGENFACES_CLASIC(poza_test, media, HQPB, proiectii, norma, etichete_antrenare):
    # Pas 1: Centram poza de test
    poza_test_centrat = poza_test - media
    
    # Pas 2: Proiectam poza de test pe HQPB
    pr_test = poza_test_centrat @ HQPB  # vector de k elemente
    
    # Pas 3: Aplicam NN pe proiectii
    # proiectii este (nrPoze x k), pr_test este (k,)
    # Trebuie sa comparam pr_test cu fiecare linie din proiectii
    nr_poze_antrenare = proiectii.shape[0]
    distante = np.zeros(nr_poze_antrenare)
    
    for i in range(nr_poze_antrenare):
        distante[i] = CALC_DISTANTA_NORMA(proiectii[i, :], pr_test, norma)
    
    pozitia = np.argmin(distante)
    
    eticheta_gasita = None
    if etichete_antrenare is not None:
        eticheta_gasita = etichete_antrenare[pozitia]
    
    return pozitia, eticheta_gasita


# Interogare pentru algoritmul Eigenfaces cu reprezentanti de clasa
# Args: poza_test - Poza de test, media - Poza medie, HQPB - High-Quality Pseudo-Basis
#       proiectii_rc - Proiectiile reprezentantilor (nrPersoane x k), norma - Norma de distanta
#       etichete_antrenare - Etichetele, nr_persoane - Numarul de persoane
# Returns: pozitia, eticheta_gasita
def ALG_EIGENFACES_REPREZENTANTI(poza_test, media, HQPB, proiectii_rc, norma, 
                                 etichete_antrenare, nr_persoane):
    # Pas 1: Centram poza de test
    poza_test_centrat = poza_test - media
    
    # Pas 2: Proiectam poza de test pe HQPB
    pr_test = poza_test_centrat @ HQPB  # vector de k elemente
    
    # Pas 3: Aplicam NN pe proiectiile reprezentantilor
    # proiectii_rc este (nrPersoane x k), pr_test este (k,)
    nr_persoane_rc = proiectii_rc.shape[0]
    distante = np.zeros(nr_persoane_rc)
    
    for i in range(nr_persoane_rc):
        distante[i] = CALC_DISTANTA_NORMA(proiectii_rc[i, :], pr_test, norma)
    
    # Gasim reprezentantul cel mai apropiat
    pozitia_reprezentant = np.argmin(distante)
    eticheta_gasita = pozitia_reprezentant
    
    # Gasim o poza de antrenare a persoanei respective
    # (similar cu kNN: dupa determinarea vecinului, calculam pozitia unei poze)
    pozitii_persoana = [i for i, et in enumerate(etichete_antrenare) if et == eticheta_gasita]
    if len(pozitii_persoana) > 0:
        pozitia = pozitii_persoana[0]
    else:
        pozitia = 0
    
    return pozitia, eticheta_gasita

