# Algoritmul Eigenfaces - Documentație

## 📚 Cuprins

1. [Ideea Principală](#ideea-principală)
2. [Preprocesarea](#preprocesarea)
3. [Interogarea](#interogarea)
4. [Variante](#variante)
5. [Parametrul k](#parametrul-k)
6. [Exemplu Concret](#exemplu-concret)

---

## 💡 Ideea Principală

Algoritmul **Eigenfaces** este o metodă de recunoaștere facială bazată pe **reducerea dimensionalității** folosind **Principal Component Analysis (PCA)** sau **Singular Value Decomposition (SVD)**.

### Conceptul de bază:

În loc să lucrăm cu toți cei **10,304 pixeli** dintr-o poză (112 × 92), algoritmul reduce dimensiunea la **k coeficienți** (unde k ∈ {20, 40, 60, 80, 100}), păstrând doar informația esențială.

**Analogia:** Dacă o poză ar fi un punct într-un spațiu 10,304-dimensional, Eigenfaces găsește un subspațiu de dimensiune k care conține cea mai multă informație relevantă.

---

## 🔧 Preprocesarea (se face o singură dată)

Preprocesarea este partea cea mai importantă și computațional costisitoare. Ea se face o singură dată pentru toate pozele de antrenare.

### Pasul 1: Calculăm poza medie

```python
media = np.mean(A, axis=1)  # media pe coloane
```

- Calculăm media fiecărui pixel peste toate pozele de antrenare
- Rezultat: o "poza medie" (fața medie a tuturor persoanelor)
- Dimensiune: vector de 10,304 pixeli

**Ce reprezintă:** Fața "tipică" sau "medie" din baza de date.

### Pasul 2: Centrăm toate pozele

```python
A_centrat = A - media.reshape(-1, 1)
```

- Scădem poza medie din fiecare poză de antrenare
- Rezultat: poze centrate în jurul mediei (diferențe față de medie)
- Dimensiune: matrice 10,304 × 320

**De ce?** Pentru a elimina variațiile comune și a evidenția diferențele între persoane.

### Pasul 3: Găsim direcțiile principale (HQPB - High-Quality Pseudo-Basis)

Aceasta este partea cheie! Găsim direcțiile principale de variație în date.

#### Varianta SVD (folosită în implementare):

```python
U, s, Vt = svd(A_centrat.T, full_matrices=False)
HQPB = Vt.T  # primele k coloane
```

**Ce se întâmplă:**
- SVD (Singular Value Decomposition) descompune matricea centrată
- Găsește direcțiile principale de variație în date
- Fiecare direcție = un **"eigenface"** (o față fantomă)
- HQPB conține **k eigenfaces** (cele mai importante)

**Eigenfaces:** Sunt "fețe fantomă" care reprezintă direcțiile principale de variație. Primele eigenfaces captează variațiile cele mai importante (forma feței, poziția ochilor, etc.), iar cele din urmă captează detalii fine sau zgomot.

**Analogia:** Dacă pozele sunt puncte într-un spațiu, eigenfaces sunt axele principale ale unui elipsoid care înconjoară aceste puncte.

### Pasul 4: Proiectăm pozele pe eigenfaces

```python
proiectii = A_centrat.T @ HQPB  # sau U @ S pentru SVD
```

- Fiecare poză devine un **vector de k coeficienți**
- În loc de 10,304 pixeli → **k numere** (ex. 40)
- Aceste coeficienți descriu poza în spațiul eigenfaces

**Dimensiuni:**
- `A_centrat.T`: (320 poze × 10,304 pixeli)
- `HQPB`: (10,304 pixeli × k eigenfaces)
- `proiectii`: (320 poze × k coeficienți)

**Ce reprezintă coeficienții:** Cât de mult "seamănă" fiecare poză cu fiecare eigenface.

### Rezultatul preprocesării:

- ✅ `media`: poza medie (10,304 pixeli)
- ✅ `HQPB`: k eigenfaces (10,304 pixeli × k)
- ✅ `proiectii`: toate pozele de antrenare proiectate (320 poze × k coeficienți)

---

## 🔍 Interogarea (căutarea)

Când vrem să identificăm o poză nouă, urmăm acești pași:

### Pasul 1: Centrăm poza de test

```python
poza_test_centrat = poza_test - media
```

- Scădem poza medie pentru a centra poza de test în același mod ca pozele de antrenare.

### Pasul 2: Proiectăm poza de test

```python
pr_test = poza_test_centrat @ HQPB  # vector de k elemente
```

- Transformăm poza de test în același spațiu de k coeficienți
- Dimensiune: vector de k elemente

**Ce obținem:** Un vector care descrie poza de test în termenii eigenfaces.

### Pasul 3: Căutăm cea mai apropiată poză

```python
for i in range(nr_poze_antrenare):
    distante[i] = CALC_DISTANTA_NORMA(proiectii[i, :], pr_test, norma)
pozitia = np.argmin(distante)
```

- Comparăm coeficienții pozei de test cu coeficienții pozelor de antrenare
- Folosim o **normă de distanță** (Manhattan, Euclidian, Infinit, Cosinus)
- Găsim cea mai apropiată poză (NN - Nearest Neighbor pe proiecții)

**Rezultat:** Identificăm persoana din poza de test!

---

## 🔀 Variante

### Eigenfaces Clasic (algoritm `3`)

- Compară poza de test cu **toate cele 320 de poze** de antrenare
- Calculează **320 de distanțe**
- Mai precis, dar mai lent

**Când să folosești:** Când vrei acuratețe maximă.

### Eigenfaces cu Reprezentanți (algoritm `4`)

- Creează câte un **reprezentant per persoană** (ex. media pozelor)
- Compară poza de test doar cu cei **40 de reprezentanți**
- Calculează doar **40 de distanțe** → **mult mai rapid!**

**Când să folosești:** Când vrei viteză și ai multe poze per persoană.

**Cum se creează reprezentanții:**
- **Varianta a) Media pozelor:** Media tuturor pozelor unei persoane
- **Varianta b) Poză aleatorie:** O poză aleatorie a persoanei

---

## 📊 Parametrul k

Parametrul **k** determină câte eigenfaces (componente principale) reținem.

### Trade-offs:

| k | Viteză | Acuratețe | Memorie | Recomandare |
|---|--------|-----------|---------|-------------|
| **20** | ⚡⚡⚡ Foarte rapid | ⭐⭐ Scăzută | 💾 Mică | Testare rapidă |
| **40** | ⚡⚡ Rapid | ⭐⭐⭐ Bună | 💾💾 Moderată | **Recomandat pentru început** |
| **60** | ⚡ Moderat | ⭐⭐⭐⭐ Foarte bună | 💾💾💾 Mare | **Recomandat pentru producție** |
| **80** | 🐌 Lent | ⭐⭐⭐⭐⭐ Excelentă | 💾💾💾💾 Foarte mare | Când ai nevoie de precizie maximă |
| **100** | 🐌🐌 Foarte lent | ⭐⭐⭐⭐⭐ Maximă | 💾💾💾💾💾 Foarte mare | Poate include zgomot |

### De ce 40 sau 60?

- **k = 40:** Bun pentru început - echilibru între viteză și acuratețe
- **k = 60:** Recomandat - acuratețe foarte bună, fără să fie prea lent

**Sfat:** Testează toate valorile (20, 40, 60, 80, 100) și vezi care dă cea mai bună rată de recunoaștere pentru datele tale!

---

## 📝 Exemplu Concret

Să zicem că ai:
- **320 de poze de antrenare** (40 persoane × 8 poze)
- **80 de poze de test** (40 persoane × 2 poze)
- **k = 40**

### Preprocesare (o singură dată):

1. ✅ Calculezi poza medie (10,304 pixeli)
2. ✅ Centrezi toate cele 320 de poze
3. ✅ Găsești 40 de eigenfaces (direcții principale)
4. ✅ Proiectezi toate cele 320 de poze → **320 vectori de 40 de coeficienți**

**Timp:** ~2-5 secunde (depinde de k)

### Interogare (pentru fiecare poză nouă):

1. ✅ Centrezi poza nouă
2. ✅ O proiectezi → obții un **vector de 40 de coeficienți**
3. ✅ Compari cu cei 320 de vectori (folosind norma)
4. ✅ Găsești cea mai apropiată poză → identifici persoana

**Timp:** ~0.001-0.01 secunde per poză

### Rezultat:

- **Rata de recunoaștere:** % de poze de test identificate corect
- **Timp mediu de interogare:** timpul pentru o singură poză

---

## 🎯 De ce funcționează?

1. **Reducerea dimensionalității:** Lucrezi cu k coeficienți în loc de 10,304 pixeli
2. **Direcțiile principale:** Eigenfaces captează variațiile importante (față, ochi, nas, etc.)
3. **Viteza:** Comparări mult mai rapide pe vectori mici
4. **Robustete:** Zgomotul din pixeli individuali este redus

---

## 📚 Referințe

- **PCA (Principal Component Analysis):** Metodă clasică de reducere a dimensionalității
- **SVD (Singular Value Decomposition):** Metodă echivalentă, mai eficientă computațional
- **Eigenfaces:** Termen introdus de Turk și Pentland (1991)

---

## 🔧 Implementare

Implementarea folosește:
- **Varianta SVD** pentru preprocesare (mai eficientă)
- **Truncated SVD** (păstrăm doar primele k componente)
- **Norme de distanță:** Manhattan, Euclidian, Infinit, Cosinus

Pentru detalii tehnice, vezi codul din `algoritmi/eigenfaces.py`.

