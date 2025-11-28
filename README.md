# Sistem de Recunoaștere Facială

Aplicație pentru recunoașterea fețelor folosind diferiți algoritmi: NN, KNN, Eigenfaces Clasic și Eigenfaces cu Reprezentanți.

## 📁 Structura Proiectului

```
M1/
├── algoritmi/          # Algoritmii de recunoaștere
│   ├── __init__.py
│   ├── nn.py          # Nearest Neighbor
│   ├── knn.py         # K-Nearest Neighbors
│   └── eigenfaces.py  # Eigenfaces (clasic și reprezentanți)
│
├── utils/              # Funcții utilitare
│   ├── __init__.py
│   ├── date.py        # Încărcare și prelucrare date
│   └── distante.py    # Calcul distanțe
│
├── ui/                 # Interfață grafică
│   ├── __init__.py
│   └── interfata.py   # Interfață cu dropdown-uri
│
├── docs/               # Documentație
│   └── EIGENFACES.md  # Documentație detaliată Eigenfaces
│
├── att_faces/          # Baza de date cu poze
├── main.py            # Punct de intrare principal
├── grafic.py          # Funcții pentru grafice
└── rezultate_recunoastere.txt  # Rezultate salvate
```

## 🚀 Utilizare

### Rulare aplicație

```bash
python main.py
```

### Interfață

Aplicația deschide o fereastră grafică unde poți selecta:

1. **Algoritm:**
   - `1 - NN` - Nearest Neighbor
   - `2 - KNN` - K-Nearest Neighbors
   - `3 - Eigenfaces Clasic`
   - `4 - Eigenfaces Reprezentanti`

2. **Norma:**
   - `1 - Manhattan`
   - `2 - Euclidian`
   - `3 - Infinit`
   - `4 - Cosinus`

3. **Parametri:**
   - **k (KNN):** Număr impar > 1 (doar pentru KNN)
   - **k Eigenfaces:** 20, 40, 60, 80, 100 (doar pentru Eigenfaces)

### Exemplu

1. Selectează: `3 - Eigenfaces Clasic`
2. Selectează: `2 - Euclidian`
3. Selectează: `40` pentru k Eigenfaces
4. Apasă **Start**

## 📊 Rezultate

Rezultatele sunt salvate în `rezultate_recunoastere.txt` și includ:
- Rata de recunoaștere (%)
- Timp mediu de interogare
- Timp de preprocesare (pentru Eigenfaces)

## 📚 Documentație

Pentru detalii despre algoritmul Eigenfaces, vezi [docs/EIGENFACES.md](docs/EIGENFACES.md).

## 🔧 Dependențe

- `numpy`
- `opencv-python` (cv2)
- `matplotlib`
- `tkinter` (inclus în Python)

## 📝 Note

- Baza de date `att_faces` conține 40 de persoane cu câte 10 poze fiecare
- 8 poze per persoană sunt folosite pentru antrenare, 2 pentru test
- Dimensiunea imaginilor: 112 × 92 pixeli (10,304 pixeli total)

