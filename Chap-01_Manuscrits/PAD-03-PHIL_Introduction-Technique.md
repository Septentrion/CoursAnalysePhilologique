# Cours 1.4 — Introduction technique : de l'image numérique au problème de vision par ordinateur

**Module 1 · Computer Vision appliquée aux manuscrits médiévaux · MD5**

---

> *« Avant de demander à un algorithme de lire, il faut lui expliquer ce qu'est une image. »*

---

## Pourquoi cette section maintenant ?

Les trois sections précédentes ont établi la nature du problème depuis le côté humain : qu'est-ce qu'un manuscrit, quelles décisions sa lecture impose, comment mesurer la qualité d'une transcription. Cette section opère le basculement du côté machine.

Elle est volontairement condensée — trente minutes en cours, un document de référence à relire à tête reposée. Elle n'entre pas dans les détails de chaque outil ou de chaque algorithme : c'est l'objet des Jours 2 à 5. Elle pose les fondements conceptuels sans lesquels les sessions suivantes manqueraient d'ancrage, et dessine la carte du territoire que nous allons explorer.

À l'issue de cette section, vous devrez être capables de répondre à ces trois questions simples :
- Qu'est-ce qu'une image numérique, concrètement ?
- Pourquoi les approches classiques d'OCR échouent-elles sur des manuscrits médiévaux ?
- Quelles sont les grandes étapes du pipeline que nous allons construire, et dans quel ordre ?

---

## 1. L'image numérique : un tableau de nombres

### 1.1 Le pixel comme unité élémentaire

Une image numérique est, fondamentalement, un **tableau rectangulaire de nombres entiers**. Chaque case de ce tableau est un **pixel** (*picture element*). La valeur numérique stockée dans chaque case représente l'intensité lumineuse à cet emplacement.

Pour une image en **niveaux de gris**, chaque pixel est un entier compris entre 0 et 255 (sur 8 bits) :
- **0** correspond au noir absolu (aucune lumière).
- **255** correspond au blanc absolu (intensité maximale).
- Toutes les valeurs intermédiaires correspondent à des nuances de gris.

```python
import numpy as np
from PIL import Image

# Charger un scan de manuscrit en niveaux de gris
image = Image.open("scan_manuscrit.tif").convert("L")  # "L" = niveaux de gris
tableau = np.array(image)

print(f"Type          : {type(tableau)}")
print(f"Type des valeurs : {tableau.dtype}")       # uint8 (entier 8 bits non signé)
print(f"Dimensions    : {tableau.shape}")           # (hauteur, largeur)
print(f"Valeur min    : {tableau.min()}")           # proche de 0 = zones noires
print(f"Valeur max    : {tableau.max()}")           # proche de 255 = zones blanches
print(f"Valeur moyenne : {tableau.mean():.1f}")     # luminosité globale

# Accéder à un pixel individuel
ligne, colonne = 120, 340
print(f"Pixel en ({ligne}, {colonne}) : {tableau[ligne, colonne]}")

# Extraire une région (par exemple, une ligne de texte)
# Supposons que la ligne commence à y=200 et a une hauteur de 50 pixels
region_ligne = tableau[200:250, :]   # Toute la largeur, lignes 200 à 249
print(f"Dimensions de la région extraite : {region_ligne.shape}")
```

> **Convention d'indexation :** en NumPy, les tableaux d'images sont indexés `[ligne, colonne]`, ce qui correspond à `[y, x]` dans le repère image (l'axe y pointe vers le bas). Cette convention — différente du repère mathématique habituel où y pointe vers le haut — est une source fréquente d'erreurs. À retenir une fois pour toutes.

### 1.2 La résolution et les DPI

La **résolution** d'une image est le nombre de pixels qu'elle contient, exprimé en hauteur × largeur. Une image de 4 000 × 3 000 pixels contient 12 millions de pixels (12 mégapixels).

Les **DPI** (*dots per inch*, points par pouce) expriment la densité de pixels par rapport à une dimension physique réelle. Un scan à 300 DPI signifie que chaque pouce physique du document original est représenté par 300 pixels dans l'image numérique.

Pour les manuscrits, la résolution de numérisation a des conséquences directes sur la qualité de l'HTR :

| Résolution | Usage typique | Remarques pour l'HTR |
|------------|--------------|----------------------|
| 72–96 DPI | Aperçu web | Insuffisant : les détails des lettres sont perdus |
| 150 DPI | Lecture à l'écran | Minimum acceptable pour les écritures lisibles |
| 300 DPI | Standard d'archivage | Bon compromis qualité/taille de fichier |
| 400–600 DPI | HTR et analyse fine | Recommandé pour les écritures difficiles |
| > 600 DPI | Conservation et paléographie | Utile pour les lettrines ornées et les détails fins |

Les scans Gallica sont généralement disponibles à 300 ou 400 DPI. Les numérisations de l'IRHT montent souvent à 400 voire 600 DPI pour les manuscrits de qualité.

```python
# Vérifier la résolution d'un scan
image = Image.open("scan_manuscrit.tif")
dpi = image.info.get('dpi', (None, None))
print(f"Résolution : {dpi[0]} × {dpi[1]} DPI")
print(f"Dimensions en pixels : {image.size[0]} × {image.size[1]}")

# Calculer la taille physique correspondante
if dpi[0]:
    largeur_pouces = image.size[0] / dpi[0]
    hauteur_pouces = image.size[1] / dpi[1]
    print(f"Taille physique estimée : {largeur_pouces:.1f}″ × {hauteur_pouces:.1f}″")
    print(f"                         ({largeur_pouces * 2.54:.1f} cm × {hauteur_pouces * 2.54:.1f} cm)")
```

### 1.3 Les images couleur : canaux et espaces colorimétriques

Un scan de manuscrit peut être capturé en couleur, même si le document lui-même est noir sur parchemin. La couleur apporte des informations supplémentaires : elle distingue l'encre noire, l'encre rouge des rubriques, le fond jaunâtre du parchemin vieilli, et parfois des traces d'encre effacée invisible en niveaux de gris.

**Le modèle RGB**

Dans une image couleur en mode **RGB** (*Red, Green, Blue*), chaque pixel est représenté non pas par un seul nombre, mais par un **triplet** de valeurs : une intensité pour le rouge, une pour le vert, une pour le bleu. Le tableau NumPy correspondant a alors trois dimensions : `(hauteur, largeur, 3)`.

```python
image_couleur = Image.open("scan_manuscrit_couleur.tif")  # Mode RGB par défaut
tableau_rgb = np.array(image_couleur)

print(f"Dimensions : {tableau_rgb.shape}")  # (hauteur, largeur, 3)

# Séparer les canaux
canal_rouge = tableau_rgb[:, :, 0]  # Valeurs 0–255 pour le rouge
canal_vert  = tableau_rgb[:, :, 1]  # Valeurs 0–255 pour le vert
canal_bleu  = tableau_rgb[:, :, 2]  # Valeurs 0–255 pour le bleu
```

Pour les manuscrits, le canal rouge est souvent le plus informatif pour le texte noir sur fond clair, car l'encre ferro-gallique absorbe fortement le rouge. Le canal bleu, en revanche, capture mieux certaines taches et dégradations. La paléographie scientifique utilise parfois des éclairages multispectaux (ultraviolet, infrarouge) pour révéler des encres effacées, mais cela dépasse le cadre de ce cours.

**La conversion en niveaux de gris**

Pour l'HTR, travailler en niveaux de gris simplifie le traitement sans perte d'information critique. La conversion depuis RGB vers niveaux de gris n'est pas une simple moyenne des trois canaux — elle utilise une pondération qui reflète la sensibilité perceptuelle de l'œil humain aux différentes longueurs d'onde :

$$L = 0.299 \cdot R + 0.587 \cdot G + 0.114 \cdot B$$

Le canal vert est pondéré plus fortement car l'œil humain y est plus sensible. Pillow applique automatiquement cette pondération lors d'une conversion `.convert("L")`.

**Autres espaces colorimétriques**

L'espace **HSV** (*Hue, Saturation, Value*) sépare la teinte (quelle couleur ?), la saturation (à quel point la couleur est-elle pure ?) et la valeur (à quel point est-ce lumineux ?). Il est utile pour isoler les rubriques rouges d'un manuscrit, car la teinte rouge est bien séparée de la teinte noire.

```python
import cv2

# Charger en BGR (convention OpenCV — noter l'inversion R/B par rapport à PIL)
image_bgr = cv2.imread("scan_manuscrit_couleur.tif")

# Convertir en HSV
image_hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)

# Isoler les rubriques rouges par seuillage HSV
# Le rouge en HSV occupe deux plages (autour de 0° et autour de 360°)
masque_rouge_bas = cv2.inRange(image_hsv, (0, 80, 80), (10, 255, 255))
masque_rouge_haut = cv2.inRange(image_hsv, (170, 80, 80), (180, 255, 255))
masque_rouge = cv2.bitwise_or(masque_rouge_bas, masque_rouge_haut)

# masque_rouge est une image binaire : 255 là où il y a du rouge, 0 ailleurs
```

### 1.4 L'histogramme : lire l'image en un coup d'œil

L'**histogramme** d'une image en niveaux de gris est un graphique qui représente, pour chaque valeur d'intensité de 0 à 255, le nombre de pixels qui ont cette valeur. C'est un outil de diagnostic rapide, extrêmement utile pour comprendre les caractéristiques d'un scan avant de le traiter.

```python
import matplotlib.pyplot as plt

def afficher_histogramme(image_gris: np.ndarray, titre: str = "") -> None:
    """
    Affiche l'histogramme d'une image en niveaux de gris,
    avec les statistiques descriptives essentielles.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # Image originale
    axes[0].imshow(image_gris, cmap='gray', vmin=0, vmax=255)
    axes[0].set_title(f"Image{' : ' + titre if titre else ''}")
    axes[0].axis('off')
    
    # Histogramme
    axes[1].hist(image_gris.ravel(), bins=256, range=(0, 255),
                 color='steelblue', alpha=0.8, edgecolor='none')
    axes[1].set_xlabel("Intensité (0 = noir, 255 = blanc)")
    axes[1].set_ylabel("Nombre de pixels")
    axes[1].set_title("Histogramme des niveaux de gris")
    axes[1].axvline(image_gris.mean(), color='red', linestyle='--',
                    label=f"Moyenne : {image_gris.mean():.0f}")
    axes[1].legend()
    
    plt.tight_layout()
    plt.show()
```

**Ce que l'histogramme révèle sur un manuscrit**

Un histogramme typique de scan de manuscrit présente deux pics bien séparés :
- Un **pic à droite** (valeurs élevées, proches de 255) : les pixels clairs du fond — le parchemin ou le papier.
- Un **pic à gauche** (valeurs basses, proches de 0) : les pixels sombres de l'encre.

La forme de ces pics et la distance qui les sépare indiquent immédiatement la qualité du scan :

```
Histogramme idéal (fort contraste) :
  ▓                          ▓▓▓▓▓▓▓▓▓▓
  ▓▓                       ▓▓▓▓▓▓▓▓▓▓▓▓
  ▓▓▓                   ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓
  ▓▓▓▓▓              ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓
  0              128                255
  (encre)                        (fond)

Histogramme problématique (faible contraste, fond sale) :
           ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓
         ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓
       ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓
  0              128                255
  Les deux populations se confondent — binarisation difficile
```

Un histogramme qui ne montre pas deux pics distincts indique un faible contraste, peut-être dû à une numérisation de mauvaise qualité, à un parchemin très uniformément coloré, ou à une encre fortement dégradée. Dans ces cas, la binarisation sera difficile et le CER de l'HTR s'en ressentira.

### 1.5 La binarisation : de l'image en niveaux de gris à l'image binaire

La **binarisation** consiste à transformer une image en niveaux de gris en une image binaire : chaque pixel devient soit noir (0) soit blanc (1). C'est une étape de prétraitement classique qui simplifie le traitement en aval.

**Le seuillage global (Otsu)**

La méthode d'Otsu cherche automatiquement le seuil qui minimise la variance intra-classe — c'est-à-dire le seuil qui sépare au mieux les deux populations de pixels (fond et encre). Elle est efficace quand l'histogramme présente deux pics bien distincts.

```python
import cv2

image = cv2.imread("scan.tif", cv2.IMREAD_GRAYSCALE)

# Seuillage global d'Otsu
# THRESH_BINARY_INV : les pixels sombres (encre) deviennent blancs,
# le fond devient noir — convention courante en HTR
seuil_otsu, binaire_otsu = cv2.threshold(
    image, 0, 255,
    cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
)
print(f"Seuil automatique d'Otsu : {seuil_otsu}")
```

**Le seuillage adaptatif (Sauvola)**

Quand l'éclairage est inégal sur le scan — parchemin plus sombre au centre qu'en bordure, taches de lumière — un seuil global unique est insuffisant. Le **seuillage de Sauvola** calcule un seuil local pour chaque pixel, en fonction de la moyenne et de l'écart-type dans une fenêtre de voisinage autour de ce pixel. Il est généralement supérieur à Otsu pour les documents anciens.

```python
from skimage.filters import threshold_sauvola

# window_size : taille de la fenêtre locale (en pixels)
# k : paramètre de sensibilité (0.2 par défaut, augmenter pour les fonds sales)
seuil_sauvola = threshold_sauvola(image, window_size=25, k=0.2)
binaire_sauvola = (image < seuil_sauvola).astype(np.uint8) * 255
```

Sauvola est plus lent qu'Otsu (calcul local vs global) mais produit des résultats significativement meilleurs sur les documents dont le fond est non uniforme — ce qui est presque toujours le cas pour les manuscrits sur parchemin vieilli.

---

## 2. Pourquoi l'OCR classique échoue sur les manuscrits médiévaux

### 2.1 Ce que suppose l'OCR classique

L'OCR (*Optical Character Recognition*) est une technologie mature pour les documents imprimés. Des outils comme Tesseract (Google), ABBYY FineReader ou Adobe Acrobat obtiennent des CER inférieurs à 1% sur du texte imprimé en police courante — une performance remarquable.

Ces performances reposent sur des hypothèses implicites qui sont toutes remplies pour un document imprimé et toutes violées pour un manuscrit médiéval.

### 2.2 Les hypothèses violées

**Hypothèse 1 — La police est connue**

L'OCR classique est entraîné sur des polices de caractères connues (Times, Arial, Garamond…). Chaque caractère a une forme fixe, reproductible à l'identique à chaque occurrence.

Dans un manuscrit médiéval, chaque lettre est tracée à la main, avec des variations d'un mot à l'autre et d'une ligne à l'autre. Pire : il n'existe pas deux manuscrits ayant la même « police ». La main du copiste est unique. Un OCR entraîné sur des polices ne peut pas reconnaître des tracés manuels sans avoir été spécifiquement adapté à ce type d'écriture — et encore, seulement après avoir appris *cette* main particulière.

**Hypothèse 2 — Les caractères sont isolables**

L'OCR classique segmente l'image en caractères individuels — chaque lettre est découpée et classée séparément. Cette segmentation est possible car dans un texte imprimé, chaque caractère est séparé des autres par de l'espace blanc.

Dans un manuscrit cursif, les lettres se connectent. Il n'existe pas de séparation physique entre *n* et *i* dans le mot *ni*, ni entre *l* et *i* dans *li*. La segmentation en caractères individuels est soit impossible, soit arbitraire.

**Hypothèse 3 — L'alignement horizontal est régulier**

L'OCR classique suppose que les lignes de texte sont horizontales et régulièrement espacées. Il peut tolérer une légère inclinaison, mais pas les variations qui caractérisent les manuscrits médiévaux : lignes légèrement courbées, espacement inter-lignes variable, lignes qui montent ou descendent progressivement.

**Hypothèse 4 — La mise en page est simple**

L'OCR classique traite le texte comme un flux continu, de gauche à droite et de haut en bas. Il ne sait pas gérer les colonnes multiples, les rubriques, les lettrines de grande taille, les notes marginales ou les illustrations intercalées dans le texte.

**Hypothèse 5 — L'alphabet est connu et fermé**

L'OCR classique classe chaque caractère dans un alphabet fini. Pour le latin moderne, cet alphabet fait 26 lettres + quelques signes diacritiques. Pour l'ancien français médiéval, il faudrait inclure les signes d'abréviation, le *s* long, les ligatures *æ*, *œ*, le signe tironien *⁊* (un 7 stylisé qui représente *et*), et d'autres signes sans équivalent dans l'alphabet moderne.

### 2.3 Pourquoi le HTR est une réponse différente

La **Handwritten Text Recognition** (HTR) ne segmente pas en caractères — elle traite la ligne entière comme une séquence. Au lieu de classifier des caractères isolés, elle génère une séquence de caractères à partir de l'image entière de la ligne, en utilisant un modèle séquentiel (RNN, Transformer) qui peut exploiter le contexte.

Ce changement de paradigme, des années 2010 aux années 2020, est la raison pour laquelle les outils modernes d'HTR (TrOCR, Kraken) sont radicalement supérieurs aux OCR classiques sur les manuscrits. Nous y reviendrons en détail au Jour 3.

---

## 3. Le pipeline que nous allons construire

### 3.1 Vue d'ensemble

Le pipeline complet que nous construirons au fil de ce cours peut être représenté comme une séquence d'étapes de transformation, chacune prenant en entrée la sortie de l'étape précédente.

```
┌─────────────────────────────────────────────────────────┐
│                    ENTRÉE                               │
│          Scan TIFF haute résolution (300–400 DPI)       │
└─────────────────────┬───────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────┐
│               ÉTAPE 1 — PRÉTRAITEMENT                   │
│  • Correction d'orientation (deskewing)                 │
│  • Normalisation du contraste (CLAHE)                   │
│  • Binarisation adaptative (Sauvola)                    │
│  • Réduction du bruit                                   │
│  Outils : OpenCV, scikit-image, Pillow                  │
└─────────────────────┬───────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────┐
│          ÉTAPE 2 — SEGMENTATION DE LAYOUT               │
│  • Détection des régions : texte / image / marge        │
│  • Classification de chaque région                      │
│  Outils : SAM (Segment Anything Model)                  │
└──────────────┬──────────────────────┬───────────────────┘
               │                      │
               ▼                      ▼
┌──────────────────────┐  ┌───────────────────────────────┐
│  BRANCHE TEXTE       │  │  BRANCHE IMAGE                │
│                      │  │                               │
│ ÉTAPE 3 — SEGM.      │  │ ÉTAPE 3b — DESCRIPTION        │
│ DE LIGNES            │  │ • Description automatique     │
│ • Détection des      │  │   des enluminures et dessins  │
│   lignes de base     │  │ Outils : CLIP, LLaVA          │
│   (baseline)         │  │                               │
│ • Découpe des        │  └───────────────────────────────┘
│   images de lignes   │
│ Outils : Kraken,     │
│ DINO                 │
└──────────┬───────────┘
           │
           ▼
┌─────────────────────────────────────────────────────────┐
│               ÉTAPE 4 — HTR PAR LIGNE                   │
│  • Reconnaissance du texte dans chaque image de ligne   │
│  • Score de confiance par caractère                     │
│  Outils : TrOCR (fine-tuné), Kraken                     │
└─────────────────────┬───────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────┐
│            ÉTAPE 5 — AGRÉGATION ET CONSENSUS            │
│  • Vote pondéré entre plusieurs modèles                 │
│  • Calcul du score de confiance global par ligne        │
│  • Détection des lignes nécessitant une révision        │
└─────────────────────┬───────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────┐
│               ÉTAPE 6 — EXPORT STRUCTURÉ                │
│  • Format JSON (pour le module NLP)                     │
│  • Format TEI XML (optionnel, pour les humanités)       │
│  • Métriques : CER estimé, lignes à réviser             │
└─────────────────────────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────┐
│                    SORTIE                               │
│     Dataset structuré → Input du module NLP             │
└─────────────────────────────────────────────────────────┘
```

### 3.2 Pourquoi cet ordre ?

Chaque étape conditionne la suivante, et une erreur à une étape précoce se propage et s'amplifie dans toutes les étapes suivantes. C'est pourquoi le prétraitement — l'étape la plus « basse » techniquement — mérite autant d'attention que les modèles sophistiqués qui viennent après.

Un exemple concret : si la binarisation laisse du bruit (des pixels parasites qui ressemblent à de l'encre), la segmentation de lignes détectera de fausses lignes. Chaque fausse ligne produit une image de ligne qui ne contient pas de texte réel. Le modèle HTR recevra ces images et produira des transcriptions de charabia. Ces transcriptions de charabia seront incluses dans le dataset livré au module NLP. Le module NLP essaiera de les corriger — et échouera.

La propagation des erreurs dans un pipeline est l'un des défis fondamentaux des systèmes de traitement de documents. On y fait face en testant chaque étape de façon isolée, en métrisant les sorties de chaque étape, et en définissant des critères d'acceptance avant de passer à l'étape suivante.

### 3.3 Ce que ce cours couvre, étape par étape

| Étape | Jour(s) | Contenu principal |
|-------|---------|-------------------|
| Prétraitement (1) | Jour 4 | OpenCV, Sauvola, CLAHE, deskewing |
| Segmentation de layout (2) | Jour 3 | SAM, segmentation promptable |
| Segmentation de lignes (3a) | Jour 3 | Kraken segment, DINO |
| Description d'images (3b) | Jour 3 | CLIP, LLaVA |
| HTR par ligne (4) | Jour 2–3 | TrOCR, fine-tuning, Kraken |
| Agrégation (5) | Jour 4 | Vote pondéré, distance de Levenshtein |
| Export (6) | Jour 4 | JSON, TEI, HuggingFace Datasets |
| Évaluation globale | Jour 5 | CER, WER, robustesse, peer review |

---

## 4. Panorama des outils

Cette section présente brièvement chaque outil que nous utiliserons, avec son rôle dans le pipeline et ses caractéristiques essentielles. Chacun sera étudié en détail dans les jours suivants.

### 4.1 OpenCV

**OpenCV** (*Open Source Computer Vision Library*) est la bibliothèque de référence pour le traitement d'images en Python (et C++). Développée depuis 1999, elle contient des milliers de fonctions couvrant le prétraitement, la détection de contours, la transformation géométrique, le flot optique, et bien plus.

Dans notre pipeline, OpenCV est utilisé principalement pour le prétraitement : correction d'orientation, normalisation du contraste, opérations morphologiques (dilatation, érosion — utiles pour nettoyer la binarisation).

```python
import cv2

# Charger, convertir en niveaux de gris, appliquer un filtre médian
img = cv2.imread("scan.tif")
gris = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
denoised = cv2.medianBlur(gris, ksize=3)  # Filtre médian, noyau 3×3
```

### 4.2 Pillow (PIL)

**Pillow** est la bibliothèque Python standard pour la manipulation d'images : ouvrir et sauvegarder des fichiers dans tous les formats courants (TIFF, JPEG, PNG, WebP…), effectuer des transformations simples (recadrage, redimensionnement, rotation), convertir entre modes colorimétriques.

Pillow et OpenCV sont complémentaires : Pillow excelle pour la gestion des fichiers et les opérations de haut niveau ; OpenCV pour le traitement numérique intensif. Le passage de l'un à l'autre se fait facilement via NumPy.

```python
from PIL import Image
import numpy as np

# PIL → NumPy
img_pil = Image.open("scan.tif").convert("L")
img_np = np.array(img_pil)

# NumPy → PIL (utile après traitement OpenCV pour sauvegarder)
img_retour = Image.fromarray(img_np)
img_retour.save("scan_traite.png")
```

### 4.3 scikit-image

**scikit-image** complète OpenCV avec des algorithmes de traitement d'images plus académiques, notamment le seuillage de Sauvola, les transformées de Hough, et les métriques de qualité d'image. Son API est cohérente avec celle de NumPy et de SciPy.

```python
from skimage.filters import threshold_sauvola
from skimage.transform import rotate
from skimage.measure import label, regionprops
```

### 4.4 Kraken

**Kraken** est un système HTR open source développé à l'École pratique des hautes études (EPHE) par Benjamin Kiessling. Contrairement aux outils génériques, il est conçu *spécifiquement* pour les documents patrimoniaux — manuscrits, documents d'archives, livres anciens.

Il effectue deux tâches dans notre pipeline :
- La **segmentation de lignes** (*kraken segment*) : détecter les lignes de base (*baseline*) du texte dans une image de page.
- L'**HTR** (*kraken ocr*) : reconnaître le texte dans les images de lignes extraites.

Kraken utilise ses propres modèles, entraînables via l'interface eScriptorium ou en ligne de commande. Plusieurs modèles pré-entraînés pour le moyen français médiéval sont disponibles via HTR-United.

```bash
# Installation
pip install kraken

# Segmentation d'une page
kraken -i scan.tif segments.json segment

# HTR avec un modèle pré-entraîné
kraken -i scan.tif sortie.txt ocr -m modele_medieval.mlmodel
```

### 4.5 TrOCR

**TrOCR** (*Transformer-based OCR*) est un modèle développé par Microsoft Research, publié en 2021. Il combine un encodeur de vision de type ViT (pré-entraîné avec BEiT) et un décodeur de langage de type GPT-2 dans une architecture encodeur-décodeur standard.

Il est distribué via la bibliothèque HuggingFace Transformers, ce qui facilite son utilisation et son fine-tuning.

```python
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image

# Charger le modèle et le processeur
processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten")

# Transcrire une image de ligne
image_ligne = Image.open("ligne_001.png").convert("RGB")
pixel_values = processor(images=image_ligne, return_tensors="pt").pixel_values
generated_ids = model.generate(pixel_values)
transcription = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

print(f"Transcription : {transcription}")
```

TrOCR pré-entraîné sur des manuscrits modernes (dataset IAM) sera notre point de départ. Nous le fine-tunerons sur des données médiévales en Jour 3.

### 4.6 SAM — Segment Anything Model

**SAM** est un modèle de segmentation développé par Meta AI et publié en 2023. Il peut segmenter n'importe quel objet dans une image à partir d'un prompt minimal : un point, une boîte englobante, ou rien du tout (segmentation automatique).

Dans notre pipeline, SAM sert à la **segmentation de layout** : identifier et délimiter les régions de la page (colonnes de texte, illustrations, rubriques, annotations marginales).

```python
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry

# Charger le modèle (nécessite le téléchargement des poids)
sam = sam_model_registry["vit_b"](checkpoint="sam_vit_b.pth")
mask_generator = SamAutomaticMaskGenerator(sam)

# Segmenter automatiquement une page de manuscrit
import numpy as np
image = np.array(Image.open("page.tif").convert("RGB"))
masques = mask_generator.generate(image)

# masques est une liste de dicts, chacun contenant :
# - 'segmentation' : masque booléen (hauteur × largeur)
# - 'bbox'         : [x, y, largeur, hauteur]
# - 'area'         : surface en pixels
# - 'predicted_iou': score de qualité estimé
print(f"Nombre de régions détectées : {len(masques)}")
```

### 4.7 DINO et DINOv2

**DINO** (*Self-Distillation with No Labels*) est un modèle d'apprentissage auto-supervisé développé par Facebook AI Research. Il apprend des représentations visuelles riches sans aucune annotation humaine, en faisant s'accorder un réseau « étudiant » sur les représentations d'un réseau « enseignant ».

Sa version améliorée, **DINOv2**, est entraînée sur un dataset curé à grande échelle et produit des features particulièrement généralisables.

Dans notre pipeline, DINOv2 est utilisé pour deux tâches :
- Le **clustering de pages par style d'écriture** : regrouper automatiquement les pages selon la main du copiste.
- L'**aide à la segmentation de lignes** : les features DINOv2 permettent de distinguer visuellement le texte du fond et des illustrations.

```python
from transformers import AutoImageProcessor, AutoModel
import torch

processor = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
model = AutoModel.from_pretrained("facebook/dinov2-base")

image = Image.open("page.tif").convert("RGB")
inputs = processor(images=image, return_tensors="pt")

with torch.no_grad():
    outputs = model(**inputs)

# features : vecteur de 768 dimensions représentant l'image entière
features = outputs.last_hidden_state[:, 0, :]  # CLS token
print(f"Dimensions du vecteur de features : {features.shape}")  # (1, 768)
```

### 4.8 CLIP

**CLIP** (*Contrastive Language-Image Pre-Training*) est un modèle développé par OpenAI, entraîné simultanément sur 400 millions de paires (image, texte). Il apprend un espace commun où les représentations d'images et de textes décrivant la même chose sont proches.

Dans notre pipeline, CLIP sert à la **description automatique des illustrations** : les enluminures, les dessins à la plume, les cartes et les diagrammes qui ne sont pas du texte mais qui font partie du document.

```python
from transformers import CLIPProcessor, CLIPModel

processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")

image_illustration = Image.open("enluminure.png").convert("RGB")

# Classification zero-shot : quelle description correspond le mieux ?
descriptions = [
    "une enluminure représentant un chevalier à cheval",
    "un diagramme astronomique avec des cercles concentriques",
    "une carte géographique médiévale",
    "une lettrine ornée avec des motifs végétaux",
    "un portrait de personnage en médaillon"
]

inputs = processor(
    text=descriptions,
    images=image_illustration,
    return_tensors="pt",
    padding=True
)
outputs = model(**inputs)
probabilites = outputs.logits_per_image.softmax(dim=1)

for desc, prob in zip(descriptions, probabilites[0]):
    print(f"{prob:.1%} — {desc}")
```

---

## 5. Vérification de l'environnement de travail

Avant les TP des jours suivants, vérifiez que votre environnement contient tous les outils nécessaires. Le script suivant effectue une vérification minimale.

```python
"""
verification_environnement.py
À exécuter une fois pour confirmer que toutes les dépendances sont installées.
"""

import sys

def verifier_import(nom_module: str, nom_affiche: str = None) -> bool:
    """Tente d'importer un module et rapporte le résultat."""
    nom_affiche = nom_affiche or nom_module
    try:
        __import__(nom_module)
        print(f"  [OK] {nom_affiche}")
        return True
    except ImportError as e:
        print(f"  [MANQUANT] {nom_affiche} — {e}")
        return False

print(f"Python {sys.version}\n")
print("=== Bibliothèques de traitement d'images ===")
verifier_import("cv2", "OpenCV (cv2)")
verifier_import("PIL", "Pillow (PIL)")
verifier_import("skimage", "scikit-image")
verifier_import("numpy", "NumPy")

print("\n=== Visualisation ===")
verifier_import("matplotlib", "Matplotlib")
verifier_import("seaborn", "Seaborn")

print("\n=== Apprentissage automatique ===")
verifier_import("torch", "PyTorch")
verifier_import("transformers", "HuggingFace Transformers")
verifier_import("datasets", "HuggingFace Datasets")

print("\n=== HTR spécialisé ===")
verifier_import("kraken", "Kraken")

print("\n=== Segmentation ===")
verifier_import("segment_anything", "SAM (segment-anything)")

print("\n=== Utilitaires ===")
verifier_import("langchain", "LangChain")
verifier_import("umap", "UMAP (umap-learn)")
```

Si des bibliothèques sont manquantes, installez-les avec :

```bash
pip install opencv-python pillow scikit-image numpy matplotlib seaborn
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install transformers datasets accelerate
pip install kraken
pip install git+https://github.com/facebookresearch/segment-anything.git
pip install langchain umap-learn
```

---

## 6. Récapitulatif du Jour 1

Le Jour 1 a posé trois types de fondements :

**Des fondements philologiques (1.1)**
La nature du manuscrit médiéval, les systèmes d'écriture, la variabilité de la langue, les obstacles à la lecture — autant de contraintes qui définissent le problème avant toute considération technique.

**Des fondements méthodologiques (1.2 et 1.3)**
La traduction du problème philologique en spécification ML : granularité de l'unité d'apprentissage (la ligne), définition de la vérité terrain, métriques d'évaluation (CER), gestion de l'ambiguïté, et construction de la baseline humaine par transcription manuelle.

**Des fondements techniques (1.4)**
La représentation numérique d'une image, les outils du pipeline, et l'architecture globale du système que nous allons construire.

Les Jours 2 à 5 approfondiront chacun des composants du pipeline, en partant de la théorie des architectures (Jour 2 — CNN et ViT) jusqu'à la livraison du dataset au module NLP (Jour 5).

---

## Bibliographie de référence

### Traitement numérique des images

- **Gonzalez, R. C., Woods, R. E.** (2018). *Digital Image Processing* (4e éd.). Pearson. — La référence encyclopédique du domaine. Chapitres 2 (Fundamentals), 3 (Intensity Transformations) et 10 (Image Segmentation) sont directement pertinents pour ce cours.

- **Bradski, G., Kaehler, A.** (2008). *Learning OpenCV: Computer Vision with the OpenCV Library*. O'Reilly Media. — Le livre de référence d'OpenCV, bien que certaines parties soient à actualiser pour l'API actuelle. La documentation officielle ([docs.opencv.org](https://docs.opencv.org)) reste la ressource la plus à jour.

- **Van der Walt, S. et al.** (2014). *scikit-image: Image Processing in Python*. PeerJ, 2, e453. — L'article de référence de scikit-image, présentant la philosophie de la bibliothèque et ses algorithmes principaux.

- **Sauvola, J., Pietikäinen, M.** (2000). *Adaptive Document Image Binarization*. Pattern Recognition, 33(2), 225–236. — L'article original du seuillage adaptatif de Sauvola, incontournable pour la binarisation de documents anciens.

- **Otsu, N.** (1979). *A Threshold Selection Method from Gray-Level Histograms*. IEEE Transactions on Systems, Man, and Cybernetics, 9(1), 62–66. — L'article fondateur du seuillage automatique d'Otsu.

### OCR et HTR : état de l'art

- **Smith, R.** (2007). *An Overview of the Tesseract OCR Engine*. Ninth International Conference on Document Analysis and Recognition (ICDAR). — Présentation de Tesseract, le moteur OCR open source de référence, et de ses limites sur les documents non standards.

- **Breuel, T. M.** (2008). *The OCRopus Open Source OCR System*. Document Recognition and Retrieval XV, 6815. — OCRopus a introduit l'approche par réseau de neurones récurrents pour l'OCR/HTR, posant les bases des approches modernes.

- **Li, M., Lyu, T., Yao, T., Ye, J., Lu, T., Zheng, Y., Huang, F.** (2021). *TrOCR: Transformer-based Optical Character Recognition with Pre-trained Models*. AAAI 2023. [arXiv:2109.10282] — L'article de référence de TrOCR.

- **Kiessling, B.** (2019). *Kraken — an Universal Text Recognizer for the Humanities*. Digital Humanities Conference (DH2019). — Présentation synthétique de Kraken par son auteur, avec les motivations de conception.

### Modèles fondateurs du pipeline

- **Dosovitskiy, A. et al.** (2020). *An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale*. ICLR 2021. [arXiv:2010.11929] — ViT, la brique de base de TrOCR et de DINO.

- **Caron, M. et al.** (2021). *Emerging Properties in Self-Supervised Vision Transformers*. ICCV 2021. [arXiv:2104.14294] — DINO.

- **Radford, A. et al.** (2021). *Learning Transferable Visual Models From Natural Language Supervision*. ICML 2021. — CLIP.

- **Kirillov, A. et al.** (2023). *Segment Anything*. ICCV 2023. [arXiv:2304.02643] — SAM.

### Ressources pratiques

- **Documentation HuggingFace Transformers** — [huggingface.co/docs/transformers](https://huggingface.co/docs/transformers). Pour TrOCR, CLIP, DINO et tous les modèles pré-entraînés.

- **Documentation Kraken** — [kraken.re](https://kraken.re). Guide complet d'installation, entraînement et utilisation.

- **Documentation SAM** — [github.com/facebookresearch/segment-anything](https://github.com/facebookresearch/segment-anything). Avec notebooks d'exemples.

---

*Support de cours rédigé pour le module Computer Vision · Promotion MD5 · Mai 2026*
*Ce document accompagne la séance 1.4 du Jour 1. Il conclut le Module 1 et prépare le Jour 2.*
*Durée estimée de lecture : 45 minutes. Durée en cours magistral : 30 minutes.*
