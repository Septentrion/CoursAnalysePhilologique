---
updated: 2026-05-17T17:24:11.612+02:00
edited_seconds: 1714
---
# Introduction technique : de l'image numérique au problème de vision par ordinateur

- **Module 1 · Computer Vision appliquée aux manuscrits médiévaux · MD5**
notes:  
Cette section opère le basculement du côté machine. Elle est condensée : 30 minutes en cours. Elle pose les fondements conceptuels pour les sessions suivantes. À l'issue, les étudiants doivent savoir : qu'est-ce qu'une image numérique, pourquoi l'OCR classique échoue, quelles sont les grandes étapes du pipeline.

À l'issue de cette section, vous devrez être capables de répondre à ces trois questions simples :
- Qu'est-ce qu'une image numérique, concrètement ?
- Pourquoi les approches classiques d'OCR échouent-elles sur des manuscrits médiévaux ?
- Quelles sont les grandes étapes du pipeline que nous allons construire, et dans quel ordre ?

---
## 1. L'image numérique : un tableau de **nombres**

--

### 1.1 Le pixel comme unité élémentaire
- Une image numérique = tableau rectangulaire de nombres entiers (NumPy Arrays)
![[Cours/Computer Vision/Michel/CoursAnalysePhilologique/metadata/Cours_1_4_Introduction_Technique-1778788655287.png]]
```python
import cv2
image = cv2.imread("photo.jpeg")
print(image.shape)
> (969, 969, 3)
```

notes:
- Chaque image étant une matrice numérique, sa forme va représenter 3 choses
	- **Height**: number of pixel rows (vertical dimension)
	- **Width**: number of pixel columns (horizontal dimension)
	- **Channels**: number of colour components per pixel
La valeur numérique stockée dans chaque case représente l'intensité lumineuse à cet emplacement.

--
![[OpenCV with Python — Comprehensive Technical Notes-1767613846470.png]]
- Chaque case correspond à une position dans la matrice numérique -> le pixel
- Chaque pixel est une combinaison de plusieurs valeurs.

--
<!-- slide template="[[plt-two-col]]" -->
::: title
### 1.1.b La valeur des pixels
Chaque pixel = 8 bits (1 byte)
:::
::: left 
![[OpenCV with Python — Comprehensive Technical Notes-1767613119318.png|300]]

- Pour les **images en couleur** ou nuances de gris es pixels seront entre ==0 et 255==
	- 0 → intensité minimum 
	- 255 → intensité maximum

:::
::: right
![[OpenCV with Python — Comprehensive Technical Notes-1767613722171.png]]

- Pour une **image binaire** un pixel peut être entre (==0 et 1==) ou (==0 et 255==).
	- 0 -> Noir
	- 1 ou 255 -> Blanc
:::
notes:
Pour une image en **niveaux de gris**, chaque pixel est un entier compris entre 0 et 255 (sur 8 bits) :
- **0** correspond au noir absolu (aucune lumière).
- **255** correspond au blanc absolu (intensité maximale).
- Toutes les valeurs intermédiaires correspondent à des nuances de gris.


pure **red**:   “255, 0, 0.” = 
Pure **green** is 0, 255, 0
pure **blue** is 0, 0, 255

any other combination can be made by mixing the three.
-> “255, 100, 150” for a particular shade of pink.


255 en binaire fait: 11111111 - 8 bit the absolute highest
8 bit -> représente 256 variation d'une couleur.
En vrai on dit 8 bit mais c'est plus 8 bit par canal 
-> 16,777,216 nuances en tout


-> nuances de gris en 8 bit peut faire jusqu'à 256 dégradé de blanc à noir.

8 bits = 24 bits C'est la même chose ! 

--
<!-- slide template="[[tpl-two-col-bottom]]" -->
::: title
### 1.2 Images couleur et espaces colorimétriques
:::

::: left
- **Espace RGB**: Chaque pixel est un triplet - Red, Green, Blue, (Attention openCV utilise BGR par défaut)
![[Cours/Computer Vision/Michel/CoursAnalysePhilologique/metadata/Cours_1_4_Introduction_Technique-1778793899489.png|350]]
:::

::: right

**Espace HSV** (Hue, Saturation, Value) 
- très pratique pour isoler des couleurs précises, moins facile à visualiser.

![[Cours/Computer Vision/Michel/CoursAnalysePhilologique/metadata/Cours_1_4_Introduction_Technique-1778789903692.png|300]]
:::
**Pour les manuscrits** :
- Canal **rouge** : bon pour l'encre (absorbe le rouge)
- Canal **bleu** : capture mieux taches et dégradations


notes:  
Montrer rapidement l'intérêt de HSV pour détecter l'encre rouge des rubriques. 

Un scan de manuscrit peut être capturé en couleur, même si le document lui-même est noir sur parchemin. La couleur apporte des informations supplémentaires : elle distingue l'encre noire, l'encre rouge des rubriques, le fond jaunâtre du parchemin vieilli, et parfois des traces d'encre effacée invisible en niveaux de gris.

- Capturer le rouge -> encre ferro-gallique tend vers le rouge
	- Et c'est l'une des couleurs les plus simple à créer donc souvent la couleur de highlight
- Capturer le bleu -> certaines taches et dégradations
	- éclairages multispectaux (ultraviolet, infrarouge) pour révéler des encres effacées

**La conversion en niveaux de gris** -> PIL & OpenCV ont des fonctions de conversion en niveau de gris
Grayscale images have a single channel.


**Autres espaces colorimétriques**
- L'espace **HSV** (*Hue, Saturation, Value*) 
	- sépare la teinte (quelle couleur ?), 
	- la saturation (à quel point la couleur est-elle pure ?) 
	- et la valeur (à quel point est-ce lumineux ?). 
Il est utile pour isoler les rubriques rouges d'un manuscrit, car la teinte rouge est bien séparée de la teinte noire.

--
### 1.4 L'histogramme : lire l'image en un coup d'œil
![[Cours/Computer Vision/Michel/CoursAnalysePhilologique/metadata/Cours_1_4_Introduction_Technique-1778795320454.png]]
- **Histogramme** : pour chaque intensité (0–255), nombre de pixels ayant cette valeur
	- Un **pic à droite** (valeurs élevées, proches de 255) : les pixels clairs du fond : le parchemin ou le papier.
	- Un **pic à gauche** (valeurs basses, proches de 0) : les pixels sombres de l'encre.

notes:
- Outil de diagnostic rapide pour classifier le scan avant de le traiter
L'**histogramme** d'une image en niveaux de gris est un graphique qui représente, pour chaque valeur d'intensité de 0 à 255, le nombre de pixels qui ont cette valeur. C'est un outil de diagnostic rapide, extrêmement utile pour comprendre les caractéristiques d'un scan avant de le traiter.

**Ce que l'histogramme révèle sur un manuscrit**
- Un histogramme typique de scan de manuscrit présente deux pics bien séparés :
- Un histogramme qui ne montre pas deux pics distincts indique un faible contraste, peut-être dû à une numérisation de mauvaise qualité, à un parchemin très uniformément coloré, ou à une encre fortement dégradée. Dans ces cas, la binarisation sera difficile et le CER de l'HTR s'en ressentira.

--
#### Histogramme espace colorimétrique - Manuscrit Voltaire 0
![[Cours/Computer Vision/Michel/CoursAnalysePhilologique/metadata/Cours_1_4_Introduction_Technique-1778795247924.png|500]]
--
### 1.5 La binarisation : de l'image en niveaux de gris à l'image binaire
- La **binarisation** consiste à transformer une image en niveaux de gris (une seule dimension) en une image binaire : 
	- chaque pixel devient soit noir (0) soit blanc (1) selon une ==valeur limite==
```python
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
seuil, binaire = cv2.threshold(gray, thresh=80, maxval=255, type=cv2.THRESH_BINARY )
```
![[Cours/Computer Vision/Michel/CoursAnalysePhilologique/metadata/Cours_1_4_Introduction_Technique-1778797340950.png|460]]

notes:
La binarisation est critique. Sur les manuscrits, Sauvola est presque toujours supérieur à Otsu. 

 C'est une étape de prétraitement classique qui simplifie le traitement en aval.
--
### **Seuillage global – Otsu** 
Optimal si deux pics distincts -calcule le seuil **maximisant la variance inter-classe**
```python
image = cv2.imread(img, cv2.IMREAD_GRAYSCALE)
seuil_otsu, binaire_otsu = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
```
![[Cours/Computer Vision/Michel/CoursAnalysePhilologique/metadata/Cours_1_4_Introduction_Technique-1778797349068.png|650]]

notes:
**Le seuillage global (Otsu)**
La méthode d'Otsu cherche automatiquement le seuil qui minimise la variance intra-classe : c'est-à-dire le seuil qui sépare au mieux les deux populations de pixels 0 à k et k 255.
-> Efficace quand l'histogramme présente deux pics bien distincts.
	-> Compare la moyenne de l'intensité de chaque classe par rapport à la moyenne globale de l'image

--
### **Seuillage adaptatif – Sauvola**
Pour des manuscrits avec plus de bruits, un éclairage inégal.
-> calcule un **seuil local pour chaque pixel**, en fonction de la **moyenne** et de l'**écart-type** dans une fenêtre de voisinage autour de ce pixel. 

```python
from skimage.filters import threshold_sauvola
seuil_sovola = threshold_sauvola(img, window_size=25, k=0.2)
binare = (seuil_sauvola > image).astype(np.unit8) * 255
```

![[Cours/Computer Vision/Michel/CoursAnalysePhilologique/metadata/Cours_1_4_Introduction_Technique-1778797361584.png|600]]

notes:

**Le seuillage adaptatif (Sauvola)**
Très pratique si variation de lumières: un seuil global unique est insuffisant. 
-> calcule un **seuil local pour chaque pixel**, en fonction de la **moyenne** et de l'**écart-type** dans une fenêtre de voisinage autour de ce pixel. 
- Généralement supérieur à Otsu pour les documents anciens.

```python
from skimage.filters import threshold_sauvola

# window_size : taille de la fenêtre locale (en pixels)
# k : paramètre de sensibilité (0.2 par défaut, augmenter pour les fonds sales)
seuil_sauvola = threshold_sauvola(image, window_size=25, k=0.2)
binaire_sauvola = (image < seuil_sauvola).astype(np.uint8) * 255
```

Sauvola est plus lent qu'Otsu (calcul local vs global) mais produit des résultats significativement meilleurs sur les documents dont le fond est non uniforme : ce qui est presque toujours le cas pour les manuscrits sur parchemin vieilli.

---
## 2. Pourquoi l'OCR classique échoue sur les manuscrits médiévaux

--
### 2.1 Ce que suppose l'OCR classique
- **Police de caractères** connue et fixe
- Caractères **isolables** (séparés par du blanc)
- Alignement horizontal régulier
- Mise en page **simple** (texte continu)
- Alphabet **fermé** (26 lettres + quelques diacritiques)

--
### 2.2 Les hypothèses violées pour un manuscrit
1. **Police inconnue** – chaque main est unique, variation d'un mot à l'autre.
2. **Caractères non isolables** – écriture cursive, ligatures, pas de séparation physique.
3. **Alignement irrégulier** – lignes courbées, espacement variable.
4. **Mise en page complexe** – colonnes, rubriques, lettrines, notes marginales.
5. **Alphabet ouvert** – signes d'abréviation, s long, ligatures (æ, œ), signe tironien (⁊).

> [!tldr] L'OCR classique atteint <1% CER sur imprimé. Sur manuscrit médiéval, >80% CER.

notes:
**Hypothèse 1 : La police est connue**: L'OCR classique est entraîné sur des polices de caractères connues (Times, Arial, Garamond…). Chaque caractère a une forme fixe, reproductible à l'identique à chaque occurrence.

Dans un manuscrit médiéval, chaque lettre est tracée à la main, 
- avec des variations d'un mot à l'autre et d'une ligne à l'autre. 
- **Pire** : il n'existe pas deux manuscrits ayant la même « police ». 
- La main du copiste est unique. Un OCR entraîné sur des polices ne peut pas reconnaître des tracés manuels sans avoir été spécifiquement adapté à ce type d'écriture : et encore, seulement après avoir appris *cette* main particulière.
- **Pire de Pire**: Un manuscrit à souvent plusieurs mains ayant contribué à sa rédaction, donc entraîner sur un manuscrit entier ne se traduit pas par une bonne précision sur tout le manuscrit.

**Hypothèse 2 : Les caractères sont isolables**

L'OCR classique segmente l'image en caractères individuels : chaque lettre est découpée et classée séparément. Cette segmentation est possible car dans un texte imprimé, chaque caractère est séparé des autres par de l'espace blanc.

Dans un manuscrit cursif, les lettres se connectent. Il n'existe pas de séparation physique entre *n* et *i* dans le mot *ni*, ni entre *l* et *i* dans *li*. La segmentation en caractères individuels est soit impossible, soit arbitraire.

**Hypothèse 3 : L'alignement horizontal est régulier**
L'OCR classique suppose que les lignes de texte sont horizontales et régulièrement espacées. Il peut tolérer une légère inclinaison, mais pas les variations qui caractérisent les manuscrits médiévaux : lignes légèrement courbées, espacement inter-lignes variable, lignes qui montent ou descendent progressivement.

**Hypothèse 4 : La mise en page est simple**

L'OCR classique traite le texte comme un flux continu, de gauche à droite et de haut en bas. Il ne sait pas gérer les colonnes multiples, les rubriques, les lettrines de grande taille, les notes marginales ou les illustrations intercalées dans le texte.

**Hypothèse 5 : L'alphabet est connu et fermé**

L'OCR classique classe chaque caractère dans un alphabet fini. Pour le latin moderne, cet alphabet fait 26 lettres + quelques signes diacritiques. Pour l'ancien français médiéval, il faudrait inclure les signes d'abréviation, le *s* long, les ligatures *æ*, *œ*, le signe tironien *⁊* (un 7 stylisé qui représente *et*), et d'autres signes sans équivalent dans l'alphabet moderne.

--
### 2.3 La réponse : HTR (Handwritten Text Recognition)
**Différence fondamentale** :
- OCR classique : segmentation en caractères → classification
- HTR : entrée = **ligne entière**, sortie = séquence de caractères (modèle séquentiel)

**Avantages** :
- Pas besoin d'isoler les caractères
- Contexte local (40–80 caractères) pour lever ambiguïtés
- Adapté aux architectures modernes (RNN, Transformer)

---
## 3. Le pipeline que nous allons construire

--
### 3.1 Vue d'ensemble
![[Pipeline HTR.png|750]]

notes:
```
┌─────────────────────────────────────────────────────────┐
│                    ENTRÉE                               │
│          Scan TIFF haute résolution (300–400 DPI)       │
└─────────────────────┬───────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────┐
│               ÉTAPE 1 : PRÉTRAITEMENT                   │
│  • Correction d'orientation (deskewing)                 │
│  • Normalisation du contraste (CLAHE)                   │
│  • Binarisation adaptative (Sauvola)                    │
│  • Réduction du bruit                                   │
│  Outils : OpenCV, scikit-image, Pillow                  │
└─────────────────────┬───────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────┐
│          ÉTAPE 2 : SEGMENTATION DE LAYOUT               │
│  • Détection des régions : texte / image / marge        │
│  • Classification de chaque région                      │
│  Outils : SAM (Segment Anything Model)                  │
└──────────────┬──────────────────────┬───────────────────┘
               │                      │
               ▼                      ▼
┌──────────────────────┐  ┌───────────────────────────────┐
│  BRANCHE TEXTE       │  │  BRANCHE IMAGE                │
│                      │  │                               │
│ ÉTAPE 3 : SEGM.      │  │ ÉTAPE 3b : DESCRIPTION        │
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
│               ÉTAPE 4 : HTR PAR LIGNE                   │
│  • Reconnaissance du texte dans chaque image de ligne   │
│  • Score de confiance par caractère                     │
│  Outils : TrOCR (fine-tuné), Kraken                     │
└─────────────────────┬───────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────┐
│            ÉTAPE 5 : AGRÉGATION ET CONSENSUS            │
│  • Vote pondéré entre plusieurs modèles                 │
│  • Calcul du score de confiance global par ligne        │
│  • Détection des lignes nécessitant une révision        │
└─────────────────────┬───────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────┐
│               ÉTAPE 6 : EXPORT STRUCTURÉ                │
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

Chaque étape conditionne la suivante, et une erreur à une étape précoce se propage et s'amplifie dans toutes les étapes suivantes. C'est pourquoi le prétraitement : l'étape la plus « basse » techniquement : mérite autant d'attention que les modèles sophistiqués qui viennent après.

Un exemple concret : si la binarisation laisse du bruit (des pixels parasites qui ressemblent à de l'encre), la segmentation de lignes détectera de fausses lignes. Chaque fausse ligne produit une image de ligne qui ne contient pas de texte réel. Le modèle HTR recevra ces images et produira des transcriptions de charabia. Ces transcriptions de charabia seront incluses dans le dataset livré au module NLP. Le module NLP essaiera de les corriger : et échouera.

La propagation des erreurs dans un pipeline est l'un des défis fondamentaux des systèmes de traitement de documents. On y fait face en testant chaque étape de façon isolée, en métrisant les sorties de chaque étape, et en définissant des critères d'acceptance avant de passer à l'étape suivante.

--
### 3.3 Ce que ce cours couvre, étape par étape

| Étape                       | Jour(s)  | Contenu principal                     |
| --------------------------- | -------- | ------------------------------------- |
| Prétraitement (1)           | Jour 4   | OpenCV, Sauvola, CLAHE, deskewing     |
| Segmentation de layout (2)  | Jour 3   | SAM, segmentation promptable          |
| Segmentation de lignes (3a) | Jour 3   | Kraken segment, DINO                  |
| Description d'images (3b)   | Jour 3   | CLIP, LLaVA                           |
| HTR par ligne (4)           | Jour 2–3 | TrOCR, fine-tuning, Kraken            |
| Agrégation (5)              | Jour 4   | Vote pondéré, distance de Levenshtein |
| Export (6)                  | Jour 4   | JSON, TEI, HuggingFace Datasets       |
| Évaluation globale          | Jour 5   | CER, WER, robustesse, peer review     |

---
## 4. Panorama des outils

notes:
Cette section présente brièvement chaque outil que nous utiliserons, avec son rôle dans le pipeline et ses caractéristiques essentielles. Chacun sera étudié en détail dans les jours suivants.

--
### 4.1 OpenCV
- Bibliothèque de référence pour le traitement d'images
- Tutoriels: https://opencv24-python-tutorials.readthedocs.io/en/latest/py_tutorials/py_tutorials.html
- Prétraitement : correction d'orientation, normalisation, opérations morphologiques

```python
import cv2
img = cv2.imread("scan.tif")
gris = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
denoise = cv2.medianBlur(gris, 3)
```

notes:
**OpenCV** (*Open Source Computer Vision Library*) est la bibliothèque de référence pour le traitement d'images en Python (et C++). Développée depuis 1999, elle contient des milliers de fonctions couvrant le prétraitement, la détection de contours, la transformation géométrique, le flot optique, et bien plus.

Dans notre pipeline, OpenCV est utilisé principalement pour le prétraitement : correction d'orientation, normalisation du contraste, opérations morphologiques (dilatation, érosion : utiles pour nettoyer la binarisation).

```python
import cv2

# Charger, convertir en niveaux de gris, appliquer un filtre médian
img = cv2.imread("scan.tif")
gris = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
denoised = cv2.medianBlur(gris, ksize=3)  # Filtre médian, noyau 3×3
```

--
### 4.2 Pillow (PIL)
- Manipulation simple et formats d'image
- Tutoriel: https://pillow.readthedocs.io/en/stable/handbook/tutorial.html
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

notes:
**Pillow** est la bibliothèque Python standard pour la manipulation d'images : ouvrir et sauvegarder des fichiers dans tous les formats courants (TIFF, JPEG, PNG, WebP…), effectuer des transformations simples (recadrage, redimensionnement, rotation), convertir entre modes colorimétriques.

Pillow et OpenCV sont complémentaires : Pillow excelle pour la gestion des fichiers et les opérations de haut niveau ; OpenCV pour le traitement numérique intensif. Le passage de l'un à l'autre se fait facilement via NumPy.

--
### 4.3 scikit-image
- Algorithmes académiques (Sauvola, transformée de Hough, métriques)
- Tutoriel: https://scikit-image.org/docs/dev/user_guide/tutorial_segmentation.html

notes:
**scikit-image** complète OpenCV avec des algorithmes de traitement d'images plus académiques, notamment le seuillage de Sauvola, les transformées de Hough, et les métriques de qualité d'image. Son API est cohérente avec celle de NumPy et de SciPy.

```python
from skimage.filters import threshold_sauvola
from skimage.transform import rotate
from skimage.measure import label, regionprops
```

---
### 4.4 Kraken
- HTR spécialisé pour documents patrimoniaux
- Segmentation de lignes + reconnaissance
- Dépôt: https://github.com/mittagessen/kraken
```bash
kraken -i scan.tif segment lines
kraken -i scan.tif ocr -m modele_medieval.mlmodel
```

notes:
**Kraken** est un système HTR open source développé à l'École pratique des hautes études (EPHE) par Benjamin Kiessling. Contrairement aux outils génériques, il est conçu *spécifiquement* pour les documents patrimoniaux : manuscrits, documents d'archives, livres anciens.

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

--
### 4.5 TrOCR
- Transformer-based OCR (Microsoft Research) 
- Disponible sur Hugging Face - dont modèles fine-tunés pour du Français.

```python

from transformers import TrOCRProcessor, VisionEncoderDecoderModel
processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten")
```

notes:

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

--
### 4.6 SAM : Segment Anything Model
- Segmentation de layout par prompt (Meta AI, 2023)

```python
from segment_anything import SamAutomaticMaskGenerator
mask_generator = SamAutomaticMaskGenerator(sam)
masks = mask_generator.generate(image_np)
```

notes:

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

--
### 4.7 DINO / DINOv2
- Représentations auto‑supervisées
- Le **clustering de pages par style d'écriture** : regr
- ouper automatiquement les pages selon la main du copiste.
- L'**aide à la segmentation de lignes** : les features DINOv2 permettent de distinguer visuellement le texte du fond et des illustrations.

```python
from transformers import AutoImageProcessor, AutoModel
model = AutoModel.from_pretrained("facebook/dinov2-base")
```

notes:
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

--
### 4.8 CLIP
- Espace commun image‑texte (OpenAI)
- Description zero‑shot des illustrations

```python
from transformers import CLIPProcessor, CLIPModel
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
```

notes:
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
    print(f"{prob:.1%} : {desc}")
```

---
## 5. Récapitulatif du Jour 1

Le Jour 1 a posé trois types de fondements :

**Des fondements philologiques (1.1)**
La nature du manuscrit médiéval, les systèmes d'écriture, la variabilité de la langue, les obstacles à la lecture : autant de contraintes qui définissent le problème avant toute considération technique.

**Des fondements méthodologiques (1.2 et 1.3)**
La traduction du problème philologique en spécification ML : granularité de l'unité d'apprentissage (la ligne), définition de la vérité terrain, métriques d'évaluation (CER), gestion de l'ambiguïté, et construction de la baseline humaine par transcription manuelle.

**Des fondements techniques (1.4)**
La représentation numérique d'une image, les outils du pipeline, et l'architecture globale du système que nous allons construire.

Les Jours 2 à 5 approfondiront chacun des composants du pipeline, en partant de la théorie des architectures (Jour 2 : CNN et ViT) jusqu'à la livraison du dataset au module NLP (Jour 5).

---

## Bibliographie de référence

--
### Traitement numérique des images

- **Gonzalez, R. C., Woods, R. E.** (2018). *Digital Image Processing* (4e éd.). Pearson. : La référence encyclopédique du domaine. Chapitres 2 (Fundamentals), 3 (Intensity Transformations) et 10 (Image Segmentation) sont directement pertinents pour ce cours.

- **Bradski, G., Kaehler, A.** (2008). *Learning OpenCV: Computer Vision with the OpenCV Library*. O'Reilly Media. : Le livre de référence d'OpenCV, bien que certaines parties soient à actualiser pour l'API actuelle. La documentation officielle ([docs.opencv.org](https://docs.opencv.org)) reste la ressource la plus à jour.

- **Van der Walt, S. et al.** (2014). *scikit-image: Image Processing in Python*. PeerJ, 2, e453. : L'article de référence de scikit-image, présentant la philosophie de la bibliothèque et ses algorithmes principaux.

- **Sauvola, J., Pietikäinen, M.** (2000). *Adaptive Document Image Binarization*. Pattern Recognition, 33(2), 225–236. : L'article original du seuillage adaptatif de Sauvola, incontournable pour la binarisation de documents anciens.

- **Otsu, N.** (1979). *A Threshold Selection Method from Gray-Level Histograms*. IEEE Transactions on Systems, Man, and Cybernetics, 9(1), 62–66. : L'article fondateur du seuillage automatique d'Otsu.

--
### OCR et HTR : état de l'art

- **Smith, R.** (2007). *An Overview of the Tesseract OCR Engine*. Ninth International Conference on Document Analysis and Recognition (ICDAR). : Présentation de Tesseract, le moteur OCR open source de référence, et de ses limites sur les documents non standards.

- **Breuel, T. M.** (2008). *The OCRopus Open Source OCR System*. Document Recognition and Retrieval XV, 6815. : OCRopus a introduit l'approche par réseau de neurones récurrents pour l'OCR/HTR, posant les bases des approches modernes.

- **Li, M., Lyu, T., Yao, T., Ye, J., Lu, T., Zheng, Y., Huang, F.** (2021). *TrOCR: Transformer-based Optical Character Recognition with Pre-trained Models*. AAAI 2023. [arXiv:2109.10282] : L'article de référence de TrOCR.

- **Kiessling, B.** (2019). *Kraken : an Universal Text Recognizer for the Humanities*. Digital Humanities Conference (DH2019). : Présentation synthétique de Kraken par son auteur, avec les motivations de conception.

--
### Modèles fondateurs du pipeline

- **Dosovitskiy, A. et al.** (2020). *An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale*. ICLR 2021. [arXiv:2010.11929] : ViT, la brique de base de TrOCR et de DINO.

- **Caron, M. et al.** (2021). *Emerging Properties in Self-Supervised Vision Transformers*. ICCV 2021. [arXiv:2104.14294] : DINO.

- **Radford, A. et al.** (2021). *Learning Transferable Visual Models From Natural Language Supervision*. ICML 2021. : CLIP.

- **Kirillov, A. et al.** (2023). *Segment Anything*. ICCV 2023. [arXiv:2304.02643] : SAM.

--
### Ressources pratiques

- **Documentation HuggingFace Transformers** : [huggingface.co/docs/transformers](https://huggingface.co/docs/transformers). Pour TrOCR, CLIP, DINO et tous les modèles pré-entraînés.

- **Documentation Kraken** : [kraken.re](https://kraken.re). Guide complet d'installation, entraînement et utilisation.

- **Documentation SAM** : [github.com/facebookresearch/segment-anything](https://github.com/facebookresearch/segment-anything). Avec notebooks d'exemples.
