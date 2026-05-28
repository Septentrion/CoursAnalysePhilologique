# Cours 4.1 — Prétraitement de scans de manuscrits

**Module 4 · Computer Vision appliquée aux manuscrits médiévaux · MD5**

---

> *« Un scan de manuscrit médiéval, c'est d'abord une photographie d'un objet physique vieux de plusieurs siècles. Avant même de penser à reconnaître du texte, il faut comprendre ce que le capteur a capturé — et ce qui, dans cette image, n'est pas du texte mais de l'histoire matérielle du document. »*

---

## Introduction : le fossé entre le capteur et le modèle

Les Jours 2 et 3 ont présenté des modèles capables de performances remarquables sur des images « propres » : des scans bien éclairés, redressés, à fort contraste. La réalité d'un corpus de manuscrits médiévaux est très différente.

Un scan issu de Gallica ou de la BVMM (Bibliothèque virtuelle des manuscrits médiévaux) arrive dans le pipeline avec ses défauts : la page n'est pas parfaitement horizontale, le parchemin a jauni de façon non uniforme, l'encre du verso se devine par transparence, une tache d'humidité couvre le quart inférieur droit, et la reliure projette une ombre sur les premières lettres de chaque ligne. Ces artefacts ne sont pas de l'information sur le texte — ce sont des traces de l'histoire physique du document.

**L'objectif du prétraitement est précis :** transformer l'image brute en une image dont les propriétés photométriques et géométriques favorisent la reconnaissance automatique du texte, tout en préservant l'information graphique utile à cette reconnaissance.

Ce chapitre est organisé autour d'une tension fondamentale : **chaque opération de prétraitement améliore certaines propriétés de l'image au prix d'en dégrader d'autres**. Comprendre ces compromis, c'est comprendre pourquoi le prétraitement n'est pas une recette universelle mais une série de décisions contextuelles.

---

## 1. Anatomie d'un scan de manuscrit médiéval

Avant d'aborder les algorithmes, il faut comprendre ce qu'un scan capture — et pourquoi cette image pose problème à un modèle HTR.

### 1.1 La formation de l'image : ce que le capteur voit

Un scanner ou un appareil photo numérique capture l'intensité lumineuse réfléchie par la surface du document dans chaque cellule de son capteur. Ce processus introduit déjà plusieurs sources de bruit :

**L'éclairage non uniforme.** Dans un scanner à plat, les lampes illuminent la page de façon légèrement asymétrique. Les bords reçoivent moins de lumière que le centre. Pour les manuscrits photographiés en bibliothèque avec des éclairages directionnels, les zones près de la reliure sont systématiquement plus sombres.

**La réponse spectrale du capteur.** Le capteur numérique est sensible à trois canaux (rouge, vert, bleu) avec des sensibilités différentes. L'encre ferro-gallique médiévale, qui a vieilli en prenant une teinte brune-dorée, reflète différemment selon le canal — ce que l'œil humain compense intuitivement, mais que le modèle doit apprendre à gérer.

**Le bruit du capteur.** Même pour une surface parfaitement uniforme, la mesure pixel par pixel fluctue légèrement (bruit de lecture, bruit thermique). Sur un parchemin vieilli avec une encre à faible contraste, ce bruit peut représenter une fraction non négligeable du signal.

### 1.2 Les propriétés physiques spécifiques au parchemin

Le parchemin (*vellum* pour le veau, *parchment* pour le mouton ou la chèvre) présente des propriétés optiques radicalement différentes du papier moderne, dont les conséquences pour la numérisation sont souvent sous-estimées.

**La translucidité.** Le parchemin est un matériau semi-transparent. Quand une page est photographiée en lumière ambiante non directionnelle, la lumière traverse partiellement la feuille et transporte l'image de l'encre du verso. On appelle ce phénomène le ***show-through*** (transparence) ou l'*éclairement en transmission*. Le résultat dans l'image : des traits fantômes, visuellement similaires aux traits d'encre du recto, mais correspondant au texte du verso lu en miroir.

Pour un algorithme de segmentation ou d'HTR, ces traits fantômes sont indiscernables des vrais traits au niveau local. Ils créent de faux positifs dans la détection de texte et de fausses lettres dans la reconnaissance.

**Le vieillissement et le jaunissement.** Le parchemin vieilli prend une teinte variant du crème clair au brun foncé, selon l'humidité, la lumière et la composition chimique des peaux utilisées. Ce jaunissement est rarement uniforme sur une page : les bords exposés à l'air vieillissent plus vite que le centre, les zones proches des marges varient selon les conditions de conservation. Le fond n'est donc pas une constante — c'est un gradient spatial qui complique la séparation encre/fond.

**Les dégradations locales.** Sur sept siècles, un parchemin peut avoir subi : des taches d'humidité (qui laissent des halos bruns caractéristiques), des moisissures (qui dissolvent localement l'encre), des insectes (piqûres d'insectes qui perforent la feuille), des grattages (le scribe qui corrige en grattant la surface), des réparations (coutures, colmatage de déchirures avec de la colle animale). Chacune de ces dégradations crée des artefacts visuels locaux qui interfèrent avec la lecture automatique.

### 1.3 Les propriétés spécifiques à l'encre médiévale

**L'encre ferro-gallique.** L'encre la plus courante dans les manuscrits médiévaux est fabriquée à partir de sulfate de fer (vitriol vert) et d'acide tannique (extrait des galles de chêne). Fraîche, elle est noire. En vieillissant, elle s'oxyde et prend une teinte brun-dorée. Mais surtout, elle est corrosive : l'oxydation du fer dans l'encre détruit progressivement les fibres du parchemin. Les zones d'encre dense peuvent se retrouver physiquement percées ou fragilisées — l'encre a littéralement mangé le support. Sur le scan, ces zones apparaissent comme des irrégularités dans la texture de l'encre.

**Les encres colorées.** Les rubriques (titres en rouge), les lettrines (initiales en rouge et bleu), et les enluminures utilisent des pigments minéraux ou organiques différents de l'encre principale : cinabre (rouge vif), azurite (bleu), minium (rouge orangé), or en feuille. Ces pigments ont des propriétés spectrales très différentes de l'encre ferro-gallique et réagissent différemment aux algorithmes de binarisation globale. L'or, en particulier, peut saturer le capteur sur les scans haute résolution — apparaissant comme des zones brillantes blanches qui masquent le texte environnant.

**Les marques d'écriture effacées.** Les palimpsestes (parchemins grattés et réutilisés) peuvent conserver des traces du texte précédent, visibles en imagerie multispectrale (ultraviolet, infrarouge) mais partiellement visibles aussi dans les scans ordinaires. Ces traces constituent un bruit supplémentaire pour les modèles HTR.

---

## 2. La déformation géométrique et sa correction

### 2.1 Les sources de déformation

Un scan de manuscrit n'est presque jamais géométriquement parfait. Plusieurs phénomènes contribuent à déformer l'image par rapport à la réalité physique du document :

**L'inclinaison de la page.** Lors de la photographie, la page n'est pas toujours parfaitement horizontale. Une inclinaison de 2 à 5 degrés est fréquente, et certains scans de bibliothèques non numérisées atteignent 10 degrés. Un modèle HTR entraîné sur des lignes horizontales voit ses performances chuter significativement sur des lignes inclinées — même légèrement.

**La distorsion perspective.** Quand la page est photographiée depuis un angle, les lignes parallèles du document convergent dans l'image (effet de perspective). Cela affecte particulièrement les manuscrits photographiés sur pupitre ou avec un capteur légèrement incliné.

**La courbure de la reliure.** Pour les codex (livres), les pages proches de la reliure (la zone centrale quand le livre est ouvert) sont courbées. Cette courbure crée une distorsion non linéaire : les lignes de texte qui suivent la courbure apparaissent en arc dans l'image plane. Les premières et dernières lettres de chaque ligne sont « tirées » vers le bord intérieur.

**Le déformations du parchemin.** Le parchemin humide puis séché se déforme. Sur des siècles, les feuillets peuvent onduler, bomber, ou se rétracter de façon complexe. Ces déformations locales sont irréductibles par des transformations géométriques globales.

### 2.2 Le deskewing : correction d'inclinaison

Le *deskewing* (ou *redressement*) corrige l'inclinaison globale de la page en l'estimant puis en appliquant une rotation inverse.

**Estimation par projection horizontale**

L'idée : dans une image de texte horizontale, les lignes de texte créent des « pics » dans la projection verticale (somme des pixels sombres par ligne horizontale). Si la page est inclinée, ces pics s'aplatissent. On cherche l'angle de rotation qui maximise la variance de cette projection — ce qui correspond aux pics les plus nets.

```python
import numpy as np
from PIL import Image
import cv2

def estimer_angle_inclinaison(
    image_gris: np.ndarray,
    plage_angles: tuple = (-15, 15),
    pas: float = 0.5
) -> float:
    """
    Estime l'angle d'inclinaison d'une image de texte manuscrit
    par maximisation de la variance de la projection horizontale.

    Args:
        image_gris   : image en niveaux de gris (0–255)
        plage_angles : intervalle d'angles testés (degrés)
        pas          : résolution angulaire (degrés)

    Returns:
        angle_optimal : angle d'inclinaison estimé (degrés)
    """
    # Binarisation préalable (Otsu) pour n'avoir que du noir et blanc
    _, binaire = cv2.threshold(
        image_gris, 0, 255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    h, w   = binaire.shape
    angles = np.arange(plage_angles[0], plage_angles[1] + pas, pas)
    variances = []

    for angle in angles:
        # Rotation de l'image binaire
        M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
        rotee = cv2.warpAffine(binaire, M, (w, h),
                               flags=cv2.INTER_NEAREST,
                               borderValue=0)
        # Projection horizontale : somme des pixels sombres par ligne
        projection = rotee.sum(axis=1)
        variances.append(projection.var())

    # L'angle optimal maximise la variance (pics les plus nets)
    idx_optimal = np.argmax(variances)
    return float(angles[idx_optimal])


def redresser_image(
    image: np.ndarray,
    angle: float,
    couleur_fond: int = 255   # Blanc pour les scans
) -> np.ndarray:
    """
    Applique une rotation de correction d'inclinaison.
    Remplit les zones découvertes avec la couleur du fond.
    """
    h, w = image.shape[:2]
    centre = (w / 2, h / 2)
    M = cv2.getRotationMatrix2D(centre, angle, 1.0)

    # Calculer les dimensions de l'image après rotation (sans cropping)
    cos_a = abs(M[0, 0])
    sin_a = abs(M[0, 1])
    new_w = int(h * sin_a + w * cos_a)
    new_h = int(h * cos_a + w * sin_a)
    M[0, 2] += (new_w - w) / 2
    M[1, 2] += (new_h - h) / 2

    redressee = cv2.warpAffine(
        image, M, (new_w, new_h),
        flags=cv2.INTER_LANCZOS4,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=couleur_fond
    )
    return redressee


# Pipeline de correction d'inclinaison
def pipeline_deskew(image_pil: Image.Image) -> tuple[Image.Image, float]:
    """
    Redresse une image de page de manuscrit.
    Retourne (image_redressée, angle_corrigé).
    """
    image_np = np.array(image_pil.convert("L"))

    angle = estimer_angle_inclinaison(image_np, plage_angles=(-10, 10), pas=0.3)
    print(f"  Angle d'inclinaison détecté : {angle:.1f}°")

    if abs(angle) < 0.3:
        print("  Inclinaison négligeable — image conservée telle quelle")
        return image_pil, angle

    image_redressee_np = redresser_image(np.array(image_pil), angle)
    return Image.fromarray(image_redressee_np), angle
```

**Limitation importante :** la méthode de projection fonctionne bien pour des inclinaisons globales uniformes. Elle échoue sur les déformations locales (courbure de reliure, parchemin ondulé) qui nécessitent des approches non linéaires — comme la correction par spline ou les méthodes de *document dewarping* apprises (DewarpNet, DocUNet).

### 2.3 Correction de la distorsion de perspective

Pour les pages photographiées en biais, une transformation homographique peut redresser la perspective. Cela nécessite de détecter les quatre coins de la page dans l'image (ou des lignes de référence) et d'appliquer une transformation projective.

```python
def corriger_perspective(
    image: np.ndarray,
    coins_source: np.ndarray,   # 4 points dans l'image brute (ordre : HG, HD, BD, BG)
    largeur_cible: int = 2000,
    hauteur_cible: int = 3000
) -> np.ndarray:
    """
    Corrige la distorsion de perspective par transformation homographique.
    coins_source : coordonnées des 4 coins de la page dans l'image.
    Peut être détecté automatiquement (contour de la page) ou fourni manuellement.
    """
    coins_cible = np.array([
        [0,               0],
        [largeur_cible-1, 0],
        [largeur_cible-1, hauteur_cible-1],
        [0,               hauteur_cible-1]
    ], dtype=np.float32)

    H = cv2.getPerspectiveTransform(
        coins_source.astype(np.float32),
        coins_cible
    )
    corrigee = cv2.warpPerspective(
        image, H, (largeur_cible, hauteur_cible),
        flags=cv2.INTER_LANCZOS4,
        borderValue=255
    )
    return corrigee
```

---

## 3. La binarisation : de l'image en niveaux de gris à l'image binaire

### 3.1 Pourquoi binariser ?

La binarisation réduit chaque pixel à deux états : encre (noir, valeur 0) ou fond (blanc, valeur 255). Cette simplification présente plusieurs avantages pour le pipeline HTR :

- **Réduction du bruit.** Les variations de teinte dans le fond (parchemin non uniforme, légère coloration) sont effacées. Le modèle HTR ne voit que la forme des tracés.
- **Compatibilité.** Les modèles de segmentation de lignes (Kraken BLLA notamment) attendent des images binarisées ou à fort contraste.
- **Réduction de la charge computationnelle.** Une image binaire nécessite 8× moins de mémoire qu'une image RGB.

Mais la binarisation a aussi des coûts :

- **Perte d'information spectrale.** Les rubriques rouges deviennent noires, indiscernables du texte principal. Pour les pipelines qui exploitent la couleur pour détecter les rubriques (via HSV), la binarisation doit être appliquée *après* cette détection.
- **Artefacts sur les zones dégradées.** Là où l'encre est pâlie, le seuillage peut la faire disparaître. Là où le fond est sali, le seuillage peut créer du bruit qui ressemble à de l'encre.

### 3.2 La méthode d'Otsu : seuillage global optimal

Présentée en section 1.4, la méthode d'Otsu cherche le seuil $t^*$ qui minimise la variance intra-classe des deux populations (pixels sombres et pixels clairs) :

$$t^* = \argmin_t \left[ w_0(t) \sigma_0^2(t) + w_1(t) \sigma_1^2(t) \right]$$

où $w_0, w_1$ sont les proportions de pixels dans chaque classe et $\sigma_0^2, \sigma_1^2$ leurs variances.

**Quand Otsu fonctionne :** histogramme clairement bimodal (deux pics distincts correspondant à l'encre et au fond). Éclairage uniforme. Image à fort contraste.

**Quand Otsu échoue :** éclairage non uniforme (coin sombre de la reliure), fond très variable (parchemin de teintes hétérogènes), zones dégradées (encre pâlie, taches). Dans ces cas, un seuil global unique ne peut pas être correct partout sur la page.

```python
def binariser_otsu(image_gris: np.ndarray) -> tuple[np.ndarray, int]:
    """Binarisation par méthode d'Otsu. Retourne (image_binaire, seuil)."""
    seuil, binaire = cv2.threshold(
        image_gris, 0, 255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )
    print(f"  Seuil Otsu : {seuil:.0f}/255")
    return binaire, int(seuil)
```

### 3.3 La méthode de Sauvola : seuillage adaptatif local

La méthode de Sauvola (2000) calcule un seuil différent pour chaque pixel, en fonction des statistiques locales dans une fenêtre de voisinage. Pour le pixel $(i, j)$, le seuil est :

$$t(i, j) = \mu(i, j) \cdot \left[1 + k \cdot \left(\frac{\sigma(i, j)}{R} - 1\right)\right]$$

où :
- $\mu(i, j)$ est la **moyenne locale** des niveaux de gris dans une fenêtre de taille $w \times w$ autour du pixel $(i,j)$
- $\sigma(i, j)$ est l'**écart-type local** dans la même fenêtre
- $R$ est la valeur maximale de l'écart-type (128 pour des images 8 bits, par convention)
- $k \in [0.2, 0.5]$ est un paramètre de sensibilité

**Intuition :** dans une zone de fond uniforme, $\sigma$ est faible et le seuil $t$ est proche de $\mu$ (seuil « laxiste »). Dans une zone de texte avec fort contraste, $\sigma$ est élevé et le seuil est plus bas (seuil « strict »). Le paramètre $k$ contrôle la pondération de $\sigma$ — un $k$ élevé rend le seuil plus sensible aux variations locales.

```python
from skimage.filters import threshold_sauvola

def binariser_sauvola(
    image_gris: np.ndarray,
    window_size: int = 25,
    k: float = 0.2
) -> np.ndarray:
    """
    Binarisation adaptative de Sauvola.
    Recommandée pour les manuscrits sur parchemin vieilli.

    Args:
        image_gris  : image en niveaux de gris
        window_size : taille de la fenêtre locale (pixels)
                      Régler à environ 1/20 de la hauteur de l'image
                      pour les manuscrits (typiquement 15–35)
        k           : sensibilité au contraste local (0.2–0.5)
                      Augmenter si l'encre pâlie est perdue,
                      diminuer si le fond sale crée du bruit

    Returns:
        image binaire (255 = encre, 0 = fond) selon convention HTR
    """
    seuil = threshold_sauvola(image_gris, window_size=window_size, k=k)
    binaire = (image_gris < seuil).astype(np.uint8) * 255
    return binaire


def choisir_binarisation(
    image_gris: np.ndarray,
    seuil_bimodalite: float = 0.5
) -> np.ndarray:
    """
    Choisit automatiquement entre Otsu et Sauvola selon le profil de l'histogramme.
    Un histogramme bimodal → Otsu ; non bimodal → Sauvola.

    La bimodalité est estimée par l'indice de bimodalité de Sarle :
    B = (skewness² + 1) / kurtosis — valeur > 0.555 → bimodal
    """
    from scipy.stats import skew, kurtosis as kurt

    hist = image_gris.flatten()
    skewness  = skew(hist)
    kurtosis_val = kurt(hist, fisher=True)
    bimodalite = (skewness**2 + 1) / (kurtosis_val + 3 + 1e-8)

    if bimodalite > seuil_bimodalite:
        print(f"  Histogramme bimodal (B={bimodalite:.3f}) → Otsu")
        binaire, _ = binariser_otsu(image_gris)
    else:
        print(f"  Histogramme non bimodal (B={bimodalite:.3f}) → Sauvola")
        binaire = binariser_sauvola(image_gris, window_size=25, k=0.2)

    return binaire
```

### 3.4 Pourquoi ne pas toujours binariser

Pour certaines étapes du pipeline, travailler sur l'image en niveaux de gris (voire en couleur) est préférable à la binarisation :

- **Détection des rubriques rouges** (section 3.2 du cours, via HSV) : la binarisation efface la distinction rouge/noir.
- **CLIP pour la description d'illustrations** : CLIP attend des images RGB. La binarisation dégraderait ses performances.
- **DINOv2 pour le clustering** : les features DINOv2 sur une image en niveaux de gris sont moins riches que sur une image couleur.

La binarisation est donc appliquée **localement**, juste avant les étapes qui en ont besoin (segmentation de lignes Kraken, certains modèles HTR légers), et non comme première transformation globale irréversible du pipeline.

---

## 4. L'amélioration du contraste : CLAHE

### 4.1 Le problème de l'éclairage non uniforme

Un manuscrit photographié en bibliothèque présente presque systématiquement des variations d'éclairage à grande échelle :

- **Vignettage** : les bords de l'image reçoivent moins de lumière que le centre (effet optique de la plupart des objectifs).
- **Ombre de la reliure** : pour un codex ouvert, la zone près de la couture est dans l'ombre de la page opposée.
- **Éclairage directionnel** : une lampe placée d'un côté crée un gradient lumineux de gauche à droite.

Ces variations d'éclairage font qu'un seuil global est insuffisant : le niveau de gris de l'encre dans la zone sombre peut être identique au niveau de gris du fond dans la zone claire. L'égalisation d'histogramme cherche à corriger cela.

### 4.2 L'égalisation d'histogramme globale et ses limites

L'égalisation d'histogramme globale (*Histogram Equalization*, HE) remapped la distribution des niveaux de gris pour qu'elle soit uniforme sur [0, 255]. Elle améliore le contraste global mais amplifie également le bruit dans les zones à faible signal — exactement ce qu'on veut éviter sur les zones dégradées d'un manuscrit.

### 4.3 CLAHE : égalisation adaptative avec limitation du contraste

**CLAHE** (*Contrast Limited Adaptive Histogram Equalization*) divise l'image en petites tuiles et applique une égalisation d'histogramme localement dans chaque tuile, avec une contrainte de limitation du contraste pour éviter l'amplification du bruit.

**Le paramètre `clipLimit`** définit la valeur maximale du contraste amplifié. Les parties de l'histogramme qui dépassent ce seuil sont redistribuées uniformément — ce qui évite les artefacts visuels caractéristiques de l'égalisation globale (zones de bruit amplifiées, halos autour des traits).

**Le paramètre `tileGridSize`** définit la taille des tuiles. Des tuiles trop petites produisent des artefacts aux frontières ; des tuiles trop grandes reviennent à une égalisation globale.

```python
def ameliorer_contraste_clahe(
    image_gris: np.ndarray,
    clip_limit: float = 2.0,
    tile_grid_size: tuple = (8, 8)
) -> np.ndarray:
    """
    Amélioration adaptative du contraste par CLAHE.

    Args:
        clip_limit     : limite de l'amplification du contraste
                         Faible (1.5–2.0) : effet modéré, risque d'artefacts réduit
                         Élevé (3.0–5.0)  : effet fort, risque d'artefacts accru
        tile_grid_size : nombre de tuiles (horizontal × vertical)
                         (8, 8) est un bon défaut pour des images 1024 × 1400

    Returns:
        image améliorée (uint8)
    """
    clahe = cv2.createCLAHE(
        clipLimit=clip_limit,
        tileGridSize=tile_grid_size
    )
    return clahe.apply(image_gris)


def ameliorer_contraste_couleur_clahe(
    image_rgb: np.ndarray,
    clip_limit: float = 2.0,
    tile_grid_size: tuple = (8, 8)
) -> np.ndarray:
    """
    Applique CLAHE sur le canal L de l'espace colorimétrique LAB.
    Améliore la luminosité sans déformer les couleurs (important pour les rubriques).
    """
    # Conversion RGB → LAB (L = luminosité, A et B = chrominance)
    lab = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)

    # CLAHE uniquement sur le canal de luminosité
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    l_ameliore = clahe.apply(l)

    # Reconstruction et retour en RGB
    lab_ameliore = cv2.merge([l_ameliore, a, b])
    return cv2.cvtColor(lab_ameliore, cv2.COLOR_LAB2RGB)
```

**Pourquoi appliquer CLAHE sur le canal L en espace LAB ?**

L'espace colorimétrique LAB sépare la **luminosité** (canal L) des **informations de couleur** (canaux A et B). En appliquant CLAHE uniquement sur L, on améliore le contraste sans modifier les teintes — les rubriques rouges restent rouges, les enluminures bleues restent bleues. Si on applique CLAHE directement sur les canaux RGB, la redistribution des intensités modifie les rapports entre canaux et fausse les couleurs.

---

## 5. La réduction du bruit

### 5.1 Les types de bruit dans les scans de manuscrits

**Bruit sel-et-poivre** (*salt-and-pepper noise*) : pixels isolés anormalement clairs ou sombres. Origine : poussières sur le scanner, défauts du capteur, petites perforations dans le parchemin. Caractérisé par des pics dans l'histogramme aux extrémités (0 et 255).

**Bruit gaussien** : fluctuations aléatoires de faible amplitude affectant tous les pixels. Origine : bruit thermique du capteur, bruit de lecture, compression JPEG. Visible comme un « grain » uniforme sur les zones de fond.

**Bruit structuré** : artefacts avec une organisation spatiale — quadrillage de compression JPEG, lignes de réglure, show-through du verso. Différent des deux types précédents car non aléatoire.

### 5.2 Le filtre médian : efficace contre le sel-et-poivre

Le filtre médian remplace chaque pixel par la **valeur médiane** de son voisinage. Contrairement au filtre de moyenne (qui ferait une somme pondérée et « étalerait » les pics), le médian est robuste aux valeurs extrêmes — un pixel isolé à 0 ou 255 dans un voisinage de 8 pixels à 180 sera remplacé par 180, sans affecter le voisinage.

```python
def filtrer_bruit_sel_poivre(
    image: np.ndarray,
    ksize: int = 3
) -> np.ndarray:
    """
    Filtre médian pour éliminer le bruit sel-et-poivre.
    ksize : taille du noyau (3 ou 5 — toujours impair)
    Note : ksize=5 élimine les taches plus larges mais commence à flouter les fins traits.
    """
    return cv2.medianBlur(image, ksize)
```

**Précaution sur les manuscripts :** les fins traits d'écriture (les déliés des lettres gothiques, les points diacritiques) ont des dimensions de 1 à 3 pixels. Un filtre médian avec `ksize=5` peut les effacer. Utiliser `ksize=3` de préférence.

### 5.3 Le filtre gaussien : pour le bruit de fond

Le filtre gaussien convole l'image avec un noyau gaussien. Il atténue le bruit de haute fréquence (variations pixel à pixel) mais floute également les contours. C'est un compromis entre réduction du bruit et préservation des détails.

```python
def filtrer_bruit_gaussien(
    image: np.ndarray,
    sigma: float = 0.8
) -> np.ndarray:
    """
    Filtre gaussien léger pour le bruit de fond.
    sigma < 1.0 : effet très léger, préserve la majorité des détails.
    sigma > 2.0 : flou significatif, perd les fins traits.
    """
    return cv2.GaussianBlur(image, (0, 0), sigmaX=sigma, sigmaY=sigma)
```

### 5.4 Opérations morphologiques : nettoyer après la binarisation

Les opérations morphologiques travaillent sur les images binaires. Elles permettent de nettoyer le résultat de la binarisation en éliminant de petits artefacts ou en connectant des traits brisés.

**L'ouverture morphologique** (érosion suivie d'une dilatation) élimine les petits objets isolés (bruit, pixels parasites) sans modifier les objets plus grands.

**La fermeture morphologique** (dilatation suivie d'une érosion) connecte des traits proches ou comble de petites lacunes dans les traits d'encre.

```python
def nettoyer_binarisation(
    image_binaire: np.ndarray,
    taille_bruit_max: int = 2,
    connecter_traits: bool = True
) -> np.ndarray:
    """
    Nettoyage post-binarisation par opérations morphologiques.

    Args:
        taille_bruit_max : taille maximale (en pixels) des artefacts à supprimer
        connecter_traits : si True, tente de connecter les traits brisés
    """
    kernel_bruit = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (taille_bruit_max * 2 + 1, taille_bruit_max * 2 + 1)
    )
    # Ouverture : supprime les petits artefacts
    nettoyee = cv2.morphologyEx(image_binaire, cv2.MORPH_OPEN, kernel_bruit)

    if connecter_traits:
        kernel_connexion = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 2))
        # Fermeture légère : connecte les traits brisés verticalement (entre ascendantes)
        nettoyee = cv2.morphologyEx(nettoyee, cv2.MORPH_CLOSE, kernel_connexion)

    return nettoyee
```

---

## 6. Le show-through : traiter la transparence du parchemin

### 6.1 Le problème en détail

La transparence du parchemin est l'une des sources de bruit les plus difficiles à traiter dans les manuscrits médiévaux. Quand la lumière traverse la feuille, elle transporte l'image du texte du verso — lue en miroir depuis le recto. Sur le scan, cela se traduit par des traits grisés qui ressemblent à de l'encre mais ne correspondent à aucun texte du recto.

La difficulté fondamentale : au niveau local, un pixel de show-through est souvent indiscernable d'un pixel d'encre pâlie du recto. La distinction n'est possible que de façon contextuelle ou avec un scan du verso.

### 6.2 Approche par soustraction du fond estimé

Quand on dispose du scan du verso, on peut construire une estimation du show-through et le soustraire :

```python
def attenuer_show_through(
    recto: np.ndarray,
    verso: np.ndarray,
    alpha: float = 0.3
) -> np.ndarray:
    """
    Atténue le show-through en soustrayant une version pondérée du verso retourné.
    Nécessite les scans recto ET verso du même feuillet.

    Args:
        recto, verso : images en niveaux de gris (même dimensions)
        alpha        : coefficient de soustraction (0.1–0.4 selon la transparence)

    Returns:
        image recto avec show-through atténué
    """
    # Retourner le verso horizontalement (miroir)
    verso_miroir = cv2.flip(verso, 1)

    # Soustraire une fraction du verso (les forts contrastes = encre verso)
    recto_float   = recto.astype(np.float32)
    verso_float   = verso_miroir.astype(np.float32)
    corrige_float = recto_float + alpha * (255 - verso_float)
    corrige = np.clip(corrige_float, 0, 255).astype(np.uint8)
    return corrige
```

**Quand le verso n'est pas disponible.** Dans la majorité des cas pratiques, on ne dispose que du scan recto. On peut alors utiliser des approches de filtrage en fréquence : le show-through tend à avoir une signature fréquentielle différente du texte principal (légèrement plus flou, contraste plus faible). Des méthodes de décomposition aveugle de sources (ICA, NMF) ont été proposées dans la littérature, mais restent peu robustes en pratique.

**Recommandation pragmatique :** si le show-through est léger (transparence < 20% de l'intensité de l'encre du recto), un seuillage de Sauvola avec un `k` légèrement réduit (0.15 au lieu de 0.2) suffit souvent à l'éliminer dans la binarisation. Si le show-through est sévère, le traitement doit être signalé dans le data contract comme cas limite nécessitant une révision humaine.

---

## 7. Gestion des cas spéciaux

### 7.1 Les enluminures et les zones saturation

Les feuilles d'or et les pigments brillants saturent le capteur : leur intensité dépasse 255, et dans l'image numérique, ces zones apparaissent comme des plages uniformément blanches. Toute information sur le texte environnant est perdue.

Le diagnostic est simple : des zones rectangulaires parfaitement blanches (valeur 255 sur tous les canaux RGB) là où se trouvent des ornements dorés ou argentés sont caractéristiques de la saturation. Ces zones doivent être :

1. **Détectées** : par seuillage haut (pixels > 250 sur tous les canaux).
2. **Signalées** : dans le JSON de sortie, comme zones non transcriptibles.
3. **Exclues** du prétraitement intense (pas de CLAHE, pas de binarisation forcée).

```python
def detecter_zones_saturees(
    image_rgb: np.ndarray,
    seuil_saturation: int = 250,
    surface_min: int = 100
) -> list[dict]:
    """
    Détecte les zones saturées (typiquement : or, argent, zones surexposées).
    Retourne la liste des régions saturées avec leurs coordonnées.
    """
    masque_sature = np.all(image_rgb >= seuil_saturation, axis=2).astype(np.uint8) * 255

    # Trouver les composantes connexes
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(masque_sature)

    zones = []
    for i in range(1, n_labels):   # 0 = fond
        surface = stats[i, cv2.CC_STAT_AREA]
        if surface >= surface_min:
            zones.append({
                "x"      : int(stats[i, cv2.CC_STAT_LEFT]),
                "y"      : int(stats[i, cv2.CC_STAT_TOP]),
                "w"      : int(stats[i, cv2.CC_STAT_WIDTH]),
                "h"      : int(stats[i, cv2.CC_STAT_HEIGHT]),
                "surface": int(surface),
                "type"   : "saturation",
            })
    return zones
```

### 7.2 La courbure de reliure et le déwarping

La déformation de page par la reliure est un problème non linéaire que les corrections géométriques simples (rotation, homographie) ne peuvent pas résoudre. Le *document dewarping* est un sujet de recherche actif.

**Approches disponibles :**

**Approche géométrique classique :** détecter les lignes de texte (par projection horizontale ou Kraken), estimer leur courbure par un polynôme de degré 2 ou une spline, et appliquer une transformation inverse qui redresse chaque ligne. Cette approche est fragile si les lignes sont peu nombreuses ou si la courbure est sévère.

**Approche par réseau neuronal (DocUNet, DewarpNet) :** un réseau U-Net prédit un champ de déformation pour chaque pixel, puis une grille de déformation inverse est appliquée. Ces modèles, entraînés sur des documents variés, généralisent bien aux manuscrits mais requièrent un GPU et un temps d'inférence non négligeable.

**Recommandation pratique pour ce projet :** pour les manuscrits bien conservés et les pages non trop courbées (courbure < 15% de la hauteur de la page), le deskewing suffit. Pour les pages fortement courbées, signaler le cas dans le data contract et traiter manuellement. L'implémentation complète du dewarping dépasse le périmètre de ce cours.

---

## 8. La simulation de dégradations : data augmentation documentaire

### 8.1 Pourquoi simuler les dégradations ?

Un modèle HTR entraîné uniquement sur des images propres (binarisées, redressées, sans show-through) sera fragile sur les images dégradées qu'il rencontrera en production. Pour le rendre robuste, on augmente le dataset d'entraînement avec des **dégradations synthétiques** : des transformations qui simulent les artefacts réels tout en restant contrôlables.

La difficulté spécifique aux documents patrimoniaux : les dégradations de manuscrits médiévaux (show-through, jaunissement, encre corrodée) sont différentes des dégradations habituellement simulées pour les images naturelles (bruit gaussien, flou, compression). Il faut donc des augmentations spécifiques au domaine.

### 8.2 Bibliothèque d'augmentations documentaires

```python
import random
from typing import Optional

class AugmenteurDocumentaire:
    """
    Bibliothèque d'augmentations simulant les dégradations
    spécifiques aux manuscripts médiévaux.
    Peut être utilisée pour augmenter les datasets d'entraînement HTR.
    """

    @staticmethod
    def simuler_jaunissement(
        image: np.ndarray,
        intensite: float = 0.3
    ) -> np.ndarray:
        """
        Simule le jaunissement du parchemin vieilli.
        Teinte l'image vers le brun-jaune caractéristique.
        """
        if image.ndim == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

        # Couleur de jaunissement : teinte brun-dorée
        teinte_jaune = np.array([[[20, 50, 120]]], dtype=np.uint8)  # BGR
        jauni = cv2.addWeighted(image, 1 - intensite,
                                np.ones_like(image) * teinte_jaune, intensite, 0)
        return np.clip(jauni, 0, 255).astype(np.uint8)

    @staticmethod
    def simuler_show_through(
        image: np.ndarray,
        image_verso: Optional[np.ndarray] = None,
        alpha: float = 0.15
    ) -> np.ndarray:
        """
        Simule la transparence du parchemin.
        Si image_verso est fourni, utilise une version miroir atténuée.
        Sinon, utilise une version miroir de l'image elle-même (approximation).
        """
        if image_verso is None:
            # Auto-show-through : miroir de l'image elle-même
            verso_approx = cv2.flip(image, 1)
        else:
            verso_approx = cv2.flip(image_verso, 1)

        # Le show-through est sombre (encre du verso) sur fond clair
        show = (255 - verso_approx) * alpha
        result = np.clip(image.astype(np.float32) - show, 0, 255).astype(np.uint8)
        return result

    @staticmethod
    def simuler_tache_humidite(
        image: np.ndarray,
        n_taches: int = 3,
        taille_max: float = 0.15
    ) -> np.ndarray:
        """
        Simule des taches d'humidité (halos bruns caractéristiques).
        Chaque tache est une ellipse avec gradient de couleur.
        """
        result = image.copy().astype(np.float32)
        h, w = image.shape[:2]

        for _ in range(n_taches):
            cx = random.randint(0, w - 1)
            cy = random.randint(0, h - 1)
            rx = random.randint(20, int(w * taille_max))
            ry = random.randint(15, int(h * taille_max))

            # Masque elliptique avec gradient
            Y, X = np.ogrid[:h, :w]
            dist = ((X - cx) / rx)**2 + ((Y - cy) / ry)**2
            masque = np.clip(1 - dist, 0, 1)

            # Coloration brun-doré
            if result.ndim == 3:
                result[:, :, 0] += masque * random.uniform(5, 20)   # B
                result[:, :, 1] += masque * random.uniform(10, 30)  # G
                result[:, :, 2] += masque * random.uniform(15, 40)  # R
            else:
                result += masque * random.uniform(10, 30)

        return np.clip(result, 0, 255).astype(np.uint8)

    @staticmethod
    def simuler_encre_pale(
        image_binaire: np.ndarray,
        fraction_pixels: float = 0.05
    ) -> np.ndarray:
        """
        Simule l'encre pâlie : supprime aléatoirement une fraction des pixels d'encre.
        Modélise les zones où l'encre a partiellement disparu par corrosion ou frottement.
        """
        result  = image_binaire.copy()
        pixels_encre = np.where(image_binaire > 128)
        n_pixels = len(pixels_encre[0])
        n_a_supprimer = int(n_pixels * fraction_pixels)

        if n_a_supprimer > 0:
            idx = np.random.choice(n_pixels, n_a_supprimer, replace=False)
            result[pixels_encre[0][idx], pixels_encre[1][idx]] = 0

        return result

    @staticmethod
    def simuler_reglure(
        image: np.ndarray,
        n_lignes: int = 30,
        couleur: int = 220   # Gris clair
    ) -> np.ndarray:
        """
        Simule les lignes de réglure (lignes directrices tracées avant l'écriture).
        Visibles sur certains manuscrits, elles constituent un bruit structuré.
        """
        result = image.copy()
        h = image.shape[0]
        espacement = h // (n_lignes + 1)

        for i in range(1, n_lignes + 1):
            y = i * espacement + random.randint(-2, 2)
            epaisseur = 1 if random.random() > 0.3 else 2
            cv2.line(result, (0, y), (image.shape[1], y),
                     couleur, epaisseur)
        return result

    def augmenter(
        self,
        image: np.ndarray,
        p_jaunissement: float = 0.4,
        p_show_through: float = 0.3,
        p_tache: float = 0.2,
        p_reglure: float = 0.3,
    ) -> np.ndarray:
        """
        Applique aléatoirement un sous-ensemble des augmentations.
        Chaque augmentation est appliquée avec sa probabilité propre.
        """
        result = image.copy()
        if random.random() < p_jaunissement:
            result = self.simuler_jaunissement(result, random.uniform(0.1, 0.4))
        if random.random() < p_show_through:
            result = self.simuler_show_through(result, alpha=random.uniform(0.05, 0.2))
        if random.random() < p_tache:
            result = self.simuler_tache_humidite(result, n_taches=random.randint(1, 4))
        if random.random() < p_reglure:
            result = self.simuler_reglure(result)
        return result
```

---

## 9. Pipeline de prétraitement complet

### 9.1 Assemblage des étapes

```python
from dataclasses import dataclass
from pathlib import Path

@dataclass
class ConfigPretraitement:
    """Paramètres configurables du pipeline de prétraitement."""
    # Deskewing
    corriger_inclinaison: bool   = True
    plage_angles: tuple          = (-10, 10)
    # Amélioration du contraste
    appliquer_clahe: bool        = True
    clahe_clip_limit: float      = 2.0
    clahe_tile_size: tuple       = (8, 8)
    # Réduction du bruit
    filtre_median: bool          = True
    median_ksize: int            = 3
    # Binarisation
    binariser: bool              = True
    methode_binarisation: str    = "auto"   # "otsu", "sauvola", "auto"
    sauvola_window: int          = 25
    sauvola_k: float             = 0.2
    # Nettoyage morphologique
    nettoyage_morpho: bool       = True
    taille_bruit_max: int        = 2


def pretraiter_scan(
    chemin_image: str,
    config: ConfigPretraitement = None,
    sauvegarder_etapes: bool = False,
    dossier_sortie: str = "preprocessed"
) -> dict[str, np.ndarray]:
    """
    Pipeline complet de prétraitement d'un scan de manuscrit.

    Returns:
        Dictionnaire {nom_etape: image_numpy}
        Les étapes intermédiaires sont disponibles pour le diagnostic.
    """
    if config is None:
        config = ConfigPretraitement()

    Path(dossier_sortie).mkdir(exist_ok=True)
    etapes = {}

    # ── Chargement ──────────────────────────────────────────────────────────────
    image_pil = Image.open(chemin_image).convert("RGB")
    image_rgb = np.array(image_pil)
    image_gris = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    etapes["original"] = image_rgb
    print(f"\n=== Prétraitement : {Path(chemin_image).name} ===")
    print(f"  Dimensions : {image_rgb.shape[1]} × {image_rgb.shape[0]} pixels")

    # ── Détection des zones saturées ────────────────────────────────────────────
    zones_saturees = detecter_zones_saturees(image_rgb)
    if zones_saturees:
        print(f"  ⚠ {len(zones_saturees)} zone(s) saturée(s) détectée(s) "
              f"(or, enluminures)")

    # ── Correction de la distorsion chromatique ─────────────────────────────────
    # (Optionnel — dépend de l'objectif)
    image_gris_ameliore = image_gris.copy()

    # ── Amélioration du contraste (CLAHE sur canal L) ──────────────────────────
    if config.appliquer_clahe:
        image_rgb = ameliorer_contraste_couleur_clahe(
            image_rgb,
            clip_limit=config.clahe_clip_limit,
            tile_grid_size=config.clahe_tile_size
        )
        image_gris_ameliore = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
        etapes["clahe"] = image_rgb
        print("  CLAHE appliqué")

    # ── Correction d'inclinaison ─────────────────────────────────────────────
    if config.corriger_inclinaison:
        image_pil_corr, angle = pipeline_deskew(
            Image.fromarray(image_gris_ameliore)
        )
        image_gris_ameliore = np.array(image_pil_corr)
        image_rgb = cv2.cvtColor(image_gris_ameliore, cv2.COLOR_GRAY2RGB)
        etapes["deskew"] = image_rgb

    # ── Réduction du bruit ──────────────────────────────────────────────────────
    if config.filtre_median:
        image_gris_debruite = filtrer_bruit_sel_poivre(
            image_gris_ameliore, ksize=config.median_ksize
        )
        etapes["denoised"] = image_gris_debruite
    else:
        image_gris_debruite = image_gris_ameliore

    # ── Binarisation ────────────────────────────────────────────────────────────
    if config.binariser:
        if config.methode_binarisation == "otsu":
            binaire, _ = binariser_otsu(image_gris_debruite)
        elif config.methode_binarisation == "sauvola":
            binaire = binariser_sauvola(image_gris_debruite,
                                        window_size=config.sauvola_window,
                                        k=config.sauvola_k)
        else:  # "auto"
            binaire = choisir_binarisation(image_gris_debruite)

        if config.nettoyage_morpho:
            binaire = nettoyer_binarisation(binaire,
                                            taille_bruit_max=config.taille_bruit_max)

        etapes["binarised"] = binaire
        print(f"  Binarisation : {config.methode_binarisation}")

    # ── Sauvegarde optionnelle des étapes ────────────────────────────────────
    if sauvegarder_etapes:
        nom = Path(chemin_image).stem
        for nom_etape, img in etapes.items():
            chemin = f"{dossier_sortie}/{nom}_{nom_etape}.png"
            Image.fromarray(img).save(chemin)
            print(f"  Sauvegardé : {chemin}")

    etapes["zones_saturees"] = zones_saturees
    return etapes
```

### 9.2 Critères de qualité d'un scan

Avant de passer un scan dans le pipeline HTR, il est utile d'évaluer sa qualité. Ces métriques permettent de filtrer automatiquement les scans trop dégradés pour être traités, et de les signaler pour une révision manuelle.

```python
def evaluer_qualite_scan(image_gris: np.ndarray) -> dict[str, float]:
    """
    Calcule des métriques de qualité pour un scan de manuscrit.
    Aide à identifier les scans trop dégradés pour une transcription automatique fiable.
    """
    # ── Contraste global : plage dynamique effective ─────────────────────────
    p5, p95 = np.percentile(image_gris, [5, 95])
    contraste = float(p95 - p5) / 255.0

    # ── Netteté : variance du Laplacien (plus élevé = plus net) ─────────────
    laplacien = cv2.Laplacian(image_gris, cv2.CV_64F)
    nettete = float(laplacien.var())

    # ── Uniformité du fond : écart-type des pixels clairs ───────────────────
    fond_mask = image_gris > np.percentile(image_gris, 75)
    uniformite_fond = float(1.0 - image_gris[fond_mask].std() / 255.0)

    # ── Fraction de pixels extrêmes (saturation + sous-exposition) ──────────
    frac_sature    = float(np.mean(image_gris >= 253))
    frac_noir_pur  = float(np.mean(image_gris <= 2))

    metriques = {
        "contraste"       : round(contraste, 3),
        "nettete"         : round(nettete, 1),
        "uniformite_fond" : round(uniformite_fond, 3),
        "frac_saturee"    : round(frac_sature, 3),
        "frac_bruit_noir" : round(frac_noir_pur, 3),
    }

    # Diagnostic
    problemes = []
    if contraste < 0.3:
        problemes.append("Faible contraste — binarisation difficile")
    if nettete < 50:
        problemes.append("Image floue — HTR dégradé probable")
    if frac_sature > 0.05:
        problemes.append("Zones saturées importantes — or ou surexposition")
    if uniformite_fond < 0.5:
        problemes.append("Fond très hétérogène — prétraitement complexe")

    metriques["problemes"] = problemes
    metriques["qualite_globale"] = "acceptable" if not problemes else "problematique"
    return metriques
```

---

## 10. Connexions avec le pipeline global

Le prétraitement est l'**étape 1** du pipeline, et ses sorties conditionnent toutes les étapes suivantes :

```
Scan brut (TIFF 300–400 DPI)
    ↓
[4.1] Prétraitement ← Cette section
    ↓
├── Image RGB améliorée → SAM (layout) / CLIP (illustrations)
├── Image niveaux de gris → CLAHE → Kraken BLLA (baselines)
└── Image binarisée → Kraken HTR / TrOCR
    ↓
[4.2] Pipeline Kraken intégré ou [4.3] Pipeline modulaire SAM + TrOCR
```

**Ce que le prétraitement garantit en aval :**
- Une inclinaison corrigée → la détection de baselines horizontales par Kraken BLLA fonctionne correctement.
- Un contraste amélioré → SAM produit des masques plus nets aux frontières texte/fond.
- Une binarisation adaptative → l'encre pâlie est préservée, le fond sale est éliminé.
- Les zones saturées sont identifiées → le pipeline ne tente pas de transcrire des régions d'enluminures.

**Ce que le prétraitement ne peut pas corriger :**
- Les lacunes physiques (trous dans le parchemin, encre totalement disparue).
- Un show-through sévère sans accès au verso.
- La courbure non linéaire extrême (nécessite un dewarping dédié).
- Une résolution de numérisation insuffisante (< 150 DPI pour les textes difficiles).

Ces limitations sont documentées dans le data contract livré au module NLP.

---

## Glossaire des termes avancés

**Binarisation adaptative**
Méthode de seuillage où le seuil varie spatialement en fonction des statistiques locales de l'image. Par opposition au seuillage global (un seul seuil pour toute l'image). Sauvola est la méthode adaptative de référence pour les documents anciens.

**CLAHE (*Contrast Limited Adaptive Histogram Equalization*)**
Extension de l'égalisation d'histogramme adaptative avec limitation du contraste pour éviter l'amplification du bruit. Opère sur des tuiles locales plutôt que sur l'image entière. Paramètres clés : `clipLimit` (amplitude de l'amplification) et `tileGridSize` (découpage en tuiles).

**Corrosion par l'encre ferro-gallique**
Phénomène chimique par lequel l'acide contenu dans l'encre ferro-gallique médiévale attaque les fibres du parchemin. Les zones d'encre dense peuvent se fragiliser jusqu'à se perforer. Sur les scans : irrégularités de texture dans les zones d'encre, parfois trous visibles.

**Déwarping (*document dewarping*)**
Correction de la déformation non linéaire des pages (courbure de reliure, ondulation du parchemin). Nécessite soit des algorithmes géométriques basés sur la détection des lignes, soit des réseaux neuronaux spécialisés (DewarpNet, DocUNet).

**Deskewing (redressement)**
Correction de l'inclinaison globale d'une image de document. Peut être estimée par maximisation de la variance de la projection horizontale, ou par la transformée de Hough. S'applique aux inclinaisons linéaires uniformes — ne corrige pas les déformations locales.

**Dilatation morphologique**
Opération qui étend les régions sombres (encre) dans une image binaire : chaque pixel sombre est « gonflé » selon un élément structurant. Utilisée pour connecter des traits proches ou combler des lacunes dans les lettres.

**Érosion morphologique**
Opération inverse de la dilatation : réduit les régions sombres. Un pixel sombre est conservé uniquement si tous ses voisins (selon l'élément structurant) sont aussi sombres. Utilisée pour supprimer les petits artefacts isolés.

**Encre ferro-gallique**
Encre médiévale fabriquée à partir de sulfate de fer et d'acide tannique. Noire à l'état frais, elle brunit en vieillissant et peut devenir corrosive pour le support. C'est l'encre dominante dans les manuscrits du Moyen Âge occidental.

**Espace colorimétrique LAB**
Modèle de couleur qui sépare la luminosité (canal L) de l'information de couleur (canaux A et B). Proche de la perception humaine : des variations égales de L correspondent à des variations perçues comme égales par l'œil. Recommandé pour appliquer des traitements qui ne doivent pas modifier les teintes (CLAHE, etc.).

**Fermeture morphologique (*closing*)**
Dilatation suivie d'une érosion. Comble les petites lacunes dans les objets et connecte les composantes proches. Utile pour corriger les ruptures dans les traits d'encre corrodée.

**Filtre médian**
Filtre non linéaire qui remplace chaque pixel par la médiane des valeurs dans une fenêtre locale. Efficace contre le bruit sel-et-poivre sans flouter les contours, car la médiane est robuste aux valeurs extrêmes.

**Histogramme bimodal**
Histogramme présentant deux pics distincts. Pour une image de document texte, le premier pic correspond aux pixels sombres (encre) et le second aux pixels clairs (fond). La méthode d'Otsu est optimale pour les histogrammes bimodaux.

**Homographie**
Transformation projective en 2D : une application linéaire dans les coordonnées homogènes qui peut représenter n'importe quelle transformation perspective d'un plan sur un autre. Utilisée pour corriger la distorsion de perspective des scans photographiés en biais.

**Indice de bimodalité de Sarle**
Mesure statistique de la bimodalité d'une distribution : $B = (\text{skewness}^2 + 1) / \kappa$, où $\kappa$ est le kurtosis. Valeur > 0.555 suggère une distribution bimodale (seuillage Otsu approprié).

**Laplacien d'image**
Opérateur différentiel du second ordre qui détecte les variations rapides d'intensité (contours, textures). La variance du Laplacien est une mesure de la netteté d'une image : une variance élevée indique une image nette ; une variance faible indique un flou.

**Ouverture morphologique (*opening*)**
Érosion suivie d'une dilatation. Supprime les petits objets isolés (artefacts, pixels de bruit) sans modifier les objets plus grands. Utile pour nettoyer une image binarisée après seuillage.

**Palimpseste**
Manuscrit dont le texte original a été gratté ou lavé pour permettre la réutilisation du parchemin. Le texte effacé (*scriptio inferior*) reste partiellement visible, notamment en imagerie multispectrale UV ou infrarouge. Sur les scans ordinaires, il constitue un bruit structuré subtil.

**Parchemin**
Support d'écriture médiéval fabriqué à partir de peau animale (mouton, chèvre, veau) traitée : trempée, tendue sur cadre, raclée, séchée. Semi-transparent, sensible à l'humidité, durable si conservé correctement. Le *vélin* (parchemin de veau) est le plus fin et le plus blanc.

**Sauvola (*méthode de*)**
Méthode de binarisation adaptative proposée par Jaakko Sauvola en 2000. Calcule un seuil local pour chaque pixel en fonction de la moyenne ($\mu$) et de l'écart-type ($\sigma$) dans une fenêtre voisine : $t = \mu [1 + k(\sigma/R - 1)]$. Paramètre $k$ : sensibilité au contraste local.

**Show-through**
Phénomène de transparence du parchemin : l'encre du verso est partiellement visible par transparence sur le recto, créant des traits fantômes qui interfèrent avec la lecture automatique du texte du recto.

**Vignettage**
Diminution de la luminosité aux bords d'une image par rapport au centre, due aux propriétés optiques de l'objectif. Source d'éclairage non uniforme dans les scans de documents.

---

## Bibliographie de référence

### Traitement des images de documents anciens

- **Sauvola, J., Pietikäinen, M.** (2000). *Adaptive Document Image Binarization*. Pattern Recognition, 33(2), 225–236. — L'article fondateur du seuillage adaptatif de Sauvola. Formulation mathématique et justification sur les documents anciens.

- **Otsu, N.** (1979). *A Threshold Selection Method from Gray-Level Histograms*. IEEE Transactions on Systems, Man, and Cybernetics, 9(1), 62–66. — La méthode de seuillage automatique global la plus utilisée.

- **Stathis, P., Kavallieratou, E., Papamarkos, N.** (2008). *An Evaluation Technique for Binarization Algorithms*. Journal of Universal Computer Science, 14(18). — Comparaison systématique des méthodes de binarisation sur des documents historiques.

- **Clausner, C., Antonacopoulos, A., Pletschacher, S.** (2019). *ICDAR2019 Competition on Recognition of Documents with Complex Layouts (RDCL2019)*. ICDAR 2019. — Évaluation standard des pipelines de segmentation de layout sur documents complexes.

### Dégradations et restauration des manuscrits

- **Tonazzini, A., Gerace, I., Martinelli, F.** (2010). *Multichannel Blind Separation and Deconvolution of Images for Document Analysis*. IEEE Transactions on Image Processing, 19(3). — Séparation du show-through par ICA dans les images de manuscrits.

- **Fernandez-Maloigne, C., Robert-Inacio, F., Macaire, L.** (dir.) (2012). *Digital Color Imaging*. Wiley-ISTE. Chapitre 10 : *Color in Cultural Heritage Imaging*. — Les propriétés optiques spécifiques des documents patrimoniaux.

- **Stork, D. G., Duarte, M.** (2009). *Image Analysis for Art Conservation and Authentication*. Dans : *Image Processing for Artist Identification*. Springer. — Techniques d'imagerie multispectrale pour les manuscrits.

### Déwarping et correction géométrique

- **Ma, K., Shu, Z., Bai, X., Wang, J., Samaras, D.** (2018). *DocUNet: Document Image Unwarping via a Stacked U-Net*. CVPR 2018. [arXiv:1804.01277] — Déwarping par réseau neuronal, état de l'art.

- **Xie, G. et al.** (2021). *DewarpNet: Single-Image Document Unwarping With Stacked 3D and 2D Regression Networks*. ICCV 2019. [arXiv:1909.09741] — Alternative à DocUNet, souvent plus performante.

### Benchmarks et datasets de binarisation

- **Pratikakis, I., Zagoris, K., Karagiannis, X., Tsochatzidis, L., Mondal, T., Marthot-Santaniello, I.** (2019). *ICDAR 2019 Competition on Document Image Binarization (DIBCO 2019)*. ICDAR 2019. — Le benchmark de référence pour l'évaluation des méthodes de binarisation de documents.

- **Tensmeyer, C., Martinez, T.** (2017). *Document Image Binarization with Fully Convolutional Neural Networks*. ICDAR 2017. — Introduction des réseaux de neurones profonds pour la binarisation de documents.

### CLAHE et amélioration du contraste

- **Pizer, S. M. et al.** (1987). *Adaptive Histogram Equalization and Its Variations*. Computer Vision, Graphics, and Image Processing, 39(3), 355–368. — CLAHE : l'article fondateur.

- **Zuiderveld, K.** (1994). *Contrast Limited Adaptive Histogram Equalization*. Dans : *Graphics Gems IV*. Academic Press. — Implémentation pratique de CLAHE, référence pour OpenCV.

### Codicologie et matériaux des manuscrits

- **Rück, P.** (éd.) (1991). *Pergament: Geschichte, Struktur, Restaurierung, Herstellung*. Thorbecke, Sigmaringen. — Référence technique sur le parchemin et ses propriétés physiques.

- **Bicchieri, M., Monti, M., Piantanida, G., Sannino, A.** (2013). *A non-invasive multi-technique survey of manuscripts: the case of the Beatus Vir*. Spectrochimica Acta. — Exemple de caractérisation multi-spectrale des encres et supports de manuscrits médiévaux.

---

*Support de cours rédigé pour le module Computer Vision · Promotion MD5 · Mai 2026*
*Ce document accompagne la séance 4.1 du Jour 4.*
*Il suppose la lecture des sections 1.1 (philologie) et 1.4 (introduction technique).*
*Durée estimée de lecture : 75–90 minutes.*
