---
updated: 2026-05-17T18:56:06.162+02:00
edited_seconds: 1650
---
# Cours 2.1 : Rappel CNN et limites structurelles pour les documents

**Module 2 · Computer Vision appliquée aux manuscrits médiévaux · MD5**

notes:
Objectif : revisiter les CNN depuis leur biais inductif pour comprendre pourquoi ils atteignent leurs limites sur les manuscrits médiévaux, et pourquoi les Vision Transformers les surpassent. Les CNN ne sont pas « mauvais » : ils sont très adaptés à certains problèmes, mais pas à la reconnaissance de texte manuscrit ancien.

Les **CNN (réseaux de neurones convolutifs)** sont des architectures fondamentales en computer vision, conçues pour traiter des données structurées en grille, comme les images. 

L'architecture est inspirée du cortex visuel: son fonctionnement est **hiérarchique** : 
- les premières couches détectent des motifs **simples** (bords, coins et textures simples.), 
- les couches intermédiaires combinent ces motifs en **formes géométriques** et **parties d’objets** (ex. : yeux, roues), et 
- les dernières couches (entièrement connectées, FC) assemblent ces parties pour reconnaître des **objets entiers** (ex. : visage, vélo).

---
## 1. La convolution 2D : formalisme et intuitions
![[Cours/Computer Vision/Michel/CoursAnalysePhilologique/metadata/Cours_2_1_CNN_et_Limites-1778841336472.png|750]]

--
### 1.1 Définition formelle
- Entrée : image (niveaux de gris)$$I∈RH×W$$ 
- Filtre (noyau) : $$K∈Rk×k$$
- Sortie (carte de caractéristiques) : $$F∈RH′×W′$$
	- $s$ = stride (pas de déplacement)
	- $b$ = biais scalaire
	- Avec padding $p$ : $H′=⌊(H−k+2p)/s⌋+1$, idem pour $W′$
$$F[i,j]=∑m=0k−1∑n=0k−1I[i⋅s+m, j⋅s+n]⋅K[m,n]+b$$


> En deep learning, on parle de « convolution » mais c'est techniquement une **corrélation croisée** (pas de retournement du filtre). Sans importance pratique puisque les filtres sont appris.

notes:  
Soit $I \in \mathbb{R}^{H \times W}$ une image en niveaux de gris (hauteur $H$, largeur $W$) et $K \in \mathbb{R}^{k \times k}$ un filtre (ou noyau) de taille $k \times k$. La convolution discrète 2D produit une **carte de caractéristiques** (*feature map*) $F \in \mathbb{R}^{H' \times W'}$ définie par :

$$F[i, j] = \sum_{m=0}^{k-1} \sum_{n=0}^{k-1} I[i \cdot s + m,\ j \cdot s + n] \cdot K[m, n] + b$$

où $s$ est le **stride** (pas de déplacement du filtre) et $b$ un biais scalaire.

Les dimensions de la sortie en fonction du padding $p$ sont :

$$H' = \left\lfloor \frac{H - k + 2p}{s} \right\rfloor + 1, \quad W' = \left\lfloor \frac{W - k + 2p}{s} \right\rfloor + 1$$

**Nota bene sur la terminologie :** en deep learning, on parle de « convolution » mais l'opération implémentée est techniquement une **corrélation croisée** (*cross-correlation*) : le filtre n'est pas retourné. La distinction est sans importance pratique puisque les filtres sont appris.
Rappeler que les filtres ne sont pas définis à la main : ils sont appris par rétropropagation. Une couche avec $Cout$​ filtres sur $Cin$​ canaux a $Cout×Cin×k2+Cout​$ paramètres.

--
### 1.2 Les couches d'une architecture CNN

--
#### a. La couche convolutionnelle
- La couche **convolutionnelle** utilise des **filtres** (noyaux) qui se déplacent sur l'image et scannent l'entrée $I$ pour détecter les motifs locaux (bords, texture) en effectuant des opérations de convolution. 
- Elle peut être réglée en ajustant la taille du filtre $F$ et le stride $S$. 
- La sortie $O$ de cette opération est appelée **_feature map_** ou aussi **_activation map**_ (carte de caractéristiques), capturant la présence d’un motif spécifique..

![[Computer Vision - MD5-1778164851290.png|750]]



--
#### b. La couche de  pooling
La couche de **pooling** (en anglais _pooling layer_) (POOL) est une opération de **sous-échantillonnage**. Elle réduit la dimension spatiale des feature maps, en conservant les informations les plus pertinentes et en **diminuant la charge computationnelle**. 
Il permet de construire une **hiérarchie de features** : les couches basses captent des détails fins, les couches profondes, après plusieurs opérations de pooling, représentent des concepts **abstraits et globaux**, indépendants de la position exacte.

![[Cours/Computer Vision/Michel/CoursAnalysePhilologique/metadata/Cours_2_1_CNN_et_Limites-1778841902304.png|750]]

--
#### c. la couche fully connected
La couche de **fully connected** (en anglais _fully connected layer_) (FC) s'applique sur une entrée préalablement **aplatie** où chaque **entrée est connectée à tous les neurones**. Les couches de fully connected sont typiquement présentes **à la fin des architectures** de CNN et peuvent être utilisées pour **optimiser des objectifs** tels que les **scores de classe**.
![[Cours/Computer Vision/Michel/CoursAnalysePhilologique/metadata/Cours_2_1_CNN_et_Limites-1778841725579.png]]

--
### 1.3 Les paramètres opérationnels

--
#### a. filtre
**Le filtre (*kernel*)**
Dans un CNN, les filtres ne sont pas définis à la main : ils sont **appris par rétropropagation**. Un filtre $3 \times 3$ contient 9 paramètres apprenables (plus le biais). Une couche convolutive avec $C_{out}$ filtres sur une entrée à $C_{in}$ canaux contient $C_{out} \times C_{in} \times F^2 + C_{out}$ paramètres.
![[Cours/Computer Vision/Michel/CoursAnalysePhilologique/metadata/Cours_2_1_CNN_et_Limites-1778840049050.png]]

--
#### b. stride
**Le stride $s$**
Contrôle le sous-échantillonnage spatial - définit le nombre de pixel
- $s=1$: pas de réduction
- $s=2$ : résolution divisée par 2 (alternative au pooling)
![[Cours/Computer Vision/Michel/CoursAnalysePhilologique/metadata/Cours_2_1_CNN_et_Limites-1778840188021.png]]


--
<!-- slide template="[[plt-three-col]]" -->

::: title
#### c. padding $p$

:::
::: col1
- _Valid_ ($p=0$) : pas de zéros ajoutés sortie plus petite que l'entrée.
![[Cours/Computer Vision/Michel/CoursAnalysePhilologique/metadata/Cours_2_1_CNN_et_Limites-1778840564151.png]]
:::
::: col2
- _Same_ ou _demi_ ($p=⌊k/2⌋$) : sortie même résolution (convention moderne)
![[Cours/Computer Vision/Michel/CoursAnalysePhilologique/metadata/Cours_2_1_CNN_et_Limites-1778840617208.png]]
:::
::: col3
- _Total_ (p=$P=F−1$): Le filtre « voit » chaque pixel comme centre → sortie plus grande
![[Cours/Computer Vision/Michel/CoursAnalysePhilologique/metadata/Cours_2_1_CNN_et_Limites-1778840621340.png]]

:::

notes:
- Padding - ajoute des 0 de autour de l'entrée (matrice numérique)


--
### 1.3.c Les paramètres opérationnels - dilatation
Insère des zéros entre les éléments du filtre  
$$F[i,j]=∑m,nI[i+d⋅m,j+d⋅n]⋅K[m,n]$$
- $d$ = taux de dilatation
- Augmente le champ réceptif sans ajouter de paramètres ni réduire la résolution
![[Cours/Computer Vision/Michel/CoursAnalysePhilologique/metadata/Cours_2_1_CNN_et_Limites-1778841006480.png|550]]<!-- element style="display: block; margin-left: auto; margin-right: auto;" -->

notes:  
La convolution dilatée est utilisée dans DeepLab, WaveNet. Elle permet d'agrandir le contexte sans perte de résolution – une idée qui préfigure l'attention.

--
### 1.4 Le champ réceptif (*receptive field*)
- **Définition** : ensemble des pixels de l'image d'entrée qui influencent la valeur d'un neurone dans une couche profonde.
- **Croissance** avec des convolutions 3×33×3, stride 1, sans pooling :  $RF(l)=1+l×(k−1)=1+2l$
- **Avec pooling par facteur 2 après chaque bloc** : croissance exponentielle $RF(l)≈kl$
- **Champ réceptif effectif** (Luo et al., 2016) : en pratique, seuls les pixels centraux contribuent significativement. L'information utile se distribue selon une gaussienne.

> **Conséquence pour les manuscrits** : une page 4000×5500 pixels nécessiterait $log⁡2(4000)≈12$ poolings pour qu'un neurone voie la page entière – soit une réduction de résolution par 4096. La perte de détails fins des lettres est irréversible.

notes:  
Montrer l'image d'un champ réceptif effectif en forme de gaussienne. C'est une limite fondamentale : les CNN sont faits pour traiter des structures locales, pas des dépendances à longue portée.

Le **champ réceptif** d'un neurone dans une couche $l$ est l'ensemble des pixels de l'image d'entrée qui influencent la valeur de ce neurone. C'est une notion centrale pour comprendre les limites des CNN.

Pour une séquence de convolutions $3 \times 3$ avec stride 1 et sans pooling, le champ réceptif croît **linéairement** avec la profondeur :

$$\text{RF}(l) = 1 + l \times (k - 1) = 1 + 2l \quad \text{(pour } k=3\text{)}$$

Avec du pooling par facteur 2 après chaque bloc, il croît exponentiellement :

$$\text{RF}_{\text{eff}}(l) \approx k^l$$

Mais ce calcul donne le **champ réceptif théorique**, pas le champ réceptif **effectif**. Des travaux de Luo et al. (2016) ont montré empiriquement que les gradients se distribuent selon une gaussienne dans le champ réceptif théorique : seul le centre contribue significativement. Le champ réceptif effectif est environ $\sqrt{\text{RF}_{\text{théorique}}}$ : soit un carré de côté 11 pour un champ théorique de 121.

> **Implication directe pour les manuscrits :** une page de manuscrit à 400 DPI fait environ 4 000 × 5 500 pixels. Pour qu'un neurone en haut du réseau « voie » la page entière, il faudrait un champ réceptif de 4 000 pixels. Avec des blocs $3 \times 3$ et du pooling ×2, cela nécessite $\log_2(4000) \approx 12$ étapes de pooling : soit une réduction de résolution par 4096. La représentation finale ne contient plus que $\approx 1$ pixel par page. L'information spatiale fine est irrémédiablement perdue.

--
![[Computer Vision - MD5-1778164851290.png]]
--
<!-- slide template="[[plt-two-col]]" -->
::: title
## 1.5 Pooling et invariance
:::

::: left
**Max pooling** : sélectionne la valeur maximale dans une fenêtre.  
![](https://stanford.edu/~shervine/teaching/cs-230/illustrations/max-pooling-a.png)

:::

::: right
**Average pooling** : calcule la moyenne.
![](https://stanford.edu/~shervine/teaching/cs-230/illustrations/average-pooling-a.png)

:::
- 
- Réduction de résolution (sous-échantillonnage)
- **Invariance locale à la translation** : un petit décalage ne change pas la sortie
notes:
Le **max pooling** sélectionne la valeur maximale dans une fenêtre locale ; l'**average pooling** en calcule la moyenne. Les deux induisent :
- Une **réduction de résolution** (sous-échantillonnage).
- Une **invariance locale à la translation** : décaler légèrement une feature dans la fenêtre de pooling ne change pas la sortie du max pooling.

Cette invariance à la translation est l'une des forces des CNN sur les images naturelles. Pour un classificateur d'objets (chat, chien, voiture…), qu'un chat soit à gauche ou à droite de l'image importe peu. Pour un HTR, la position est au contraire porteuse d'information : la deuxième lettre d'un mot n'est pas interchangeable avec la cinquième. L'invariance devient un obstacle.

La couche de **pooling** (en anglais _pooling layer_) (POOL) est une opération de **sous-échantillonnage**. Elle réduit la dimension spatiale des feature maps, en conservant les informations les plus pertinentes et en **diminuant la charge computationnelle**. 
Il permet de construire une **hiérarchie de features** : les couches basses captent des détails fins, les couches profondes, après plusieurs opérations de pooling, représentent des concepts **abstraits et globaux**, indépendants de la position exacte.

Les types de pooling les plus populaires sont le max et l'**average pooling**, où les valeurs maximales et moyennes sont prises, respectivement.
- **Max pooling**: Chaque opération de pooling sélectionne la valeur maximale dans chaque fenêtre. Il **préserve les caractéristiques les plus saillantes** (ex. : bords, textures fortes) et renforce la **translation invariance**. Très utilisé dans les couches intermédiaires.

--
<!-- slide template="[[plt-two-col]]" -->
::: title
### 1.6 Normalisation et stabilisation de l'entraînement
:::

::: left
**Batch Normalization** (Ioffe & Szegedy, 2015) : normalise sur le mini-batch  
- Stabilise les gradients, accélère la convergence, régularise
- Limite : mauvais pour petits batch sizes (8>N)

![[Cours/Computer Vision/Michel/CoursAnalysePhilologique/metadata/Cours_2_1_CNN_et_Limites-1778850356804.png]]

:::

::: right
**Layer Normalization** (Ba et al., 2016) : normalise sur les dimensions de caractéristiques
- Indépendante de la taille du batch
- Préférée dans les Transformers (séquences de longueur variable)

![[Cours/Computer Vision/Michel/CoursAnalysePhilologique/metadata/Cours_2_1_CNN_et_Limites-1778850367260.png|350]]

:::
notes:  
BatchNorm est omniprésent dans les CNN modernes. LayerNorm s'impose dans les architectures Transformer car elle s'applique à chaque token indépendamment.

**Batch Normalization (BatchNorm, Ioffe & Szegedy, 2015)**: Pour chaque canal $c$, **BatchNorm** normalise les activations sur le mini-batch courant :

$$\hat{x}_c = \frac{x_c - \mu_c}{\sqrt{\sigma_c^2 + \epsilon}}$$

puis applique une transformation affine apprise : $y_c = \gamma_c \hat{x}_c + \beta_c$.

Effets : stabilise les gradients, accélère la convergence, permet des taux d'apprentissage plus élevés, et agit comme une régularisation implicite (réduisant le besoin de dropout).

Limite : se comporte mal pour les petits batch sizes ($N < 8$) et n'est pas applicable séquence par séquence (contrairement à Layer Normalization, préférée dans les Transformers).

Le **Batch Normalization (BatchNorm)** stabilise le **flux du gradient** en réduisant le **changement de distribution des activations** (covariance shift) entre les couches. En normalisant les activations de chaque mini-lot à **moyenne nulle et variance unitaire**, il empêche les gradients de **disparaître ou d’exploser**, facilitant ainsi l’entraînement de réseaux profonds.
**Effet sur les activations** :
- Elles sont **centrées et réduites** par lot.
- Des paramètres apprenables (γ, β) permettent au réseau de **réajuster l’échelle et le décalage**, préservant sa capacité d’expression.
- Cela **lisse le paysage d’optimisation**, permet d’utiliser des **taux d’apprentissage plus élevés** et agit comme une **régularisation implicite**.

**Layer Normalization**
Normalise sur les dimensions de caractéristiques (plutôt que sur le batch). Indépendante de la taille du batch : c'est pourquoi elle s'impose dans les Transformers, où les séquences ont des longueurs variables.

--
### 1.7. Fonctions d’activation

| Fonction       | Formule                           | Propriétés                                             |
| -------------- | --------------------------------- | ------------------------------------------------------ |
| **ReLU**       | $g(z)=max⁡(0,z)$                  | Non‑linéaire, rapide, gradients non saturés pour $z>0$ |
| **Leaky ReLU** | $g(z)=max⁡(ϵz,z)$ avec $ϵ≪ϵ≪1$    | Évite le « dying ReLU »                                |
| **ELU**        | $g(z)=max⁡(α(ez−1),z)$ avec $α≪1$ | Dérivable partout, moyenne proche de 0                 |
| **Softmax**    | $pi=exi∑jexj​​$                   | Convertit des scores en probabilités (couche finale)   |

---
#### **ReLU** : la plus utilisée dans les CNN.  
- $g(z)=max⁡(0,z)$
- Non‑linéaire, rapide, gradients non saturés pour $z>0$
![[Cours/Computer Vision/Michel/CoursAnalysePhilologique/metadata/Cours_2_1_CNN_et_Limites-1778850835547.png|450]]<!-- element style="display: block; margin-left: auto; margin-right: auto;" -->

notes:
**ReLU**: 
	- Si le résultat de la somme est **positif** → on le garde tel quel.
	- Si le résultat est **négatif** → on le transforme en **0**.
- Le « dying ReLU » : si un neurone ReLU n’est jamais activé (sortie 0), son gradient est nul et il ne se met plus jamais à jour. Leaky ReLU et ELU apportent un faible gradient pour les valeurs négatives.
--
#### Leaky ReLU
- $g(z)=max⁡(ϵz,z)$ avec $ϵ≪ϵ≪1$
- Évite le « dying ReLU »
![[Cours/Computer Vision/Michel/CoursAnalysePhilologique/metadata/Cours_2_1_CNN_et_Limites-1778850856733.png|450]]<!-- element style="display: block; margin-left: auto; margin-right: auto;" -->

notes:

**Leaky ReLU** : au lieu de mettre à 0 les valeurs négatives, on les multiplie par un tout petit nombre (0,01 par exemple).  
Leaky ReLU a un petit coefficient (souvent 0,01). ELU est dérivable partout, ce qui peut aider dans certains cas mais ne change pas la vie pour les débutants.
→ Le neurone reste un peu actif même en négatif, il ne « meurt » jamais.

--
#### **ELU**
- $g(z)=max⁡(α(ez−1),z)$ avec $α≪1$
- Dérivable partout, moyenne proche de 0

![[Cours/Computer Vision/Michel/CoursAnalysePhilologique/metadata/Cours_2_1_CNN_et_Limites-1778850879492.png|450]]<!-- element style="display: block; margin-left: auto; margin-right: auto;" -->

notes:

**ELU** (Exponential Linear Unit) : pour les négatifs, on utilise une courbe exponentielle douce.  
→ Les sorties sont en moyenne proches de zéro, ce qui aide l’apprentissage.

**À retenir** : ce sont des variantes de ReLU, plus robustes mais un peu plus coûteuses. Dans 90% des cas, la ReLU standard suffit.

--
#### **Softmax** : toujours en dernière couche pour la classification multi‑classe.
- $pi=exi∑jexj​​$
- Convertit des scores en probabilités (couche finale)
![[Cours/Computer Vision/Michel/CoursAnalysePhilologique/metadata/Cours_2_1_CNN_et_Limites-1778851117616.png|450]]<!-- element style="display: block; margin-left: auto; margin-right: auto;" -->

notes:  

La **Softmax** n’est pas une activation « interne » comme ReLU.  
Elle s’utilise **en dernière couche** d’un classifieur.

**Ce qu’elle fait** :
- Prend une liste de nombres (les scores bruts du réseau).
- Les transforme en **probabilités** (tous entre 0 et 1, somme = 1).

**Exemple** : sortie brute [2.0, 1.0, 0.1] → après Softmax → [0.65, 0.24, 0.11]  
→ le réseau est sûr à 65% que c’est la classe 1.

La fonction d'activation permet au réseau de s'adapter à des relation compliquées (non-linéaires). Sans ça le modèle ne serait qu'un modèle linéaire

---
## 2. Les grandes architectures CNN : innovations et enseignements

--
![[Computer Vision - MD5-1778166797302.png]]

--
### 2.1 LeNet-5 (LeCun et al., 1998) - Chiffres manuscrits
- La **première architecture CNN** opérationnelle à grande échelle, conçue pour la reconnaissance de chiffres manuscrits (dataset MNIST). 
- Architecture : 2 couches convolutives (5×5) + 2 couches de pooling moyen + 3 couches fully connected. 
	- Total : ~60 000 paramètres.
- **Innovation** : preuve que les features apprises surpassent les features artisanales.

![[Cours/Computer Vision/Michel/CoursAnalysePhilologique/metadata/Cours_2_1_CNN_et_Limites-1778841243820.png|650]]<!-- element style="display: block; margin-left: auto; margin-right: auto;" -->
notes:
- **Innovation :** démonstration que les features hiérarchiques apprises par rétro propagation surpassent les features artisanales pour la reconnaissance de caractères.
- **Limite :** impossible à entraîner efficacement sur GPU à l'époque : le passage à l'échelle attendait AlexNet - 14 ans.
- 

--
### 2.2 AlexNet (Krizhevsky, Sutskever & Hinton, 2012) - Le point d'inflexion
- Vainqueur d'ImageNet 2012 (erreur top-5 15,3% vs 26,2% pour le second)
- 5 couches convolutives (11×11, 5×5, 3×3) + 3 fully connected
	- ~62M paramètres
- **Innovations** : entraînement sur GPU, ReLU, Dropout, data augmentation
![[Cours/Computer Vision/Michel/CoursAnalysePhilologique/metadata/Cours_2_1_CNN_et_Limites-1778841252017.png]]<!-- element style="display: block; margin-left: auto; margin-right: auto;" -->

notes:  
AlexNet a déclenché la révolution du deep learning en vision. À retenir : l'importance du GPU et de ReLU.
Le point d'inflexion historique de la vision par ordinateur moderne. Vainqueur de l'ImageNet Large Scale Visual Recognition Challenge (ILSVRC) 2012 avec un taux d'erreur top-5 de 15,3% contre 26,2% pour le second : une réduction d'erreur relative de 41%.

**Innovations :**
- Entraînement sur GPU (deux GTX 580 en parallèle).
- Fonction d'activation ReLU à la place de tanh/sigmoïde : convergence 6× plus rapide.
- Dropout (taux 0.5) sur les couches fully connected : régularisation efficace sans surcoût computationnel.
- Data augmentation (flips, crops, perturbations couleur) : expansion artificielle du dataset.
- Local Response Normalization (précurseur de BatchNorm, abandonné par la suite).

**Architecture :** 5 couches convolutives (noyaux 11×11, 5×5, 3×3) + 3 couches fully connected. Total : ~62 millions de paramètres.

--
### 2.3 VGGNet (Simonyan & Zisserman, 2014) - La profondeur comme clé
- **Idée** : empiler des petits filtres 3×33×3 plutôt que de grands filtres
    - Deux filtres 3×33×3 ont le même champ réceptif qu'un 5×55×5, avec deux fois moins de paramètres et deux non-linéarités
- VGG-16 : 16 couches à poids, 138M paramètres
- **Héritage** : utilisé comme extracteur de features et backbone pour la segmentation de documents.
![[Cours/Computer Vision/Michel/CoursAnalysePhilologique/metadata/Cours_2_1_CNN_et_Limites-1778841263211.png|650]]
notes:


**Innovation principale :** démonstration que la profondeur est le facteur clé de la qualité des représentations, et que des petits filtres $3 \times 3$ empilés sont plus efficaces que de grands filtres. Deux filtres $3 \times 3$ ont le même champ réceptif qu'un filtre $5 \times 5$, mais deux fois moins de paramètres et deux non-linéarités au lieu d'une.

**Architectures :** VGG-16 (16 couches à poids) et VGG-19. Extrêmement régulières : blocs de convolutions $3 \times 3$ séparés par du max pooling. Paramètres : 138M pour VGG-16.

**Héritage :** VGGNet est encore utilisé comme extracteur de features et comme backbone dans des tâches de segmentation, notamment pour l'analyse de documents. Sa régularité la rend facile à comprendre et à modifier.

--
### 2.4 ResNet (He et al., 2015) – Les skip connections
**Problème résolu** : la dégradation – les réseaux très profonds (>20 couches) entraînés naïvement ont une erreur d'entraînement _plus élevée_ que les réseaux moins profonds.

$y=F(x,{Wi})+x$

Le réseau apprend la **résiduelle** $F(x)=y−x$. Si $H(x)≈x$, alors $F(x)≈0$ est plus facile à apprendre.
**Résultat** : ResNet-152 (152 couches) atteint 3,57% d'erreur top-5 (sous le niveau humain).
![[Cours/Computer Vision/Michel/CoursAnalysePhilologique/metadata/Cours_2_1_CNN_et_Limites-1778841269235.png]]

notes:  
ResNet est l'article le plus cité de l'histoire de la vision (>100k citations). Les skip connections sont aussi utilisées dans les Transformers.

La contribution architecturale la plus influente de la décennie 2010. 
Résout le **problème de dégradation** : contrairement à l'intuition, les réseaux très profonds (> 20 couches) entraînés naïvement produisent des erreurs d'entraînement *plus élevées* que les réseaux moins profonds : non pas à cause du surapprentissage, mais à cause de la difficulté à optimiser des fonctions profondes.

La sortie est $y = F(x, \{W_i\}) + x$ : le réseau apprend la **résiduelle** $F(x) = y - x$ plutôt que la transformation directe $y = H(x)$.

**Pourquoi ça marche :** si la transformation optimale est proche de l'identité, apprendre $F(x) \approx 0$ est plus facile qu'apprendre $H(x) \approx x$. Les skip connections fournissent des **raccourcis de gradient** qui permettent la rétropropagation à travers des centaines de couches sans dégradation.

**Résultats :** ResNet-152 (152 couches) a remporté ILSVRC 2015 avec 3,57% d'erreur top-5 : sous le niveau humain estimé à ~5%.

**Variantes :** ResNeXt (convolutions groupées), Wide ResNet (plus large, moins profond), ResNet-D (modifications des couches de stride pour améliorer le flux de l'information).

--
### 2.5 DenseNet (Huang et al., 2016) – Connexions denses
Chaque couche reçoit les feature maps de **toutes** les couches précédentes (concaténation) :
$$xl=Hl([x0,x1,…,xl−1])$$

- **Avantages** : réutilisation maximale des features, gradients très courts, régularisation naturelle.
- **Pertinence pour les documents** : les features de bas niveau (bords, texture d'encre) restent accessibles aux couches profondes.

--
![[Cours/Computer Vision/Michel/CoursAnalysePhilologique/metadata/Cours_2_1_CNN_et_Limites-1778851688396.png]]
notes:
- Dans un bloc dense, **chaque couche reçoit les feature maps de toutes les couches précédentes** (concaténation, pas addition) :

$$x_l = H_l([x_0, x_1, \ldots, x_{l-1}])$$

**Avantages :** réutilisation maximale des features, gradients courts, régularisation naturelle, réduction du nombre de paramètres par rapport aux réseaux de même profondeur.

**Pertinence pour les documents :** les features de bas niveau (bords, textures locales d'encre) restent accessibles aux couches profondes sans dégradation : ce qui peut être utile pour la détection fine de tracés d'écriture.

--
### 2.6 EfficientNet (Tan & Le, 2019) – Scaling composé
Au lieu d'augmenter profondeur, largeur ou résolution séparément, on les augmente **ensemble** selon un facteur $ϕ$, c'est le **scaling composé** (*compound scaling*) :

$$\text{profondeur} : d = \alpha^\phi, \quad \text{largeur} : w = \beta^\phi, \quad \text{résolution} : r = \gamma^\phi$$

avec $α⋅β2⋅γ2≈2$ (contrainte de budget computationnel).

**Résultat** : EfficientNet-B7 atteint 84,4% de précision top-1 sur ImageNet avec 8× moins de paramètres que ResNet.

--
![[Cours/Computer Vision/Michel/CoursAnalysePhilologique/metadata/Cours_2_1_CNN_et_Limites-1778851950073.png]]
notes:
Propose un **scaling composé** (*compound scaling*) : plutôt que d'augmenter la profondeur, la largeur ou la résolution séparément, on les augmente ensemble selon un facteur de scaling $\phi$ :

$$\text{profondeur} : d = \alpha^\phi, \quad \text{largeur} : w = \beta^\phi, \quad \text{résolution} : r = \gamma^\phi$$

avec $\alpha \cdot \beta^2 \cdot \gamma^2 \approx 2$ (contrainte de budget computationnel).

Le backbone est basé sur des **MBConv** (Mobile Inverted Bottleneck Convolution) avec squeeze-and-excitation, hérités de MobileNetV3.

**Résultat :** EfficientNet-B7 atteint 84,4% de précision top-1 sur ImageNet avec 8× moins de paramètres que les meilleurs modèles ResNet comparables.

**EfficientNetV2** (2021) remplace certains blocs MBConv par des blocs Fused-MBConv pour accélérer l'entraînement.

--
### 2.7 Architectures spécialisées pour la segmentation
**U-Net** (Ronneberger et al., 2015) : architecture encodeur-décodeur
- Encodeur : sous-échantillonne (perd la résolution, gagne en sémantique)
- Décodeur : suréchantillonne
- **Skip connections** entre encodeur et décodeur au même niveau de résolution – transmettent les détails spatiaux fins

> U-Net est la référence pour la segmentation de layout dans les documents. Variante : ResUNet (encodeur ResNet pré-entraîné).

notes:  
Les skip connections sont essentielles pour récupérer l'information spatiale perdue par le pooling. C'est une idée qui réapparaît dans les modèles Transformer.
Les architectures précédentes sont des classificateurs (une étiquette par image). Pour la segmentation (une étiquette par pixel), il faut des architectures encodeur-décodeur.

**U-Net (Ronneberger et al., 2015)**
Architecture en U : un encodeur qui sous-échantillonne (perd la résolution pour gagner la sémantique) + un décodeur qui suréchantillonne + des **skip connections** entre encodeur et décodeur au même niveau de résolution. Ces connections transmettent les détails spatiaux fins que l'encodeur aurait autrement perdus.

Conçu initialement pour la segmentation d'images médicales, U-Net est la référence pour la segmentation de layout dans les documents. Une variante avec un encodeur ResNet-50 pré-entraîné est souvent appelée **ResUNet**

--
![[Cours/Computer Vision/Michel/CoursAnalysePhilologique/metadata/Cours_2_1_CNN_et_Limites-1778852012755.png]]
