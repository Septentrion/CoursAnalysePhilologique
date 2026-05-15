---
updated: 2026-05-15T12:05:08.759+02:00
edited_seconds: 1330
---
# Cours 1.2 : Traduire la philologie en problème d'apprentissage automatique

**Module 1 · Computer Vision appliquée aux manuscrits médiévaux · MD5**

notes:
- L'objectif de ce cours est de comprendre les différentes unités d'apprentissage disponibles pour une tâche de reconnaissance d'écriture manuscrite, leurs avantages et inconvénients.

---
## 1. La structure d'un problème d'apprentissage supervisé
--
### 1.1 Quel est l'input et l'output dans un problème de HTR ?

> *Définir un problème, c'est déjà le résoudre à moitié. Le définir mal, c'est s'assurer de ne jamais le résoudre du tout.*

notes:
**L'entrée (*input* ou *x*)**
Ce que le modèle reçoit. Dans notre cas : une image, ou une portion d'image.

**La sortie (*output*, *label* ou *y*)**
Ce que le modèle doit prédire. Dans notre cas : une chaîne de caractères (la transcription du texte visible dans l'image).

**La fonction de perte (*loss function*)**
Une mesure de l'écart entre la prédiction du modèle et la vérité terrain. C'est ce que le modèle minimise pendant l'entraînement. Le choix de cette fonction est lié : mais pas identique : au choix de la métrique d'évaluation.

Ces trois éléments sont contraints les uns par les autres. Changer l'unité d'entrée (passer de la ligne au mot, par exemple) change la nature des sorties attendues, les architectures envisageables, les stratégies d'augmentation de données, et les métriques pertinentes. C'est pourquoi le choix de la **granularité** est la première décision à prendre, et la plus structurante.

--
### 1.2 Qu'est-ce qu'un *label* pour un manuscrit ?
En apprentissage supervisé classique : (image, label) – label = réponse correcte unique.

**Pour un manuscrit médiéval, trois différences majeures :**
1. **Lenteur** : 1 000–3 000 mots/jour pour un expert → 50 000 lignes = années-personnes
2. **Qualification** : formation en paléographie, abréviations, dialectes, contexte
3. **Ambiguïté** : deux experts peuvent diverger pour 10% des caractères

→ Ces contraintes imposent : granularité adaptée, réduction du besoin d'annotation, métriques tenant compte de l'ambiguïté.


notes:  
L'annotation n'est pas un travail mécanique. C'est pourquoi on utilisera le **transfert learning** et l'agrégation de transcriptions.

Dans le cadre de l'apprentissage supervisé classique, chaque exemple d'entraînement est un couple (entrée, label). Le **label est une réponse correcte connue**, fournie par **un annotateur humain**.

Pour la reconnaissance d'images naturelles (chats, chiens, chaises), la notion de label est simple : un annotateur regarde l'image et écrit « chat ». Il y a peu d'ambiguïté, l'annotation est rapide, et plusieurs annotateurs convergent presque toujours vers la même réponse.

- **L'annotation est lente.** 
	- Un paléographe expérimenté transcrit entre **1 000 et 3 000** mots par jour sur un manuscrit difficile. 
	- Pour constituer un dataset d'entraînement de 50 000 lignes, il faudrait des années-personnes de travail.
- **L'annotation est qualifiée.** 
	- Lire un manuscrit médiéval ne s'improvise pas. Il faut connaître le système d'écriture de l'époque, les conventions d'abréviation, les dialectes possibles, et souvent le contexte du texte pour lever les ambiguïtés. Les annotateurs doivent être formés.
- **L'annotation est ambiguë.** Comme expliqué en 1.1, deux paléographes experts produisent des transcriptions qui divergent sur 3 à 8% des caractères en moyenne sur les passages difficiles. Il n'existe pas toujours de « bonne » réponse unique.


- Elles imposent de **réfléchir soigneusement** à la granularité (pour maximiser l'utilité de chaque heure d'annotation), à la **constitution des datasets** (pour réduire le besoin d'annotations humaines), et aux **métriques** (pour rendre compte de l'ambiguïté plutôt que de la masquer).

---
## 2. Le choix de la granularité : quelle est l'unité d'apprentissage ?
Première décision architecturale du projet: ==La granularité==. La hiérarchie naturelle d'un manuscrit est la suivante :
- **Document**: Ensemble de pages
+ **Page** : L'unité physique ou numérisée complète, contenant un ou plusieurs blocs de texte.
+ **Bloc** (ou **région** de texte) : Un ensemble de lignes formant une unité logique ou physique sur la page (par exemple, un paragraphe, une colonne, une annotation marginale).
+ **Ligne** : Une suite de mots ou de caractères alignés horizontalement sur le document. C'est une unité cruciale pour la segmentation des documents manuscrits.
+ **Mot** : Une séquence de caractères séparée par des espaces ou des signes de ponctuation.
+ **Caractère ou graphème** : L'unité alphabétique ou typographique de base (lettre, chiffre, signe de ponctuation). C'est l'unité fondamentale que les modèles HTR tentent de reconnaître.
	+ **Glyphe** : La forme graphique concrète d'un caractère (par exemple, la façon dont un 'a' est tracé sur un manuscrit). Un même **caractère** (l'unité abstraite) peut avoir plusieurs glyphes.
- **Trait**: Mouvement continue entre deux levés de plume
notes:
À chaque niveau correspond une famille d'approches, avec ses avantages et ses contraintes.
- Les Glyphes sont adaptés à des CNN, mais modèles séquentiels nécessaires pour les mots
https://medium.com/apache-mxnet/page-segmentation-with-gluon-dcb4e5955e2
https://medium.com/apache-mxnet/handwriting-ocr-line-segmentation-with-gluon-7af419f3a3d8

- Travail au niveau du **trait** (Stroke) - mouvement continue entre 2 levés de plume - en HTR pratiquement impossible et très différent selon le style d'écriture

--
### 2.1 Le niveau du caractère
**Approche classique de l'OCR** : segmenter en caractères, classer chaque caractère selon un alphabet prédéfini (classification multi-classe)
**Pourquoi cela ne fonctionne pas** :
- Segmentation impossible : lettres se touchent, ligatures, pas de frontière physique
- Ambiguïté contextuelle : un jambage vertical isolé peut être *i*, *n*, *m*, *u* ou partie de *r*

> [!tldr]
> Le niveau du caractère isolé est adapté aux textes imprimés, pas aux manuscrits médiévaux. Il est abandonné au profit d'approches séquentielles.

notes:  
Le problème des minimes rend la classification de caractères isolés absurde. Un modèle aveugle au contexte ne peut pas lever l'ambiguïté.


L'approche classique de l'OCR (Optical Character Recognition) pour les documents imprimés : segmenter l'image en caractères individuels, classer chaque caractère parmi un alphabet défini.

**Avantages**
- **Datasets de référence existants** pour certains alphabets (MNIST pour les chiffres, EMNIST pour les lettres latines modernes).
- **Interprétabilité** : on sait exactement où le modèle a fait une erreur.

**Pourquoi cela ne fonctionne pas pour les manuscrits médiévaux**
- *Premier obstacle : la segmentation.* 
	- Pour classer les caractères individuellement, il faut d'abord les isoler. Dans un texte imprimé avec une police régulière, cela est aisé : chaque caractère est séparé des autres par de l'espace blanc. 
	- Dans un manuscrit avec des écritures cursives ou gothiques, les lettres se touchent, se chevauchent et se fondent dans des ligatures. Il n'existe pas de frontière physique entre deux caractères adjacents. La segmentation en caractères est un problème aussi difficile : sinon plus : que la reconnaissance elle-même.
- *Second obstacle : l'ambiguïté contextuelle.* 
	- Le problème des minimes, décrit en 1.1, illustre un fait fondamental : **dans les écritures médiévales, beaucoup de caractères ne sont lisibles qu'en contexte**. Un segment vertical isolé ne peut être identifié comme *i*, *n*, *m*, *u* ou partie de *r* que par un modèle qui considère simultanément ses voisins. Un modèle de classification de caractères isolés est, par construction, aveugle à ce contexte.

--
### 2.2 Le niveau du mot
Une unité intermédiaire: segmenter l'image en images de mots, et prédire le mot pour chaque image.
**Avantages**
- Unité linguistiquement signifiante: lexiques médiévaux peuvent aider à la correction
- Plus petit qu'une phrase entière, réduit les besoin en mémoire pendant l'inférence.
**Difficultés** :
+ Segmentation en mots peu fiable (espaces irréguliers, mots agglutinés), abbreviations présentes
+ Vocabulaire ouvert : l'ancien français sans norme orthographique produit un nombre potentiellement infini de formes

> [!tldr]
> Le niveau du mot est une option théorique rarement implémentée en pratique. Il est généralement abandonné au profit du niveau de la ligne.

notes:

**Avantages**
- Le mot est une unité linguistiquement signifiante : un lexique médiéval peut potentiellement aider à la correction.
- Les images de mots sont plus petites que les images de lignes, ce qui réduit les besoins en mémoire.

**Pourquoi ce niveau est rarement utilisé en HTR**
- *La segmentation en mots est difficile.* L'espace entre les mots n'est pas toujours présent ni régulier dans les manuscrits médiévaux. Les mots peuvent se toucher. Les abréviations créent des mots graphiquement courts mais linguistiquement complexes. La segmentation en mots est un problème qui n'est pas résolu de façon robuste pour les écritures médiévales.
- *Le vocabulaire est ouvert et non normalisé.* Pour des textes imprimés modernes, on peut envisager un dictionnaire de mots connus. Pour l'ancien français sans norme orthographique, le nombre de formes distinctes d'un même mot est potentiellement illimité. Un modèle entraîné à prédire des mots dans un vocabulaire fermé sera incapable de traiter les formes inconnues.

--
### 2.3 Le niveau de la ligne : le choix canonique de l'HTR
1. **Standard de la communauté HTR depuis les années 2010**
	- Entrée : [image de ligne ~= 60px × 1200px]
	- Sortie : `"En cel tems que li rois Artus regnoit en la grant Bretaigne"`
2. **Pourquoi ce niveau est optimal** :
	- Évite la segmentation en caractères
	- Fournit assez de contexte local (40–80 caractères)
	- Correspond à une unité naturelle : les lignes sont séparées par de l'espace blanc
	- Annotations raisonnablement rapides : 300–600 lignes/heure
3. **Conséquence architecturale** : pipeline en deux étapes
	1. Segmentation de layout et de lignes (SAM, Kraken segment, U-Net)
	2. HTR sur chaque ligne



notes:  
C'est **le standard de la communauté HTR** depuis le début des années 2010. L'entrée est une image de ligne de texte (une bande horizontale extraite de la page) ; la sortie est la chaîne de caractères correspondant à cette ligne.

```
Entrée : [image de ligne ~= 60px × 1200px]
Sortie : "En cel tems que li rois Artus regnoit en la grant Bretaigne"
```

**Pourquoi ce niveau est optimal**
- *Il évite la segmentation en caractères.* Pas besoin d'isoler les lettres individuelles : le modèle reçoit la ligne entière et produit la transcription entière.
- *Il fournit assez de contexte.* Une ligne de texte contient typiquement 40 à 80 caractères : suffisamment pour que le modèle exploite le contexte local pour lever les ambiguïtés graphiques.
- *Il correspond à une unité naturelle du document.* La ligne est la structure élémentaire de la page manuscrite : les scripteurs écrivent ligne par ligne, et les lignes sont séparées par de l'espace blanc : même si cet espace est irrégulier, il est en général suffisant pour une segmentation automatique.
- *Les annotations au niveau de la ligne sont raisonnablement rapides.* Un annotateur peut produire entre 300 et 600 annotations de lignes par heure pour des textes médiévaux relativement lisibles : soit 3 à 6 fois plus vite qu'une transcription intégrale de page avec structuration.
- *Les architectures modernes sont conçues pour ce niveau.* TrOCR, Kraken, HTR+ et la plupart des modèles HTR de référence opèrent au niveau de la ligne.

**La conséquence architecturale**
Travailler au niveau de la ligne implique un **pipeline en deux étapes** :
1. **Segmentation de layout et de lignes** : identifier et extraire les lignes de texte dans la page. C'est un problème de vision par ordinateur distinct, traité en Jour 2–3 avec des outils comme SAM, Kraken segment, ou des modèles U-Net.
2. **HTR proprement dit** : reconnaître le texte dans chaque image de ligne extraite. C'est le problème principal du module.

Ces deux étapes ont leurs propres sources d'erreur, et les erreurs de la première se propagent dans la seconde (une ligne mal segmentée : coupée en deux, ou incluant du texte de la ligne suivante : produit une transcription dégradée même si le modèle HTR est excellent).

> [!danger] Attention
> Les erreurs de segmentation de lignes se propagent à l'étape HTR. Une ligne mal coupée dégrade la transcription même avec un excellent modèle HTR.

--
### 2.4 Le niveau de la région
- **Classification de type de document** (charte vs roman, XIIIe vs XVe siècle)
- **Description des illustrations** (enluminures)

→ Outils : DINO, CLIP (vus au Jour 3)

notes:

Pour certaines tâches, on travaille à un niveau supérieur à la ligne : la région (colonne, paragraphe, bloc de texte). C'est le niveau pertinent pour la **classification de type de document** (charte ou roman ? XIIIe ou XVe siècle ?) et pour la **description des illustrations** (que représente cette enluminure ?).

Ces tâches seront traitées avec DINO et CLIP en Jour 3.

--
### 2.5 Synthèse : granularités et tâches associées

| Granularité | Tâche                       | Outil principal        | Difficulté spécifique                        |
| ----------- | --------------------------- | ---------------------- | -------------------------------------------- |
| Trait       | Analyse du geste graphique  | Non traité ici         | Requiert des données d'écriture en ligne     |
| Caractère   | Classification              | Non retenu             | Segmentation impossible en cursive           |
| Mot         | Reconnaissance              | Non retenu             | Vocabulaire ouvert, segmentation difficile   |
| Chunk       | HTR segmentée               | CNN + RNN, TrOCR       | Découpage arbitraire, perte de contexte      |
| **Ligne**   | **HTR**                     | **TrOCR, Kraken**      | **Segmentation de lignes (étape préalable)** |
| Région/page | Classification, description | DINO, CLIP             | Annotations de niveau supérieur              |
| Document    | Identification du manuscrit | Métadonnées + features | Hors périmètre du pipeline HTR               |

---
## 3. La vérité terrain : une notion à redéfinir

--
### 3.1 La vérité terrain en apprentissage supervisé classique

![[Cours_1_2_Philologie_vers_ML-1778770143695.png]]

notes:
- Gold standard - vérité terrain (ground truth): réponse correcte, unique, fournie par un expert
- Hypothèse rarement questionnée

Pour 98% des images du dataset ImageNet, il n'y a pas de débat sur l'étiquette correcte : un chien est un chien.

Cette unicité de la vérité terrain est une **hypothèse de travail, rarement questionnée** en apprentissage automatique classique. Elle est profondément remise en cause par les manuscrits médiévaux.

--
### 3.2 Pourquoi la vérité terrain est problématique ici

![[Cours_1_2_Philologie_vers_ML-1778770860688.png]]


notes:
- Regardons la profession (dernière colonne)
	- Cultivat(eur)
	- Ménagère
	- Cultivat.
	- Cultiv.
	- id.

- Colonne du milieu est la nationalité: id (idem) car première ligne (hors capture d'écran) écrit Française.
	- Est-ce qu'il faut écrire NAtionalité = id ? idem ? Français/Française
	- Est-ce que la profession est cultiv? cultivat ? id ?


- **Ambiguïté graphique irréductible.** Sur certains caractères, même un expert ne peut pas trancher avec certitude. Il peut écrire *e* ou *o*, *d* ou *a*, parce que le signe sur le parchemin est équivoque. La vérité terrain elle-même est incertaine.
- **Choix éditoriaux divergents.** Développer une abréviation d'une façon ou d'une autre (par exemple, le signe qui note *con-*, *com-*, *cum-* ou *cun-* selon les cas) relève d'une interprétation linguistique, pas d'une lecture graphique pure. Deux experts font des choix différents, tous deux légitimes.
- **Variabilité inter-annotateur mesurable.** -> Projets collaboratifs ont prouvé que l'accord inter-annotateur peut descendre à 92% pour certains caractères. 
	- peut paraître faible mais 3 à 8 caractères sur 100 sachant qu'il y a moins de 100 caractères en une ligne veut dire plus d'une erreur par ligne.
- Les études empiriques menées dans des projets collaboratifs (Transkribus, eScriptorium, Wikisource) mesurent systématiquement ce que l'on appelle le **taux d'accord inter-annotateurs** (*inter-annotator agreement* ou IAA). Pour les manuscrits médiévaux difficiles, ce taux descend à 92–97% au niveau du caractère : ce qui peut sembler élevé, mais représente une divergence de 3 à 8 caractères sur cent, soit plusieurs erreurs par ligne.
--
### 3.2.b Mesurer l’accord entre annotateurs : Kappa de Cohen et Alpha de Krippendorff

- **Kappa de Cohen** (pour 2 annotateurs)  
  $$\kappa = \frac{P_o - P_e}{1 - P_e}$$  
  - $P_o$ = proportion d’accord observé  
  - $P_e$ = proportion d’accord attendu par hasard  
  - $\kappa=1$ : accord parfait ; $\kappa=0$ : accord aléatoire ; $\kappa<0$ : désaccord systématique

- **Alpha de Krippendorff** (généralise à plus de 2 annotateurs, variables nominales/ordinales)  
  - Utilisé en HTR pour évaluer la fiabilité des transcriptions multiples.

> **Seuils usuels** : $\kappa > 0.8$ = excellent accord ; $0.6-0.8$ = bon ; $<0.6$ = discutable.

notes:
Les projets collaboratifs (Transkribus, eScriptorium) calculent systématiquement l’accord inter-annotateurs. Un faible Kappa indique des règles d’annotation floues ou un manuscrit très ambigu. Cela aide à décider si on peut utiliser ces données comme vérité terrain.

--
### 3.3 Ce que cela implique pour la conception du dataset

**Première implication : plusieurs annotations valent mieux qu'une seule.**
Si une seule personne annote chaque ligne, les erreurs et les choix idiosyncratiques de cet annotateur contaminent tout le dataset. Il est préférable d'avoir plusieurs annotateurs indépendants, même moins experts, dont on combine les transcriptions.

Cette approche est adoptée par les grands projets de transcription collaborative (Zooniverse, FromThePage) : une foule de contributeurs non-spécialistes (*crowdsourcing*) produit de nombreuses transcriptions par ligne, et un algorithme de consensus détermine la transcription de référence.

**Deuxième implication : la transcription de référence doit encoder l'incertitude.**
Plutôt que de choisir arbitrairement une transcription parmi plusieurs divergentes, on peut encoder l'incertitude explicitement :

```xml
<choice>
  <sic>o</sic>
  <corr cert="medium">e</corr>
</choice>
```

En format JSON pour notre pipeline :
```json
{
  "line_id": "l047",
  "transcription": "En cel tems que li rois",
  "confidence": 0.84,
  "uncertain_chars": [
    {"position": 7, "candidates": ["e", "o"], "probabilities": [0.72, 0.28]}
  ]
}
```

**Troisième implication : les métriques doivent tenir compte de la variabilité.**
Un modèle évalué contre une seule transcription de référence sera pénalisé pour des erreurs qui ne sont pas des erreurs : des choix différents mais également valides. Une évaluation plus honnête compare le modèle contre *plusieurs* transcriptions de référence et retient le meilleur score, ou calcule une moyenne.

--
### 3.4 Niveaux de fidélité de la transcription et leurs vérités terrain respectives

Un point crucial, souvent source de confusion dans les projets HTR pour les humanités : **la vérité terrain dépend du niveau de fidélité choisi**.

**Transcription diplomatique stricte**
On transcrit exactement ce qu'on voit, y compris les signes d'abréviation, sans les développer. Le signe tilde au-dessus d'un *o* est transcrit comme `õ` ou avec une convention balisée. Les lettres malformées sont reproduites telles quelles.

Cette vérité terrain est la plus *reproductible* : deux annotateurs qui se sont mis d'accord sur les conventions de notation produiront des transcriptions plus proches. Elle est la moins *interprétative*.

**Transcription semi-diplomatique**
On développe les abréviations les plus claires (celles dont le développement est conventionnellement admis), on signale les incertitudes, mais on ne normalise pas l'orthographe. C'est le format le plus courant dans les datasets HTR professionnels.

**Transcription normalisée**
On normalise l'orthographe, on développe toutes les abréviations, on corrige les erreurs manifestes du scribe. Cette transcription est plus lisible mais plus subjective. Elle est utile comme cible pour le module NLP, moins comme vérité terrain pour l'HTR.

> **Pour notre projet :** nous visons une **transcription semi-diplomatique** comme vérité terrain : développement des abréviations claires, conservation de l'orthographe médiévale, signalement des incertitudes. Ce choix sera documenté dans le data contract livré au module NLP.

---
## 4. Les métriques d'évaluation

Choisir une métrique, c'est définir ce qu'on entend par « le modèle fonctionne bien ». Un mauvais choix de métrique conduit à optimiser la mauvaise quantité. Dans ce domaine comme dans beaucoup d'autres en ML, les métriques ne sont pas neutres : elles reflètent des jugements sur ce qui compte.

--
### 4.1.1 Le Character Error Rate (CER)
Le CER est **la métrique standard de toute transcription** (ou traduction). Il mesure le taux d'erreur au niveau du caractère. 
II est calculé à partir de la **distance de Levenshtein** entre la transcription prédite et la transcription de référence: le nombre minimal d'opérations élémentaires nécessaires pour transformer l'une en l'autre.
- $S$ -**Substitution** : remplacer un caractère par un autre (*a* → *o*).
- $I$ - **Insertion** : ajouter un caractère absent de la référence.
- $D$ - **Suppression** (ou délétion) : supprimer un caractère présent dans la référence.
- $N$ = nombre total de caractères dans la **référence****
$$\text{CER \% } = \frac{S + I + D}{N}$$


notes:
- Le CER est exprimé en pourcentage. Un CER de 0% signifie que la transcription prédite est identique à la référence. Un CER de 100% signifie que la prédiction est aussi éloignée de la référence qu'une chaîne entièrement différente.

**Note importante** : le CER peut dépasser 100%. Si la prédiction est beaucoup plus longue que la référence (nombreuses insertions), $S + I + D$ peut être supérieur à $N$.

**Pourquoi le CER et pas l'exactitude par caractère ?**

L'exactitude (*accuracy*) serait le complément du taux d'erreur : $1 - \text{CER}$. On pourrait utiliser cette mesure à la place. En pratique, le CER est préféré parce qu'il est *additif* sur plusieurs lignes : on peut sommer les $(S + I + D)$ et les $N$ de toutes les lignes d'un document, et calculer un CER global qui reflète fidèlement la performance sur l'ensemble du document.

---
### 4.1 .2 - CER en pratique
![[Cours_1_2_Philologie_vers_ML-1778613285756.jpg]]

```
manuscripts
de Voltaire
Recueil
de
Lettres de Voltaire.
Années 1722 à 1778.

X
Comme-lipienne.

A Paris,
Jan. m. 1° de la République.
(1795.)
```

$$\text{CER \% } = \frac{S + I + D}{129}$$
--
### 4.2 Le Word Error Rate (WER)

Le WER applique exactement la même formule que le CER, mais à l'échelle des **mots** plutôt que des caractères.

$$\text{WER} = \frac{S_w + I_w + D_w}{N_w}$$

où les indices $w$ indiquent que les opérations sont comptées au niveau du mot, et $N_w$ est le nombre de mots dans la référence.

**Exemple numérique**

Référence : `"li rois Artus regnoit"`  
Prédiction : `"li rous Artus regnoit"`

*rois* est prédit comme *rous* : une substitution de mot. Trois mots corrects.

$$\text{WER} = \frac{1}{4} = 25\%$$

alors que le CER sur le même passage serait :

$$\text{CER} = \frac{1}{20} = 5\%$$

**Le WER est une métrique plus sévère que le CER**, et sensiblement plus sévère dans le contexte médiéval, pour deux raisons :

*Première raison : les mots courts.* En ancien français, de nombreux mots fréquents sont très courts (*li*, *la*, *le*, *de*, *en*, *et*…). Une substitution d'un seul caractère dans *li* → *la* est une erreur d'un caractère (faible CER), mais une erreur d'un mot entier (fort impact sur le WER).

*Seconde raison : l'orthographe variable.* Si le modèle prédit *rei* et la référence est *roi*, il s'agit d'une substitution d'un mot entier selon le WER, alors que le CER ne compte qu'un seul caractère différent. Or *rei* est une forme authentique et légitime de *roi* en ancien français : l'erreur est discutable.

**Verdict :** le WER est une métrique légitime pour évaluer la lisibilité globale d'une transcription, mais il pénalise trop sévèrement les variations orthographiques légitimes de l'ancien français. Dans notre projet, le **CER est la métrique principale**. Le WER est calculé à titre indicatif.

**Bonnes pratiques**
- Utiliser des scripts normalisés (ex. suppression de la ponctuation, mise en minuscules).
- Compléter par des métriques alternatives comme le **bWER** (ignorant l’ordre des mots) ou le **CERberus**, qui pondère les erreurs selon leur impact (ex.  `é` vs `e` moins grave que `e` vs `0`).
- https://github.com/WHaverals/CERberus/tree/main

---
### 4.2 .2 - CER en pratique
![[Cours_1_2_Philologie_vers_ML-1778613285756.jpg]]

```
manuscripts
de Voltaire
Recueil
de
Lettres de Voltaire.
Années 1722 à 1778.

X
Comme-lipienne.

A Paris,
Jan. m. 1° de la République.
(1795.)
```

--
### 4.2.b Orthographe historique : pas de « faute » avant le XVIIIe siècle

Avant la seconde moitié du XVIIIe siècle, **il n’existe aucune autorité normative** (Académie française, dictionnaire unifié) imposant une orthographe unique.  
- Les formes *roy / roi / rey / roix* coexistent légitimement.  
- Un scribe normand écrit différemment d’un scribe picard, sans que l’un ou l’autre soit « faux ».  
- Ce que l’on appellerait aujourd’hui une « faute » (ex : *j’ai deu* pour *j’ai dû*) est simplement une graphie ancienne.

> **Conséquence pour l’HTR** : ne jamais corriger l’orthographe d’un manuscrit médiéval sous prétexte qu’elle est « erronée ». La transcription doit être **fidèle** à la graphie du scribe. La normalisation viendra après, en NLP, avec un dictionnaire historique.

notes:
C’est un point souvent mal compris par les informaticiens. Le CER calculé contre une référence « corrigée » pénaliserait injustement le modèle. Il faut donc que la vérité terrain utilise la même convention orthographique que l’original – d’où l’importance de la transcription diplomatique ou semi-diplomatique.

--
### 4.3 Les limites des métriques dans le contexte médiéval
Même un CER bien conçu présente des limites structurelles importantes dans notre contexte.

**Le problème de l'orthographe variable**
Le CER suppose l'existence d'une référence univoque. Si le modèle prédit `roi` et que la référence est `rey`, le CER compte une substitution. Mais `rei`, `roi`, `roy`, `rex` sont toutes des formes attestées du même mot en ancien français. La pénalité est injuste.

Une approche partielle consiste à **normaliser** à la fois la référence et la prédiction avant de calculer le CER : convertir toutes les formes en une forme canonique. Mais cela suppose un dictionnaire de normalisation de l'ancien français : qui n'existe pas de façon exhaustive, précisément parce que l'orthographe médiévale est ouverte.

**Le problème des abréviations développées**

Si la vérité terrain développe les abréviations et que le modèle les laisse contractées (ou inversement), le CER pénalise fortement des différences qui sont des choix de représentation, pas des erreurs de lecture.

Solution : documenter clairement le niveau de fidélité de la vérité terrain, et s'assurer que le modèle est évalué contre une référence du même niveau de fidélité que ce qu'il a appris à produire.

**Le problème des espaces et de la segmentation en mots**

En ancien français, certains mots sont souvent agglutinés sans espace dans les manuscrits (*del*, *au*, *quel*…). Si le modèle insère ou omet un espace, le CER compte une opération, mais le WER peut compter plusieurs mots différents. Ce traitement asymétrique des espaces est une source de distorsion dans les métriques.

**Le problème de la longueur inégale des lignes**

Le CER agrège tous les caractères de toutes les lignes. Une ligne de 80 caractères contribue 8 fois plus qu'une ligne de 10 caractères. Les lignes courtes (lettrines isolées, titres de chapitres) ont souvent un CER élevé par nature (peu de contexte), mais leur contribution au CER global est faible. Il faut examiner les distributions d'erreur par type de ligne, pas seulement le CER global.

--
### 4.3.b Multilinguisme : latin, grec, vernaculaire mêlés

Dans les manuscrits médiévaux, plusieurs langues cohabitent fréquemment sur la même page, voire dans la même ligne.

**Exemples** :  
- Textes religieux : latin + gloses en ancien français  
- Traités scientifiques : latin + grec translittéré ou non  
- Chartes : latin + quelques formules en français  
- Registres : mixte selon la colonne

**Stratégies pour l’HTR** :  
- Détecter la **langue dominante par région** (bloc ou ligne) avant la transcription.  
- Utiliser des modèles de langue spécifiques (latin, vieux français, moyen français) en post-traitement.  
- Pour les caractères grecs : conserver les glyphes ou translittérer selon une norme (ISO 843).

> Dans le pipeline, on peut ajouter une étape de classification de langue par région (avec un petit CNN ou DINO) avant d’appliquer le modèle HTR approprié.

notes:
Le multilinguisme est une difficulté supplémentaire. Les modèles pré-entraînés (TrOCR) sont principalement entraînés sur du latin moderne / anglais, donc peu robustes. L’idéal est d’avoir des modèles spécialisés par langue (via HTR-United).

--
### 4.4  Éléments non textuels : rubriques, lettrines, notes marginales, illustrations

- **Rubriques** (titres en rouge) : utilisent un canal couleur distinct. En computer vision, segmentation par seuillage HSV (teinte) pour isoler le rouge.
- **Lettrines** : grandes initiales souvent ornées. Leur forme ne correspond pas à la police standard. Il faut les détecter comme région spéciale et les transcrire comme la lettre correspondante (parfois avec métadonnées).
- **Notes marginales** :
  - **Position** : marge haute, basse, latérale.  
  - **Orientation** : parfois écrites à 90° (rotation du texte). Nécessite une détection de ligne non horizontale (algorithme de skew local).
- **Illustrations** : doivent être segmentées (boîte englobante) et traitées par un modèle vision-langage (CLIP, LLaVA) pour générer une description.

> Le pipeline de segmentation de layout (SAM) doit distinguer ces régions et les étiqueter (texte normal, rubrique, lettrine, note, illustration). Chaque type aura un traitement différent.

notes:
Les outils comme Transkribus ou eScriptorium intègrent ces fonctionnalités. Pour les notes marginales à 90°, on peut tourner la région avant de passer le modèle HTR. La classification par couleur (rouge) est très fiable pour les rubriques.


--
### 4.5 Objectifs de précision pour le projet
Les valeurs suivantes sont des **références de la communauté HTR**. Elles correspondent aux performances observées sur des manuscrits médiévaux européens dans des projets récents (Transkribus, CREMMA, eScriptorium).

| Niveau                                    | CER cible | Interprétation                                 |
| ----------------------------------------- | --------- | ---------------------------------------------- |
| Pipeline de base (zéro fine-tuning)       | 25–40%    | Modèle pré-entraîné sur des données génériques |
| Fine-tuning minimal (< 500 lignes)        | 15–25%    | Amélioration significative avec peu de données |
| Fine-tuning standard (1 000–5 000 lignes) | 8–15%     | Niveau opérationnel pour la recherche          |
| **Cible projet (avant NLP)**              | **< 12%** | **Objectif du module CV**                      |
| Après correction NLP                      | 3–7%      | Objectif du module NLP                         |
| Expert humain (inter-annotateur)          | 3–8%      | Plafond indicatif de la performance humaine    |

> **Un CER de 12% signifie concrètement** : sur une ligne de 60 caractères, environ 7 caractères sont incorrects. Le texte reste généralement lisible par un lecteur humain qui connaît le domaine, mais nécessite une révision pour une utilisation académique rigoureuse.

---
## 5. Le problème de l'annotation rare : stratégies pour constituer un dataset

--
### 5.1 L'apprentissage par transfert comme point de départ

La contrainte fondamentale du projet est la rareté des données annotées. Il est hors de portée de constituer, dans le temps du cours, un dataset de plusieurs milliers de lignes annotées par des experts.

La réponse de l'état de l'art est le **transfert d'apprentissage** (*transfer learning*) : utiliser un modèle pré-entraîné sur une large quantité de données d'une tâche voisine, et le spécialiser (*fine-tuner*) sur les données spécifiques à notre problème.

Concrètement :
- TrOCR a été pré-entraîné sur des millions de lignes de texte manuscrit moderne (dataset IAM, synthétique…). Il a appris des représentations génériques de l'écriture manuscrite.
- On lui soumet quelques centaines à quelques milliers de lignes de manuscrits médiévaux annotées, et on ajuste ses paramètres pour qu'il s'adapte à ce sous-domaine.

La quantité de données nécessaires pour le fine-tuning est **bien inférieure** à ce qu'il faudrait pour entraîner le modèle de zéro. Des résultats acceptables peuvent être obtenus avec 500 à 2 000 lignes annotées pour un scripteur ou un corpus relativement homogène.

--
### 5.2 La data augmentation pour l'HTR

La data augmentation consiste à générer des variantes synthétiques des données existantes pour augmenter artificiellement la taille du dataset et la robustesse du modèle.

Pour les images de lignes de manuscrits, les augmentations pertinentes sont :

**Augmentations géométriques**
- Légères rotations (±2°) pour simuler des lignes légèrement inclinées.
- Déformation élastique : une déformation locale aléatoire des pixels, qui simule l'irrégularité naturelle du geste d'écriture.
- Modification de l'aspect ratio horizontal pour simuler des lignes comprimées ou étirées.

**Augmentations photométriques**
- Variations de luminosité et de contraste pour simuler l'hétérogénéité des numérisations.
- Ajout de bruit gaussien pour simuler des imperfections de numérisation.
- Simulation de taches d'encre ou de dégradations locales.
- Simulation du *show-through* (transparence du parchemin) en superposant partiellement une image de ligne miroir.

**Augmentations de style**
Des techniques plus avancées permettent de générer de nouvelles images de lignes en synthétisant un texte connu dans un style d'écriture appris à partir d'exemples. Ces techniques (GANs, modèles de diffusion) sont actives dans la recherche HTR, mais dépassent le périmètre de ce cours.

Une règle empirique : la data augmentation améliore la robustesse mais ne remplace pas les données réelles. Elle est particulièrement efficace pour les transformations que le modèle rencontrera réellement en production.

--
### 5.3 L'utilisation de datasets existants : HTR-United et CREMMA

La voie la plus rapide pour constituer un dataset d'entraînement solide est de partir des datasets existants, ouverts, annotés par des experts. Comme détaillé dans l'Annexe du plan de cours général, les ressources clés sont :

- **HTR-United** : catalogue de datasets HTR sous licences ouvertes, incluant plusieurs corpus de vieux et moyen français médiéval.
- **CREMMA Médiéval** : corpus spécifiquement constitué pour l'HTR en ancien et moyen français, au format PAGE XML, compatible avec Kraken et eScriptorium.

Ces datasets peuvent être utilisés directement comme données d'entraînement, ou comme point de départ pour un fine-tuning supplémentaire sur le sous-corpus spécifique du projet.

---
## 6. Des estimateurs faibles à l'estimateur fort : le pooling de transcriptions

--
### 6.1 L'intuition
Revenons au constat de départ : les experts humains ne s'accordent pas parfaitement, et les modèles font des erreurs différentes selon leur architecture et leur entraînement. Un modèle entraîné sur des chartes du XIIIe siècle sera plus performant que les autres sur ce type de documents, mais moins bon sur des romans du XIVe siècle. Un autre modèle présentait le profil inverse.

Au lieu de chercher un seul modèle parfait (qui n'existe pas), on peut combiner plusieurs modèles imparfaits de façon à ce que leurs erreurs se compensent partiellement. C'est le principe des **méthodes d'ensemble** (*ensemble methods*) en apprentissage automatique, appliqué ici à la transcription.

Cette idée n'est pas nouvelle en philologie non plus : l'édition critique d'un texte médiéval consiste précisément à comparer plusieurs manuscrits (les *témoins*) d'un même texte pour reconstituer la forme la plus proche de l'original : en supposant que les erreurs de copie sont distribuées de façon aléatoire entre les témoins indépendants, et qu'un passage confirmé par la majorité des témoins a plus de chances d'être correct.

--
### 6.1.a Théorème de Condorcet (1785)
**Principe** : si chaque votant a une probabilité $p > 0.5$ de prendre la bonne décision (indépendamment), alors le **vote majoritaire** augmente la probabilité de la bonne décision. Plus le groupe est grand, plus cette probabilité tend vers 1.
- **Hypothèses fortes** : indépendance des jugements, deux choix seulement, tous les votants ont la même compétence $p$.
- En ML, cela justifie les **méthodes d’ensemble** : combiner plusieurs modèles faibles pour obtenir un modèle fort.
notes:
Contre-exemple : si $p < 0.5$, le groupe est pire qu’un individu seul. C’est pourquoi on n’agrège pas des modèles systématiquement mauvais. Pour la transcription, on n’utilise que des modèles dont le CER est < 50% (ce qui est toujours le cas après pré-entraînement).

--
### 6.2 Le vote majoritaire

La méthode la plus simple : on dispose de $n$ transcriptions d'une même ligne (produites par $n$ modèles différents, ou par $n$ annotateurs humains). Pour chaque position dans la séquence de caractères, on retient le caractère le plus fréquent.

**Exemple** avec 3 transcriptions :

```
Transcription 1 : "li rois Artus regnoit"
Transcription 2 : "li rous Artus regnoit"
Transcription 3 : "li rois Artus regnoit"
```

Position 4 (*i* vs *o*) : vote 2 pour *i*, 1 pour *o* → on retient *i*. Résultat correct.

**Limite du vote simple.** Les transcriptions ne sont généralement pas alignées caractère par caractère de façon triviale. Des insertions ou des suppressions dans l'une des transcriptions décalent toute la suite. Il faut d'abord **aligner** les transcriptions avant de voter.

--
### 6.1.b Sagesse des foules (wisdom of the crowd)

- Phénomène général : l’agrégation d’opinions indépendantes de nombreux individus (même non experts) peut surpasser l’expert isolé.
- **Conditions** : diversité des opinions, indépendance, décentralisation, agrégation (vote/moyenne).
- **Application directe en HTR** :  
  - Crowdsourcing de transcriptions (Zooniverse, FromThePage)  
  - Plusieurs annotations par ligne, puis consensus (vote majoritaire ou modèle probabiliste)

> Dans notre TP, le CER inter-annotateurs entre étudiants est la base de la « sagesse de la classe » : une transcription combinée par vote serait meilleure que la moyenne individuelle.

notes:
C’est le principe des projets comme « Transcribe Bentham » : des centaines de volontaires non-paléographes produisent des transcriptions qui, aggrégées, atteignent la qualité experte. Cependant, la condition d’indépendance est cruciale (pas de copie, pas de discussion).

--
### 6.3 L'alignement multiple de séquences

L'alignement de deux séquences de caractères s'effectue avec l'algorithme de Needleman-Wunsch (ou ses variantes, notamment Smith-Waterman pour l'alignement local). C'est un algorithme de programmation dynamique classique, identique dans son principe à celui utilisé en bioinformatique pour l'alignement de séquences d'ADN.

Pour trois transcriptions ou plus, on peut utiliser un alignement multiple progressif (aligner d'abord les deux transcriptions les plus proches, puis ajouter les suivantes) ou un algorithme dédié à l'alignement multiple de séquences courtes.

Une fois les transcriptions alignées, le vote position par position est bien défini.

--
### 6.4 Le vote pondéré par la confiance

Tous les modèles ne sont pas également fiables. Si l'on sait a priori que le modèle A est meilleur que le modèle B sur un type de document donné, il serait dommage de les traiter à égalité dans le vote.

La plupart des modèles HTR produisent, en plus de la transcription, un **score de confiance** pour chaque caractère ou chaque token : typiquement la probabilité softmax de la prédiction. Ces scores peuvent servir à pondérer le vote :

```
Transcription A, "rois", confiance = 0.92
Transcription B, "rous", confiance = 0.61
Transcription C, "rois", confiance = 0.88
```

Vote pondéré pour *rois* : $0.92 + 0.88 = 1.80$  
Vote pondéré pour *rous* : $0.61$  
→ On retient *rois* avec une confiance agrégée de $1.80 / (1.80 + 0.61) \approx 0.75$.

Ce score de confiance agrégé peut être propagé dans le JSON de sortie comme indicateur de fiabilité de la transcription : information précieuse pour le module NLP qui décidera quelles lignes méritent une révision prioritaire.

--
### 6.4.b Biais systématique vs bruit aléatoire

| Type d’erreur | Définition | Exemple en transcription | Correction |
|---------------|------------|--------------------------|-------------|
| **Biais systématique** | Erreur cohérente, répétée par un annotateur/modèle | Un annotateur lit toujours *u* pour *n* en gothique | Recalibration, pondération faible |
| **Bruit aléatoire** | Variations incohérentes, non prédictibles | Une même lettre tantôt *c* tantôt *e* selon le contexte | Augmentation de données, consensus |

> Pour l’agrégation, il faut détecter et pénaliser les annotateurs biaisés, mais le bruit aléatoire se corrige par le vote majoritaire.

notes:
Dans les modèles comme Dawid-Skene, on estime la matrice de confusion de chaque annotateur (biais) et on l’utilise pour pondérer. Le bruit aléatoire augmente la variance mais peut être moyenné.


--
### 6.4.c Modèles d’agrégation d’annotations : Dawid-Skene, GLAD, MACE

**Dawid-Skene (1979)** – Modèle probabiliste de référence  
- Estime simultanément la **vérité cachée** et la **compétence** de chaque annotateur (taux d’erreur par classe).  
- Algorithme EM (Expectation-Maximisation) : itère entre estimer la vérité (pondérée par compétence) et estimer la compétence (à partir de la vérité).

**GLAD (Generative Model of Labels, Abilities, and Difficulties)** – Étend Dawid-Skene  
- Ajoute un paramètre de **difficulté de l’item** (ex : une ligne très dégradée sera difficile même pour un bon annotateur).

**MACE (Majority Agreement and Competence Estimation)** – Robuste aux annotateurs malveillants ou biaisés  
- Modélise la probabilité que chaque annotateur suive une stratégie cohérente (biais) plutôt qu’aléatoire.

> Ces modèles sont utilisés dans les plateformes de crowdsourcing (Amazon Mechanical Turk) et peuvent être appliqués aux transcriptions HTR pour produire une vérité terrain de meilleure qualité que le simple vote.

notes:
**Attention** : la description de MACE dans le document original était erronée (il n’est pas spécialisé pour les séquences temporelles). Ici, on corrige : MACE est général pour annotations catégorielles, avec détection d’annotateurs biaisés.

--
### 6.4.d Lien avec l’ensemble learning (bagging, boosting)

- **Bagging** (Bootstrap Aggregating) – Réduit la variance  
  - Entraîne plusieurs modèles sur des échantillons bootstrap de données.  
  - Agrège par vote ou moyenne.  
  - Exemple : Random Forest.  
  - **Application HTR** : entraîner plusieurs TrOCR sur des sous-ensembles de lignes, puis voter.

- **Boosting** – Réduit le biais  
  - Entraîne séquentiellement des modèles faibles, chaque nouveau modèle corrige les erreurs du précédent (pondération des exemples mal classés).  
  - Exemple : AdaBoost, Gradient Boosting.  
  - **Application HTR** : moins courant, mais possible pour la correction de caractères spécifiques.

> Dans notre pipeline, le **vote pondéré** entre plusieurs modèles HTR (TrOCR, Kraken, etc.) relève du bagging (indépendance). Le boosting pourrait être utilisé pour affiner les décisions sur les caractères ambigus.

notes:
Le bagging suppose que les modèles sont indépendants (erreurs non corrélées). On obtient cette indépendance en variant les architectures, les pré-entraînements, ou les augmentations de données.

--
### 6.5 Intégration des transcriptions humaines comme signal de supervision

Les transcriptions manuelles produites en TP (section 1.3) ne sont pas de simples exercices pédagogiques. Elles constituent une **donnée de supervision** de haute qualité : une annotation humaine est, par définition, le signal le plus fiable disponible.

Dans le pipeline final, on peut :
- Utiliser les transcriptions humaines comme vérité terrain pour le fine-tuning.
- Les intégrer dans le vote pondéré avec un poids élevé (par exemple, 3 fois le poids d'une transcription automatique).
- S'en servir comme référence pour calibrer les scores de confiance des modèles automatiques.

Cette intégration fait le lien entre l'activité de transcription manuelle du TP et l'usage des données en entraînement : ce n'est pas un hasard si le cours commence par une heure de transcription à la main.

---
## 7. Synthèse : spécifications techniques du projet

À l'issue de cette section, nous pouvons formaliser les spécifications du projet HTR.

--
### 7.1 Résumé des choix fondamentaux

| Dimension | Choix | Justification |
|-----------|-------|---------------|
| **Unité d'apprentissage** | Ligne de texte | Standard HTR, évite la segmentation en caractères, fournit assez de contexte |
| **Type de transcription** | Semi-diplomatique | Reproductible, non normalisée, compatible avec le module NLP |
| **Métriques principales** | CER (principal), WER (secondaire) | CER standard HTR ; WER indicatif de lisibilité |
| **Cible de performance** | CER < 12% | Niveau opérationnel pour la recherche, atteignable avec fine-tuning |
| **Gestion de l'ambiguïté** | Confiance par caractère + flag de révision | Propagation de l'incertitude vers le module NLP |
| **Stratégie d'ensemble** | Vote pondéré par confiance, 2–3 modèles | Compense les erreurs spécifiques à chaque modèle |
| **Format de sortie** | JSON + TEI P5 optionnel | Lisible par les outils des humanités numériques |

--
### 7.2 Ce que le module NLP recevra

Le module NLP recevra un dataset de transcriptions avec les caractéristiques suivantes :
- Chaque ligne est transcrite en vieux ou moyen français semi-diplomatique.
- Chaque ligne est accompagnée d'un score de confiance global et d'une liste de positions incertaines.
- Le dataset contient des métadonnées par page : siècle estimé, type de document, qualité de numérisation.
- Un flag `needs_review` signale les lignes où la confiance est inférieure à un seuil défini.

Le module NLP pourra alors travailler à la normalisation orthographique, au développement des abréviations restantes, et à la correction contextuelle : en sachant exactement quelles parties de la transcription sont fiables et lesquelles sont incertaines.

---
## Bibliographie de référence

--
### Métriques et évaluation en HTR

- **Romero, V., Toselli, A. H., Vidal, E.** (2012). *Multimodal Interactive Handwritten Text Transcription*. World Scientific. : Définition rigoureuse du CER et du WER dans le contexte HTR, avec discussion des cas limites.

- **Levenshtein, V. I.** (1966). *Binary codes capable of correcting deletions, insertions, and reversals*. Soviet Physics Doklady, 10(8), 707–710. : L'article fondateur de la distance d'édition qui sous-tend le calcul du CER.

- **Papineni, K., Roukos, S., Ward, T., Zhu, W.-J.** (2002). *BLEU: A Method for Automatic Evaluation of Machine Translation*. ACL 2002, 311–318. : L'article original du score BLEU, pour comprendre ses fondements avant de l'adapter à l'HTR.

--
### Apprentissage supervisé et transfer learning

- **Goodfellow, I., Bengio, Y., Courville, A.** (2016). *Deep Learning*. MIT Press. Chapitres 5 (Machine Learning Basics) et 7 (Regularization). : Référence générale sur les fondements de l'apprentissage supervisé.

- **Pan, S. J., Yang, Q.** (2010). *A Survey on Transfer Learning*. IEEE Transactions on Knowledge and Data Engineering, 22(10), 1345–1359. : Revue systématique des approches de transfert d'apprentissage.

- **Howard, J., Ruder, S.** (2018). *Universal Language Model Fine-tuning for Text Classification*. ACL 2018. : Formalisation des pratiques de fine-tuning, applicables aux modèles HTR.

--
### Annotation, accord inter-annotateurs et vérité terrain

- **Artstein, R., Poesio, M.** (2008). *Inter-Coder Agreement for Computational Linguistics*. Computational Linguistics, 34(4), 555–596. : Revue complète des mesures d'accord inter-annotateurs (kappa de Cohen, alpha de Krippendorff…).

- **Snow, R., O'Connor, B., Jurafsky, D., Ng, A. Y.** (2008). *Cheap and Fast : But is it Good? Evaluating Non-Expert Annotations for Natural Language Tasks*. EMNLP 2008. : Sur le crowdsourcing comme alternative aux annotations expertes.

- **Sabou, M. et al.** (2014). *Corpus Annotation through Crowdsourcing: Towards Best Practice Guidelines*. LREC 2014. : Meilleures pratiques pour la collecte d'annotations multiples et le calcul de consensus.

--
### Méthodes d'ensemble et pooling

- **Dietterich, T. G.** (2000). *Ensemble Methods in Machine Learning*. LNCS 1857, 1–15. Springer. : Introduction aux méthodes d'ensemble (bagging, boosting, stacking) et aux principes théoriques qui les sous-tendent.

- **Surowiecki, J.** (2004). *The Wisdom of Crowds*. Doubleday. : Accessible. Sur les conditions dans lesquelles l'agrégation d'opinions indépendantes surpasse l'expert individuel. Donne l'intuition derrière le crowdsourcing et les méthodes d'ensemble.

--
### HTR et datasets médiévaux

- **Kiessling, B., Stökl Ben Ezra, D., Miller, M. T.** (2019). *BADAM: A Public Dataset for Baseline Detection in Arabic-script Manuscripts*. Document Analysis and Recognition Workshop (GREC). : Sur la segmentation de lignes comme étape préalable à l'HTR, par les développeurs de Kraken.

- **Pinche, A., Camps, J.-B., Clérice, T.** (2022). *HTR-United, Mutualisons la vérité de terrain !* Billet de blog CREMMA. [hal-03693079] : Présentation du projet HTR-United et des enjeux de partage des données d'entraînement pour les manuscrits anciens.

- **Camps, J.-B., Fischer, F., Jänicke, S., Schöch, C., Viehhauser, G.** (2021). *From analogue to digital philology: old manuscripts, new approaches*. Digital Humanities Quarterly, 15(1). : Panorama des enjeux de la philologie numérique, avec une perspective sur les manuscrits médiévaux.

---

*Support de cours rédigé pour le module Computer Vision · Promotion MD5 · Mai 2026*
*Ce document accompagne la séance 1.2 du Jour 1. Il suppose la lecture préalable du document 1.1.*
