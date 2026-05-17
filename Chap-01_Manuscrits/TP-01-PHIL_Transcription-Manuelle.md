---
updated: 2026-05-15T17:57:00.109+02:00
edited_seconds: 19
---
# Cours 1.3 — TP : Transcription manuelle et baseline humaine

**Module 1 · Computer Vision appliquée aux manuscrits médiévaux · MD5**

---

> *« Il faut avoir fait la chose à la main pour comprendre ce qu'on demande à la machine de faire. »*

---
## Avant de commencer : pourquoi ce TP existe

Ce travail pratique est le premier acte du projet. Il pourrait sembler anachronique : pourquoi passer du temps à transcrire des manuscrits à la main dans un cours de Machine Learning ? La réponse est précise.

**Premièrement**, vous produirez des données. Les transcriptions que vous déposerez à l'issue de ce TP serviront de **baseline humaine** : une référence contre laquelle évaluer les modèles automatiques tout au long du projet. Ces données ont une valeur réelle.

**Deuxièmement**, vous ferez l'expérience directe des ambiguïtés et des décisions que vous demanderez ensuite à un modèle de résoudre. Il est difficile de spécifier correctement un problème qu'on n'a jamais résolu soi-même. La paléographie n'est pas un domaine qu'on comprend bien en lisant des descriptions : il faut pratiquer.

**Troisièmement**, vous mesurerez l'accord entre transcripteurs humains. Cette mesure : le CER inter-annotateurs : est le plafond théorique de ce qu'on peut espérer obtenir d'un modèle. Un modèle qui ferait mieux que la moyenne des humains entraînés serait exceptionnel ; un modèle qui s'en approche est excellent ; un modèle qui reste très en dessous doit être amélioré.

Il est organisé en quatre étapes progressives, qui passent du travail individuel au travail collectif, puis à l'analyse quantitative.

---
## Matériel fourni

### Les documents de travail

Vous recevrez **trois scans distincts**, représentant trois types de documents et trois périodes différentes. Cette diversité est délibérée : elle vous exposera à la variabilité réelle des manuscrits médiévaux, qui est l'une des difficultés centrales du projet.

**Document A : Charte du XIIIe siècle**
Une charte de donation ou de vente, probablement en latin mêlé de quelques formules en vieux français. Écriture gothique textualis de chancellerie, relativement régulière. Abréviations très denses (les actes juridiques latins abrèvent massivement les formules récurrentes). Mise en page simple : une seule colonne, sans illustration, peu de notes marginales.

*Intérêt pédagogique :* la régularité de l'écriture facilite la lecture des lettres individuelles, mais la densité des abréviations impose des décisions de développement constantes. C'est un bon point d'entrée pour comprendre les systèmes d'abréviation.

**Document B : Extrait de roman arthurien (XIVe siècle)**
Quelques pages d'un roman en prose en moyen français. Écriture gothique cursive ou bâtarde, plus personnelle et plus rapide que la textualis de chancellerie. Mise en page à une colonne avec une lettrine ornée en début de section et quelques rubriques. Texte narratif : moins d'abréviations que la charte, mais une langue moins prévisible.

*Intérêt pédagogique :* la cursive et ses ligatures illustrent le problème des frontières de lettres. La lettrine soulève la question du traitement des éléments graphiques non standards.

**Document C : Registre de comptes (XVe siècle)**
Quelques folios d'un registre administratif ou comptable, probablement en moyen français avec des chiffres et des montants. Écriture cursive rapide, peu soignée, probablement de plusieurs mains. Mise en page tabulaire ou semi-tabulaire. Présence de chiffres romains et arabes. Notes marginales fréquentes.

*Intérêt pédagogique :* la mise en page complexe et le mélange texte/chiffres posent la question de la segmentation de layout. Les notes marginales posent la question de l'ordre de lecture.

### Les outils autorisés
Pendant la phase de transcription individuelle, vous disposez de :
- Le scan imprimé ou affiché à l'écran (en haute définition : zoomez autant que nécessaire).
- Un stylo et du papier, ou un éditeur de texte simple (pas de correcteur orthographique automatique).
- *Aucun outil de traduction, aucun dictionnaire, aucune aide extérieure.* L'objectif est de mesurer ce que vous pouvez déchiffrer par vous-même, pas ce que vous pouvez reconstituer par le sens.

---
## Étape A : Transcription individuelle

### A.1 Conventions obligatoires

Avant de commencer, prenez connaissance des conventions suivantes. Leur respect est indispensable pour que les transcriptions soient comparables entre étudiants : et donc pour que le calcul du CER inter-annotateurs soit significatif.

Ces conventions sont délibérément simples, bien en deçà de ce que préconise le TEI P5 dans toute sa rigueur. Le TEI sera introduit à l'Étape B. Pour l'instant, l'objectif est la praticabilité.

---
**Convention 1 : Respect de l'orthographe du scribe**

Reproduisez l'orthographe telle qu'elle apparaît sur le document, sans la moderniser ni la « corriger ». Si le scribe écrit *rei* là où l'on attendrait *roi*, vous écrivez *rei*. Si le scribe alterne *u* et *v* de façon apparemment aléatoire, vous reproduisez ses choix lettre par lettre.

Cela inclut les majuscules : respectez la casse telle qu'elle apparaît, même si elle ne correspond pas aux règles typographiques modernes.

---
**Convention 2 : Les abréviations : trois options, une seule à choisir par document**

C'est la décision la plus importante à prendre avant de commencer. Trois options sont possibles, et vous devez en choisir une et l'appliquer de façon cohérente sur tout votre document.

*Option 2a : Transcription diplomatique des abréviations :* vous reproduisez le signe abréviatif tel quel, en utilisant les conventions suivantes :

| Ce que vous voyez | Ce que vous écrivez |
|-------------------|---------------------|
| Tilde (~) au-dessus d'une lettre | La lettre suivie de `~` (ex. : *o~* pour *on* abrégé) |
| Barre horizontale au-dessus | La lettre suivie de `=` (ex. : *q=* pour *que* abrégé) |
| Lettre finale en exposant | La lettre en minuscule entre crochets (ex. : *fr[s]* pour *francs*) |
| Signe spécial non identifié | `[abr]` |

*Option 2b : Développement systématique :* vous développez toutes les abréviations en toutes lettres, selon votre meilleure interprétation. Si vous n'êtes pas certain du développement, vous inscrivez votre proposition entre parenthèses : *(que)*.

*Option 2c : Signalement sans développement :* vous reproduisez le mot jusqu'au signe d'abréviation, puis marquez `[...]` pour indiquer que le reste a été omis. Ex. : *fr[...]* pour un mot abrégé dont vous ne savez pas développer la fin.

**Notez votre choix en tête de votre transcription** : `Convention abréviations : 2a / 2b / 2c`.

> *Remarque pour plus tard :* ces trois options correspondent à trois niveaux de fidélité différents, avec des implications différentes pour l'entraînement d'un modèle. L'option 2a produit des données qui reflètent ce que le modèle voit réellement sur l'image. L'option 2b produit des données plus utiles pour le module NLP, mais encode des décisions humaines qui peuvent être erronées. Nous en discuterons à l'Étape B.

---
**Convention 3 : Les zones illisibles**

Quand vous ne parvenez pas à déchiffrer un caractère ou un groupe de caractères, utilisez `[?]` pour un caractère incertain et `[...]` pour une zone plus longue. Évitez de deviner sans signaler votre incertitude.

Si vous avez une proposition mais n'en êtes pas sûr, écrivez votre proposition entre crochets avec un point d'interrogation : `[rois?]`.

---
**Convention 4 : Les fins de ligne**

Marquez chaque fin de ligne du document par un saut de ligne dans votre transcription. Respectez la division en lignes telle qu'elle apparaît sur le document, même si une ligne est incomplète ou si une phrase commence au milieu d'une ligne.

Si un mot est coupé en fin de ligne (avec ou sans signe de coupure), transcrivez-le en deux parties : la fin de la première ligne contient le début du mot, la ligne suivante contient la suite. N'essayez pas de reconstituer le mot entier sur une seule ligne.

Exemple :
```
... et il lor dist que ce-
ste chose est vrai...
```

---
**Convention 5 : Les rubriques**

Les textes écrits à l'encre rouge (rubriques, titres de chapitre, indications liturgiques) font partie du texte et doivent être transcrits. Indiquez leur présence en les encadrant par `<R>` et `</R>`.

Exemple : `<R>Ci commence le roman dou Graal</R>`

---
**Convention 6 : Les notes marginales**

Transcrivez les notes marginales en fin de page, séparées du corps du texte par une ligne vide, précédées de `[MARGE :]`. Si la position de la note par rapport au texte principal est importante (elle annote une ligne précise), indiquez-le : `[MARGE l.7 :]`.

---
**Convention 7 : Les illustrations et les espaces non textuels**

Signalez les illustrations, enluminures, lettrines ornées et espaces vides par une balise descriptive :
- `[LETTRINE : lettre]` pour une grande initiale (ex. : `[LETTRINE : E]`)
- `[ILLUSTRATION : description courte]` pour une image ou un dessin (ex. : `[ILLUSTRATION : chevalier à cheval]`)
- `[ESPACE VIDE]` pour un espace laissé blanc (peut-être pour une illustration jamais réalisée)

---
**Convention 8 : Les chiffres**

Transcrivez les chiffres romains tels qu'ils apparaissent (*xij*, *lx*, *MCCxliij*…). Transcrivez les chiffres arabes tels qu'ils apparaissent (leurs formes médiévales peuvent différer des nôtres : notamment le 4, le 5 et le 7). Ne convertissez pas les chiffres romains en chiffres arabes.

---
### A.2 Consignes de travail

**Travaillez en silence et de façon individuelle.** L'objectif de cette phase est de mesurer la variabilité naturelle entre transcripteurs non spécialisés : ce que vous faites les uns des autres biaiserait cette mesure.

**Consacrez l'essentiel de votre temps au document B** (le roman arthurien), qui est le plus représentatif du type de manuscrit que le projet ciblera. Vous transcrirez une demi-page complète, c'est-à-dire environ 15 à 25 lignes selon la densité de l'écriture.

**Si vous avez du temps**, commencez le document A ou C, en choisissant 5 à 10 lignes représentatives.

**Notez vos hésitations.** Au fil de la transcription, notez sur un brouillon les passages où vous avez hésité, les lettres que vous n'avez pas reconnues, les abréviations dont vous ignorez le développement. Ces notes alimenteront le débriefing collectif.

**Ne retranscrivez pas ce que vous croyez que le texte devrait dire.** C'est la tentation la plus courante : le contexte suffit parfois à deviner un mot, même si ses lettres individuelles ne sont pas toutes lisibles. Dans ce cas, signalez votre incertitude avec `[mot?]` plutôt que d'écrire le mot sans signalement. Un modèle, lui, n'a pas accès au « sens » pour compenser les lettres illisibles : du moins pas de la même façon.

---
### A.3 Exemples annotés de cas difficiles

Les cas suivants illustrent des situations fréquentes et la façon d'y répondre selon les conventions.

---
**Cas 1 : Minimes indistinguables**

Vous voyez une suite de jambages verticaux dans un mot que vous ne reconnaissez pas. Vous comptez sept jambages. Les possibilités sont nombreuses : *minimum*, *miminim*, *mmimm*… Vous ne savez pas.

→ Écrivez `[?????]` (autant de `?` que de caractères approximatifs, ou `[...]` si vous renoncez à estimer le nombre).

---
**Cas 2 : Abréviation avec tilde, option 2b choisie**

Vous voyez le mot *hõe* (le *o* est surmonté d'une vague : un tilde médiéval).

→ Vous développez : *home* (la forme attendue en ancien français pour *homme*). Mais si vous n'êtes pas sûr, écrivez *(home)* entre parenthèses.

→ Si vous avez choisi l'option 2a, vous écrivez *ho~e*.

---
**Cas 3 : Lettre suspecte : *s* long ou *f* ?**

En début de mot, vous voyez un signe vertical avec une petite traverse asymétrique à mi-hauteur. Est-ce un *s* long (forme normale du *s* en début et milieu de mot dans les écritures médiévales) ou un *f* ?

→ Examinez la barre : si elle traverse entièrement la haste des deux côtés, c'est probablement *f* ; si elle n'est visible que d'un côté (la droite), c'est probablement *s* long, que vous transcrivez *s*. Si vous n'êtes vraiment pas sûr : `[s/f]`.

→ Note : le *s* final de mot a une forme différente (*s* rond ou *s* en sigma ς) : ce n'est pas le même problème.

---
**Cas 4 : *u* ou *n* ?**

Deux jambages adjacents dans un mot que vous lisez autrement sans problème. Le mot serait *uenir* ou *nenir* : aucun des deux n'existe. Vous relisez le contexte : la phrase parle d'un personnage qui se déplace.

→ Probabilité maximale : *venir* (le *v* initial a deux formes en ancien français, *u* et *v*). Mais vous n'êtes pas certain que le premier signe est un *u* ou *n*, et vous hésitez sur le troisième. Écrivez *[v/u]e[n/u]ir* pour indiquer vos hésitations.

→ Rappel : dans nos conventions, vous **n'utilisez pas le contexte pour « corriger »** ce que vous lisez. Vous signalez l'ambiguïté.

---
**Cas 5 : Rature et correction par le scribe**

Le scribe a rayé un mot et en a écrit un autre au-dessus ou à la suite.

→ Transcrivez le mot rayé entre crochets barrés : `[~~mot~~]`, puis le mot de correction. Si la rature est illisible, écrivez `[~~...~~]`.

---
**Cas 6 : Ligne trop dégradée pour être transcrite**

Une ligne entière est effacée, tachée ou manquante.

→ Écrivez `[LIGNE ILLISIBLE]` sur la ligne correspondante.

---
## Étape B : Débriefing collectif et conventions TEI

### B.1 Mise en commun des transcriptions
Quelques étudiants présentent brièvement (2 à 3 minutes) :
1. **Les trois passages les plus difficiles** qu'il a rencontrés, et les choix qu'il a faits.
2. **Un passage où il est certain** de sa transcription et un passage où il l'est peu.
3. **Les questions que le document lui a posées** et auxquelles les conventions fournies n'apportaient pas de réponse satisfaisante.

On recensera au tableau les types de difficultés rencontrées. On distinguera systématiquement :
- Les difficultés **graphiques** (lettres ambiguës, ligatures, dégradations).
- Les difficultés **linguistiques** (mot inconnu, abréviation inconnue, langue non identifiée).
- Les difficultés **structurelles** (que faire de cet élément ? dans quel ordre le transcrire ?).

### B.2 Introduction aux conventions TEI P5

La **Text Encoding Initiative (TEI)** est le standard international pour l'encodage numérique des textes patrimoniaux. Son format XML est utilisé par les grandes bibliothèques numériques (BnF, British Library, Herzog August Bibliothek…) et par la majorité des projets d'édition numérique académique.

Comprendre les grandes balises TEI sert deux objectifs dans notre projet : d'une part, les datasets HTR professionnels utilisent fréquemment le format PAGE XML ou ALTO XML, qui partagent une philosophie voisine du TEI ; d'autre part, le format de sortie de notre pipeline peut être lu directement par les outils des humanités numériques si nous adoptons des conventions compatibles.

Voici les balises essentielles pour notre contexte, avec leur équivalent dans nos conventions simplifiées :

---
**Structure physique du document**

```xml
<pb n="1r"/>        <!-- Page break : début du folio 1 recto -->
<cb n="1"/>         <!-- Column break : début de la colonne 1 -->
<lb n="1"/>         <!-- Line break : début de la ligne 1 -->
```

Le *recto* est le côté avant d'un feuillet (face visible quand on ouvre le livre à droite) ; le *verso* est le côté arrière. Un feuillet est numéroté *1r* (recto) et *1v* (verso).

---
**Abréviations**

```xml
<!-- Option 1 : noter abréviation et développement séparément -->
<choice>
    <abbr>dns</abbr>
    <expan>dominus</expan>
</choice>

<!-- Option 2 : développement avec indication des lettres ajoutées -->
<expan>d<ex>ominu</ex>s</expan>
```

La balise `<ex>` (pour *editorial expansion*) marque les lettres qui sont dans le développement mais pas dans le signe abréviatif visible : ce qui permet de distinguer, dans le texte développé, les lettres effectivement écrites de celles qui ont été restituées.

---
**Zones illisibles et lacunes**

```xml
<!-- Zone illisible, nombre de caractères incertain -->
<gap reason="illegible"/>

<!-- Zone illisible, environ 3 caractères -->
<gap reason="illegible" unit="chars" quantity="3"/>

<!-- Lacune physique (le parchemin est endommagé) -->
<gap reason="damage" unit="chars" quantity="5"/>
```

La distinction entre `illegible` (le texte existe mais ne peut être lu) et `damage` (le support est physiquement détruit) est importante : dans le second cas, aucun algorithme de traitement d'image ne permettra de récupérer l'information.

---
**Corrections et interventions du scribe**

```xml
<!-- Rature -->
<del rend="strikethrough">mot barré</del>

<!-- Addition supralinéaire (au-dessus de la ligne) -->
<add place="supralinear">mot ajouté</add>

<!-- Addition interlinéaire (entre deux lignes) -->
<add place="interlinear">mot ajouté</add>

<!-- Addition marginale -->
<add place="margin">mot ajouté</add>

<!-- Le scribe se corrige : barré puis réécrit -->
<subst>
    <del rend="strikethrough">première version</del>
    <add place="supralinear">version corrigée</add>
</subst>
```

---
**Notes marginales et rubriques**

```xml
<!-- Note marginale (contemporaine du texte ou postérieure) -->
<note place="margin" hand="#main">annotation marginale</note>

<!-- Rubrique (titre en rouge) -->
<head rend="rubric">Ci commence le premier chapitre</head>

<!-- Lettrine -->
<hi rend="initial">E</hi>n cel tems...
```

L'attribut `hand` permet d'identifier la main qui a écrit : `#main` pour la main principale, `#corrector` pour un correcteur, `#secondaryHand` pour une main postérieure. Cette identification est précieuse pour les projets de paléographie comparée.

---
**Incertitude sur une lecture**

```xml
<!-- Lecture incertaine, avec degré de certitude -->
<unclear cert="low">rois</unclear>
<unclear cert="medium">reis</unclear>

<!-- Plusieurs lectures possibles -->
<choice>
    <unclear>roi</unclear>
    <unclear>rei</unclear>
</choice>
```

---
### B.3 La question philosophique centrale

Le débriefing collectif aura mis en évidence un fait fondamental : des transcripteurs humains, face aux mêmes documents, font des choix différents. Pas seulement sur les passages difficiles : sur des passages que chacun trouvait parfaitement lisible.

Cela pose une question que tout le projet devra maintenir présente à l'esprit :

**Peut-on demander à un modèle de faire des choix que des humains n'arrivent pas à standardiser entre eux ?**

La réponse courte est : oui, si on lui apprend à faire *un* choix cohérent : pas *le bon* choix, mais un choix documenté et reproductible. Un modèle entraîné sur des données où l'abréviation *dns* est toujours développée en *dominus* apprendra cette convention, et la reproduira de façon cohérente : peut-être mieux que la majorité des annotateurs humains, dont certains auraient développé *domininus* par inadvertance.

La valeur d'un modèle HTR n'est donc pas d'être « meilleur » qu'un humain au sens absolu : elle est de produire des transcriptions **cohérentes à grande échelle**, ce qu'aucune équipe humaine ne peut faire de façon économiquement viable pour des corpus de plusieurs milliers de pages.

---
## Étape C : Analyse quantitative des divergences

### C.1 Objectif

Calculer le **CER inter-annotateurs** : mesurer, quantitativement, à quel point les transcriptions des étudiants divergent les unes des autres. Ce chiffre est crucial pour deux raisons :

- Il révèle la **difficulté intrinsèque** du document : un CER inter-annotateurs de 15% sur un passage signifie que même des humains ne s'accordent pas. On ne peut pas raisonnablement demander à un modèle d'être beaucoup plus précis.
- Il constitue le **plafond de performance** contre lequel évaluer les modèles automatiques : un modèle qui atteint le niveau d'accord inter-annotateurs est considéré comme ayant atteint la performance humaine.

### C.2 Protocole de collecte des transcriptions

Chaque étudiant dépose sa transcription dans un fichier texte nommé selon la convention :

```
transcription_[NOM]_[DOCUMENT].txt
```

Exemple : `transcription_dupont_documentB.txt`

Le fichier contient :
- En en-tête : le nom de l'étudiant, la date, le document transcrit, et la convention d'abréviation choisie (2a, 2b ou 2c).
- Le corps de la transcription, une ligne du document par ligne du fichier.

Les fichiers sont déposés dans un dossier partagé accessible à tous.

### C.3 Calcul du CER inter-annotateurs : approche pas à pas

Voici le code Python complet pour calculer le CER entre toutes les paires de transcriptions et en tirer des statistiques.

---
**Étape 1 : La distance de Levenshtein**

La distance de Levenshtein est l'algorithme fondamental du CER. Elle calcule le nombre minimal d'opérations (insertions, suppressions, substitutions) pour transformer une chaîne en une autre.

```python
def levenshtein(s1: str, s2: str) -> int:
    """
    Calcule la distance d'édition (Levenshtein) entre deux chaînes.
    Retourne le nombre minimal d'insertions, suppressions et
    substitutions pour transformer s1 en s2.
    """
    # Matrice de programmation dynamique
    # dp[i][j] = distance entre s1[:i] et s2[:j]
    m, n = len(s1), len(s2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    # Cas de base : transformer une chaîne vide
    for i in range(m + 1):
        dp[i][0] = i  # i suppressions
    for j in range(n + 1):
        dp[0][j] = j  # j insertions
    
    # Remplissage de la matrice
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s1[i-1] == s2[j-1]:
                dp[i][j] = dp[i-1][j-1]  # Pas d'opération nécessaire
            else:
                dp[i][j] = 1 + min(
                    dp[i-1][j],    # Suppression dans s1
                    dp[i][j-1],    # Insertion dans s1
                    dp[i-1][j-1]   # Substitution
                )
    return dp[m][n]
```

---

**Étape 2 : Le CER entre deux transcriptions**

```python
def cer(hypothesis: str, reference: str) -> float:
    """
    Calcule le Character Error Rate entre une hypothèse et une référence.
    
    Args:
        hypothesis: La transcription à évaluer.
        reference:  La transcription de référence.
    
    Returns:
        CER comme float entre 0.0 et (potentiellement > 1.0).
    
    Note : les espaces comptent comme des caractères.
    """
    if len(reference) == 0:
        # Cas dégénéré : référence vide
        return float('inf') if len(hypothesis) > 0 else 0.0
    
    distance = levenshtein(hypothesis, reference)
    return distance / len(reference)
```

---
**Étape 3 : CER ligne par ligne et CER global**

Pour comparer deux transcriptions complètes (plusieurs lignes), on a deux options :

*Option A : CER global (recommandée) :* on concatène toutes les lignes (ou on les traite comme une seule grande chaîne) et on calcule la distance globale. C'est la méthode standard de la communauté HTR.

*Option B : CER par ligne puis moyenne :* on calcule le CER de chaque ligne et on fait la moyenne. Cette méthode est utile pour identifier les lignes problématiques, mais elle traite toutes les lignes à égalité indépendamment de leur longueur.

```python
def cer_global(transcription_1: list[str], transcription_2: list[str]) -> float:
    """
    Calcule le CER global entre deux transcriptions multi-lignes.
    
    Args:
        transcription_1, transcription_2 : listes de chaînes (une par ligne).
    
    Returns:
        CER global comme float.
    """
    # Concaténation avec un séparateur de ligne
    texte_1 = "\n".join(transcription_1)
    texte_2 = "\n".join(transcription_2)
    return cer(texte_1, texte_2)


def cer_par_ligne(transcription_1: list[str],
                  transcription_2: list[str]) -> list[float]:
    """
    Calcule le CER pour chaque ligne, retourne une liste.
    Les lignes sans correspondance (longueurs différentes) sont ignorées.
    """
    resultats = []
    for l1, l2 in zip(transcription_1, transcription_2):
        resultats.append(cer(l1, l2))
    return resultats
```

---
**Étape 4 : Matrice d'accord entre tous les étudiants**

```python
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def charger_transcription(chemin_fichier: str) -> list[str]:
    """Charge une transcription depuis un fichier texte, une ligne par item."""
    with open(chemin_fichier, 'r', encoding='utf-8') as f:
        lignes = f.readlines()
    
    # Supprimer l'en-tête (les 3 premières lignes par convention)
    corps = lignes[3:]
    
    # Nettoyer les espaces et les lignes vides
    return [ligne.strip() for ligne in corps if ligne.strip()]


def matrice_cer(dossier: str, document: str = "documentB") -> None:
    """
    Calcule et visualise la matrice de CER entre toutes les paires
    de transcriptions pour un document donné.
    
    Args:
        dossier : chemin vers le dossier contenant les fichiers .txt
        document : identifiant du document (filtre sur le nom de fichier)
    """
    # Collecter tous les fichiers du document cible
    fichiers = [
        f for f in os.listdir(dossier)
        if f.endswith('.txt') and document in f
    ]
    fichiers.sort()
    
    n = len(fichiers)
    noms = [f.replace(f'transcription_', '').replace(f'_{document}.txt', '')
            for f in fichiers]
    
    # Charger toutes les transcriptions
    transcriptions = {}
    for fichier in fichiers:
        nom = fichier.replace('transcription_', '').replace(f'_{document}.txt', '')
        transcriptions[nom] = charger_transcription(os.path.join(dossier, fichier))
    
    # Calculer la matrice de CER (symétrique)
    matrice = np.zeros((n, n))
    for i, nom_i in enumerate(noms):
        for j, nom_j in enumerate(noms):
            if i != j:
                matrice[i][j] = cer_global(
                    transcriptions[nom_i],
                    transcriptions[nom_j]
                )
    
    # Affichage sous forme de heatmap
    plt.figure(figsize=(max(8, n), max(6, n - 1)))
    sns.heatmap(
        matrice,
        xticklabels=noms,
        yticklabels=noms,
        annot=True,
        fmt='.1%',
        cmap='YlOrRd',    # Jaune (accord) → Rouge (désaccord)
        vmin=0,
        vmax=0.25,         # Adapter selon les résultats observés
        linewidths=0.5
    )
    plt.title(f'CER inter-annotateurs : {document}', fontsize=14)
    plt.xlabel('Référence')
    plt.ylabel('Hypothèse')
    plt.tight_layout()
    plt.savefig(f'cer_inter_annotateurs_{document}.png', dpi=150)
    plt.show()
    
    # Statistiques descriptives
    # On ne prend que les valeurs hors diagonale (pas de comparaison avec soi-même)
    valeurs_hors_diag = matrice[matrice > 0]
    print(f"\n--- Statistiques CER inter-annotateurs ({document}) ---")
    print(f"Nombre de transcripteurs : {n}")
    print(f"CER moyen    : {np.mean(valeurs_hors_diag):.1%}")
    print(f"CER médian   : {np.median(valeurs_hors_diag):.1%}")
    print(f"CER minimum  : {np.min(valeurs_hors_diag):.1%}")
    print(f"CER maximum  : {np.max(valeurs_hors_diag):.1%}")
    print(f"Écart-type   : {np.std(valeurs_hors_diag):.1%}")
```

---
**Étape 5 : Identification des lignes les plus divergentes**

Cette analyse permet de localiser précisément les passages où l'accord entre transcripteurs est le plus faible : c'est-à-dire les passages qui seront les plus difficiles pour les modèles automatiques.

```python
def lignes_les_plus_divergentes(
    transcriptions: dict[str, list[str]],
    n_top: int = 10
) -> list[tuple[int, float, list[str]]]:
    """
    Identifie les lignes sur lesquelles l'accord inter-annotateurs
    est le plus faible.
    
    Args:
        transcriptions : dict {nom_annotateur: liste_de_lignes}
        n_top : nombre de lignes les plus divergentes à retourner
    
    Returns:
        Liste de tuples (numéro_de_ligne, CER_moyen, variantes)
    """
    noms = list(transcriptions.keys())
    
    # Trouver la longueur minimale commune (lignes communes à tous)
    longueur_min = min(len(t) for t in transcriptions.values())
    
    divergences = []
    for idx in range(longueur_min):
        variantes = [transcriptions[nom][idx] for nom in noms]
        
        # Calculer le CER de chaque variante contre chaque autre
        cer_ligne = []
        for i in range(len(variantes)):
            for j in range(i + 1, len(variantes)):
                c = cer(variantes[i], variantes[j])
                cer_ligne.append(c)
        
        cer_moyen = np.mean(cer_ligne) if cer_ligne else 0.0
        divergences.append((idx + 1, cer_moyen, variantes))
    
    # Trier par CER décroissant et retourner les n_top pires
    divergences.sort(key=lambda x: x[1], reverse=True)
    return divergences[:n_top]


# Exemple d'utilisation
top_divergentes = lignes_les_plus_divergentes(transcriptions, n_top=5)
print("\n--- Lignes les plus divergentes ---")
for num_ligne, cer_moyen, variantes in top_divergentes:
    print(f"\nLigne {num_ligne} : CER moyen inter-annotateurs : {cer_moyen:.1%}")
    for i, (nom, variante) in enumerate(zip(transcriptions.keys(), variantes)):
        print(f"  [{nom}] : {variante}")
```

---
### C.4 Interprétation des résultats

**CER inter-annotateurs typiques pour des manuscrits médiévaux**

Les valeurs suivantes sont issues de la littérature et de projets collaboratifs de référence. Elles vous permettent de situer vos résultats dans un contexte plus large.

| Type de document                                        | Annotateurs               | CER inter-annotateurs typique |
| ------------------------------------------------------- | ------------------------- | ----------------------------- |
| Chartes latines XIIIe s., annotateurs experts           | Paléographes spécialisés  | 3–6%                          |
| Romans en vieux français XIVe s., annotateurs experts   | Médiévistes               | 5–10%                         |
| Registres comptables XVe s., annotateurs experts        | Historiens + paléographes | 7–15%                         |
| Documents médiévaux variés, annotateurs non spécialisés | Crowdsourcing Zooniverse  | 12–25%                        |
| **Documents de ce TP, non-spécialistes (estimation)**   | **Vous**                  | **15–35%**                    |

Un CER inter-annotateurs de 20–30% parmi des non-spécialistes n'est pas un échec : c'est le résultat attendu, et il justifie précisément l'utilité des modèles HTR entraînés sur des données expertes : leur CER contre une référence experte sera bien inférieur à ce qu'un non-spécialiste obtiendrait.

**Ce que révèlent les résultats ligne par ligne**

Examinez ensemble les lignes identifiées comme les plus divergentes. Pour chacune :

- Est-ce une difficulté **graphique** (une lettre ambiguë, une ligature) ? → Elle sera difficile pour tous les modèles, et le CER sur ces lignes restera élevé.
- Est-ce une difficulté **de convention** (abréviation développée de façons différentes) ? → Elle disparaîtra si tous les annotateurs adoptent la même convention. C'est un problème d'annotation, pas un problème de lecture.
- Est-ce une difficulté **linguistique** (mot inconnu dont on ne sait pas interpréter les lettres) ? → Elle pourrait être résolue par un modèle de langage en post-traitement.

Cette classification des sources d'erreur est directement utile pour la conception du pipeline : chaque type d'erreur appelle un remède différent.

---
## Étape D : Problématisation et formalisation du projet

### D.1 Discussion : à quoi sert la transcription automatique ?

Avant de passer à la technique, une question d'usage : pourquoi vouloir transcrire automatiquement des manuscrits médiévaux ? La réponse conditionne les exigences de qualité et les compromis acceptables.

**Usage 1 : La recherche plein texte dans les archives**

Des institutions comme la BnF, l'IRHT ou les Archives nationales possèdent des millions de documents numérisés, dont une infime fraction a été transcrite par des humains. Une transcription automatique imparfaite : disons, CER de 15% : suffit pour rendre ces documents *cherchables* par mots-clés. Un historien qui cherche toutes les mentions d'un nom propre ou d'une formule juridique dans dix mille chartes peut travailler avec une transcription approximative.

*Exigence de qualité :* CER < 20% est souvent suffisant. La précision compte plus que le rappel (mieux vaut ne pas trouver quelques occurrences que d'être submergé de faux positifs).

**Usage 2 : L'alignement avec des éditions critiques existantes**

Pour les textes littéraires qui ont déjà fait l'objet d'une édition critique imprimée (la quasi-totalité des grands textes médiévaux), la transcription automatique permet d'aligner le texte du scan avec le texte édité, et ainsi de localiser précisément chaque mot dans l'image. Cette fonctionnalité est à la base des outils de lecture assistée.

*Exigence de qualité :* CER < 10% est souhaitable pour un alignement fiable.

**Usage 3 : L'alimentation d'un modèle de langage en aval (notre cas)**

La transcription produite par notre pipeline sera l'entrée du module NLP, qui cherchera à normaliser l'orthographe, résoudre les abréviations et annoter le texte linguistiquement. Pour cet usage, le taux d'erreur se propage et s'amplifie : une erreur de transcription peut entraîner une erreur d'analyse NLP, qui entraîne une erreur d'interprétation. La qualité doit être suffisamment haute pour que le modèle NLP puisse « se débrouiller » avec les erreurs résiduelles.

*Exigence de qualité :* CER < 12% est notre cible, avec un objectif post-NLP de < 5%.

**Usage 4 : La production d'éditions numériques académiques**

Pour une édition académique destinée à la publication, une transcription automatique doit être révisée intégralement par un expert. Dans ce cas, la transcription automatique n'est pas une fin : c'est un outil d'aide à la saisie, qui réduit le temps de transcription de l'expert de 60 à 80% selon les études.

*Exigence de qualité :* pas de minimum absolu, mais plus la qualité est haute, moins la révision est longue.

---

### D.2 Formalisation du projet

À l'issue de la discussion, le projet est formalisé collectivement selon les dimensions suivantes.

**Le corpus de travail**

Nous utiliserons principalement des manuscrits en moyen français du XIVe–XVe siècle, issus du corpus CREMMA disponible via HTR-United, complété par des scans de Gallica pour les pages de test. La diversité des mains et des types de documents est un objectif explicite : le pipeline doit généraliser, pas seulement fonctionner sur un seul scripteur.

**La cible linguistique**

Vieux et moyen français (IXe–XVe siècle). Latin exclus du périmètre principal (mais présent dans les chartes du corpus). L'objectif n'est pas de comprendre la langue : c'est l'affaire du module NLP : mais de la reproduire fidèlement au niveau graphique.

**La convention de transcription**

Pour le projet complet, nous adoptons la **transcription semi-diplomatique** avec développement des abréviations les plus communes. Les abréviations ambiguës sont signalées avec un flag d'incertitude. Cette décision sera documentée dans le data contract.

**Les métriques**

CER comme métrique principale. WER calculé à titre indicatif. CER inter-annotateurs comme plafond de référence.

**L'objectif chiffré**

CER < 12% contre la vérité terrain experte sur les pages de test, avant toute correction NLP.

**Le format de livraison**

JSON structuré (voir le schéma défini en Jour 4), avec optionnellement une conversion TEI P5 pour compatibilité avec les outils des humanités numériques.

---

## Livrable : Ce que vous déposez à l'issue du Jour 1

**Fichier de transcription :** votre transcription du document B (et si possible A ou C), selon les conventions définies ci-dessus, au format `.txt` encodé en UTF-8.

**Feuille de choix :** un court document (5 à 10 lignes) précisant :
1. La convention d'abréviation choisie et les raisons de ce choix.
2. Les trois passages les plus incertains, avec le choix effectué et son justification.
3. Les questions auxquelles vous n'avez pas trouvé de réponse dans les conventions fournies.

Ces documents ne seront pas notés sur leur qualité philologique : vous n'êtes pas médiévistes. Ils seront évalués sur leur cohérence interne (les conventions sont-elles appliquées de façon homogène ?) et sur la qualité des questionnements soulevés.

Ils alimenteront directement le calcul du CER inter-annotateurs présenté à l'Étape C, et seront réutilisés comme données de supervision et de validation tout au long du projet.

---
## Pour aller plus loin

### Ressources pratiques pour la lecture de manuscrits médiévaux

**Outils de lecture en ligne**
- **e-Scriptorium** : [escriptorium.fr](https://escriptorium.fr) : plateforme collaborative d'annotation HTR, développée par l'EPHE. Interface similaire à Transkribus mais open source. Vous pourrez y déposer vos transcriptions et les comparer avec les prédictions des modèles.
- **Transkribus** : [transkribus.eu](https://transkribus.eu) : outil commercial avec une version gratuite limitée. Donne accès à des modèles HTR pré-entraînés pour les manuscrits médiévaux. Utile pour obtenir une première transcription automatique à comparer avec les vôtres.
- **Initiale : base de données des manuscrits enluminés** : [initiale.irht.cnrs.fr](http://initiale.irht.cnrs.fr) : pour identifier les types d'enluminures et de lettrines rencontrés dans vos documents.

**Guides de lecture pratiques**
- **École nationale des chartes : Cours de paléographie en ligne** : des exercices de lecture de manuscrits français classés par époque et par difficulté, avec corrections. Non disponibles publiquement pour tous les niveaux, mais accessibles sur demande via l'IRHT.
- **Muzerelle, D.** : *Vocabulaire codicologique* (version électronique) : [codicologia.irht.cnrs.fr](http://codicologia.irht.cnrs.fr). Pour tout terme technique du domaine.
- **Album de paléographie** de l'IRHT : recueil de planches commentées couvrant les principales écritures médiévales françaises du IXe au XVe siècle. Disponible en bibliothèque universitaire spécialisée.

---
## Bibliographie de référence
### Méthodes d'annotation et crowdsourcing pour les humanités
- **Causer, T., Wallace, V.** (2012). *Building A Volunteer's Sense of the Project and Commitment to a Task: Testing Commitment and Task Complexity*. Literary and Linguistic Computing, 27(2), 153–167. : Sur les facteurs qui déterminent la qualité des transcriptions crowdsourcées. Première étude empirique large sur Transcribe Bentham.

- **Lejeune, G. et al.** (2015). *Automatic Transcription of Handwritten Medieval Documents: first experiments on the DAHN Project*. Document Analysis Systems (DAS). : Comparaison des transcriptions humaines et automatiques sur des manuscrits français médiévaux ; données chiffrées sur le CER inter-annotateurs.

- **Voss, J., Terras, M., Bunt, G.** (2017). *Transcription of Medieval Manuscripts*. Dans : Schreibman, S., Siemens, R., Unsworth, J. (dir.), *A New Companion to Digital Humanities*. Wiley-Blackwell, 332–346. : Revue complète des pratiques de transcription numérique des manuscrits, avec discussion des enjeux d'annotation.

- **Sabou, M. et al.** (2014). *Corpus Annotation through Crowdsourcing: Towards Best Practice Guidelines*. LREC 2014. : Meilleures pratiques pour la collecte d'annotations multiples et le calcul de consensus dans des projets de crowdsourcing.

### Accords inter-annotateurs
- **Artstein, R., Poesio, M.** (2008). *Inter-Coder Agreement for Computational Linguistics*. Computational Linguistics, 34(4), 555–596. : Revue complète et rigoureuse des mesures d'accord inter-annotateurs. Section 3 sur les mesures basées sur la distance d'édition, directement applicable au CER.

- **Krippendorff, K.** (2004). *Content Analysis: An Introduction to Its Methodology* (2e éd.). Sage Publications. Chapitre 11 : *Reliability*. : L'alpha de Krippendorff est une mesure d'accord généralisée qui étend le kappa de Cohen à plus de deux annotateurs et à des échelles ordinales ou continues. Utilisable pour mesurer l'accord sur des transcriptions.

### Distance d'édition et algorithmes d'alignement
- **Levenshtein, V. I.** (1966). *Binary codes capable of correcting deletions, insertions, and reversals*. Soviet Physics Doklady, 10(8), 707–710. : L'article fondateur.

- **Needleman, S. B., Wunsch, C. D.** (1970). *A general method applicable to the search for similarities in the amino acid sequence of two proteins*. Journal of Molecular Biology, 48(3), 443–453. : L'algorithme d'alignement global utilisé pour comparer deux séquences. Bien que développé pour la bioinformatique, il est directement applicable à l'alignement de transcriptions.

- **Wagner, R. A., Fischer, M. J.** (1974). *The String-to-String Correction Problem*. Journal of the ACM, 21(1), 168–173. : Généralisation de la distance de Levenshtein avec des coûts variables selon les opérations. Utile si l'on veut pondérer différemment les substitutions de lettres visuellement proches.

### Projets de référence en transcription collaborative
- **Causer, T., Terras, M.** (2014). *"Many Hands Make Light Work. Many Hands Together Make Merry Work": Transcribe Bentham and Crowdsourcing Manuscript Collections*. Dans : Ridge, M. (dir.), *Crowdsourcing Our Cultural Heritage*. Ashgate, 57–88. : Bilan du projet Transcribe Bentham, un des projets fondateurs de la transcription collaborative en humanités numériques.

- **Organisciak, P. et al.** (2016). *Researching Large-scale Community Transcription of Manuscript Material*. Dans : Proceedings of the Joint Conference on Digital Libraries. : Analyse quantitative de la qualité des transcriptions dans des projets Zooniverse de grande échelle, avec des données comparatives sur le CER.

- **Pinche, A.** (2022). *Guide de transcription pour les manuscrits du projet CREMMA*. École nationale des chartes / EPHE. Disponible sur HTR-United. : Le guide de transcription du projet CREMMA : conventions détaillées pour la transcription semi-diplomatique de manuscrits médiévaux français. **À lire avant de travailler avec les datasets CREMMA.**

---
*Support de cours rédigé pour le module Computer Vision · Promotion MD5 · Mai 2026*
*Ce document accompagne la séance 1.3 du Jour 1. Il suppose la lecture préalable des documents 1.1 et 1.2.*
*Les fichiers de transcription déposés à l'issue de ce TP seront réutilisés tout au long du projet.*
