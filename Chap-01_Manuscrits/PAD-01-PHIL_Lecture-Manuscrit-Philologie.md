---
updated: 2026-05-17T13:22:43.115+02:00
edited_seconds: 0
---
# Cours 1.1 : Lecture de manuscrit : approche philologique et paléographique

- **Module 1 · Computer Vision appliquée aux manuscrits médiévaux · MD5**

---

> *« Un manuscrit médiéval n'est pas un livre. C'est le résidu matériel d'un acte d'écriture singulier, situé dans un temps, un lieu et une main. »*
> Inspiré de la tradition paléographique de l'École nationale des chartes

---

## Pourquoi commencer par la philologie ?

Avant d'écrire la première ligne de code, il faut comprendre ce qu'on demande à un modèle d'apprentissage automatique de faire. Reconnaître du texte manuscrit dans des documents médiévaux n'est pas un problème de vision par ordinateur ordinaire. C'est un problème d'interprétation, au sens fort du terme : il faut décider ce qui est texte et ce qui ne l'est pas, quelle lettre se cache derrière un trait ambigu, quel mot se cache derrière une abréviation, et quelle langue est parlée à tel endroit de la page.

Ces décisions sont, en régime humain, le travail de plusieurs disciplines : la **paléographie** (science de l'écriture ancienne), la **codicologie** (science du livre manuscrit comme objet matériel), la **philologie** (science des textes dans leur transmission) : accumulées sur plusieurs siècles. Un modèle de vision par ordinateur devra apprendre à faire, de façon approximative et statistique, ce que des spécialistes font après des années de formation.

Ce cours d'introduction a deux objectifs :

1. Vous donner les outils conceptuels pour comprendre la nature réelle du problème.
2. Vous permettre de traduire ce problème en spécifications techniques : car un ingénieur qui ne comprend pas le domaine produira des métriques qui mesurent autre chose que ce qui compte.

---

## 1. Le manuscrit médiéval comme objet matériel

### 1.1 Qu'est-ce qu'un manuscrit ?

Le mot *manuscrit* vient du latin *manu scriptum* : « écrit à la main ». Il désigne tout document rédigé manuellement, par opposition aux textes imprimés (la typographie à caractères mobiles n'est introduite en Europe occidentale qu'autour de 1450, par Gutenberg). Entre le IXe et le XVe siècle, tout document écrit est donc un manuscrit.

Mais tous les manuscrits ne se ressemblent pas. On distingue principalement :

**Le codex**
C'est la forme qui ressemble le plus à notre livre moderne : des feuilles pliées, assemblées en cahiers (*quaternions*, *sénions*…), cousues et reliées. C'est le support de prédilection pour les textes longs : bibles, textes liturgiques, romans, traités philosophiques ou scientifiques, compilations juridiques. Le codex s'est imposé progressivement face au rouleau entre le IIe et le IVe siècle de notre ère.

**Le rotulus**
Un rouleau de parchemin ou de papyrus, utilisé pour des documents administratifs, des actes, des textes liturgiques destinés à être lus debout. Il se développe verticalement ou horizontalement et impose une lecture séquentielle linéaire, sans possibilité de navigation dans le texte.

**La charte**
Document juridique attestant un acte (donation, vente, accord, privilège). La charte est généralement un feuillet unique, écrit d'un seul côté (*recto*), avec des formules juridiques très standardisées. Elle porte souvent un sceau (en cire, en plomb) qui lui confère sa valeur légale.

**Le registre**
Livre dans lequel une institution (chancellerie royale, évêché, tribunal, municipalité) consigne chronologiquement ses actes. Les registres sont souvent écrits par des clercs professionnels formés à une écriture standardisée dite de chancellerie.

> **Pourquoi cette distinction compte pour le projet ?** Les modèles HTR (Handwritten Text Recognition) entraînés sur des chartes ne fonctionneront probablement pas bien sur des romans arthuriens, et vice versa. Le type de document détermine le type d'écriture, le registre de langue, la densité des abréviations, la mise en page. Un bon pipeline doit d'abord classifier le type de document avant de choisir quel modèle appliquer.

### 1.2 Le support d'écriture

**Le parchemin**
C'est le support dominant du Moyen Âge occidental jusqu'au XIVe siècle environ. Il est fabriqué à partir de peau d'animal (mouton, chèvre, veau : le veau donne le *vélin*, plus fin et plus blanc) : la peau est trempée, tendue sur un cadre, raclée, séchée. Le résultat est une surface lisse, résistante, translucide par endroits. Le parchemin est cher, rare, réutilisable (en grattant la surface : ce qui donne les *palimpsestes*, où un texte premier partiellement effacé est recouvert d'un nouveau).

Ses propriétés physiques ont des conséquences directes pour la numérisation :
- Il jaunit et se déforme avec l'âge, ce qui crée des variations de luminosité sur le scan.
- La transparence (ou *show-through*) fait apparaître l'écriture du verso en filigrane sur le recto : une source de bruit considérable pour les algorithmes de segmentation.
- Les grattages et les retouches laissent des traces visuellement ambiguës.

**Le papier**
Introduit en Europe occidentale au XIIIe siècle (d'abord en Espagne, puis en Italie et en France), le papier se répand progressivement et devient dominant au XVe siècle. Il est moins cher que le parchemin, mais moins durable. Pour les modèles, le papier vieilli présente des problèmes différents : jaunissement, taches d'humidité, encre qui saigne, déchirures.

**L'encre**
L'encre médiévale la plus courante est l'encre ferro-gallique : un mélange de sulfate de fer et d'acide gallique (extrait des galles de chêne), qui produit une encre noire au dépôt, mais qui avec le temps peut ronger le support par oxydation. On obtient ainsi des pages où l'encre a litteralement percé le parchemin : ce qu'aucun prétraitement d'image ne peut restaurer.

### 1.3 La mise en page

La page d'un manuscrit médiéval est rarement un simple bloc de texte continu. Elle est structurée selon des conventions qui varient selon l'époque, l'origine géographique et le type de texte :

- **Le réglage** : avant d'écrire, le copiste trace des lignes directrices (à la pointe sèche, à la mine de plomb, ou à l'encre) pour guider son écriture. Ces lignes sont parfois encore visibles sur les scans.
- **La mise en colonnes** : les bibles et les textes liturgiques sont souvent disposés sur deux, voire trois colonnes. Les romans et les chroniques sont généralement à une colonne. Les registres ont des colonnes spécialisées (dates, sommes, noms).
- **Les rubriques** : titres, sous-titres ou indications de structure écrits à l'encre rouge (*rubrum* en latin). Ils appartiennent au texte mais ont un statut différent du corps principal.
- **Les lettrines** : grandes lettres initiales de chapitres ou de sections, souvent ornées, parfois enluminées. Elles posent un problème spécifique : elles font partie du texte, mais un modèle doit les traiter différemment des lettres ordinaires.
- **Les annotations marginales** : le lecteur médiéval annotait abondamment. Ces annotations peuvent être contemporaines du texte, ou ajoutées des décennies ou des siècles plus tard, par d'autres mains, parfois dans d'autres langues. Sont-elles du texte à transcrire ? Dans quel ordre ?
- **Les illustrations et enluminures** : des miniatures peintes, des dessins à la plume, des diagrammes, des cartes. Elles sont physiquement présentes sur la page mais ne sont pas du texte. Un modèle de segmentation doit apprendre à les isoler.

---

## 2. La chaîne de transmission : copistes, scriptoria, variabilité

### 2.1 Qui écrivait ?

Pendant une grande partie du Moyen Âge (VIIe–XIIe siècle surtout), l'écriture est quasi-exclusivement le fait de clercs : des hommes d'Église formés dans des écoles monastiques ou cathédrales. L'écriture est alors indissociable de la culture religieuse : on copie des textes sacrés, des textes patristiques (les Pères de l'Église), des textes liturgiques.

À partir du XIIIe siècle, avec l'essor des villes, du commerce et de l'administration royale, des **scribes laïcs professionnels** apparaissent : les *écrivains publics*, les *clercs de notaire*, les employés des chancelleries. L'écriture devient progressivement un métier civil.

Ce glissement a des conséquences directes sur les écritures : les scriptoria monastiques produisent des écritures soignées, calligraphiées, relativement standardisées au sein d'un atelier. Les scribes laïcs produisent des écritures plus rapides, plus personnelles, plus variables.

### 2.2 Le scriptorium et ses normes

Un **scriptorium** est l'atelier d'écriture d'une abbaye ou d'une grande institution. Les moines copistes y travaillent selon des normes locales : une tradition graphique propre à l'établissement, des conventions d'abréviation partagées, un style de mise en page reconnaissable. On peut parfois identifier l'origine d'un manuscrit à son écriture seule : c'est l'objet de la paléographie comparée.

Cette notion de norme locale est importante : un modèle entraîné sur des manuscrits de l'abbaye de Saint-Denis ne sera pas nécessairement performant sur des manuscrits de l'abbaye de Cluny. La variabilité inter-scriptorium est une source de défi majeur pour la généralisation des modèles.

### 2.3 La copie comme acte d'interprétation

Un point fondamental, souvent mal compris par qui approche les manuscrits médiévaux depuis l'informatique : **copier un texte au Moyen Âge n'est pas un acte mécanique**. Le copiste n'est pas une photocopieuse.

Lorsqu'un moine copie un texte, il le lit, le comprend (partiellement), et le retranscrit dans sa propre écriture. Ce faisant, il introduit inévitablement des variations : il modernise l'orthographe (selon les conventions de son époque et de sa région), il corrige ce qui lui semble être des erreurs, il développe ou contracte des abréviations selon ses habitudes, il saute des lignes par inadvertance (*saut du même au même* : quand deux passages commencent par les mêmes mots, le copiste peut passer directement du premier au second), il insère des gloses (commentaires) dans le texte lui-même.

Résultat : il n'existe pratiquement jamais deux manuscrits identiques d'un même texte. La **variabilité est la norme**, pas l'exception. C'est précisément l'objet de la philologie traditionnelle que de reconstituer, à partir de tous ces témoins divergents, quelque chose d'approchant le texte « original » : travail appelé l'**édition critique**.

> **Implication pour le ML** : il n'existe pas de « bonne » transcription unique d'un manuscrit médiéval, mais des transcriptions plus ou moins fidèles à ce qu'on voit sur la page, et des éditions critiques qui reflètent des choix éditoriaux. Les données d'entraînement et les métriques d'évaluation doivent tenir compte de cette ambiguïté fondamentale.

---

## 3. Les systèmes d'écriture médiévaux

La **paléographie** est la discipline qui étudie les écritures anciennes : leur morphologie, leur évolution historique, leurs variantes régionales, leurs systèmes d'abréviation. C'est une science qui s'apprend en plusieurs années, par la pratique intensive sur des documents originaux.

Voici un panorama des principaux systèmes d'écriture que vous rencontrerez dans les manuscrits français du IXe au XVe siècle.

### 3.1 L'écriture caroline (IXe–XIIe siècle)

La **minuscule carolingienne** est la grande réforme graphique de l'époque carolingienne (fin VIIIe–IXe siècle). Initiée dans l'entourage de Charlemagne, elle cherche à standardiser l'écriture à l'échelle de l'empire pour faciliter la lecture et la communication administrative.

Ses caractéristiques :
- Lettres **séparées** les unes des autres (peu de ligatures, c'est-à-dire de connexions entre les lettres).
- Formes **régulières et arrondies** : le *a* est formé de deux compartiments, le *d* est droit, le *g* est caractéristique.
- Haste des lettres longues bien marquées (*b*, *d*, *h*, *l*, *p*, *q*).
- **Relativement lisible** pour un lecteur moderne : c'est l'écriture médiévale la plus accessible.

Pour les modèles d'HTR, la caroline est généralement plus facile que les écritures gothiques : les formes sont moins ambiguës, les ligatures moins nombreuses. Les datasets d'écriture caroline (IXe–XIe siècle) sont cependant rares, car cette période a produit moins de manuscrits survivants.

### 3.2 La gothique textualis (XIIIe–XVe siècle)

L'écriture **gothique** (appelée aussi *littera textualis*, *textura*, ou : à tort mais couramment : *black letter*) se développe progressivement à partir du XIIe siècle et domine aux XIIIe–XVe siècles. Elle marque une rupture stylistique profonde avec la caroline.

Ses caractéristiques :
- Formes **anguleuses et brisées** : les courbes de la caroline sont remplacées par des traits droits, des angles et des losanges. On parle de *fracture* de la lettre.
- **Condensation** de l'écriture : les lettres sont plus étroites et plus hautes, permettant d'écrire plus de texte sur moins de parchemin : une économie délibérée quand le support coûte cher.
- **Ligatures systématiques** : les lettres se connectent entre elles selon des règles précises. La suite *de*, *do*, *be*, *bo*, *pe*, *po* forme une ligature caractéristique appelée *bowing* où les deux lettres partagent un trait commun.
- **Ambiguïtés graphiques majeures** : le segment vertical (jambage) est l'unité de base de toutes les lettres. Un *m* est trois jambages, un *n* est deux, un *u* est deux, un *i* est un. Hors contexte, *minim*, *mnim*, *nnm*, *nim* sont graphiquement indistinguables. Ce problème, connu des paléographes sous le nom de **problème des minimes**, est l'une des difficultés techniques les plus redoutables pour les modèles.

> **Exemple** : le mot *minimum* en gothique textualis est une suite de dix-huit jambages parfaitement identiques. Sa lecture est entièrement contextuelle : elle dépend de la connaissance du latin et du vocabulaire probable dans le passage concerné.

La gothique textualis existe elle-même en plusieurs variantes : la *textualis formata* (ou *quadrata*) est la plus soignée, utilisée pour les manuscrits de prestige (bibles, livres d'heures) ; la *textualis semi-quadrata* et la *textualis rotunda* (plus arrondie, développée en Italie) en sont des déclinaisons.

### 3.3 Les écritures cursives (XIVe–XVe siècle)

À partir du XIVe siècle, la croissance de l'administration royale et des échanges commerciaux crée un besoin d'écriture rapide. Les écritures **cursives** se développent pour répondre à ce besoin : les lettres se connectent de manière fluide, le calame (puis la plume) ne se lève presque plus du parchemin.

Les grandes familles cursives françaises :
- **La cursive gothique française** : écriture des actes royaux et des registres administratifs. Très rapide, très liée, avec des *hastes* (traits verticaux supérieurs) très allongés vers la gauche, formant parfois des enchevêtrements en tête de ligne.
- **La bâtarde** (*littera bastarda*) : écriture hybride entre la textualis et la cursive, développée au XVe siècle en France et dans les Pays-Bas bourguignons. Elle est utilisée pour les manuscrits de luxe en langue vernaculaire : romans, chroniques. C'est l'écriture des grands manuscrits de la cour de Bourgogne.
- **La lettre de forme** : variante soignée utilisée pour les chartes solennelles.

Les cursives posent des problèmes radicalement différents de la textualis pour les modèles : les lettres sont rarement isolables, les formes individuelles des caractères varient énormément d'un scripteur à l'autre et selon la vitesse d'écriture. La segmentation en caractères est souvent impossible : c'est pourquoi l'HTR moderne travaille au niveau de la **ligne** entière, et non au niveau du caractère.

### 3.4 Les grandes familles d'écriture : tableau de synthèse

| Époque | Écriture dominante | Lisibilité | Défi principal |
|--------|-------------------|------------|---------------|
| IXe–XIe siècle | Caroline | Élevée | Rareté des datasets |
| XIIe siècle | Transition caroline → pré-gothique | Moyenne | Formes hybrides |
| XIIIe–XIVe siècle | Gothique textualis | Faible à moyenne | Problème des minimes, ligatures |
| XIVe–XVe siècle | Cursive gothique, bâtarde | Faible | Ligatures, variabilité inter-scripteur |

---

## 4. L'ancien et le moyen français : une langue sans norme

### 4.1 Qu'entend-on par « ancien français » et « moyen français » ?

Ces dénominations sont des conventions de linguistes historiques, établies rétrospectivement. Il n'existait pas de conscience de parler une langue distincte du latin : on parlait et on écrivait selon les usages de son milieu, de son époque et de sa région.

- **Le latin** reste la langue de l'Église, de l'université, de la diplomatie internationale et d'une grande partie de la production écrite savante tout au long du Moyen Âge. De nombreux manuscrits mêlent le latin et le vernaculaire dans la même page, voire dans la même phrase.

- **L'ancien français** (environ 842–1340) désigne l'ensemble des variétés de la langue d'oïl parlées et écrites dans le nord de la France médiévale. Le terme « langue d'oïl » signifie « langue où oui se dit *oïl* » (par opposition à la langue d'oc du sud, où oui se dit *oc*).

- **Le moyen français** (environ 1340–1550) désigne la période intermédiaire entre l'ancien français et le français classique. La langue change rapidement : la déclinaison à deux cas disparaît (vers le XIVe siècle), le vocabulaire se latinise massivement, les constructions syntaxiques évoluent.

> **Attention à l'anachronisme** : projeter nos catégories sur le Moyen Âge induit des erreurs. Un clerc du XIIIe siècle ne « parlait pas l'ancien français » : il parlait sa langue, point. Les dénominations d'époque sont : *romanz* (langue romane, par opposition au latin), *françois*, *langage françoys*, *vulgaire*.

### 4.2 L'absence de norme orthographique

C'est probablement le point le plus déstabilisant pour qui aborde les manuscrits médiévaux avec des outils conçus pour des langues modernes normées.

Avant le XVIe–XVIIe siècle, **il n'existe pas d'orthographe normalisée du français**. Pas de dictionnaire de référence, pas d'académie, pas de règles codifiées unanimement reconnues. Chaque scripteur écrit selon son habitude, sa formation, son dialecte et les conventions de son atelier.

Quelques conséquences pratiques :

**Un même mot peut s'écrire de dizaines de façons différentes**
Le mot *roi* (qui désigne le souverain) s'écrit selon les scribes et les périodes : *rey*, *rei*, *roi*, *roy*, *roix*, *roe*, *rez*, *rex* (latinisation)… et plusieurs de ces graphies peuvent coexister dans le même manuscrit, voire dans la même page.

**L'orthographe reflète la phonétique locale**
Un scribe normand écrira différemment d'un scribe de l'Île-de-France, parce que certains sons de leur dialecte ne sont pas identiques. Ces différences dialectales sont précieuses pour les linguistes (elles permettent de localiser géographiquement un manuscrit), mais elles complexifient l'entraînement de modèles de langue.

**L'étymologie entre parfois en concurrence avec la phonétique**
À partir du XIIIe–XIVe siècle, certains scripteurs introduisent des lettres « étymologiques » : des lettres qui ne se prononcent pas mais rappellent l'origine latine du mot. Exemple : *doigt* (du latin *digitus*) au lieu de *doi* ou *doit*. Ces latinisations graphiques constituent une couche supplémentaire d'ambiguïté.

> **Implication directe pour les métriques** : le CER (Character Error Rate) est calculé par comparaison avec une transcription de référence. Mais si deux paléographes expérimentés transcrivent le même mot de deux façons différentes (l'un développe une abréviation d'une façon, l'autre d'une autre façon), lequel a raison ? La notion de « vérité terrain » est philosophiquement problématique dans ce contexte, et il faut en tenir compte dans la conception des évaluations.

### 4.3 La variation dialectale

L'ancien français n'est pas une langue unique mais un continuum de dialectes. Les principaux dialectes littéraires sont :

- **Le francien** (Île-de-France) : qui finira par devenir la base du français moderne, mais qui n'a pas au Moyen Âge de statut de prestige supérieur aux autres dialectes.
- **Le normand** : important dialecte de la Normandie, langue de l'administration anglo-normande après 1066, présent dans une masse considérable de textes juridiques et littéraires.
- **Le picard** : dialecte du nord de la France et de la Belgique actuelle, doté d'une remarquable tradition littéraire (trouvères, farces). Caractéristiques graphiques distinctives : *c* pour *ch*, *ki* pour *qui*, etc.
- **L'anglo-normand** : variété du français parlée et écrite en Angleterre après la Conquête normande. Nombreux documents juridiques, traités, textes religieux.
- **Le champenois**, l'**orléanais**, le **bourguignon**, le **lorrain**… chacun avec ses spécificités graphiques.

Pour les modèles de langue utilisés en post-traitement HTR, ces dialectes posent un problème sérieux : les modèles de langage entraînés sur le français moderne (ou même sur le français médiéval de l'Île-de-France) seront moins performants sur un texte picard, où les probabilités de séquences de caractères sont différentes.

### 4.4 Les abréviations : un système à part entière

L'abréviation est un phénomène massif dans les manuscrits médiévaux. Elle n'est pas un accident ou une paresse : c'est un **système codifié**, appris par les scripteurs au cours de leur formation, qui permettait d'écrire plus vite et d'économiser le parchemin.

On distingue plusieurs grands types :

**La suspension**
On écrit le début du mot et on coupe, en signalant la coupure par un signe (une barre, un point, une lettre finale en exposant). Exemple : *fris* avec une barre au-dessus pour *francorum*. La règle générale est que la première syllabe ou les premières consonnes sont écrites.

**La contraction**
On écrit le début et la fin du mot en supprimant le milieu, là encore signalé par un signe. Exemple : *dns* pour *dominus* (seigneur). Ce procédé est particulièrement fréquent pour les mots latins très courants dans les textes religieux et juridiques.

**Les signes spéciaux**
Certains mots ou syllabes très fréquents ont un signe propre, non dérivable des lettres ordinaires. Le plus célèbre est le **tilde** (ou tilde médiéval, qui ressemble à une vague ou un tiret ondulé au-dessus d'une lettre) : placé au-dessus d'une lettre, il signifie généralement qu'une lettre nasale (*m* ou *n*) a été omise. Exemple : *hõe* avec tilde au-dessus du *o* pour *home* (*homme*). D'autres signes signalent *con-*, *com-*, *per-*, *pro-*, *-us*, *-et*…

**Les ligatures de lettres**
Certaines combinaisons de lettres sont écrites en un seul geste, formant un signe composite. *&* (esperluette) est à l'origine une ligature de *e* et *t* (*et* en latin). *æ* est une ligature de *a* et *e*.

> **Pour le modèle** : les abréviations posent un problème fondamental. Un système OCR classique transcrit ce qu'il voit. Un système HTR pour les manuscrits médiévaux doit décider s'il transcrit le signe abréviatif tel quel (transcription *diplomatique*, fidèle à la surface du document) ou s'il le développe en toutes lettres (transcription *normalisée*). Ces deux choix correspondent à des cas d'usage différents et à des métriques d'évaluation différentes. Il faut choisir : et documenter ce choix.

---

## 5. Les obstacles à la lecture : une cartographie des difficultés

### 5.1 Obstacles au niveau du caractère

**Le problème des minimes**
Décrit en section 3.2 pour la gothique textualis : les lettres *m*, *n*, *u*, *i* partagent la même unité graphique (le jambage vertical). Leur distinction est purement contextuelle : morphologique ou lexicale, jamais graphique au niveau local.

**L'ambiguïté *u* / *v***
Au Moyen Âge, *u* et *v* ne sont pas deux lettres distinctes : c'est une seule lettre avec deux formes positionnelles. En début de mot, on trouve généralement *v* ; à l'intérieur du mot, *u*. La distinction entre les valeurs vocalique (*u*) et consonantique (*v*) n'est pas systématiquement notée graphiquement. Même ambiguïté entre *i* et *j*.

**La distinction majuscule / minuscule**
Elle n'est pas codifiée de la même façon qu'aujourd'hui. Les grandes lettres (appelées *litterae maiores* ou *capitales*) n'indiquent pas nécessairement un nom propre ou un début de phrase : elles peuvent marquer une rupture rythmique, une emphase, ou simplement une fantaisie du scribe.

**Les lettres homophones**
*c* et *t* sont souvent graphiquement proches, surtout en gothique tardive. *s* long (ʃ, le *s* initial médiéval qui ressemble à un *f*) peut être confondu avec *f*. *r* et *n* en fin de certaines combinaisons peuvent être ambigus.

### 5.2 Obstacles au niveau du mot

**Les frontières de mots**
Dans de nombreux manuscrits médiévaux, les mots ne sont pas toujours clairement séparés par des espaces. La scriptio continua (écriture continue sans espaces) est plus fréquente dans les textes anciens ; mais même aux XIIIe–XVe siècles, les espaces sont irréguliers, et des mots fréquents peuvent être agglutinés (*del* pour *de le*, *au* pour *à le*, mais pas de façon systématique).

**Les formes inattendues**
La lexicographie médiévale n'est pas la lexicographie moderne. Des mots qui semblent familiers ont des formes imprévues ; des mots inconnus du lecteur moderne sont fréquents. Un modèle de langage qui a été entraîné sur du français moderne sera peu fiable pour corriger les transcriptions d'ancien français.

### 5.3 Obstacles au niveau de la ligne

**Les ratures et les corrections**
Le scribe se corrige : il rature, il gratte, il ajoute des lettres ou des mots en interligne (*addition supralinéaire*) ou en marge. Ces corrections appartiennent au texte mais brouillent la lecture séquentielle d'une ligne.

**Les fins de ligne et les coupures de mots**
La coupure syllabique en fin de ligne n'est pas normalisée. Un mot peut être coupé n'importe où, parfois sans aucun signe de coupure, parfois avec un tiret ou un signe spécial.

**Le cadrage et la justification**
Certains scribes cherchent à justifier leur ligne à droite (comme un texte imprimé moderne) en comprimant ou en étirant les espaces, ou en ajoutant des signes de remplissage. D'autres laissent des lignes inachevées.

### 5.4 Obstacles au niveau de la page

**La structure de la page**
Un système de segmentation de layout doit identifier correctement les régions de la page : corps du texte principal, rubriques, annotations marginales, illustrations, lettrines. Ces régions ont des statuts textuels différents et doivent être traitées séparément.

**La question de l'ordre de lecture**
Sur une page avec deux colonnes de texte, des notes en marge, une rubrique en tête et une lettrine, quel est l'ordre de lecture ? Pour un humain entraîné, c'est souvent intuitif. Pour un modèle, il faut définir des règles explicites : et ces règles peuvent être fausses pour certains documents.

**Les dégradations physiques**
- Taches d'encre, d'eau, de cire de bougie.
- Moisissures qui dissolvent partiellement l'encre ou le parchemin.
- Déchirures, lacunes physiques.
- Transparence : le texte du verso apparaît en miroir sur le recto.
- Reliure trop serrée : le texte proche de la gouttière (le centre du livre ouvert) disparaît dans l'ombre ou la courbure.
- Dorures et pigments qui saturent les capteurs de numérisation.

---

## 6. Ce que la tradition philologique apporte à la vision par ordinateur

### 6.1 La distinction entre transcription diplomatique et transcription normalisée

La philologie a développé des conventions précises pour répondre à une question que tout modèle HTR doit également trancher : **que transcrit-on exactement ?**

**La transcription diplomatique** (ou *fac-similé diplomatique*) reproduit le texte tel qu'il apparaît sur le document, sans développer les abréviations, sans corriger les erreurs apparentes du scribe, sans moderniser l'orthographe. Elle note chaque signe abréviatif tel quel, signale les lacunes, les corrections et les ajouts. C'est la transcription la plus fidèle au document matériel.

**La transcription normalisée** (ou *transcription critique*) développe les abréviations, standardise l'orthographe selon des conventions décidées à l'avance, corrige les erreurs manifestes du scribe. Elle est plus lisible mais plus interprétative.

Pour notre projet, la transcription diplomatique est la cible prioritaire : c'est celle qui est objectivement évaluable (on peut calculer un CER contre une référence sans débat interprétatif sur le développement des abréviations). La normalisation sera le travail du module NLP.

### 6.2 Le TEI comme pont entre philologie et données

La **Text Encoding Initiative** (TEI) est un consortium international qui développe depuis 1987 des recommandations pour l'encodage numérique des textes. Son format, le **TEI P5**, est le standard de facto des éditions numériques de textes patrimoniaux dans les humanités numériques.

Le TEI utilise un format XML avec des balises spécialisées pour encoder la structure et le contenu des documents. Quelques balises fondamentales pour notre contexte :

```xml
<lb/>          <!-- Saut de ligne (line break) -->
<pb/>          <!-- Saut de page (page break) -->
<cb/>          <!-- Saut de colonne (column break) -->

<gap reason="illegible" unit="chars" quantity="3"/>
               <!-- Lacune illisible, environ 3 caractères -->

<abbr>dns</abbr>
<expan>dominus</expan>
               <!-- Abréviation et son développement -->

<add place="supralinear">mot ajouté</add>
               <!-- Addition supralinéaire (interlinéaire) -->

<del rend="strikethrough">mot barré</del>
               <!-- Rature -->

<note place="margin">annotation marginale</note>
               <!-- Note en marge -->

<fw type="catchword">premier</fw>
               <!-- Réclame (mot répété en bas de page) -->
```

Le TEI est un excellent format de sortie pour notre pipeline : il est suffisamment expressif pour encoder l'incertitude, la structure et les métadonnées, et il est lisible par les outils des humanités numériques.

### 6.3 La question de la vérité terrain et des annotations divergentes

En apprentissage automatique supervisé, on suppose l'existence d'une vérité terrain (*ground truth*) : pour chaque image d'entrée, il existe une transcription correcte avec laquelle évaluer le modèle. Cette hypothèse est commode mais inexacte pour les manuscrits médiévaux.

Des expériences menées dans des projets de transcription collaborative (Zooniverse, Transkribus, eScriptorium) montrent systématiquement que deux paléographes experts, transcrivant le même passage, produisent des transcriptions qui diffèrent sur 3 à 8% des caractères en moyenne : et davantage sur les passages difficiles.

La philologie propose plusieurs approches pour gérer cette ambiguïté :
- **L'édition collaborative** : plusieurs relecteurs, consensus par discussion.
- **L'apparat critique** : noter explicitement les divergences entre témoins ou entre interprétations.
- **La probabilisation** : dans les éditions numériques modernes, certains systèmes encodent un *degré de certitude* (`<certainty>`) sur une lecture.

Pour le ML, cela se traduit par :
- L'utilisation d'annotations multiples et de mécanismes de consensus (vote, score de confiance).
- L'évaluation contre plusieurs transcriptions de référence (retenir la meilleure ou la moyenne).
- L'encodage explicite de l'incertitude dans le format de sortie (`"confidence": 0.72, "needs_review": true`).

---

## 7. Synthèse : traduire les contraintes philologiques en spécifications techniques

Le tableau suivant résume les correspondances entre les problèmes identifiés par la philologie et les choix techniques qu'ils impliquent.

| Contrainte philologique | Choix technique impliqué |
|------------------------|--------------------------|
| Variabilité de mise en page (colonnes, rubriques, marges) | Segmentation de layout avant l'HTR : deux étapes distinctes |
| Problème des minimes (ambiguïté graphique locale) | HTR au niveau de la ligne entière, pas du caractère : modèles séquentiels (TrOCR, Kraken) |
| Abréviations | Choisir entre transcription diplomatique (reproduire le signe) et normalisée (développer) : documenter le choix |
| Absence de norme orthographique | Métriques permissives ; évaluation contre plusieurs références ; pas de correction orthographique à ce stade |
| Variation dialectale | Segmentation du corpus par dialecte si possible ; sinon, accepter une dégradation des performances |
| Mélange latin / vieux français | Détection de langue par région ; modèles séparés pour le latin et le vernaculaire |
| Dégradations physiques | Prétraitement robuste (binarisation adaptative, correction de transparence) ; augmentation de données synthétique |
| Annotations divergentes | Annotations multiples ; mécanismes de vote ; encodage de la confiance dans le format de sortie |
| Ordre de lecture ambigu | Règles de layout explicites ; flag d'incertitude quand la règle ne s'applique pas |
| Illustrations et enluminures | Segmentation distincte du texte et de l'image ; description automatique (CLIP/LLaVA) pour les zones non-textuelles |

---

## Bibliographie de référence

### Paléographie et codicologie

- **Muzerelle, D.** (1985). *Vocabulaire codicologique : répertoire méthodique des termes français relatifs aux manuscrits*. CEMI, Paris. : La référence terminologique fondamentale en français. Version électronique disponible : [codicologia.irht.cnrs.fr](http://codicologia.irht.cnrs.fr)

- **Derolez, A.** (2003). *The Palaeography of Gothic Manuscript Books: From the Twelfth to the Early Sixteenth Century*. Cambridge University Press. : La somme de référence sur les écritures gothiques, avec une description systématique et illustrée de chaque famille.

- **Stiennon, J.** (1973). *Paléographie du Moyen Âge*. Armand Colin, Paris. : Introduction classique en langue française, accessible aux non-spécialistes.

- **Bischoff, B.** (1990). *Latin Palaeography: Antiquity and the Middle Ages*. Cambridge University Press. (Traduction de l'allemand *Paläographie des römischen Altertums und des abendländischen Mittelalters*, 1979.) : Référence pour la période ancienne (carolingienne et antérieure).

- **Parkes, M. B.** (1992). *Pause and Effect: An Introduction to the History of Punctuation in the West*. Scolar Press, Aldershot. : Sur la ponctuation médiévale, essentielle pour la segmentation en phrases.

### Linguistique de l'ancien et du moyen français

- **Zink, G.** (1992). *Phonétique historique du français*. Presses Universitaires de France (coll. Linguistique nouvelle). : Introduction rigoureuse à l'évolution phonétique du français médiéval.

- **Moignet, G.** (1988). *Grammaire de l'ancien français*. Klincksieck, Paris. : Grammaire de référence pour l'ancien français, morphologie et syntaxe.

- **Marchello-Nizia, C.** (1997). *La langue française aux XIVe et XVe siècles*. Armand Colin. : Référence pour le moyen français.

- **Buridant, C.** (2000). *Grammaire nouvelle de l'ancien français*. SEDES, Paris. : Grammaire complète et actualisée.

- **Trésor de la langue française informatisé (TLFi)** : Base de données lexicographique qui couvre le français du IXe au XXe siècle. Accessible en ligne : [atilf.atilf.fr/tlfi.htm](http://atilf.atilf.fr/tlfi.htm)

- **Dictionnaire du Moyen Français (DMF)**, ATILF/CNRS. : Dictionnaire spécialisé pour le moyen français (1330–1500). En ligne : [atilf.atilf.fr/dmf/](http://atilf.atilf.fr/dmf/)

### Philologie et humanités numériques

- **Guyotjeannin, O., Vielliard, F.** (dir.) (2001). *Conseils pour l'édition des textes médiévaux*, 3 vol. Comité des Travaux Historiques et Scientifiques / École nationale des chartes, Paris. : La méthode philologique française de référence pour l'édition des textes médiévaux.

- **Burnard, L., Sperberg-McQueen, C. M.** (1994, mis à jour régulièrement). *TEI P5: Guidelines for Electronic Text Encoding and Interchange*. Text Encoding Initiative Consortium. : La spécification de référence du format TEI. En ligne : [tei-c.org/guidelines/](https://tei-c.org/guidelines/)

- **Stutzmann, D.** (2011). *Outils d'analyse de la dynamique des écritures médiévales par ordinateur*. Document numérique, 14(1), 81–107. Lavoisier. : Article fondateur sur l'application de la vision par ordinateur à la paléographie médiévale en France. À lire en priorité.

- **Camps, J.-B., Clérice, T., Pinche, A.** (2021). *Evaluating Data Augmentation for Historical Handwritten Text Recognition*. Document and Image Recognition : Proceedings of ICDAR 2021. : Sur les stratégies d'augmentation de données pour l'HTR médiéval.

- **Terras, M.** (2006). *Image to Interpretation: An Intelligent System to Aid Historians in Reading the Vindolanda Texts*. Oxford Studies in Ancient Documents. Oxford University Press. : Étude de cas pionnière sur la reconnaissance automatique d'écriture ancienne.

### Projets de recherche et ressources en ligne

- **IRHT (Institut de Recherche et d'Histoire des Textes)**, CNRS. Carnet de recherche *Écriture médiévale & numérique* : [irht.hypotheses.org](https://irht.hypotheses.org) : Veille sur l'application de l'informatique aux manuscrits médiévaux, par les spécialistes français de référence.

- **eScriptorium** (EPHE / INRIA) : plateforme collaborative d'annotation et d'entraînement de modèles HTR, développée spécifiquement pour les documents patrimoniaux. [escriptorium.fr](https://escriptorium.fr)

- **HTR-United** : initiative collaborative pour le partage de modèles et de données HTR pour les manuscrits anciens. [htrunited.github.io](https://htrunited.github.io)

- **Projet CREMMA** (Consortium pour la Reconnaissance et l'Édition de Manuscrits Médiévaux Anciens), ANR : corpus de référence pour l'HTR en vieux et moyen français. Direction : Ariane Pinche (EPHE), Jean-Baptiste Camps (École nationale des chartes).

- **Gallica**, Bibliothèque nationale de France : [gallica.bnf.fr](https://gallica.bnf.fr) : Plus de 7 millions de documents numérisés, API IIIF, notice descriptive de chaque manuscrit.

- **e-Codices** (Bibliothèque virtuelle des manuscrits médiévaux de Suisse) : [e-codices.unifr.ch](https://www.e-codices.unifr.ch) : Numérisation intégrale de haute qualité, métadonnées riches.

---

*Support de cours rédigé pour le module Computer Vision · Promotion MD5 · Mai 2026*
*Ce document accompagne la séance 1.1 du Jour 1. Il est conçu pour des étudiants sans formation préalable en linguistique ou en paléographie.*
