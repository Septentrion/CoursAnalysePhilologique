---
updated: 2026-05-17T16:22:19.053+02:00
edited_seconds: 4338
---
# Lecture de manuscrit : approche philologique et paléographique
- **Module 1 · Computer Vision appliquée aux manuscrits médiévaux · MD5**

---
<!-- slide template="[[tpl-img-left]]" -->

::: title
## Pourquoi commencer par la philologie ?
:::

::: image
![[Cours/Computer Vision/Michel/CoursAnalysePhilologique/metadata/Cours_1_1_Lecture_Manuscrit_Philologie-1778578979558.png]]
:::
+ la **paléographie** (science de l'écriture ancienne)
+ la **codicologie** (science du livre manuscrit comme objet matériel)
+ la **philologie** (science des textes dans leur transmission)

notes:
- philologie - étude de la langue orale et écrite dans les ressources historiques -> science historique (contrairement à la linguistique qui est une science naturelle)

L'HTR n'est pas un simple problème de vision ordinateur

Un modèle HTR doit apprendre :
- à distinguer texte / non-texte
- à interpréter des signes ambigus, des traits, du bruit
- à gérer des variantes linguistiques, des abbreviations
- à traiter des documents non standardisés, différentes langues mélangées.

- Ces décisions sont, en régime humain, le travail de plusieurs disciplines, accumulées sur plusieurs siècles. 
- Un modèle de vision par ordinateur devra apprendre à faire, de façon approximative et statistique, ce que des spécialistes font après des années de formation.

Ce cours d'introduction a deux objectifs :
1. Vous donner les outils conceptuels pour comprendre la nature réelle du problème.
2. Comprendre conceptuellement le problème avant d’essayer de le résoudre.
3. Vous permettre de traduire ce problème en spécifications techniques : car un ingénieur qui ne comprend pas le domaine produira des métriques qui mesurent autre chose que ce qui compte.

---
## 1. Le manuscrit médiéval comme objet matériel

--
### 1.1 Qu'est-ce qu'un manuscrit ?
- Le mot *manuscrit* vient du latin *manu scriptum* : 
« écrit à la main ». 
+ ==1450==
	+ Invention de l'**imprimerie** à caractère mobiles - **Johannes Gutemberg**

notes:
Il désigne **tout document rédigé manuellement**, par opposition aux textes imprimés (la typographie à caractères mobiles n'est introduite en Europe occidentale qu'autour de 1450, par Gutenberg). 

Jusqu'au 15e siècle tout document écrit est donc un manuscrit.

Mais tous les manuscrits ne se ressemblent pas...

--
<!-- slide template="[[tpl-img-bottom-2col]]" -->

::: title
#### Le codex
:::

::: image
![[Cours/Computer Vision/Michel/CoursAnalysePhilologique/metadata/Cours_1_1_Lecture_Manuscrit_Philologie-1778591114273.png|*Codex Argenteus - Bible Goths couvrée d'argent.*|750]]

:::

::: left
**Ancêtre direct du livre moderne**
- Cahiers reliés
- Lecture non linéaire possible
- Support des textes longs
:::

::: right
**Conséquences en Computer Vision**
- courbure près de la reliure
- ombres de gouttière
- variations de page
:::

notes:
C'est la forme qui ressemble le plus à notre livre moderne : des feuilles pliées, assemblées en cahiers (*quaternions*, *sénions*…), cousues et reliées. 
C'est le support de prédilection pour les textes longs : bibles, textes liturgiques, romans, traités philosophiques ou scientifiques, compilations juridiques. 

Le codex s'est imposé progressivement face au rouleau entre le IIe et le IVe siècle de notre ère. -> plus résitant, protégé du temps par sa couverture, donc mieux conservé, plus de codex encore existant comparé aux rouleaux.

--
<!-- slide template="[[tpl-img-left-lg]]" -->

::: title
#### Le rotulus
:::

::: image 
![[Cours/Computer Vision/Michel/CoursAnalysePhilologique/metadata/Cours_1_1_Lecture_Manuscrit_Philologie-1778591354860.png]]

:::
- **Format**
	- Rouleau vertical ou horizontal
	- Lecture séquentielle continue
	- Usage administratif ou liturgique
- **Conséquences en Computer Vision**
	- segmentation inhabituelle
	- scans fragmentés
	- ordre de lecture spécifique


notes:
Un rouleau de **parchemin** ou de **papyrus**, utilisé pour des documents administratifs, des actes, des textes liturgiques destinés à être lus debout. Il se développe verticalement ou horizontalement et impose une lecture séquentielle linéaire, sans possibilité de navigation dans le texte.


- Les **Rouleaux des morts**, utilisés par les monastères pour transmettre les noms des défunts afin de prier pour eux. -> rouleau porté pendant un voyage qui pouvait durer plus d'un an à travers l’Europe
- Le rouleau mortuaire de Jean II et de Gautier III, abbés de Saint-Bavon de Gand a une longueur de 30 m
 https://www.uvsq.fr/le-projet-tituli-etude-des-rouleaux-des-morts

--
<!-- slide template="[[tpl-img-left-lg]]" -->

::: title
#### La charte

:::

::: image
![[Cours/Computer Vision/Michel/CoursAnalysePhilologique/metadata/Cours_1_1_Lecture_Manuscrit_Philologie-1778714015999.png|Ordonnances du treschrestien roy de France]]
:::
- **Format**
	- Document juridique
	- Forte standardisation
	- Souvent un seul recto
	- Présence fréquente de sceaux
	- écriture rapide
	- formules répétitives
	- mise en page compacte

notes:
Document juridique attestant un acte (donation, vente, accord, privilège). La charte est généralement un feuillet unique, écrit d'un seul côté (*recto*), avec des formules juridiques très standardisées. Elle porte souvent un sceau (en cire, en plomb) qui lui confère sa valeur légale.

--
![[Cours/Computer Vision/Michel/CoursAnalysePhilologique/metadata/Cours_1_1_Lecture_Manuscrit_Philologie-1778591784224.png]]

notes:
- Page de frontispice du manuscrit du « Décret de Gratien » (Concordia discordantium canonum ou Decretum Gratiani) datant de 1140 et régulant les lois et canons de l'Église catholique romaine. 14e siècle Sienne, Biblioteca degli Intronati
Le **Décret de Gratien** (_Decretum Gratiani_), rédigé vers **1140** par le moine Gratien à **Bologne**, est l'œuvre majeure du droit canonique du XIIe siècle.  Il constitue la première partie du **Corpus Iuris Canonici** et est resté en vigueur jusqu'à la promulgation du **Code de droit canonique de 1917**. 

- **Titre original** : _Concordia discordantium canonum_ (« Concorde des canons discordants »). 
- **Contenu** : Compilation d'environ **3 800 textes** (canons apostoliques, décrets conciliaires, décrétales pontificales, lois romaines et écrits patristiques). 
- **Méthode** : Utilisation de la technique dialectique du _sic et non_ pour résoudre les contradictions entre les sources juridiques.

--
<!-- slide template="[[tpl-img-bottom-2col]]" -->

::: title
#### Le registre

:::

::: image
![[Cours/Computer Vision/Michel/CoursAnalysePhilologique/metadata/Cours_1_1_Lecture_Manuscrit_Philologie-1778592193503.png|**Grand Thalamus** de Montpellier|650]]
:::

::: left 
- **Format**
	- Compilation chronologique
	- Colonnes spécialisées
	- Écriture standardisée
:::
::: right 
- **Intérêt en Computer Vision**
	- layouts répétitifs
	- adapté à la segmentation supervisée
:::

notes:
Livre dans lequel une **institution** (chancellerie royale, évêché, tribunal, municipalité) consigne **chronologiquement** ses actes. Les registres sont souvent écrits par des **clercs professionnels** formés à une écriture standardisée dite de chancellerie.

Le **Grand Thalamus** de Montpellier (manuscrit AA4 des Archives municipales) est un **grand cartulaire** volumineux de **387 feuillets** en parchemin, rédigé à partir de **1221** et complété jusqu'en **1675**.  Il contient les textes diplomatiques fondamentaux de la ville, notamment les **coutumes de 1204**, 
	- les **privilèges accordés** par les papes et les rois d'Aragon, de Majorque et de France, ainsi que les **règlements de métiers et les titres de propriété.**

> **Pourquoi cette distinction compte pour le projet ?** 
> 	- Les modèles HTR (Handwritten Text Recognition) entraînés sur des chartes ne fonctionneront probablement pas bien sur des romans arthuriens, et vice versa. 
> 	- Le **type de document** détermine le type d'écriture, le registre de langue, la densité des abréviations, la mise en page. 
> 	- Un bon pipeline doit d'abord **classifier le type de document** avant de choisir quel modèle appliquer.

- Le type de document n'est pas la seule chose influant le modèle.
--
<!-- slide template="[[plt-two-col]]" -->

::: title
### 1.2.a Le support d'écriture
:::

::: left
##### **Le parchemin** 
- Support dominant jusqu'au XIVe siècle
- Réutilisable → palimpsestes
+ Peau animale préparée

**Conséquences pour la numérisation**  
- Il jaunit et se déforme avec l'âge -> variation luminteuses
- La transparence fait apparaître l'écriture du verso en filigrane sur le recto
- Les grattages et les retouches laissent des traces
:::

::: right
##### **Le papier**  
- Inventé en Asie au IIe siècle av. J-C, fait à partir de **fibres végétales** (chiffons, lin)
- Introduit en Europe au XIIIe siècle
- Moins coûteux que le parchemin, il devient dominant au XVe siècle

**Conséquences pour la numérisation**  
- humidité, tâches, encre diffusée, déchirures

:::

notes:

Support dominant du Moyen Âge occidental jusqu'au **XIVe** siècle environ. Il est fabriqué à partir de peau d'animal (mouton, chèvre, veau : le veau donne le *vélin*, plus fin et plus blanc) : la peau est trempée, tendue sur un cadre, raclée, séchée. Le résultat est une surface lisse, résistante, translucide par endroits. Le parchemin est cher, rare, réutilisable (en grattant la surface : ce qui donne les *palimpsestes*, où un texte premier partiellement effacé est recouvert d'un nouveau).


**Conséquences pour la numérisation**  
- Il jaunit et se déforme avec l'âge -> ce qui crée des variations de luminosité sur le scan.  
- La transparence (ou *show-through*) fait apparaître l'écriture du verso en filigrane sur le recto : une source de bruit considérable pour les algorithmes de segmentation.  
- Les grattages et les retouches laissent des traces visuellement ambiguës.

"Des destinées de l'âme", écrit par l'auteur français du **XIXe** siècle Arsène Houssaye - Fabriqué en peau humaine prise sur le dos du corps non réclamé d'une patiente enfermée dans un hôpital psychiatrique français, qui est morte soudainement d'apoplexie."

- Encêtre du papier: le papyrus inventé vers 3000 avant JC par les Egyptiens.
	- Fabriqué à partir de lamelles de papyrus, pressées, sechées.

Introduit en Europe occidentale au XIIIe siècle (d'abord en Espagne, puis en Italie et en France), le papier se répand progressivement et devient dominant au XVe siècle. Il est moins cher que le parchemin, mais moins durable. Pour les modèles, le papier vieilli présente des problèmes différents : taches d'humidité, encre qui saigne, déchirures.

--
<!-- slide template="[[tpl-two-col-bottom]]" -->
::: title
### 1.2.b Le support d'écriture
:::

::: left
##### **Encres au carbone**
- **Antiquité** et **Haut Moyen Âge** - XIIe
- Encres mélangeant charbon ou suie et un agent liant glucidique, protéinique ou lipidique
- Très stables dans le temps, mais peuvent s'écailler.
:::

::: right 
##### **Encre métallo-gallique**
- **Moyen Âge Central** à **l'époque moderne** - XIIe-XIXe
- Encres fait à partir d'extraits végétaux (noix de galle), de vitriole (sel métallique - fer ou cuivre), agent liant lipidique  et additifs optionnels
- Encre très corrosive pour le support
:::
##### **Encre de synthèse**
- **Époque contemporaine (après 1850)** : 
- Encres faites à base de **colorants chimiques** dissous dans de l'eau ou de l'alcool.

notes:
**L'encre**  
L'encre médiévale la plus courante est l'encre ferro‑gallique : un mélange de sulfate de fer et d'acide gallique (extrait des galles de chêne), qui produit une encre noire au dépôt, mais qui avec le temps peut ronger le support par oxydation. On obtient ainsi des pages où l'encre a littéralement percé le parchemin : ce qu'aucun prétraitement d'image ne peut restaurer.

L'évolution des encres utilisées pour les manuscrits en Europe a marqué une transition majeure, passant de l'encre au **carbone** à l'**encre métallo-gallique**, puis aux encres de synthèse.

- **Antiquité et Haut Moyen Âge** : Jusqu'au XIIe siècle, on utilisait principalement des **encres au carbone** (noir de fumée ou charbon) liées à de la gomme arabique, du miel ou du blanc d'œuf.  Ces encres étaient stables mais peu adhérentes et parfois sujettes à l'écaillage.
	    - glucidique (gomme d'arbres, gomme arabique, miel), protéinique (blanc d'œuf, gélatine, colle de peau) ou bien lipidique (huiles).
	    - Se présentaient sous forme solide
- **Moyen Âge Central à l'époque moderne (XIIe-XIXe siècles)** : L'**encre métallo-gallique** (ou ferro-gallique) devient la norme, remplaçant progressivement les encres au carbone.  Composée d'extrait de **noix de galle** (riche en tanins) et de **sulfate de fer** (vitriol), elle offre un noir profond et indélébile.  Bien qu'elle ait permis la préservation de nombreux documents, sa composition corrosive a endommagé de nombreux parchemins et papiers au fil des siècles.
	- Noix de galle (haut en tanins) - décoction/macération. -> encre rougeatre
	- Ajout de sulfate de cuivre ou de fer (vitriole)  -> encre noire
	- Un liant lipidique pour tenir les pigments en suspention -> augmentait la viscosité du liquide
	- Ajout potentiel d'autres pigments comme lapiz, certaines poudres pour donner des reflets de couleur.
	  Très corosiqves
- **Époque contemporaine (après 1850)** : Avec l'avènement de la chimie organique, les encres de synthèse à base de **colorants chimiques** dissous dans l'eau ou l'alcool remplacent les formules traditionnelles, résolvant les problèmes de corrosivité tout en offrant une plus grande variété de couleurs et de fluidités.


--

![[Cours/Computer Vision/Michel/CoursAnalysePhilologique/metadata/Cours_1_1_Lecture_Manuscrit_Philologie-1778607474190.png]]

--
<!-- slide template="[[tpl-top-content-2col]]" -->

::: title
### 1.3 La mise en page
:::

::: top
La page d'un manuscrit médiéval varie selon l'**époque**, l'origine **géographique** et le type de texte (comme vu précédemment).
:::

::: left
- ==Le réglage== : lignes directrices tracées pour guider l'écriture.
- ==La mise en colonnes== : 
	- **Bible** et **textes liturgiques** - deux à trois colonnes 
	- **Romans** et les **chroniques** - généralement une colonne. 
	- **Registres** - multiples colonnes spécialisées (dates, noms...).
- ==Les rubriques== : titres, sous-titres ou indications écrits à l'encre rouge. Ils appartiennent au texte mais ont un statut.
:::

::: right
- ==Les lettrines== : grandes lettres initiales de chapitres ou de sections, souvent ornées ou enluminées. font partie du texte, mais à traiter individuellement.
- ==Les annotations marginales== : précisions, corrections ou idées notées dans la marge par les différents possesseurs.
- ==Les illustrations et enluminures== : miniatures peintes, dessins à la plume, diagrammes, cartes. - elles sont à isoler par un modèle de segmentation
:::
notes:
- **Le réglage** : lignes directrices tracées pour guider l'écriture.
	- avant d'écrire, le copiste trace des lignes directrices (à la pointe sèche, à la mine de plomb, ou à l'encre) pour guider son écriture. Ces lignes sont parfois encore visibles sur les scans.
- **La mise en colonnes** : 
	- Bible et textes liturgiques - deux à trois colonnes les bibles et les textes liturgiques sont souvent disposés sur deux, voire trois colonnes. Les romans et les chroniques sont généralement à une colonne. Les registres ont des colonnes spécialisées (dates, sommes, noms).
- **Les rubriques** : titres, sous-titres ou indications de structure écrits à l'encre rouge (*rubrum* en latin). Ils appartiennent au texte mais ont un statut différent du corps principal.

- **Les lettrines** : grandes lettres initiales de chapitres ou de sections, souvent ornées, parfois enluminées. Elles posent un problème spécifique : elles font partie du texte, mais un modèle doit les traiter différemment des lettres ordinaires.
- **Les annotations marginales** : le lecteur médiéval annotait abondamment. Ces annotations peuvent être contemporaines du texte, ou ajoutées des décennies ou des siècles plus tard, par d'autres mains, parfois dans d'autres langues. Sont-elles du texte à transcrire ? Dans quel ordre ?
- **Les illustrations et enluminures** : des miniatures peintes, des dessins à la plume, des diagrammes, des cartes. Elles sont physiquement présentes sur la page mais ne sont pas du texte. Un modèle de segmentation doit apprendre à les isoler.

---
## 2. Chaîne de transmission : copistes et scriptoria

--
### 2.1 Qui écrivait ?
- ==VIIe–XIIe== : clercs (moines, écoles cathédrales) – écriture indissociable du religieux
- ==À partir du XIIIe== : scribes laïcs professionnels (écrivains publics, clercs de notaire, chancelleries)

**Scriptorium** : atelier d'écriture d'une abbaye
- Normes locales : tradition graphique, conventions d'abréviation, mise en page
- On peut identifier l'origine d'un manuscrit à son écriture (paléographie comparée)

> [!tldr] **Conséquence ML**: Un modèle entraîné sur une abbaye précise sera moins performant sur une autre. La variabilité inter‑scriptorium est un défi majeur pour la généralisation.

notes: 
Insister sur le fait que l'écriture n'est pas neutre. Les scriptoria produisaient des "polices de caractères" avant l'heure, mais locales et non standardisées à l'échelle européenne.


Pendant une grande partie du Moyen Âge (VIIe–XIIe siècle surtout), l'écriture est quasi-exclusivement le fait de **clercs** : des hommes d'Église formés dans des écoles monastiques ou cathédrales. L'écriture est alors indissociable de la culture religieuse : on **copie des textes sacrés, des textes** patristiques (les Pères de l'Église), des textes **liturgiques**.

À partir du XIIIe siècle, avec l'essor des villes, du commerce et de l'administration royale, des **scribes laïcs professionnels** apparaissent : les *écrivains publics*, les *clercs de notaire*, les employés des chancelleries. L'écriture devient progressivement un métier civil.

Ce glissement a des conséquences directes sur les écritures : 
- les scriptoria monastiques produisent des écritures soignées, calligraphiées, relativement standardisées au sein d'un atelier. 
- Les scribes laïcs produisent des écritures plus rapides, plus personnelles, plus variables.

Un **scriptorium** est l'atelier d'écriture d'une abbaye ou d'une grande institution. Les moines copistes y travaillent selon des normes locales : une tradition graphique propre à l'établissement, des conventions d'abréviation partagées, un style de mise en page reconnaissable. On peut parfois identifier l'origine d'un manuscrit à son écriture seule : c'est l'objet de la paléographie comparée.

Cette notion de norme locale est importante : un modèle entraîné sur des manuscrits de l'abbaye de Saint-Denis ne sera pas nécessairement performant sur des manuscrits de l'abbaye de Cluny. La variabilité inter-scriptorium est une source de défi majeur pour la généralisation des modèles.

--
### 2.2 La copie comme acte d'interprétation

**Un copiste n'est pas une photocopieuse.** En recopiant, il :
- modernise l'orthographe (selon son époque, sa région)
- corrige ce qui lui semble être des erreurs
- développe ou contracte les abréviations selon ses habitudes
- saute des lignes (_saut du même au même_)
- ajoute des gloses (commentaires) dans le texte

**Résultat** : il n'existe pratiquement jamais deux manuscrits identiques d'un même texte. La **variabilité est la norme**.

> [!tldr] **Conséquence ML**:  il n'existe pas de "bonne" transcription unique d'un manuscrit médiéval, mais des transcriptions plus ou moins fidèles. Les données d'entraînement et les métriques doivent tenir compte de cette ambiguïté.

notes: 
La philologie traditionnelle reconstruit un texte "original" par édition critique – un travail de conjecture. Pour le HTR, nous ne cherchons pas l'original mais une transcription diplomatique fiable. À rappeler quand les étudiants s'étonneront de voir des divergences entre annotateurs.

Un point fondamental, souvent mal compris par qui approche les manuscrits médiévaux depuis l'informatique : **copier un texte au Moyen Âge n'est pas un acte mécanique**. Le copiste n'est pas une photocopieuse.

Lorsqu'un moine copie un texte, 
- il le lit, le comprend (partiellement), et le retranscrit dans sa propre écriture. 
- Ce faisant, il introduit inévitablement des variations : il modernise l'orthographe (selon les conventions de son époque et de sa région), 
- il corrige ce qui lui semble être des erreurs, 
- il développe ou contracte des abréviations selon ses habitudes, 
- il saute des lignes par inadvertance (*saut du même au même* : quand deux passages commencent par les mêmes mots, le copiste peut passer directement du premier au second), 
- il insère des gloses (commentaires) dans le texte lui-même.

Résultat : **il n'existe jamais deux manuscrits identiques** d'un même texte. 
La **variabilité est la norme**, pas l'exception. C'est précisément l'objet de la philologie traditionnelle que de reconstituer, à partir de tous ces témoins divergents, quelque chose d'approchant le texte « original » : travail appelé l'**édition critique**.

> **Implication pour le ML** : il n'existe pas de « bonne » transcription unique d'un manuscrit médiéval, mais des transcriptions plus ou moins fidèles à ce qu'on voit sur la page, et des éditions critiques qui reflètent des choix éditoriaux. Les données d'entraînement et les métriques d'évaluation doivent tenir compte de cette ambiguïté fondamentale.

---
## 3. Les systèmes d'écriture médiévaux

notes:
La **paléographie** est la discipline qui étudie les écritures anciennes : leur **morphologie**, leur **évolution** historique, leurs **variantes** régionales, leurs **systèmes** d'abréviation. 
C'est une science qui s'apprend en plusieurs années, par la pratique intensive sur des documents originaux.

Voici un panorama des principaux systèmes d'écriture que vous rencontrerez dans les manuscrits français du IXe au XVe siècle.

--
<!-- slide template="[[plt-three-col]]" -->

::: title
### Origine Dynastie Carolingienne

:::

::: col1 

![[Cours/Computer Vision/Michel/CoursAnalysePhilologique/metadata/Cours_1_1_Lecture_Manuscrit_Philologie-1778608229543.png]]
:::
::: col2

![[Cours/Computer Vision/Michel/CoursAnalysePhilologique/metadata/Cours_1_1_Lecture_Manuscrit_Philologie-1778608330150.png]]
:::

::: col3

![[Cours/Computer Vision/Michel/CoursAnalysePhilologique/metadata/Cours_1_1_Lecture_Manuscrit_Philologie-1778608882713.png]]
:::

notes:
- Couronnement de pépin le pref - premier roi de la dynastie carolingienne
- Couronnement de Charlemagne
- Exemple Écriture Mérovingienne (que va remplacer la minuscule caroline)
--
<!-- slide template="[[tpl-img-left-sm]]" -->

::: title
### 3.1 L'écriture caroline (IXe–XIIe siècle)

:::

::: image

![[Cours/Computer Vision/Michel/CoursAnalysePhilologique/metadata/Cours_1_1_Lecture_Manuscrit_Philologie-1778608986023.png]]

:::

- **Minuscule carolingienne** – Grande réforme graphique sous Charlemagne (fin VIIIe–IXe)
	- Standardisation pour faciliter la lecture et l'administration
- **Caractéristiques** :
	- Lettres séparées (peu de ligatures)
	- Formes régulières et arrondies (a à deux compartiments, d droit, g caractéristique)
	- Hastes bien marquées
	- Relativement lisible pour un moderne


notes: 

> **Pour l'HTR** : la caroline est plus facile que la gothique. Mais les datasets de cette période (IXe–XIe) sont rares – moins de manuscrits survivants.


Montrer des exemples visuels (images fournies dans la base). Insister sur la rupture stylistique par rapport aux écritures mérovingiennes antérieures. La caroline est la base de notre minuscule moderne.

La **minuscule carolingienne** est la grande réforme graphique de l'époque carolingienne (fin VIIIe–IXe siècle). Initiée dans l'entourage de Charlemagne, elle cherche à standardiser l'écriture à l'échelle de l'empire pour faciliter la lecture et la communication administrative.

Ses caractéristiques :
- Lettres **séparées** les unes des autres (peu de ligatures, c'est-à-dire de connexions entre les lettres).
- Formes **régulières et arrondies** : le *a* est formé de deux compartiments, le *d* est droit, le *g* est caractéristique.
- Haste des lettres longues bien marquées (*b*, *d*, *h*, *l*, *p*, *q*).
- **Relativement lisible** pour un lecteur moderne : c'est l'écriture médiévale la plus accessible.

Pour les modèles d'HTR, la caroline est généralement plus facile que les écritures gothiques : les formes sont moins ambiguës, les ligatures moins nombreuses. Les datasets d'écriture caroline (IXe–XIe siècle) sont cependant rares, car cette période a produit moins de manuscrits survivants.


--
<!-- slide template="[[tpl-img-left-md]]" -->

::: title
### 3.2 La gothique textualis (XIIIe–XVe siècle)

:::

::: image
![[Cours/Computer Vision/Michel/CoursAnalysePhilologique/metadata/Cours_1_1_Lecture_Manuscrit_Philologie-1778609792181.png|600]]
:::

- **Gothique textualis** (_littera textualis, textura, black letter_)
	- Domine aux XIIIe–XVe siècles, rupture avec la caroline
- **Caractéristiques** :
	- Formes anguleuses, brisées, _fracture_ de la lettre
	- Condensation : lettres plus étroites et hautes – économie de parchemin
	- Ligatures systématiques (_de, do, be, bo, pe, po_ – _bowing_)
- **Problème des minimes** : _m, n, u, i_ = jambages identiques. _minimum_ = 18 jambages indistinguables. Lecture entièrement contextuelle.


:::

notes: 
L'écriture **gothique** (appelée aussi *littera textualis*, *textura*, ou : à tort mais couramment : *black letter*) se développe progressivement à partir du XIIe siècle et domine aux XIIIe–XVe siècles. Elle marque une rupture stylistique profonde avec la caroline.

Ses caractéristiques :
- Formes **anguleuses et brisées** : les courbes de la caroline sont remplacées par des traits droits, des angles et des losanges. On parle de *fracture* de la lettre.
- **Condensation** de l'écriture : les lettres sont plus étroites et plus hautes, permettant d'écrire plus de texte sur moins de parchemin : une économie délibérée quand le support coûte cher.
- **Ligatures systématiques** : les lettres se connectent entre elles selon des règles précises. La suite *de*, *do*, *be*, *bo*, *pe*, *po* forme une ligature caractéristique appelée *bowing* où les deux lettres partagent un trait commun.
- **Ambiguïtés graphiques majeures** : le segment vertical (jambage) est l'unité de base de toutes les lettres. Un *m* est trois jambages, un *n* est deux, un *u* est deux, un *i* est un. Hors contexte, *minim*, *mnim*, *nnm*, *nim* sont graphiquement indistinguables. Ce problème, connu des paléographes sous le nom de **problème des minimes**, est l'une des difficultés techniques les plus redoutables pour les modèles.

> **Exemple** : le mot *minimum* en gothique textualis est une suite de dix-huit jambages parfaitement identiques. Sa lecture est entièrement contextuelle : elle dépend de la connaissance du latin et du vocabulaire probable dans le passage concerné.

La gothique textualis existe elle-même en plusieurs variantes : la *textualis formata* (ou *quadrata*) est la plus soignée, utilisée pour les manuscrits de prestige (bibles, livres d'heures) ; la *textualis semi-quadrata* et la *textualis rotunda* (plus arrondie, développée en Italie) en sont des déclinaisons.

- exemple gothique textualis -> Miracles de sainte Catherine de Fierbois, datant du 15ème siècle
- Exemple gothique rontunda -> Somme Théologique Saint Thomas D'Aquin 14e
- Exemple Gothique Bâtarde (2criture d'origine bourgignone) -> Vie et Miracle de Notre Dame, Fait pour le duc de Bourgogne Philippe III (15e)

--
<!-- slide template="[[plt-two-col]]" -->

::: title

### 3.2.b Variantes - Gothique Rotunda et Gothique Bâtarde

:::

::: left
![[Slides_01_Lecture_Manuscrit_Philologie-1779017919306.png|500]]
:::
::: right
![[Slides_01_Lecture_Manuscrit_Philologie-1779018014893.png]]
:::

notes:
**Besoin d'écriture rapide** (administration, commerce) → les lettres se connectent, la plume ne se lève presque plus.

**Familles cursives françaises** :

- **Cursive gothique française** : actes royaux, registres. Hastes très allongées vers la gauche.
    
- **Bâtarde** (_littera bastarda_) : hybride textualis/cursive. Manuscrits de luxe en vernaculaire (cour de Bourgogne).
    
- **Lettre de forme** : variante soignée pour chartes solennelles.
    

> **Défi pour les modèles** : segmentation en caractères souvent impossible. Les formes varient énormément d'un scripteur à l'autre selon la vitesse. D'où le choix du niveau de la **ligne** entière.

Comparer avec les écritures cursives modernes. Insister sur la variabilité inter‑scripteur – c'est le pire cas pour la généralisation. Les modèles actuels (TrOCR, Kraken) traitent des lignes entières.

À partir du XIVe siècle, la croissance de l'administration royale et des échanges commerciaux crée un besoin d'écriture rapide. Les écritures **cursives** se développent pour répondre à ce besoin : les lettres se connectent de manière fluide, le calame (puis la plume) ne se lève presque plus du parchemin.

Les grandes familles cursives françaises :
- **La cursive gothique française** : écriture des actes royaux et des registres administratifs. Très rapide, très liée, avec des *hastes* (traits verticaux supérieurs) très allongés vers la gauche, formant parfois des enchevêtrements en tête de ligne.
- **La bâtarde** (*littera bastarda*) : écriture hybride entre la textualis et la cursive, développée au XVe siècle en France et dans les Pays-Bas bourguignons. Elle est utilisée pour les manuscrits de luxe en langue vernaculaire : romans, chroniques. C'est l'écriture des grands manuscrits de la cour de Bourgogne.
- **La lettre de forme** : variante soignée utilisée pour les chartes solennelles.

Les cursives posent des problèmes radicalement différents de la textualis pour les modèles : les lettres sont rarement isolables, les formes individuelles des caractères varient énormément d'un scripteur à l'autre et selon la vitesse d'écriture. La segmentation en caractères est souvent impossible : c'est pourquoi l'HTR moderne travaille au niveau de la **ligne** entière, et non au niveau du caractère.

--
### 3.4 Les systèmes d'écriture après Gutenberg (XVe–XVIIIe siècle)

notes:
- **Avant 1450** : tout texte est manuscrit.
- **Après 1450** : l’écriture manuscrite devient un acte **personnel** (lettres, notes, brouillons) ou **administratif** (registres, actes notariés, livres de comptes).
- Les textes imprimés diffusent massivement, mais les manuscrits continuent d’exister, surtout dans les sphères privées, juridiques et religieuses.
- **Conséquence pour l’HTR** : les manuscrits de cette période sont souvent plus cursifs, plus rapides, avec des abréviations moins systématiques. La variabilité inter‑scripteur augmente.

--
#### 2.1 La cursive humanistique  (XVe–XVIe)
- Née en Italie, inspirée de la minuscule caroline (redécouverte par les humanistes).
- **Caractéristiques** : arrondie, aérée, lettres détachées ou peu liées.
- **Usage** : copie de textes classiques, correspondance savante.
- **Évolution** : donne naissance à l’**écriture italique** (penchée, rapide) qui inspirera les polices d’imprimerie.
![[Slides_01_Lecture_Manuscrit_Philologie-1779006601367.png|650]]<!-- element style="display: block; margin-left: auto; margin-right: auto;" -->
notes:
aussi appelée _chancellerie_
Léonard de vinci utilisait la minuscule humaniste cursive

--
#### 2.2 La « bâtarde » française (XVe–XVIe)
- Déjà présente au XVe siècle, elle persiste.
- Hybride entre gothique et cursive.
- **Usage** : manuscrits de luxe en langue vernaculaire, documents judiciaires.
- Déclin progressif au XVIe au profit de la cursive humanistique.
![[Slides_01_Lecture_Manuscrit_Philologie-1779006728043.png|600]]<!-- element style="display: block; margin-left: auto; margin-right: auto;" -->   

notes:
L’écriture bâtarde est d’usage courant durant tout le 15e siècle principalement pour les textes en français, romans, chroniques royales ou princières. Elle est caractérisée par un tracé lourd, qui oppose de manière contrastée les traits fins et épais, et enjolive les extrémités des lettres (boucles aux hastes montantes des _b_, _h_, _l_ ; traits de fuite pour les _m_ et _n_), et par ses _f_ et ses _s_ descendants au-dessous de la ligne. Elle est particulièrement remarquable dans les manuscrits bourguignons.

L’écriture bâtarde sera la base des écritures cursives des 17e et 18e siècles.

--
<!-- slide template="[[tpl-img-left-md]]" -->

::: title
#### 2.3 La « _cursive financière_ » (XVIe–XVIIIe)

:::
- **Caractéristiques** : très rapide, très liée, avec de nombreuses boucles et ligatures.
- **Usage** : actes notariés, registres d’état civil, correspondance administrative.

::: image
![[Slides_01_Lecture_Manuscrit_Philologie-1779007833370.png|475]]
:::

notes:
 - très utilisé par l'administration notamment sous Colbert
 - provient de la cursive gothique
--
#### 2.4 L’écriture « ronde » (XVIIe–XVIIIe)
- **Caractéristiques** : lettres rondes, régulières, verticales, peu inclinées.
- **Usage** : livres de comptes, registres commerciaux, manuscrits soignés.
- **Avantage** : plus lisible que la courante, plus facile à transcrire automatiquement.

![[Slides_01_Lecture_Manuscrit_Philologie-1779008381944.png|700]]

--
#### 2.5 L’écriture « anglaise » (fin XVIIIe)
- **Caractéristiques** : très inclinée, fines pleins et déliés, influences de l’écriture anglaise.
- **Usage** : correspondance privée, documentation commerciale.
- Annonce les écritures du XIXe siècle.
![[Slides_01_Lecture_Manuscrit_Philologie-1779008455840.png|650]]<!-- element style="display: block; margin-left: auto; margin-right: auto;" -->
notes:  
La période 1450-1800 est très riche. Pour le projet, on croise surtout de la cursive humanistique, de la courante et de la ronde. Les modèles HTR pré‑entraînés sur du XIXe siècle (ex: IAM) sont parfois décalés car les écritures médiévales et de la Renaissance sont différentes.

--
### 3.5 Les grandes familles d'écriture : tableau de synthèse

| Époque            | Écriture dominante                 | Lisibilité       | Défi principal                                  |
| ----------------- | ---------------------------------- | ---------------- | ----------------------------------------------- |
| IXe–XIe siècle    | Caroline                           | Élevée           | Rareté des datasets                             |
| XIIe siècle       | Transition caroline → pré-gothique | Moyenne          | Formes hybrides                                 |
| XIIIe–XIVe siècle | Gothique textualis                 | Faible à moyenne | Problème des minimes, ligatures                 |
| XIVe–XVe siècle   | Cursive gothique, bâtarde          | Faible           | Ligatures, variabilité inter-scripteur          |
| XVe–XVIe          | Cursive humanistique               | Bonne            | Lettres détachées mais inclinaison variable     |
| XVe–XVIe          | Bâtarde                            | Moyenne          | Encore des formes gothiques (s long, ligatures) |
| XVIe–XVIIIe       | Courante (procédure)               | Faible           | Liures serrées, boucles, absence de séparation  |
| XVIIe–XVIIIe      | Ronde                              | Bonne            | Peu de défis, forme régulière                   |
| fin XVIIIe        | Anglaise                           | Moyenne          | Inclinaison forte, déliés fins                  |


---
## 4. L'ancien et le moyen français : une langue sans norme

--
<!-- slide template="[[tpl-img-left-md]]" -->
::: title
### 4.1 Que sont le *français ancien* et le *moyen français* ?

:::

- **Latin** : langue de l'Église, université, diplomatie – très présent, souvent mêlé au vernaculaire.
- **Ancien français** (842–1340) : ensemble des variétés de la langue d'oïl.  
- **Moyen français** (1340–1550) : disparition de la déclinaison à deux cas, latinisation du vocabulaire.
- ==Absence d'orthographe normalisée== (avant XVIe–XVIIe)
	- Un même mot peut s'écrire de dizaines de façons : _roi, rey, rei_
	- L'orthographe reflète la phonétique locale.
	- Latinisations étymologiques : _doigt_ (de _digitus_) au lieu de _doi_


::: image 
![[Slides_01_Lecture_Manuscrit_Philologie-1778964408739.png|450]]

:::

notes: 
C'est le point le plus déstabilisant pour des informaticiens. Insister sur le fait que nos métriques standard (CER) comparent à une référence arbitraire. À utiliser avec précaution.

- **Le latin** reste la langue de l'Église, de l'université, de la diplomatie internationale et d'une grande partie de la production écrite savante tout au long du Moyen Âge. De nombreux manuscrits mêlent le latin et le vernaculaire dans la même page, voire dans la même phrase.
- **L'ancien français** (environ 842–1340) désigne l'ensemble des variétés de la langue d'oïl parlées et écrites dans le nord de la France médiévale. Le terme « langue d'oïl » signifie « langue où oui se dit *oïl* » (par opposition à la langue d'oc du sud, où oui se dit *oc*).
	- 842 première trace écrite en langue romane (français), et en langue tudesque (allemand) avec les serments de Strasbourg premier texte avéré écrit en langue romane - proto-Français. Aujourd'hui à la BNF
		- premier écrit qui ne peut plus être identifié comme du latin.
		- Serment qui scelle une alliance entre 2 petits fils de Charlemagne
			- Charles le Chauve - Charles II
			- Et Louis le germanique
		- Serment écrit en langue vulgaire pour que toutes les troupes comprennent.
		- difficilement compréhensible par un lecteur moderne, car syntaxe et sémantique ont évolué au fil du temps
- **Le moyen français** (environ 1340–1550) désigne la période intermédiaire entre l'ancien français et le français classique. La langue change rapidement : la déclinaison à deux cas disparaît (vers le XIVe siècle), le vocabulaire se latinise massivement, les constructions syntaxiques évoluent.
	- relativement compréhensible, vocabulaire exotique, mais la structure de la phrase est assez semblable.
	-  La transition se définit par l'**effondrement du système de déclinaison** hérité du latin, la disparition progressive de la distinction cas sujet/cas régime, et la simplification de la phonétique (comme l'affaiblissement du [ə] atone).

--
<!-- slide template= "[[plt-two-col]]"-->

::: title
### Ancien Français 
==_Yvain ou le Chevalier au Lion._==
:::

::: left 
**Ancien Français**


Artus, li boens rois de Bretaingne,  
La cui proesce nos enseigne  
Que nos soiens preu et cortois,  
Tint cort si riche come rois  
A cele feste qui tant coste,  
Qu'an doit clamer la Pantecoste.  
Li rois fu a Carduel en Gales ;  
Aprés mangier, parmi ces sales,  
Cil chevalier s'atropelerent  
La ou dames les apelerent  
Ou dameiseles ou puceles.
:::
::: right
**Français contemporain**


Arthur, le bon roi de Bretagne,  
Dont la vaillance nous enseigne  
À être preux et courtois,  
Tant noble et riche comme roi,  
À cette fête qui coûte si cher,  
Qu'on doit célébrer la Pentecôte.  
Le roi était à Carduel, en Galles ;  
Après le repas, au milieu de ses salles,  
Les chevaliers s'assemblèrent  
Là où les dames les appelaient,  
Ou les damoiselles ou les jeunes filles.

:::

--
<!-- slide template="[[tpl-img-left-sm]]" -->
::: title
### 4.2 La variation dialectale
L'ancien français : un continuum de dialectes, chacun avec ses spécificités graphiques.
:::


::: image
![[Slides_01_Lecture_Manuscrit_Philologie-1779008874630.png]]
:::


**Principaux dialectes littéraires** :
- **Francien** (Île-de-France) → base du français moderne, mais sans prestige au Moyen Âge
- **Normand** : administration anglo‑normande après 1066
- **Picard** : *c* pour _ch_, _ki_ pour _qui_
- **Anglo‑normand** : français parlé en Angleterre
- Champenois, orléanais, bourguignon, lorrain…    

> **Pour les modèles de langue** : un modèle entraîné sur du francien sera moins performant sur du picard. Segmenter le corpus par dialecte.

notes: 
Si on utilise un modèle de langue en post‑traitement, il faut soit disposer de modèles spécialisés, soit accepter une dégradation des performances. Projets comme CREMMA (ANR) construisent des ressources pour l'ancien français.

L'ancien français n'est pas une langue unique mais un continuum de dialectes. Les principaux dialectes littéraires sont :

- **Le francien** (Île-de-France) : qui finira par devenir la base du français moderne, mais qui n'a pas au Moyen Âge de statut de prestige supérieur aux autres dialectes.
- **Le normand** : important dialecte de la Normandie, langue de l'administration anglo-normande après 1066, présent dans une masse considérable de textes juridiques et littéraires.
- **Le picard** : dialecte du nord de la France et de la Belgique actuelle, doté d'une remarquable tradition littéraire (trouvères, farces). Caractéristiques graphiques distinctives : *c* pour *ch*, *ki* pour *qui*, etc.
- **L'anglo-normand** : variété du français parlée et écrite en Angleterre après la Conquête normande. Nombreux documents juridiques, traités, textes religieux.
- **Le champenois**, l'**orléanais**, le **bourguignon**, le **lorrain**… chacun avec ses spécificités graphiques.

Pour les modèles de langue utilisés en post-traitement HTR, ces dialectes posent un problème sérieux : les modèles de langage entraînés sur le français moderne (ou même sur le français médiéval de l'Île-de-France) seront moins performants sur un texte picard, où les probabilités de séquences de caractères sont différentes.

Pour résumer rapidement la situation linguistique, on peut dire que les habitants de la France parlaient, selon les régions:

- diverses variétés de langues d'oïl: françois, picard, gallo, poitevin, saintongeais, normand, morvandiau, champenois, etc.

- diverses variétés des langues d'oc (gascon, languedocien, provençal, auvergnat-limousin, alpin-dauphinois, etc.) ainsi que le catalan;

- diverses variétés du franco-provençal: bressan, savoyard, dauphinois, lyonnais, forézien, chablais, etc., mais aussi, en Suisse, genevois, vaudois, neuchâtelois, valaisan, fribourgeois et, en Italie, le valdôtain.

- des langues germaniques: francique, flamand, alsacien, etc.

- le breton ou le basque.

--
### 4.4 Les abréviations : un système à part entière
**Phénomène massif** – système codifié, appris, pour écrire plus vite et économiser le parchemin.

**Types** :
- **Suspension** : début du mot + signe de coupure (barre, point, lettre exposante). Ex: _fris_ surmonté d'une barre = _francorum_
- **Contraction** : début + fin, milieu omis, signalé. Ex: _dns_ = _dominus_
- **Signes spéciaux** : tilde (vague au‑dessus d'une lettre) → nasale omise (_hõe_ = _home_)
- **Ligatures** : _&_ (et), _æ_

notes: 
Distinguer clairement. Beaucoup de projets HTR se perdent en essayant de développer les abréviations dans le modèle de vision. Mieux vaut produire une transcription diplomatique et confier le développement à une étape séparée (règles ou petit modèle).
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
## 5. Obstacles à la lecture

- **Caractère** : minimes (m/n/u/i), ambiguïté u/v, i/j, confusion c/t, s long/f, r/n.
- **Mot** : frontières de mots floues (scriptio continua), formes lexicographiques inattendues.
- **Ligne** : ratures, corrections supralinéaires, coupures de mots non standardisées, justification forcée.
- **Page** : structure (colonnes, marges, lettrines), ordre de lecture ambigu, dégradations physiques (taches, transparence, déchirures, gouttière, dorures saturantes).

> [!tldr] **Conséquence ML** : un pipeline HTR doit intégrer une segmentation de layout robuste, gérer l'incertitude sur l'ordre de lecture, et ne pas supposer une propreté parfaite.

*Pour plus de détail, référez-vous au support de cours.*

---
## 6. Ce que la tradition philologique apporte à la vision par ordinateur

--
<!-- slide template="[[tpl-top-content-2col]]" -->

::: title

### 6.1 Deux philosophies de transcription
:::

::: top
La philologie a développé des conventions précises pour répondre à une question que tout modèle HTR doit également trancher : **que transcrit-on exactement ?**

:::

::: left
###### **Transcription diplomatique**
- Reproduit le texte ==au plus près de l’original==, sans développer les abréviations, sans corriger les erreurs apparentes du scribe, sans moderniser l'orthographe. 
- Essentielle pour les **travaux de philologie**, **paléographie** où chaque détail peut avoir une **valeur sémantique** ou historique.

:::

::: right
###### **Transcription normalisée**
- Restitue le texte idéal. les abréviations sont développées,  l'orthographe est standardisée selon des conventions décidées à l'avance, corrige les erreurs manifestes du scribe. Elle est plus lisible mais plus interprétative.
- Ce type de transcription **facilite la lecture, la recherche** et l’analyse linguistique, mais au prix d’une **perte d’information** paléographique.  
::: 

notes:
**Diplomatique**
- Par exemple, une rature peut indiquer un doute du scripteur, une correction ou une censure.

- Normalisée - (ou *transcription critique*) 
- Le choix du niveau de normalisation (partielle ou complète) dépend des objectifs du projet et du public cible.
→ Dans ce projet : **transcription diplomatique** en sortie du pipeline CV,  
**transcription normalisée** en sortie du pipeline NLP.

Pour notre projet, la transcription diplomatique est la cible prioritaire : c'est celle qui est objectivement évaluable (on peut calculer un CER contre une référence sans débat interprétatif sur le développement des abréviations). La normalisation sera le travail du module NLP.

--
<!-- slide template="[[plt-two-col]]" -->

::: title
###  6.2 Normes de transcription existantes

:::

::: left
- La **Text Encoding Initiative** (TEI) **P5** est un **projet universitaire pluridisciplinaire** visant à uniformiser autant que possible le codage de documents.  La TEI utilise un format XML, et propose des balises précises pour représenter les particularités des manuscrits (ratures, abréviations, notes marginales) 

:::

::: right
 - **CREMMA** (Consortium pour la Reconnaissance des Écritures Manuscrites des Matériaux Anciens) : Produit des corpus de vérité terrain pour l’HTR. Ses **règles de transcription** privilégient une approche **graphématique** (normalisation des graphies comme ſ/s, u/v), conservent les abréviations, et codent les ratures 
:::

--

![[Cours/Computer Vision/Michel/CoursAnalysePhilologique/metadata/Cours_1_1_Lecture_Manuscrit_Philologie-1778750573238.png]]

--

![[Slides_01_Lecture_Manuscrit_Philologie-1779019884621.png|650]]
notes:
```xml
<div xml:id="n26" type="chapitre" n="23">
 <fw place="top-left" type="pageNum">164</fw>
 <head>
  <lb/>Comment <persName>Panurge</persName> faict discours
 <lb/>pour retourner a <persName>Raminagro-
  <lb rend="hyphen"/>bis</persName>.<space unit="cm" quantity="1.5"/>Chap. 23.
 </head>
 <p>
  <lb/>
  <hi rend="larger">R</hi>Etournons (dist <persName>Panurge</persName> continuant)
 <lb/>l'admonester de son salut. Allons on
 <lb/>nom, allons en la vertus Dieu. Ce
 <lb/>sera oeuvre charitable a nous faicte. Au
 <lb/>moins s'il perd le corps &amp; la vie, qu'il
 <lb/>ne damne son <sic>asne</sic>. Nous le induirons
 <lb/>a contrition de son peché: a requerir par- 
 <lb rend="hyphen"/>don es dictz tant beatz peres absens com-
 <lb rend="hyphen"/>me praesens. Et en prendrons acte, affin
 <lb/>qu'apres son trespas ilz ne le declairent
 <lb/>hereticque &amp; damné comme les Farfa-
 <lb rend="hyphen"/>detz feirent de la praevoste d'<placeName type="ville">Orleans</placeName>: &amp;
 <lb/>leurs satisfaire de l'oultraige, ordon-
 <lb rend="hyphen"/>nant par tous les couvens de ceste pro-
 <lb rend="hyphen"/>vince aux bons peres religieulx force bri-
 <lb rend="hyphen"/>bes, force messes, force obitz &amp; anni-
 <lb rend="hyphen"/>versaires. Et que au jour de son trespas
 <lb/>sempiternellement ilz ayent tous quin-
 <lb rend="hyphen"/>tuple pitance: &amp; que le grand bourra-
 </p>
</div>
```


- Et avec tout ça, on est toujours pas sûr que le label soit correct, il ya  toujours de la variation inter-annotateur.
---
## 7. Traduire les contraintes philologiques en spécifications techniques

--

| Contrainte philologique                                   | Choix technique impliqué                                                                                           |
| --------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------ |
| Variabilité de mise en page (colonnes, rubriques, marges) | Segmentation de layout avant l'HTR : deux étapes distinctes                                                        |
| Problème des minimes (ambiguïté graphique locale)         | HTR au niveau de la ligne entière, pas du caractère : modèles séquentiels (TrOCR, Kraken)                          |
| Abréviations                                              | Choisir entre transcription diplomatique (reproduire le signe) et normalisée (développer) : documenter le choix    |
| Absence de norme orthographique                           | Métriques permissives ; évaluation contre plusieurs références ; pas de correction orthographique à ce stade       |
| Variation dialectale                                      | Segmentation du corpus par dialecte si possible ; sinon, accepter une dégradation des performances                 |
| Mélange latin / vieux français                            | Détection de langue par région ; modèles séparés pour le latin et le vernaculaire                                  |
| Dégradations physiques                                    | Prétraitement robuste (binarisation adaptative, correction de transparence) ; augmentation de données synthétique  |
| Annotations divergentes                                   | Annotations multiples ; mécanismes de vote ; encodage de la confiance dans le format de sortie                     |
| Ordre de lecture ambigu                                   | Règles de layout explicites ; flag d'incertitude quand la règle ne s'applique pas                                  |
| Illustrations et enluminures                              | Segmentation distincte du texte et de l'image ; description automatique (CLIP/LLaVA) pour les zones non-textuelles |

---
## Bibliographie de référence

--
#### Paléographie et codicologie

- **Muzerelle, D.** (1985). *Vocabulaire codicologique : répertoire méthodique des termes français relatifs aux manuscrits*. CEMI, Paris. : La référence terminologique fondamentale en français. Version électronique disponible : [codicologia.irht.cnrs.fr](http://codicologia.irht.cnrs.fr)

- **Derolez, A.** (2003). *The Palaeography of Gothic Manuscript Books: From the Twelfth to the Early Sixteenth Century*. Cambridge University Press. : La somme de référence sur les écritures gothiques, avec une description systématique et illustrée de chaque famille.

- **Stiennon, J.** (1973). *Paléographie du Moyen Âge*. Armand Colin, Paris. : Introduction classique en langue française, accessible aux non-spécialistes.

- **Bischoff, B.** (1990). *Latin Palaeography: Antiquity and the Middle Ages*. Cambridge University Press. (Traduction de l'allemand *Paläographie des römischen Altertums und des abendländischen Mittelalters*, 1979.) : Référence pour la période ancienne (carolingienne et antérieure).

- **Parkes, M. B.** (1992). *Pause and Effect: An Introduction to the History of Punctuation in the West*. Scolar Press, Aldershot. : Sur la ponctuation médiévale, essentielle pour la segmentation en phrases.
--
#### Linguistique de l'ancien et du moyen français

- **Zink, G.** (1992). *Phonétique historique du français*. Presses Universitaires de France (coll. Linguistique nouvelle). : Introduction rigoureuse à l'évolution phonétique du français médiéval.

- **Moignet, G.** (1988). *Grammaire de l'ancien français*. Klincksieck, Paris. : Grammaire de référence pour l'ancien français, morphologie et syntaxe.

- **Marchello-Nizia, C.** (1997). *La langue française aux XIVe et XVe siècles*. Armand Colin. : Référence pour le moyen français.

- **Buridant, C.** (2000). *Grammaire nouvelle de l'ancien français*. SEDES, Paris. : Grammaire complète et actualisée.

- **Trésor de la langue française informatisé (TLFi)** : Base de données lexicographique qui couvre le français du IXe au XXe siècle. Accessible en ligne : [atilf.atilf.fr/tlfi.htm](http://atilf.atilf.fr/tlfi.htm)

- **Dictionnaire du Moyen Français (DMF)**, ATILF/CNRS. : Dictionnaire spécialisé pour le moyen français (1330–1500). En ligne : [atilf.atilf.fr/dmf/](http://atilf.atilf.fr/dmf/)
--
#### Philologie et humanités numériques

- **Burnard, L., Sperberg-McQueen, C. M.** (1994, mis à jour régulièrement). *TEI P5: Guidelines for Electronic Text Encoding and Interchange*. Text Encoding Initiative Consortium. : La spécification de référence du format TEI. En ligne : [tei-c.org/guidelines/](https://tei-c.org/guidelines/)

- **Stutzmann, D.** (2011). *Outils d'analyse de la dynamique des écritures médiévales par ordinateur*. Document numérique, 14(1), 81–107. Lavoisier. : Article fondateur sur l'application de la vision par ordinateur à la paléographie médiévale en France. À lire en priorité.

- **Camps, J.-B., Clérice, T., Pinche, A.** (2021). *Evaluating Data Augmentation for Historical Handwritten Text Recognition*. Document and Image Recognition : Proceedings of ICDAR 2021. : Sur les stratégies d'augmentation de données pour l'HTR médiéval.

- **Terras, M.** (2006). *Image to Interpretation: An Intelligent System to Aid Historians in Reading the Vindolanda Texts*. Oxford Studies in Ancient Documents. Oxford University Press. : Étude de cas pionnière sur la reconnaissance automatique d'écriture ancienne.
--
#### Projets de recherche et ressources en ligne

- **IRHT (Institut de Recherche et d'Histoire des Textes)**, CNRS. Carnet de recherche *Écriture médiévale & numérique* : [irht.hypotheses.org](https://irht.hypotheses.org) : Veille sur l'application de l'informatique aux manuscrits médiévaux, par les spécialistes français de référence.

- **eScriptorium** (EPHE / INRIA) : plateforme collaborative d'annotation et d'entraînement de modèles HTR, développée spécifiquement pour les documents patrimoniaux. [escriptorium.fr](https://escriptorium.fr)

- **HTR-United** : initiative collaborative pour le partage de modèles et de données HTR pour les manuscrits anciens. [htrunited.github.io](https://htrunited.github.io)

- **Projet CREMMA** (Consortium pour la Reconnaissance et l'Édition de Manuscrits Médiévaux Anciens), ANR : corpus de référence pour l'HTR en vieux et moyen français. Direction : Ariane Pinche (EPHE), Jean-Baptiste Camps (École nationale des chartes).

- **Gallica**, Bibliothèque nationale de France : [gallica.bnf.fr](https://gallica.bnf.fr) : Plus de 7 millions de documents numérisés, API IIIF, notice descriptive de chaque manuscrit.

- **e-Codices** (Bibliothèque virtuelle des manuscrits médiévaux de Suisse) : [e-codices.unifr.ch](https://www.e-codices.unifr.ch) : Numérisation intégrale de haute qualité, métadonnées riches.
