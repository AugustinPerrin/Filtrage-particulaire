# Filtrage particulaire

TP 2 du cours de filtrage bayésien consacré à la simulation de lois aléatoires et à leur utilisation pour l'estimation d'un état par filtrage particulaire.

## Objectifs

Un filtre bayésien estime l'état caché d'un système à partir de son modèle d'évolution et des observations disponibles. Le filtre particulaire représente la loi de l'état par un ensemble de particules pondérées :

- **Prédiction** : chaque particule est propagée selon le modèle de mouvement et le contrôle mesuré, avec ajout de bruit.
- **Mise à jour** : le poids de chaque particule est calculé à partir de la vraisemblance de l'observation associée à cette particule.
- **Estimation** : la moyenne pondérée des particules fournit une estimation de l'état.
- **Rééchantillonnage** : les particules sont tirées à nouveau en fonction de leurs poids afin de concentrer l'ensemble sur les états les plus plausibles.

La simulation des lois aléatoires est au cœur de cette approche. Le **sampling** consiste à tirer directement des réalisations lorsque le générateur de la loi est disponible. La méthode d'**acceptation-rejet** permet, quant à elle, de simuler une loi cible en tirant des candidats selon une loi facile à échantillonner, puis en les acceptant avec une probabilité adaptée. Dans le script fourni, les bruits gaussiens sont échantillonnés directement avec `numpy.random.randn` ; l'acceptation-rejet est un thème du TP, mais n'est pas implémentée comme algorithme séparé dans ce script.

## Démonstration : localisation d'un robot

[`ParticleFilter_Localization.py`](./ParticleFilter_Localization.py) simule un robot mobile se localisant par rapport à des amers connus. L'état du robot est sa pose `(x, y, theta)`. À chaque pas, le programme :

1. simule le mouvement réel du robot et une odométrie bruitée ;
2. propage un nuage de 1 000 particules avec le modèle de mouvement ;
3. génère une observation bruitée de la distance et de l'angle vers un amer ;
4. pondère les particules selon un modèle de bruit gaussien ;
5. calcule la pose estimée et rééchantillonne les particules avec une méthode à faible variance.

La visualisation compare la trajectoire réelle, l'odométrie et l'estimation du filtre. Des graphiques présentent également les erreurs sur les trois composantes de la pose et les écarts-types estimés par les particules. À la fin de l'exécution, le programme affiche l'erreur moyenne et sa variance, puis enregistre la figure dans `ParticleFilter_Localization.png`.

## Prérequis

- Python 3
- NumPy
- Matplotlib

Installation des dépendances si nécessaire :

```bash
python -m pip install numpy matplotlib
```

## Exécution

Depuis le répertoire du projet :

```bash
python ParticleFilter_Localization.py
```

Une fenêtre Matplotlib affiche la simulation. La figure est aussi enregistrée dans le répertoire courant. Le script utilise des tirages aléatoires ; les résultats peuvent donc varier d'une exécution à l'autre.

## Paramètres de simulation

Les principaux paramètres sont définis dans le script :

- `nParticles` : nombre de particules (1 000 par défaut) ;
- `nSteps` : nombre de pas de simulation (1 000 par défaut) ;
- `Map` : positions aléatoires des 20 amers ;
- `QTrue` et `PYTrue` : covariances des bruits de mouvement et d'observation simulés ;
- `QEst` et `PYEst` : covariances supposées par le filtre.



