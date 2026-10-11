import numpy as np
import numpy.random as rd
import matplotlib.pyplot as plt
from statistics import NormalDist

def partie_1():
    """
    Fonction répondant aux questions de la partie 1
    """
    # Variance théorique
    mu = 2/np.pi
    V = 0.5 - mu**2
    V_n = lambda n: V/n

    # estimation empirique de la variance
    print(f"Variance V_50: {V_n(50)}")
    X = np.linspace(1,50,50)
    Y = [V_n(x) for x in X]

    # vérification numérique du TCL
    g = lambda x: np.cos((np.pi * x)/2)
    mu_n = lambda n: np.sum(g(rd.uniform(0, 1, n)))*(1/n)
    realisation_mu_n = [mu_n(i) for i in range(1,51)]
    print(
        f"mu_n = {realisation_mu_n}; mu = {mu}, "
        f"erreur = {np.abs(np.array(realisation_mu_n) - mu)}"
    )

    # Vérification du TCL pour n=50 sur plusieurs réalisations indépendantes.
    n_tcl = 50
    nombre_realisations = 1000
    estimations = np.mean(
        g(rd.uniform(0, 1, size=(nombre_realisations, n_tcl))),
        axis=1,
    )
    estimations_normalisees = (estimations - mu) / np.sqrt(V_n(n_tcl))
    probabilites = (np.arange(1, nombre_realisations + 1) - 0.5) / nombre_realisations
    loi_normale = NormalDist()
    quantiles_normaux = np.array(
        [loi_normale.inv_cdf(p) for p in probabilites]
    )
    
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(17, 5))
    ax1.plot(X, Y)
    ax1.set_title("Convergence de V")
    ax1.set_xlabel("N")
    ax1.set_ylabel("V")

    ax2.plot(range(1, 51), realisation_mu_n, label="Réalisation de mu_n")
    ax2.axhline(mu, color="red", linestyle="--", label="mu théorique")
    ax2.set_title("Réalisation de mu_n")
    ax2.set_xlabel("n")
    ax2.set_ylabel("mu_n")
    ax2.legend()

    ax3.scatter(quantiles_normaux, np.sort(estimations_normalisees), s=12, alpha=0.6)
    ax3.plot(
        [quantiles_normaux[0], quantiles_normaux[-1]],
        [quantiles_normaux[0], quantiles_normaux[-1]],
        color="red",
        linestyle="--",
        label="y = x",
    )
    ax3.set_title(f"QQ-plot de mu_n (n={n_tcl})")
    ax3.set_xlabel("Quantiles théoriques N(0, 1)")
    ax3.set_ylabel("Quantiles observés standardisés")
    ax3.legend()

    fig.tight_layout()
    plt.show()

    # # calcul de l'intégrale en choisissant une bonne densité
    # force brute
    q = rd.uniform

    # autre choix
    q = lambda x: (3/2) * (1-x**2)

    # évaluation de l'intégrale
    
if __name__ == "__main__":
    partie_1()
