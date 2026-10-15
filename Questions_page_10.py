import numpy as np
import numpy.random as rd
import matplotlib.pyplot as plt
from statistics import NormalDist
import scipy.integrate as integrate
import scipy.special as special


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

    # tirage de l'estimateur
    g = lambda x: np.cos((np.pi * x)/2)
    mu_n = lambda n: np.sum(g(rd.uniform(0, 1, n)))*(1/n)
    N = 100
    I = [-(np.sqrt(V)/np.sqrt(N))*1.96+mu,(np.sqrt(V)/np.sqrt(N))*1.96+mu]
    print(
        f"Estimateur \mu = {mu_n(N)}\n"
        f"Intervalle de confiance : P[\mu_N \in I]=0.95, I = [{I[0]};{I[1]}]"
    )

    # vérification numérique du TCL
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
    
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
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

    # Tracé des fonctions q, g*q, q_tilde
    
    # g(x) : défini au-dessus
    q = lambda x: 1 if x<=1 and x>=0 else 0
    q_tilde = lambda x: (3/2)*(1-x**2)

    X = np.linspace(0,1,50)
    Q = [q(x) for x in X]
    GQ = [q(x)*g(x) for x in X]
    Q_tilde = [q_tilde(x) for x in X]

    plt.plot(X,Q, label = "q(x)")
    plt.plot(X, GQ, label = "g(x)*q(x)")
    plt.plot(X, Q_tilde, label = "q_tilde(x)")
    plt.xlabel("x")
    plt.title("Tracé des lois de probabilité")
    plt.legend()
    plt.grid()
    plt.show()


# def methode_rejet():
#     """
#     implémentation de la méthode des rejets
#     """

#     g = lambda x: np.cos((np.pi * x)/2)
#     # Majorant de g
#     C = 1

#     # Probabilité d'acceptation
#     P = (1/C) * (2/np.pi) 
#     # Couples acceptés
#     c_a = []
#     # Couples rejetés
#     c_r = []
#     # On tire X ~ q
#     X = rd.uniform(0,1)
#     U = rd.uniform(0,1)

#     if g(X) > C*U:
#         c_a.append([X,U])
#     elif rd.uniform(0,1) < P:
#         c_a.append([X,U])
#     else:
#         c_r.append([X,U])


def methode_rejet():
    """
    Compare la méthode du rejet avec une proposition uniforme et q_tilde.
    """
    g = lambda x: np.cos((np.pi * x)/2)

    nombre_tirages = 100

    # Rejet avec une proposition uniforme sur [0, 1].
    X = rd.uniform(0, 1, nombre_tirages)
    U = rd.uniform(0, 1, nombre_tirages)
    acceptes = U <= g(X)
    rejetes = ~acceptes

    x_courbe = np.linspace(0, 1, 500)
    plt.scatter(X[acceptes], U[acceptes], color="yellow", s=12, label="Points acceptés")
    plt.scatter(X[rejetes], U[rejetes], color="red", s=12, label="Points rejetés")
    plt.plot(x_courbe, g(x_courbe), color="blue", label="g(x)")
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.xlabel("x")
    plt.ylabel("u")
    plt.title("Méthode du rejet force brute")
    plt.legend()
    plt.grid()
    plt.show()

    proba_empirique = np.mean(acceptes)
    print(
        "Rejet uniforme : "
        f"probabilité d'acceptation empirique = {proba_empirique:.4f}, "
        f"théorique = {2 / np.pi:.4f}"
    )

    # Rejet avec q_tilde comme densité de proposition.
    q_tilde = lambda x: (3/2)*(1-x**2)

    def gqqtilde(x):
        """Rapport entre la cible non normalisée et la densité q_tilde."""
        x = np.asarray(x, dtype=float)
        rapport = np.full_like(x, np.pi / 6)
        np.divide(
            g(x),
            q_tilde(x),
            out=rapport,
            where=(x != 1),
        )
        return rapport

    # Inversion de F(x) = (3x - x**3) / 2, la fonction de répartition de q_tilde.
    y = rd.uniform(0, 1, nombre_tirages)
    X = 2 * np.cos((2 * np.pi - np.arccos(-y)) / 3)
    U = rd.uniform(0, 1, nombre_tirages)
    constante_rejet = 2 / 3
    acceptes = U <= gqqtilde(X) / constante_rejet
    rejetes = ~acceptes

    plt.scatter(X[acceptes], U[acceptes], color="yellow", s=12, label="Points acceptés")
    plt.scatter(X[rejetes], U[rejetes], color="red", s=12, label="Points rejetés")
    plt.plot(
        x_courbe,
        gqqtilde(x_courbe) / constante_rejet,
        color="blue",
        label="g(x) / (C q_tilde(x))",
    )
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.xlabel("x")
    plt.ylabel("u")
    plt.title("Méthode du rejet avec q_tilde")
    plt.legend()
    plt.grid()
    plt.show()

    proba_empirique = np.mean(acceptes)
    proba_theorique = (2 / np.pi) / constante_rejet
    print(
        "Rejet avec q_tilde : "
        f"probabilité d'acceptation empirique = {proba_empirique:.4f}, "
        f"théorique = {proba_theorique:.4f}"
    )

    # Variance de l'estimateur par fréquence d'acceptation.
    nombre_realisations = 100
    mu_numerique = integrate.quad(lambda x: g(x), 0, 1)[0]

    X_uniforme = rd.uniform(0, 1, (nombre_realisations, nombre_tirages))
    U_uniforme = rd.uniform(0, 1, (nombre_realisations, nombre_tirages))
    estimations_uniformes = np.mean(U_uniforme <= g(X_uniforme), axis=1)

    X_qtilde = 2 * np.sin(
        np.arcsin(rd.uniform(0, 1, (nombre_realisations, nombre_tirages))) / 3
    )
    U_qtilde = rd.uniform(0, 1, (nombre_realisations, nombre_tirages))
    estimations_qtilde = constante_rejet * np.mean(
        U_qtilde <= gqqtilde(X_qtilde) / constante_rejet,
        axis=1,
    )

    variance_uniforme_theorique = (
        mu_numerique * (1 - mu_numerique) / nombre_tirages
    )
    variance_qtilde_theorique = (
        constante_rejet * mu_numerique - mu_numerique**2
    ) / nombre_tirages
    print(
        f"Variances de l'estimateur de l'intégrale (n={nombre_tirages}, "
        f"{nombre_realisations} répétitions) :\n"
        f"  uniforme : théorique numérique = {variance_uniforme_theorique:.6g}, "
        f"empirique = {np.var(estimations_uniformes, ddof=1):.6g}\n"
        f"  q_tilde  : théorique numérique = {variance_qtilde_theorique:.6g}, "
        f"empirique = {np.var(estimations_qtilde, ddof=1):.6g}"
    )

    # Variance des estimateurs d'importance g(X) et g(X) / q_tilde(X).
    estimations_importance_uniforme = np.mean(g(X_uniforme), axis=1)
    estimations_importance_qtilde = np.mean(gqqtilde(X_qtilde), axis=1)
    variance_g_uniforme = (
        integrate.quad(lambda x: g(x) ** 2, 0, 1)[0] - mu_numerique**2
    )
    variance_g_qtilde = (
        integrate.quad(
            lambda x: float(q_tilde(x) * gqqtilde(x) ** 2),
            0,
            1,
        )[0]
        - mu_numerique**2
    )
    print(
        f"Variances des estimateurs d'importance (n={nombre_tirages}, "
        f"{nombre_realisations} répétitions) :\n"
        f"  uniforme : théorique numérique = "
        f"{variance_g_uniforme / nombre_tirages:.6g}, "
        f"empirique = {np.var(estimations_importance_uniforme, ddof=1):.6g}\n"
        f"  q_tilde  : théorique numérique = "
        f"{variance_g_qtilde / nombre_tirages:.6g}, "
        f"empirique = {np.var(estimations_importance_qtilde, ddof=1):.6g}"
    )

    # comparaison des lois (vérification du TCL)
    return None
    
if __name__ == "__main__":
    # partie_1()
    methode_rejet()
