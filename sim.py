import numpy as np
from matplotlib import pyplot as plt
import matplotlib.animation as animation


def runge_kutta_4(fun: callable, ti: float, tf: float, sol0: np.ndarray, h: float) -> tuple[np.ndarray, np.ndarray]:
    tps = np.arange(ti, ti + tf, h)
    n = tps.size
    sol = np.zeros((n, sol0.size))
    sol[0] = sol0

    for i in range(n - 1):
        t2 = tps[i] + h / 2
        y2 = sol[i] + h / 2 * fun(tps[i], sol[i])
        y3 = sol[i] + h / 2 * fun(t2, y2)
        y4 = sol[i] + h * fun(t2, y3)
        sol[i + 1] = sol[i] + h / 6 * (fun(tps[i], sol[i]) + 2 * fun(t2, y2) + 2 * fun(t2, y3) + fun(tps[i + 1], y4))

    return tps, sol


dT_ext = lambda t: 5 * 2 * np.pi / 24 * np.sin(2 * np.pi * t / 24)
D = 4  # Diffusivité
U_int = 3e-1  # Conductance à l'interface mur/air intérieur
# U_ext = 1e-4  # Conductance à l'interface mur/air extérieur

ti = 0  # Temps initial
tf = 24 * 4 # temps final
h = 1/60  # Pas de temps
nb_couches = 30  # Nombre de couches dans le mur
T_init_int = 273 + 25  # Température initiale à l'intérieur
T_init_ext = 273 + 5  # Température initiale à l'extérieur
sol0 = np.linspace(T_init_int, T_init_ext, nb_couches + 2)  # Etat initial


def equation(t: float, T: np.ndarray) -> np.ndarray:
    dT = np.zeros_like(T)

    dT[-1] = dT_ext(t)
    dT[1:-1] = D * (T[0:-2] - 2 * T[1:-1] + T[2:])
    dT[0] = U_int * (T[1] - T[0])
    # dT[-2] = U_ext * (T[-1] - T[-2])

    return dT


tps, sol = runge_kutta_4(equation, ti, tf, sol0, h)
sol -= 273

fig, ax = plt.subplots()
plt.axis("off")
plt.tight_layout()
X = np.concatenate(([0], np.linspace(1, 2, nb_couches + 1), [3]))
Y = np.array((0, 1))

artists = []
for i in range(0, sol.shape[0], 10):
    C = sol[i].reshape((1, nb_couches + 2))
    artists.append((ax.pcolorfast(X, Y, C, cmap="afmhot", vmin=sol.min(), vmax=sol.max()),))

ani = animation.ArtistAnimation(fig=fig, artists=artists, interval=50)
ani.save("mur.gif")
plt.show()
