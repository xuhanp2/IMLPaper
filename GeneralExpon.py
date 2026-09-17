import numpy as np
import matplotlib.pyplot as plt
PD_params = {'R': 6, 'S': 3, 'T': 12, 'P': 5}

w_I = 0.01
w_G = 0.01

Lambda_list = np.linspace(1, 200, 100)

t_end = 1000
dt = 0.01

def rk4_step(f, t, y, dt):
    k1 = f(t, y)
    k2 = f(t + 0.5 * dt, y + 0.5 * dt * k1)
    k3 = f(t + 0.5 * dt, y + 0.5 * dt * k2)
    k4 = f(t + dt, y + dt * k3)
    return y + dt * (k1 + 2*k2 + 2*k3 + k4) / 6


def solve_to_steady(system, y0):
    y = y0.copy()
    t = 0.0
    while t < t_end:
        y = rk4_step(system, t, y, dt)
        s = y.sum()
        if s > 0:
            y /= s
        t += dt
    return y

def calculate_payoffs(n, R, S, T, P):
    """
    Return:
        pi_C[j]: payoff of a cooperator in a j-cooperator group
        pi_D[j]: payoff of a defector in a j-cooperator group
        G[j]:    average payoff of a j-cooperator group
    """
    pi_C = np.zeros(n + 1)
    pi_D = np.zeros(n + 1)
    G = np.zeros(n + 1)

    # Cooperator payoff
    for j in range(0, n + 1):
        pi_C[j] = (R * (j - 1) / (n - 1)+ S * (n - j) / (n - 1))
    # Defector payoff
    for j in range(0, n + 1):
        pi_D[j] = (T * j / (n - 1)+P * (n - j - 1) / (n - 1))
    # Group-average payoff
    for j in range(0, n + 1):
        G[j] = ((j / n) * pi_C[j]+ ((n - j) / n) * pi_D[j])
    return pi_C, pi_D, G


def individual_selection(f,pi_C,pi_D,w_I):
    n = len(f) - 1
    df_individual = np.zeros(n + 1)
    for j in range(n + 1):
        # Flow from j-1 to j
        if j > 1:
            k = j - 1
            fitness_C = np.exp(w_I * pi_C[k])
            fitness_D = np.exp(w_I * pi_D[k])
            denominator = (k * fitness_C + (n - k) * fitness_D)
            df_individual[j] += (f[k] * (n - k) / n * k * fitness_C / denominator)
        # Flow from j+1 to j
        if j < n - 1:
            k = j + 1
            fitness_C = np.exp(w_I * pi_C[k])
            fitness_D = np.exp(w_I * pi_D[k])
            denominator = (k * fitness_C + (n - k) * fitness_D)
            df_individual[j] += (f[k] * k / n * (n - k) * fitness_D / denominator)

        # Combined flow out of j
        if 0 < j < n:
            fitness_C = np.exp(w_I * pi_C[j])
            fitness_D = np.exp(w_I * pi_D[j])
            denominator = (j * fitness_C + (n - j) * fitness_D)
            df_individual[j] -= (f[j] * j * (n - j) / n * (fitness_C + fitness_D) / denominator)
    return df_individual
def group_selection(f,G,w_G,Lambda):
    group_fitness = np.exp(w_G * G)
    D = np.dot(f, group_fitness)
    df_group = (Lambda * f * (group_fitness / D - 1.0))
    return df_group

#General ODE
def TN_general(t,f,pi_C,pi_D,G,w_I,w_G,Lambda):
    df_individual = individual_selection(f,pi_C,pi_D,w_I)
    df_group = group_selection(f,G,w_G,Lambda)
    df = (df_individual + df_group)
    return df

def make_TN_system_general(n,R,S,T,P,w_I,w_G,Lambda):
    pi_C, pi_D, G = calculate_payoffs(n, R, S, T, P)
    def system(t, f):
        return TN_general(t,f,pi_C,pi_D,G,w_I,w_G,Lambda)
    return system, G


def simulate_different_Lambda(n,params,Lambda_list,w_I,w_G):
    steady_states = []
    cooperation_values = []
    payoff_values = []

    # Initial condition
    y_init = np.full(
        n + 1,
        0.01 / n
    )
    y_init[n] = 0.99
    # y_init = np.array([0.01, 0.01, 0.98])

    x_values = np.arange(n + 1) / n

    for Lam in Lambda_list:
        system, G = make_TN_system_general(n=n,**params,w_I=w_I,w_G=w_G,Lambda=Lam)
        f_steady = solve_to_steady(system, y_init)
        steady_states.append(f_steady)
        average_cooperation = np.dot(x_values, f_steady)

        average_payoff = np.dot(G,f_steady)
        cooperation_values.append(average_cooperation)
        payoff_values.append(average_payoff)

    return (
        np.array(steady_states),
        np.array(cooperation_values),
        np.array(payoff_values)
    )

states_n2, cooperation_n2, payoff_n2 = (simulate_different_Lambda(n=2,params=PD_params,Lambda_list=Lambda_list,w_I=w_I,w_G=w_G))

states_n3, cooperation_n3, payoff_n3 = (simulate_different_Lambda(n=3,params=PD_params,Lambda_list=Lambda_list,w_I=w_I,w_G=w_G))

states_n4, cooperation_n4, payoff_n4 = (simulate_different_Lambda(n=4,params=PD_params,Lambda_list=Lambda_list,w_I=w_I,w_G=w_G))

states_n20, cooperation_n20, payoff_n20 = (simulate_different_Lambda(n=20,params=PD_params,Lambda_list=Lambda_list,w_I=w_I,w_G=w_G))


# first graph: average cooperation
colors = {
    2: "#D95F59",   # Red
    3: "#4769A6",   # Blue
    4: "#3C7A6B",   # Green
    20: "#8A62A6",  # Purple
}

plt.figure(figsize=(8, 5.5))
plt.plot(
    Lambda_list,
    cooperation_n2,
    color=colors[2],
    linewidth=2.5,
    label=r'$n=2$'
)

plt.plot(
    Lambda_list,
    cooperation_n3,
    color=colors[3],
    linewidth=2.5,
    label=r'$n=3$'
)

plt.plot(
    Lambda_list,
    cooperation_n4,
    color=colors[4],
    linewidth=2.5,
    label=r'$n=4$'
)

plt.plot(
    Lambda_list,
    cooperation_n20,
    color=colors[20],
    linewidth=2.5,
    label=r'$n=20$'
)

plt.xlabel(
    r'Speed of group-level selection $\Lambda$',
    fontsize=12
)

plt.ylabel(
    r'Average cooperation $\langle x\rangle_f$',
    fontsize=12
)

plt.title(
    'Average Cooperation vs. '
    r'$\Lambda$ in the Traulsen PD Model',
    fontsize=13
)

# The data begin at Lambda = 1
plt.xlim(0, 100)
plt.ylim(0, 1)

plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()


# second graph, average payoff
plt.figure(figsize=(8, 5.5))

plt.plot(Lambda_list,payoff_n2,color=colors[2],linewidth=2.5,label=r'$n=2$')

plt.plot(
    Lambda_list,
    payoff_n3,
    color=colors[3],
    linewidth=2.5,
    label=r'$n=3$'
)

plt.plot(
    Lambda_list,
    payoff_n4,
    color=colors[4],
    linewidth=2.5,
    label=r'$n=4$'
)

plt.plot(
    Lambda_list,
    payoff_n20,
    color=colors[20],
    linewidth=2.5,
    label=r'$n=20$'
)

plt.xlabel(r'Speed of group-level selection $\Lambda$', fontsize=12)

plt.ylabel(r'Average payoff $\langle G\rangle_f$', fontsize=12)

plt.title('Average Payoff vs. ' r'$\Lambda$ in the Traulsen PD Model', fontsize=13)

plt.xlim(0, 100)
plt.ylim(4.5, 8.0)
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()



# third graph
state_results = {
    2: states_n2,
    3: states_n3,
    4: states_n4,
    20: states_n20
}

fig, axes = plt.subplots(2, 2, figsize=(13, 9), sharex=True, sharey=True)

axes = axes.flatten()

for ax, n in zip(axes, [2, 3, 4, 20]):

    steady_states = state_results[n]

    state_colors = plt.cm.viridis(
        np.linspace(0, 1, n + 1)
    )

    for i in range(n + 1):
        ax.plot(
            Lambda_list,
            steady_states[:, i],
            color=state_colors[i],
            linewidth=2.0,
            label=rf"$f_{{{i}}}$"
        )

    ax.set_title(
        rf"$n={n}$",
        fontsize=13
    )

    ax.set_xlim(0, 100)
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.3)

    if n <= 4:
        ax.legend(
            fontsize=9,
            loc="best"
        )

    else:
        normalization = plt.Normalize(
            vmin=0,
            vmax=n
        )

        color_mapping = plt.cm.ScalarMappable(
            norm=normalization,
            cmap="viridis"
        )

        color_mapping.set_array([])

        colorbar = fig.colorbar(
            color_mapping,
            ax=ax,
            pad=0.02
        )

        colorbar.set_label(
            r"Number of cooperators $i$"
        )

        colorbar.set_ticks([0, 5, 10, 15, 20])

axes[2].set_xlabel(
    r"Speed of group-level selection $\Lambda$",
    fontsize=12
)

axes[3].set_xlabel(
    r"Speed of group-level selection $\Lambda$",
    fontsize=12
)

axes[0].set_ylabel(
    r"Steady-state frequency $f_i^*$",
    fontsize=12
)

axes[2].set_ylabel(
    r"Steady-state frequency $f_i^*$",
    fontsize=12
)

fig.suptitle(
    "Steady-State Group Distributions "
    r"vs. $\Lambda$ in the Traulsen PD Model",
    fontsize=15,
    y=1.01
)

plt.tight_layout()
plt.show()
