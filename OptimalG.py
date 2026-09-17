from GeneralExpon import simulate_different_Lambda
import numpy as np
import matplotlib.pyplot as plt
PD_params = {'R': 6, 'S': 3, 'T': 12, 'P': 5}

w_I = 0.01
w_G = 0.01

Lambda_list = np.linspace(1, 200, 100)

t_end = 1000
dt = 0.01
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

def calculate_optimal_G(
    n_values,
    params
):
    """
    For each group size n, calculate:

        G_opt[n]:
            Maximum possible group-average payoff.

        i_opt[n]:
            Number of cooperators that maximizes G.

        x_opt[n]:
            Optimal fraction of cooperators.
    """

    optimal_results = {}
    for n in n_values:
        _, _, G = calculate_payoffs(
            n=n,
            **params
        )

        i_opt = np.argmax(G)
        G_opt = G[i_opt]
        x_opt = i_opt / n

        optimal_results[n] = {
            "G_opt": G_opt,
            "i_opt": i_opt,
            "x_opt": x_opt
        }

    return optimal_results

colors = {
    2: "#D95F59",   # Red
    3: "#4769A6",   # Blue
    4: "#3C7A6B",   # Green
    20: "#8A62A6",  # Purple
}
states_n2, cooperation_n2, payoff_n2 = (simulate_different_Lambda(n=2,params=PD_params,Lambda_list=Lambda_list,w_I=w_I,w_G=w_G))
states_n3, cooperation_n3, payoff_n3 = (simulate_different_Lambda(n=3,params=PD_params,Lambda_list=Lambda_list,w_I=w_I,w_G=w_G))
states_n4, cooperation_n4, payoff_n4 = (simulate_different_Lambda(n=4,params=PD_params,Lambda_list=Lambda_list,w_I=w_I,w_G=w_G))
states_n20, cooperation_n20, payoff_n20 = (simulate_different_Lambda(n=20,params=PD_params,Lambda_list=Lambda_list,w_I=w_I,w_G=w_G))


n_values = [2, 3, 4, 20]

optimal_results = calculate_optimal_G(
    n_values=n_values,
    params=PD_params
)

for n in n_values:
    print(
        f"n = {n}: "
        f"G_opt = {optimal_results[n]['G_opt']:.6f}, "
        f"i_opt = {optimal_results[n]['i_opt']}, "
        f"x_opt = {optimal_results[n]['x_opt']:.6f}"
    )



plt.figure(figsize=(8, 6))

for n in n_values:

    _, _, G = calculate_payoffs(
        n=n,
        **PD_params
    )

    i_values = np.arange(n + 1)

    x_values = i_values / n

    plt.scatter(
        x_values,
        G,
        s=60,
        label=f"n = {n}"
    )



plt.xlabel(r"Fraction of cooperators $x=i/n$")
plt.ylabel(r"Group average payoff $G_n(x)$")
plt.title("Group Average Payoff")

plt.legend()
plt.grid(alpha=0.3)

plt.tight_layout()
plt.show()



# Eq point distribution

selected_Lambdas = [20, 100, 200]

state_results = {
    2: states_n2,
    3: states_n3,
    4: states_n4,
    20: states_n20
}

fig, axes = plt.subplots(
    len(selected_Lambdas),
    4,
    figsize=(16, 9),
    sharey=True
)

for row, selected_Lambda in enumerate(selected_Lambdas):
    lambda_index = np.argmin(
        np.abs(Lambda_list - selected_Lambda)
    )

    actual_Lambda = Lambda_list[lambda_index]

    for col, n in enumerate([2, 3, 4, 20]):

        ax = axes[row, col]

        equilibrium_fraction = (
            state_results[n][lambda_index]
        )

        cooperation_fraction = (
            np.arange(n + 1) / n
        )

        ax.bar(
            cooperation_fraction,
            equilibrium_fraction,
            width=0.7 / n,
            color=colors[n],
            alpha=0.8
        )

        ax.set_title(
            rf"$n={n},\ \Lambda={actual_Lambda:.1f}$"
        )

        ax.set_xlabel(
            r"Cooperation fraction $i/n$"
        )

        if col == 0:
            ax.set_ylabel(
                r"Equilibrium fraction $f_i^*$"
            )

        ax.set_xlim(-0.05, 1.05)
        ax.grid(axis="y", alpha=0.3)

plt.suptitle(
    "Equilibrium Distribution of Group Types",
    fontsize=15
)

plt.tight_layout()
plt.show()
