# Ising-KMC: Simulation of Non-equilibrium Dynamics in the 2D Ising Model

Non-equilibrium systems, such as those undergoing relaxation dynamics, are crucial for understanding real-world phenomena. The 2D Ising model is a fundamental framework for studying phase transitions and spin dynamics in statistical mechanics. This repo contains codes that simulate and analyze the magnetization dynamics of the 2D Ising model under various temperatures $T$, focusing on the behavior of relaxation towards equilibrium and the phase transition during altering the temperature $T$.  Both discrete- and continuous-time approaches -- the random-sequential update method and the next-reaction method -- are implemented to study the dynamics of the system.

## Prerequisites

You need to have Python 3.9 installed along with the following libraries to run the simulations:

- `numpy` 1.26.4
- `matplotlib` 3.10.0

## Installation

Clone the repository:

```bash
git clone https://github.com/bowenyu066/ising-kmc.git
cd ising-kmc
```

Setting up a virtual environment is recommended:

```bash
conda create -n ising-kmc python=3.9
conda activate ising-kmc
pip install numpy==1.26.4 matplotlib==3.10.0
```

Or, you can install the dependencies from `requirements.txt` in your current environment:

```bash
pip install -r requirements.txt
```

## Simulation

Source codes are located in the `src/ising-kmc` directory. You can simulate the 2D Ising model dynamics using either the random-sequential update method:

```bash
python src/ising-kmc/random_sequential.py --beta 0.6 --N 20 --J 1.0 --H 0.0 --MAX_TIME 25000
```

Or the next-reaction method:

```bash
python src/ising-kmc/next_reaction.py --beta 0.6 --N 20 --J 1.0 --H 0.0 --MAX_TIME 5000
```

Here, `--beta` is the inverse temperature, `--N` refers to the size of the lattice $N \times N$ (defaults to 20), `--J` is the coupling constant $J$ (defaults to 1.0), `--H` is the external magnetic field $\mathcal{H}$ (defaults to 0.0), and `--MAX_TIME` is the maximum simulation steps (defaults to 5000 for next-reaction method and 25000 for random-sequential method).

## Results

The simulation was conducted on a $20 \times 20$ lattice with periodic boundary conditions. The coupling constant was set to $J = 1$ and the external magnetic field was $\mathcal{H} = 0$. Several simulation results are provided in the `examples/` directory, including:

- `beta=0.3.gif` to `beta=0.7.gif` for both random-sequential and next-reaction methods: the relaxation dynamics of the magnetization $M$ towards equilibrium at different inverse temperatures $\beta$
- `random-sequential.png` and `next-reaction.png`: The magnetization $M(t)$ as a function of time $t$ for the random-sequential update method and the next-reaction method, respectively, at different inverse temperatures $\beta$
- `random-sequential-eq.png` and `next-reaction-eq.png`: The equilibrium magnetization $M$ as a function of inverse temperature $\beta$ for the random-sequential update method and the next-reaction method, respectively

## Methodology

### Model

The 2D Ising model consists of a lattice of spins $s_i = \pm 1$ on a square lattice, with nearest-neighbor interactions. The Hamiltonian is given by

$$
    H = -J \sum_{\langle i, j \rangle} s_i s_j - \mathcal{H} \sum_i s_i
$$

where $J$ is the coupling constant, $\langle i, j \rangle$ denotes the sum over nearest neighbors, and $\mathcal{H}$ is the external magnetic field. When $J >0$, the system exhibits ferromagnetic behavior. The magnetization $M$ is defined as

$$
    M = \frac{1}{N} \sum_i s_i
$$

where $N$ is the total number of spins. The magnetic susceptibility $\chi$ is defined as

$$
    \chi = \frac{\partial M}{\partial \mathcal{H}} = \beta \left( \langle M^2 \rangle - \langle M \rangle^2 \right)
$$

where $\beta \equiv \frac{1}{k_B T}$ is the inverse temperature, and $\langle \cdot \rangle$ denotes the average over repeated experiments, i.e., the ensemble average.

The 2D Ising model is the first model that was shown to exhibit a non-trivial phase transition at thermodynamic limit $N \to \infty$ from a continuous partition function. When the external field $\mathcal{H} = 0$, the 2D Ising model undergoes a phase transition at the critical temperature $T_c$ where the magnetization $M$ changes from zero to a non-zero value. In 1944, Onsager first showed that the critical temperature for the 2D Ising model is

$$
    T_c = \frac{2J}{k_B \ln(1 + \sqrt{2})} \approx 2.269 \frac{J}{k_B}, \quad \beta_c = \frac{\ln (1 + \sqrt{2})}{2} \approx 0.441
$$

When the system approaches phase transition, the magnetization $M$ and the magnetic susceptibility $\chi$ exhibit critical behavior:

$$
    M \sim (T - T_c)^\beta, \quad \chi \sim |T - T_c|^{-\gamma}
$$

where $\beta$ and $\gamma$ are critical exponents. Onsager showed that for the 2D Ising model, the critical exponents are

$$
    \beta = \frac{1}{8},\quad \gamma = \frac{7}{4}
$$

Regarding the simulations, we focus on the relaxation behavior of the magnetization $M$ towards equilibrium, and the phase transition behavior of the equilibrium magnetization $M$ under different temperatures $T$ within zero external field $\mathcal{H} = 0$.

### Random-Sequencial Update Method

The random-sequential update method is a discrete-time method where each spin is updated sequentially. The algorithm is as follows:

1. Initialize the $N \times N$ lattice with random spins $s_{(i, j)} = \pm 1$.
2. Set the flipping rate $w_{(i, j)}$ for each spin $s_{(i, j)}$:
- Calculate the change in energy $\Delta E$ if $s_{(i, j)}$ is flipped:

$$
    \Delta E_{(i, j)} = 2 J s_{(i, j)} (s_{(i-1, j)} + s_{(i+1, j)} + s_{(i, j-1)} + s_{(i, j+1)})
$$

- Set the flipping rate $w_{(i, j)}$ as follows:

$$
    w_{(i, j)} = \begin{cases}
        e^{-\beta \Delta E} & \text{if } \Delta E > 0, \\
        1 & \text{if } \Delta E \leq 0.
    \end{cases}
$$

- Update the maximum flipping rate $w = \max_{i, j} w_{(i, j)}$.
3. Select a random spin $(i, j)$, and calculate the flipping probability $p = w_{(i, j)} / w$ within the time step $\Delta t = 1 / w$. Generate a random number $r$ from a uniform distribution $[0, 1]$. If $r < p$, flip the spin $s_{(i, j)}$. Otherwise, do nothing.
4. Update the time $t = t + \Delta t / N^2$, where $N^2$ is the total number of spins.
5. Repeat steps 2-4 until the system reaches equilibrium.

### Next-Reaction Method

The next-reaction method is a continuous-time method where the time to the next reaction (i.e., spin flip) is calculated. The algorithm is as follows:

1. Initialize the lattice with random spins $s_{(i, j)} = \pm 1$. Set the initial time $t = 0$.
2. Set the flipping rate $w_{(i, j)}$ and the reaction time $\tau_{(i, j)}$ for each spin $s_{(i, j)}$:
- Calculate the change in energy $\Delta E$ if $s_{(i, j)}$ is flipped:

$$
    \Delta E_{(i, j)} = 2 J s_{(i, j)} (s_{(i-1, j)} + s_{(i+1, j)} + s_{(i, j-1)} + s_{(i, j+1)})
$$

- Set the flipping rate $w_{(i, j)}$ as follows:

$$
    w_{(i, j)} = \begin{cases}
        e^{-\beta \Delta E} & \text{if } \Delta E > 0, \\
        1 & \text{if } \Delta E \leq 0.
    \end{cases}
$$

- Generate a random number $r$ from a uniform distribution $[0, 1]$. Set the reaction time $\tau_{(i, j)} = -\ln r / w_{(i, j)}$.
- Store the reaction time $\tau_{(i, j)}$ in an indexed min-heap.
3. Select the spin $(i, j)$ with the minimum reaction time $\tau_{(i, j)}$ from the min-heap. Flip the spin $s_{(i, j)}$.
4. Update the time $t = \tau_{(i, j)}$.
5. Update the reaction time for the neighboring spins and the spin $(i, j)$ itself as follows: $\tau_{(i, j)} = t + \Delta \tau_{(i, j)}$, where $\Delta \tau_{(i, j)} = -\ln r / w_{(i, j)}$, $r \sim \text{Unif}[0, 1]$. Update the min-heap.
6. Repeat steps 3-5 until the system reaches equilibrium.

