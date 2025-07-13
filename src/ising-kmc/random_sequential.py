import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

TIME = 0

# Initialize the flipping rate
def flipping_rate(lattice, i, j, H, J, beta):
    """
    Calculate the flipping rate for a given spin.

    Parameters:
        lattice (np.ndarray): The spin lattice.
        i (int): Row index of the spin.
        j (int): Column index of the spin.
        H (float): External magnetic field.
        J (float): Coupling constant.
        beta (float): Inverse temperature.

    Returns:
        float: The flipping rate.
    """
    neighbors = [((i - 1) % N, j), ((i + 1) % N, j), (i, (j - 1) % N), (i, (j + 1) % N)]
    dE = (
        2 * J * lattice[i, j] * sum(lattice[x, y] for x, y in neighbors)
        + 2 * H * lattice[i, j]
    )
    rate = min(1, np.exp(-beta * dE))
    return rate


def avg_magnetization(lattice):
    """
    Calculate the average magnetization of the lattice.

    Parameters:
        lattice (np.ndarray): The spin lattice.

    Returns:
        float: The average magnetization.
    """
    return np.mean(lattice)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Simulate the 2D Ising model using the next reaction method.')
    parser.add_argument('--beta', type=float, default=0.3, help='Inverse temperature')
    parser.add_argument('--N', type=int, default=20, help='Size of the lattice (default: 20)')
    parser.add_argument('--J', type=float, default=1.0, help='Coupling constant (default: 1.0)')
    parser.add_argument('--H', type=float, default=0.0, help='External magnetic field (default: 0.0)')
    parser.add_argument('--MAX_TIME', type=float, default=25000, help='Maximum simulation time (default: 25000)')
    
    args = parser.parse_args()
    beta = args.beta
    N = args.N
    J = args.J
    H = args.H
    MAX_TIME = args.MAX_TIME
    lattice = np.random.choice([-1, 1], size=(N, N))
    
    flip_rate = np.zeros((N, N))
    max_flip_rate = 0
    for i in range(N):
        for j in range(N):
            flip_rate[i, j] = flipping_rate(lattice, i, j, H, J, beta)
    max_flip_rate = np.max(flip_rate)

    # Initialize the animation
    fig, ax = plt.subplots()
    im = ax.imshow(lattice, cmap="coolwarm", interpolation="nearest")
    ax.axis("off")
    ax.set_title(f"Ising Model at beta = {beta:.2f}")
    time_text = ax.text(
        0.02,
        0.95,
        "",
        transform=ax.transAxes,
        color="white",
        fontsize=12,
        bbox=dict(facecolor="black", alpha=0.7),
    )
    magnetization_text = ax.text(
        0.02,
        0.88,
        "",
        transform=ax.transAxes,
        color="white",
        fontsize=12,
        bbox=dict(facecolor="black", alpha=0.7),
    )
    
    def update(frame):
        """
        Update the lattice.

        Parameters:
            frame (int): The current frame number.

        Returns:
            matplotlib.image.AxesImage: The updated lattice.
        """
        global lattice, flip_rate, max_flip_rate, TIME
        i, j = np.random.randint(N), np.random.randint(N)
        normalized_flip_rate = flip_rate[i, j] / (
            max_flip_rate
        )  # The flipping probability at given time Delta t = 1/(max_flip_rate)

        if np.random.rand() < normalized_flip_rate:
            lattice[i, j] *= -1
            # update the flipping rate of its neighbors and itself
            neighbors = [
                ((i - 1) % N, j),
                ((i + 1) % N, j),
                (i, (j - 1) % N),
                (i, (j + 1) % N),
            ]
            for x, y in neighbors:
                flip_rate[x, y] = flipping_rate(lattice, x, y, H, J, beta)
            flip_rate[i, j] = flipping_rate(lattice, i, j, H, J, beta)
            max_flip_rate = np.max(flip_rate)

        TIME += 1 / (N**2 * max_flip_rate)
        im.set_array(lattice)
        time_text.set_text(f"Time: {TIME:.2f}")
        magnetization = avg_magnetization(lattice)
        magnetization_text.set_text(f"Magnetization: {magnetization:.3f}")
        return im, magnetization

    ani = FuncAnimation(fig, update, frames=MAX_TIME, interval=1)  # Simulate the lattice
    plt.show()
    # writer = PillowWriter(fps=50)
    # import os
    # os.makedirs("ising-kmc/examples/random_sequential", exist_ok=True)
    # with writer.saving(fig, f"ising-kmc/examples/random_sequential/beta={beta}.gif", dpi=200):
    #     for k in range(MAX_TIME):
    #         update(k)
    #         if k % 50 == 0:
    #             writer.grab_frame()
