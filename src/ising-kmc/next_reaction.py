import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

# # Initialize lattice
# N = 20  # Size of lattice
# J = 1.0  # Coupling constant
# H = 0.0  # External magnetic field
# beta = 0.6  # Inverse temperature
# MAX_TIME = 3000  # Maximum simulation time
TIME = 0

class IndexedHeap(object):
    def __init__(self):
        self.heap = [] # List of (tau, (i, j)) pairs
        self.pos = {} # Mapping from (i, j) to heap index

    def push(self, tau, i, j):
        # Push the element into the heap
        entry = (tau, (i, j))
        self.pos[(i, j)] = len(self.heap)
        self.heap.append(entry)
        self._siftup(0, len(self.heap)-1)

    def get_smallest(self):
        # Get the element with the smallest tau
        return self.heap[0]

    def pop(self):
        # Pop the element with the smallest tau
        last = self.heap.pop()
        if len(self.heap) > 0:
            return_item = self.heap[0]
            self.heap[0] = last
            self.pos[last[1]] = 0
            self._siftdown(0)
            return return_item
        return last
    
    def remove(self, i, j):
        # Remove the element at position (i, j)
        index = self.pos[(i, j)]
        self._swap(index, len(self.heap)-1)
        self.heap.pop()
        self._siftup(0, index)
        self._siftdown(index)

    def replace(self, tau, i, j):
        # Replace the element at position (i, j) with a new tau
        index = self.pos[(i, j)]
        self.heap[index] = (tau, (i, j))
        self._siftup(0, index)
        self._siftdown(index)

    def _siftup(self, start, index):
        # Move the element at index up the heap
        heap = self.heap
        while index > start:
            parent_index = (index - 1) >> 1
            parent = heap[parent_index]
            if parent[0] <= heap[index][0]:
                break
            self._swap(index, parent_index)
            index = parent_index

    def _siftdown(self, index):
        # Move the element at index down the heap
        heap = self.heap
        start = 0
        end = len(heap)
        child_index = 2 * index + 1
        while child_index < end:
            right_index = child_index + 1
            if right_index < end and not heap[child_index][0] < heap[right_index][0]:
                child_index = right_index
            if heap[index][0] < heap[child_index][0]:
                break
            self._swap(index, child_index)
            index = child_index
            child_index = 2 * index + 1

    def _swap(self, i, j):
        # Swap the elements at indices i and j
        heap = self.heap
        pos = self.pos
        heap[i], heap[j] = heap[j], heap[i]
        pos[heap[i][1]] = i
        pos[heap[j][1]] = j

def exp(rate):
    """
    Simulate the exponential distribution.

    Parameters:
        rate (float): The escape rate.

    Returns:
        float: The time until the next event. 
    """
    return -np.log(np.random.rand()) / rate

# Function to calculate energy change for flipping a spin
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
    neighbors = [((i-1)%N, j), ((i+1)%N, j), (i, (j-1)%N), (i, (j+1)%N)] 
    dE = 2 * J * lattice[i, j] * sum(lattice[x, y] for x, y in neighbors) + 2 * H * lattice[i, j]
    rate = min(1, np.exp(-beta * dE))
    return rate

def initialize(reaction_time: IndexedHeap, lattice, size, H, J, beta):
    """
    Initialize the reaction times for all spins.

    Parameters:
        reaction_time (IndexedHeap): The min-heap for reaction times.
        lattice (np.ndarray): The spin lattice.
        size (int): Size of the lattice.
        H (float): External magnetic field.
        J (float): Coupling constant.
        beta (float): Inverse temperature.

    Returns:
        None
    """
    for i in range(size):
        for j in range(size):
            rate = flipping_rate(lattice, i, j, H, J, beta)
            tau = exp(rate)
            reaction_time.push(tau, i, j)

def update(reaction_time: IndexedHeap, lattice, H, J, MAX_TIME=1e8):
    """
    Update the lattice.

    Parameters:
        reaction_time (IndexedHeap): The min-heap for reaction times.
        lattice (np.ndarray): The spin lattice.
        H (float): External magnetic field.
        J (float): Coupling constant.
        MAX_TIME (float): Maximum simulation time.

    Returns:
        float: The time of the next flipping event.
    """
    tau, (i, j) = reaction_time.get_smallest()
    lattice[i, j] *= -1
    # Update reaction times for neighbors (including itself)
    neighbors = [((i-1)%N, j), ((i+1)%N, j), (i, (j-1)%N), (i, (j+1)%N)]
    for x, y in neighbors:
        rate = flipping_rate(lattice, x, y, H, J, beta)
        new_tau = exp(rate) + tau
        reaction_time.replace(new_tau, x, y)
    rate = flipping_rate(lattice, i, j, H, J, beta)
    new_tau = exp(rate) + tau
    reaction_time.replace(new_tau, i, j)

    return tau

def avg_magnetization(lattice):
    """
    Calculate the average magnetization of the lattice.

    Parameters:
        lattice (np.ndarray): The spin lattice.

    Returns:
        float: The average magnetization.
    """
    return np.mean(lattice)

def plot_magnetization(beta, times, magnetization, ratio=1):
    assert len(times) == len(magnetization)
    intv = int(1 / ratio)
    plt.plot(times[::intv], magnetization[::intv])
    plt.legend([f'Beta = {beta}'])
    plt.xlabel('Time')
    plt.ylabel('Average Magnetization')
    plt.title('Average Magnetization vs Time')

def simulate(beta, H, J, N, MAX_TIME=1e8, ratio=1):
    """
    Main function for simulating the 2D Ising model.

    Parameters:
        beta (float): Inverse temperature.
        H (float): External magnetic field.
        J (float): Coupling constant.
        N (int): Size of the lattice.
        MAX_TIME (float): Maximum simulation time.

    Returns:
        None
    """
    # Initialize the lattice and the reaction times
    time = 0
    times, magnetization = [], []
    is_equilibrium = False
    MAX_LENGTH = 30
    auxiliary_M, auxiliary_M_avg = [], [] # auxiliary lists, used to determine whether the system has reached equilibrium
    reaction_time = IndexedHeap()
    initialize(reaction_time, lattice, N, H, J, beta)

    # Simulation loop
    i, j = 0, 0
    while time < MAX_TIME:
        tau = update(reaction_time, lattice, H, J, MAX_TIME)
        time = tau
        M = avg_magnetization(lattice)
        magnetization.append(M)
        times.append(time)
        if time > i * 50: 
            # Append the average magnetization to the auxiliary list for every 50 time steps
            auxiliary_M.append(M)
            if is_equilibrium:
                # If the system has reached equilibrium, calculate the average magnetization over the last 50 time steps
                if j >= 50:
                    break
                j += 1
            elif len(auxiliary_M) > MAX_LENGTH:
                # Check if the system has reached equilibrium; if the standard deviation of the average magnetization over the last 30 (MAX_LENGTH) time steps is less than 1e-2, the system has reached equilibrium
                avg_m = np.mean(auxiliary_M[-MAX_LENGTH:])
                auxiliary_M_avg.append(avg_m)
                std_avg_m = np.std(auxiliary_M_avg[-MAX_LENGTH:])
                if len(auxiliary_M_avg) > MAX_LENGTH and std_avg_m < 1e-2:
                    print(f'Equilibrium reached at time {time}')
                    is_equilibrium = True
            i += 1

    if is_equilibrium:
        return np.abs(np.mean(auxiliary_M[-50:]))
    else:
        return np.abs(np.mean(magnetization))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Simulate the 2D Ising model using the next reaction method.')
    parser.add_argument('--beta', type=float, default=0.3, help='Inverse temperature')
    parser.add_argument('--N', type=int, default=20, help='Size of the lattice (default: 20)')
    parser.add_argument('--J', type=float, default=1.0, help='Coupling constant (default: 1.0)')
    parser.add_argument('--H', type=float, default=0.0, help='External magnetic field (default: 0.0)')
    parser.add_argument('--MAX_TIME', type=float, default=5000, help='Maximum simulation time (default: 5000)')
    
    args = parser.parse_args()
    beta = args.beta
    N = args.N
    J = args.J
    H = args.H
    MAX_TIME = args.MAX_TIME
    
    fig, ax = plt.subplots()
    ax.axis("off")
    lattice = np.random.choice([-1, 1], size=(N, N))
    reaction_time = IndexedHeap()
    initialize(reaction_time, lattice, N, H, J, beta)

    im = ax.imshow(lattice, cmap="PiYG", interpolation="nearest")
    ax.set_title(f'Ising Model at beta = {beta:.2f}')
    time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes, color='white', fontsize=12, bbox=dict(facecolor='black', alpha=0.7))
    magnetization_text = ax.text(0.02, 0.88, '', transform=ax.transAxes, color='white', fontsize=12, bbox=dict(facecolor='black', alpha=0.7))

    def update_frame(frame):
        # Update the lattice
        global lattice, reaction_time, TIME
        new_TIME = update(reaction_time, lattice, H, J, MAX_TIME)
        im.set_data(lattice)
        time_text.set_text(f'Time: {TIME:.2f}')
        TIME = new_TIME
        magnetization = avg_magnetization(lattice)
        magnetization_text.set_text(f'Magnetization: {magnetization:.3f}')

    ani = FuncAnimation(fig, update_frame, frames=1000, interval=10) # Simulate the lattice
    plt.show()
    # writer = PillowWriter(fps=50)
    # import os
    # os.makedirs("ising-kmc/examples/next_reaction", exist_ok=True)
    # with writer.saving(fig, f"ising-kmc/examples/next_reaction/beta={beta}.gif", dpi=200):
    #     for k in range(MAX_TIME):
    #         update_frame(k)
    #         if k % 10 == 0:
    #             writer.grab_frame()