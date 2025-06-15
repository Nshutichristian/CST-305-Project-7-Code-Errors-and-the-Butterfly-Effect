import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.widgets import Slider, Button
from matplotlib.animation import FuncAnimation
from scipy.integrate import odeint
import random

"""
# Lorenz System and Queueing Theory Simulation for CST-305 Project 7
# Programmer: Christian Nshuti Manzi
# Grand Canyon University | Summer 2025
# This script includes:
# - Part 1: Interactive Lorenz System with Butterfly Effect
# - Part 2: Queueing Theory Analysis with All Required Problems


Part 1: Interactive Lorenz System with Butterfly Effect Demonstration
Part 2: Comprehensive Queueing Theory Analysis with Mathematical Derivations
"""

# ================== PART 1: LORENZ SYSTEM WITH BUTTERFLY EFFECT ==================
class LorenzSystem:
    def __init__(self):
        # Standard Lorenz parameters
        self.sigma = 10    # Prandtl number (σ)
        self.rho = 28      # Rayleigh number (ρ)
        self.beta = 8/3    # Dimensionless parameter (β)
        
        # Simulation parameters
        self.dt = 0.01
        self.steps = 5000
        self.time = np.arange(0, self.steps * self.dt, self.dt)
        
        # Initial conditions (two slightly different states for butterfly effect)
        self.initial_state1 = [1.0, 1.0, 1.0]
        self.initial_state2 = [1.01, 1.0, 1.0]  # 1% difference in x
        
        # Solution trajectories
        self.solution1 = None
        self.solution2 = None
        
        # Set up figure and interactive controls
        self.setup_interactive_plot()
        
    def lorenz_equations(self, state, t):
        """The Lorenz system equations"""
        x, y, z = state
        dxdt = self.sigma * (y - x)
        dydt = x * (self.rho - z) - y
        dzdt = x * y - self.beta * z
        return [dxdt, dydt, dzdt]
    
    def solve_system(self):
        """Solve the Lorenz system for both initial conditions"""
        self.solution1 = odeint(self.lorenz_equations, self.initial_state1, self.time)
        self.solution2 = odeint(self.lorenz_equations, self.initial_state2, self.time)
    
    def setup_interactive_plot(self):
        """Create interactive plot with parameter controls"""
        self.fig = plt.figure(figsize=(15, 10))
        plt.subplots_adjust(bottom=0.3)
        
        # 3D plot for Lorenz attractor
        self.ax1 = self.fig.add_subplot(121, projection='3d')
        self.ax1.set_title('Lorenz Attractor')
        
        # 2D plot for butterfly effect demonstration
        self.ax2 = self.fig.add_subplot(122)
        self.ax2.set_title('Butterfly Effect Demonstration')
        self.ax2.set_xlabel('Time')
        self.ax2.set_ylabel('X Difference')
        
        # Parameter sliders
        ax_sigma = plt.axes([0.25, 0.2, 0.65, 0.03])
        ax_rho = plt.axes([0.25, 0.15, 0.65, 0.03])
        ax_beta = plt.axes([0.25, 0.1, 0.65, 0.03])
        
        self.slider_sigma = Slider(ax_sigma, 'σ (Sigma)', 1, 20, valinit=self.sigma)
        self.slider_rho = Slider(ax_rho, 'ρ (Rho)', 1, 50, valinit=self.rho)
        self.slider_beta = Slider(ax_beta, 'β (Beta)', 0.1, 5, valinit=self.beta)
        
        # Update function for sliders
        def update(val):
            self.sigma = self.slider_sigma.val
            self.rho = self.slider_rho.val
            self.beta = self.slider_beta.val
            self.solve_system()
            self.update_plots()
        
        self.slider_sigma.on_changed(update)
        self.slider_rho.on_changed(update)
        self.slider_beta.on_changed(update)
        
        # Reset button
        reset_ax = plt.axes([0.8, 0.025, 0.1, 0.04])
        self.reset_button = Button(reset_ax, 'Reset', color='lightgoldenrodyellow')
        self.reset_button.on_clicked(self.reset_parameters)
        
        # Solve and plot initial state
        self.solve_system()
        self.update_plots()
        
        # Add parameter explanations
        param_text = (
            "Lorenz System Parameters:\n"
            "σ (Sigma): Prandtl number (viscous/thermal diffusivity ratio)\n"
            "ρ (Rho): Rayleigh number (temperature difference driving convection)\n"
            "β (Beta): Dimensionless parameter (system geometry)"
        )
        plt.figtext(0.1, 0.02, param_text, fontsize=10)
    
    def update_plots(self):
        """Update both plots with current solutions"""
        # Clear and update 3D plot
        self.ax1.clear()
        self.ax1.plot(self.solution1[:, 0], self.solution1[:, 1], self.solution1[:, 2], 
                     lw=0.5, label='Original')
        self.ax1.plot(self.solution2[:, 0], self.solution2[:, 1], self.solution2[:, 2], 
                     lw=0.5, color='red', label='Perturbed')
        self.ax1.set_xlabel('X')
        self.ax1.set_ylabel('Y')
        self.ax1.set_zlabel('Z')
        self.ax1.set_title(f'Lorenz Attractor (σ={self.sigma:.1f}, ρ={self.rho:.1f}, β={self.beta:.2f})')
        self.ax1.legend()
        
        # Update butterfly effect plot
        self.ax2.clear()
        x_diff = np.abs(self.solution1[:, 0] - self.solution2[:, 0])
        self.ax2.semilogy(self.time, x_diff, color='purple')
        self.ax2.set_title('Butterfly Effect: X Difference Over Time')
        self.ax2.set_xlabel('Time')
        self.ax2.set_ylabel('Absolute X Difference (log scale)')
        self.ax2.grid(True)
        
        self.fig.canvas.draw_idle()
    
    def reset_parameters(self, event):
        """Reset parameters to default values"""
        self.slider_sigma.reset()
        self.slider_rho.reset()
        self.slider_beta.reset()
    
    def animate_lorenz(self):
        """Create an animated visualization of the Lorenz system"""
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Initialize line objects
        line1, = ax.plot([], [], [], lw=0.5, label='Original')
        line2, = ax.plot([], [], [], lw=0.5, color='red', label='Perturbed')
        ax.set_xlim(-20, 20)
        ax.set_ylim(-30, 30)
        ax.set_zlim(0, 50)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('Animated Lorenz Attractor')
        ax.legend()
        
        def init():
            line1.set_data([], [])
            line1.set_3d_properties([])
            line2.set_data([], [])
            line2.set_3d_properties([])
            return line1, line2
        
        def animate(i):
            # Animate first 1000 steps for performance
            idx = min(i, 1000)
            line1.set_data(self.solution1[:idx, 0], self.solution1[:idx, 1])
            line1.set_3d_properties(self.solution1[:idx, 2])
            line2.set_data(self.solution2[:idx, 0], self.solution2[:idx, 1])
            line2.set_3d_properties(self.solution2[:idx, 2])
            return line1, line2
        
        ani = FuncAnimation(fig, animate, frames=200, init_func=init,
                            blit=True, interval=50, repeat=True)
        plt.show()
        return ani

# ================== PART 2: QUEUEING THEORY ANALYSIS ==================
class QueueingAnalysis:
    def __init__(self):
        # Problem 1 data
        self.arrival_times = [0, 2.5, 3.6, 7.0, 7.2, 8.7, 12.1, 13.2, 13.6, 14.1]
        self.service_durations = [2.0, 0.7, 1.3, 0.2, 1.1, 0.6, 1.2, 0.9, 0.4, 1.3]
        
    def problem1_fcfs_queue(self):
        """Problem 1: FCFS Queue Analysis with Correct Calculations"""
        print("\n=== Problem 1: FCFS Queue Analysis ===")
        
        # Calculate service start and end times
        service_start = []
        service_end = []
        for i in range(len(self.arrival_times)):
            if i == 0:
                start = self.arrival_times[i]
            else:
                start = max(self.arrival_times[i], service_end[i-1])
            end = start + self.service_durations[i]
            service_start.append(start)
            service_end.append(end)
        
        # Calculate time in queue for each customer
        queue_times = [start - arr for start, arr in zip(service_start, self.arrival_times)]
        
        # Calculate number in system and queue at arrival times
        num_in_system = []
        num_in_queue = []
        for i in range(len(self.arrival_times)):
            if i == 0:
                num_sys = 0
                num_q = 0
            else:
                # Number in system is count of customers who have arrived but not left
                num_sys = sum(1 for j in range(i) if service_end[j] > self.arrival_times[i])
                # Number in queue is count of customers waiting (arrived but service not started)
                num_q = sum(1 for j in range(i) if service_start[j] > self.arrival_times[i])
            num_in_system.append(num_sys)
            num_in_queue.append(num_q)
        
        # Calculate Lq (time average number in queue)
        events = sorted(set(self.arrival_times + service_start + service_end))
        q_times = []
        q_counts = []
        current_q = 0
        
        for i in range(len(events)-1):
            t_start = events[i]
            t_end = events[i+1]
            
            # Check if this event time is an arrival or departure
            if t_start in self.arrival_times:
                current_q += 1
            if t_start in service_start:
                current_q -= 1
            
            q_times.append(t_end - t_start)
            q_counts.append(current_q)
        
        total_time = 15.27  # Last customer exits at 15.27
        Lq = sum(t * q for t, q in zip(q_times, q_counts)) / total_time
        
        # Lq_A is average number in queue seen by arrivals
        Lq_A = np.mean(num_in_queue)
        
        # Print hand calculations
        print("\nHand Calculations:")
        print("Customer | Arrival | Service Start | Service End | Time in Queue | Num in System | Num in Queue")
        for i in range(len(self.arrival_times)):
            print(f"{i+1:8d} | {self.arrival_times[i]:7.2f} | {service_start[i]:14.2f} | {service_end[i]:11.2f} | "
                  f"{queue_times[i]:13.2f} | {num_in_system[i]:14d} | {num_in_queue[i]:13d}")
        
        print(f"\nLq (Time avg in queue) = {sum(t*q for t,q in zip(q_times,q_counts))} / {total_time} = {Lq:.4f}")
        print(f"Lq_A (Avg seen by arrivals) = Mean of {num_in_queue} = {Lq_A:.4f}")
        
        # Plotting
        plt.figure(figsize=(15, 10))
        
        # Plot 1: Arrival time vs Service start time
        plt.subplot(2, 3, 1)
        plt.plot(self.arrival_times, service_start, 'bo-')
        plt.xlabel('Arrival Time')
        plt.ylabel('Service Start Time')
        plt.title('Service Start vs Arrival Time')
        
        # Plot 2: Arrival time vs Exit time
        plt.subplot(2, 3, 2)
        plt.plot(self.arrival_times, service_end, 'ro-')
        plt.xlabel('Arrival Time')
        plt.ylabel('Exit Time')
        plt.title('Exit Time vs Arrival Time')
        
        # Plot 3: Arrival time vs Time in queue
        plt.subplot(2, 3, 3)
        plt.bar(range(len(self.arrival_times)), queue_times)
        plt.xticks(range(len(self.arrival_times)), [f"C{i+1}" for i in range(len(self.arrival_times))])
        plt.xlabel('Customer')
        plt.ylabel('Time in Queue')
        plt.title('Queue Time by Customer')
        
        # Plot 4: Arrival time vs Number in system
        plt.subplot(2, 3, 4)
        plt.step(self.arrival_times, num_in_system, where='post')
        plt.xlabel('Arrival Time')
        plt.ylabel('Number in System')
        plt.title('System Occupancy at Arrival')
        
        # Plot 5: Arrival time vs Number in queue
        plt.subplot(2, 3, 5)
        plt.step(self.arrival_times, num_in_queue, where='post')
        plt.xlabel('Arrival Time')
        plt.ylabel('Number in Queue')
        plt.title('Queue Length at Arrival')
        
        plt.tight_layout()
        plt.show()
        
        return Lq, Lq_A
    
    def problem2_mm1_gateway(self):
        """Problem 2: M/M/1 Gateway Analysis"""
        print("\n=== Problem 2: M/M/1 Gateway Analysis ===")
        
        λ = 125  # packets per second (arrival rate)
        service_time = 0.002  # 2 ms per packet
        μ = 1/service_time  # service rate (500 packets/sec)
        
        ρ = λ / μ  # utilization factor
        
        print("\nMathematical Analysis:")
        print(f"Arrival rate (λ) = {λ} packets/sec")
        print(f"Service rate (μ) = 1/{service_time} = {μ} packets/sec")
        print(f"Utilization (ρ) = λ/μ = {ρ:.4f}")
        
        # Probability of buffer overflow with 12 buffers
        # P(N > 12) = ρ^(13) for M/M/1
        P_overflow = ρ ** 13
        print(f"\nP(buffer overflow with 12 buffers) = P(N > 12) = ρ^13 = {P_overflow:.4e}")
        
        # Buffers needed for loss < 1 per million
        # Find smallest B where P(N > B) < 1e-6 => ρ^(B+1) < 1e-6
        B = np.ceil(np.log(1e-6)/np.log(ρ)) - 1
        print(f"\nFor P(loss) < 1e-6, need P(N > B) < 1e-6")
        print(f"Solve ρ^(B+1) < 1e-6 => B > log(1e-6)/log(ρ) - 1")
        print(f"B > {np.log(1e-6)/np.log(ρ)} - 1 => B ≥ {B:.0f} buffers")
        
        return ρ, P_overflow, B
    
    def problem3_scaling_mm1(self):
        """Problem 3: Scaling M/M/1 System"""
        print("\n=== Problem 3: Scaling M/M/1 System ===")
        
        # Initial parameters
        λ = 3  # jobs/sec
        μ = 4  # jobs/sec
        k_values = np.linspace(1, 5, 20)  # Scaling factors
        
        print("\nMathematical Analysis:")
        print("When scaling both λ and μ by factor k:")
        print("a) Utilization ρ = (kλ)/(kμ) = λ/μ (unchanged)")
        print("b) Throughput X = min(kλ, kμ) = k*min(λ,μ) (scales by k)")
        print("c) Mean number in system E[N] = ρ/(1-ρ) (unchanged)")
        print("d) Mean time in system E[T] = 1/(kμ - kλ) = (1/k)*1/(μ-λ) (reduces by factor k)")
        
        # Calculate metrics for each scaling factor
        utilizations = []
        throughputs = []
        mean_numbers = []
        mean_times = []
        
        for k in k_values:
            λ_scaled = λ * k
            μ_scaled = μ * k
            ρ = λ_scaled / μ_scaled
            
            utilizations.append(ρ)
            throughputs.append(min(λ_scaled, μ_scaled))
            mean_numbers.append(ρ/(1-ρ) if ρ < 1 else float('inf'))
            mean_times.append(1/(μ_scaled - λ_scaled) if ρ < 1 else float('inf'))
        
        # Plotting
        plt.figure(figsize=(12, 8))
        
        plt.subplot(2, 2, 1)
        plt.plot(k_values, utilizations)
        plt.title('Utilization (ρ) vs Scaling Factor')
        plt.xlabel('Scaling Factor (k)')
        plt.ylabel('Utilization')
        plt.axhline(y=1, color='r', linestyle='--')
        
        plt.subplot(2, 2, 2)
        plt.plot(k_values, throughputs)
        plt.title('Throughput (X) vs Scaling Factor')
        plt.xlabel('Scaling Factor (k)')
        plt.ylabel('Throughput')
        
        plt.subplot(2, 2, 3)
        plt.plot(k_values, mean_numbers)
        plt.title('Mean Number in System (E[N]) vs Scaling Factor')
        plt.xlabel('Scaling Factor (k)')
        plt.ylabel('E[N]')
        
        plt.subplot(2, 2, 4)
        plt.plot(k_values, mean_times)
        plt.title('Mean Time in System (E[T]) vs Scaling Factor')
        plt.xlabel('Scaling Factor (k)')
        plt.ylabel('E[T]')
        
        plt.tight_layout()
        plt.show()
        
        return utilizations, throughputs, mean_numbers, mean_times
    
    def problem4_max_arrival_rate(self):
        """Problem 4: Maximum Arrival Rate for M/M/1"""
        print("\n=== Problem 4: Maximum Arrival Rate ===")
        
        service_demand = 3  # minutes per job
        μ = 1/service_demand  # jobs per minute
        max_wait = 6  # minutes (E[Tq] < 6)
        
        print("\nGiven:")
        print(f"Mean job size (service demand) = 3 minutes => μ = {μ:.4f} jobs/min")
        print(f"Mean waiting time E[Tq] must be < {max_wait} minutes")
        
        print("\nMathematical Analysis:")
        print("For M/M/1 queue:")
        print("E[Tq] = ρ/(μ(1-ρ)) < 6")
        print("Where ρ = λ/μ")
        print("Solving for ρ:")
        print("ρ < 6μ(1-ρ)")
        print("ρ < 6μ - 6μρ")
        print("ρ + 6μρ < 6μ")
        print("ρ(1 + 6μ) < 6μ")
        print("ρ < 6μ/(1 + 6μ)")
        
        max_ρ = (6 * μ) / (1 + 6 * μ)
        max_λ = max_ρ * μ
        
        print(f"\nMaximum ρ = {6*μ}/(1 + {6*μ}) = {max_ρ:.4f}")
        print(f"Maximum λ = ρ*μ = {max_ρ:.4f} * {μ:.4f} = {max_λ:.4f} jobs/min")
        print(f"= {max_λ * 60:.2f} jobs/hour")
        
        return max_λ
    
    def problem5_tcmp_model(self):
        """Problem 5: TCMP Queueing Model"""
        print("\n=== Problem 5: TCMP Queueing Model ===")
        
        print("\nQueueing Model Description:")
        print("1. Each Processing Element (PE) has:")
        print("   - Task pool with sleep time μ(i,0)^-1")
        print("   - CPU with service rate μ(i,1)")
        print("   - Bus Interface Unit (BIU) with service rate μ(i,2)")
        print("2. All BIUs share a single bus (contention point)")
        print("3. Global shared memory accessible via bus")
        print("4. Branching probabilities:")
        print("   - p(i,1): Task → CPU")
        print("   - p(i,2): CPU → BIU")
        print("   - p(i,3): BIU → CPU (after memory access)")
        
        # Visualization
        fig, ax = plt.subplots(figsize=(14, 7))
        
        # Draw 3 Processing Elements as example
        for i in range(3):
            # Task pool
            ax.add_patch(plt.Rectangle((i*5, 3), 2, 1.5, fc='lightblue', ec='black'))
            plt.text(i*5+1, 3.75, f'Task Pool PE{i}\nSleep time: 1/μ({i},0)', ha='center')
            
            # CPU
            ax.add_patch(plt.Rectangle((i*5, 1), 2, 1.5, fc='lightgreen', ec='black'))
            plt.text(i*5+1, 1.75, f'CPU PE{i}\nService rate: μ({i},1)', ha='center')
            
            # BIU
            ax.add_patch(plt.Rectangle((i*5, -1), 2, 1.5, fc='lightcoral', ec='black'))
            plt.text(i*5+1, -0.25, f'BIU PE{i}\nService rate: μ({i},2)', ha='center')
            
            # Arrows for task flow
            plt.arrow(i*5+1, 2.9, 0, -0.7, head_width=0.3, color='black')
            plt.text(i*5+1.5, 2.4, f'p({i},1)', fontsize=10)
            
            plt.arrow(i*5+1, 0.9, 0, -0.7, head_width=0.3, color='black')
            plt.text(i*5+1.5, 0.4, f'p({i},2)', fontsize=10)
            
            plt.arrow(i*5+3, 0.5, 1, 0, head_width=0.2, color='black')
            plt.arrow(i*5+3, -0.5, 1, 0, head_width=0.2, color='black')
        
        # Shared bus
        plt.plot([0, 16], [-2, -2], 'k-', linewidth=3)
        plt.text(8, -2.5, 'Shared Bus', ha='center', fontsize=12)
        
        # Global memory
        ax.add_patch(plt.Rectangle((13, -3.5), 3, 2, fc='purple', ec='black'))
        plt.text(14.5, -2.5, 'Global Shared Memory', ha='center', color='white')
        
        # Return arrows from memory
        plt.arrow(16, -2, -1, 2, head_width=0.2, color='black')
        plt.text(15, -0.5, f'p(i,3)', fontsize=10)
        
        plt.xlim(-1, 17)
        plt.ylim(-4, 5)
        plt.axis('off')
        plt.title('Single Bus Tightly Coupled Multiprocessor (SBTCMP) Queueing Model')
        plt.show()

# ================== MAIN EXECUTION ==================
def main():
    print("CST-305 Project 7: Code Errors and Queueing Theory\n")
    
    # Part 1: Lorenz System with Butterfly Effect
    print("=== PART 1: LORENZ SYSTEM AND BUTTERFLY EFFECT ===")
    lorenz = LorenzSystem()
    
    # Uncomment to see animation (may slow down execution)
    # print("\nGenerating animation...")
    # lorenz.animate_lorenz()
    
    # Part 2: Queueing Theory Analysis
    print("\n=== PART 2: QUEUEING THEORY ANALYSIS ===")
    qa = QueueingAnalysis()
    
    print("\nProblem 1: FCFS Queue Analysis")
    Lq, Lq_A = qa.problem1_fcfs_queue()
    
    print("\nProblem 2: M/M/1 Gateway Analysis")
    ρ, P_overflow, B = qa.problem2_mm1_gateway()
    
    print("\nProblem 3: Scaling M/M/1 System")
    qa.problem3_scaling_mm1()
    
    print("\nProblem 4: Maximum Arrival Rate")
    max_λ = qa.problem4_max_arrival_rate()
    
    print("\nProblem 5: TCMP Queueing Model")
    qa.problem5_tcmp_model()

if __name__ == "__main__":
    main()