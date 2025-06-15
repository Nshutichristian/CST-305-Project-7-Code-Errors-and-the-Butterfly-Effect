# CST-305 Project 7: Code Errors and Queueing Theory

## Project Description
This project demonstrates the butterfly effect in computational systems and analyzes various queueing theory models. The implementation includes:

1. **Part 1**: Interactive visualization of the Lorenz system showing how small changes in initial conditions lead to dramatically different outcomes (the butterfly effect)
2. **Part 2**: Comprehensive queueing theory analysis including:
   - FCFS queue analysis with hand calculations
   - M/M/1 gateway analysis
   - Scaling analysis of M/M/1 systems
   - Maximum arrival rate calculations
   - TCMP (Tightly Coupled Multiprocessor) queueing model visualization

## Installation
1. Clone this repository:
   ```bash
   git clone https://github.com/Nshutichristian/CST-305-Project-7-Code-Errors-and-the-Butterfly-Effect.git
   cd CST305-Project7
   ```
2. Install required Python packages:
   ```bash
   pip install numpy matplotlib scipy
   ```

## Usage
Run the main script to execute both parts of the project:

```bash
python Lorenz Queue Analysis.py
```

## Part 1: Lorenz System
- Interactive 3D visualization of the Lorenz attractor
- Slider controls to adjust system parameters (σ, ρ, β)
- Demonstration of the butterfly effect through perturbed initial conditions
- Animated visualization option (uncomment in `main()` to enable)

## Part 2: Queueing Theory Analysis
- **Problem 1**: FCFS queue analysis with detailed hand calculations and visualizations
- **Problem 2**: M/M/1 gateway analysis with buffer overflow probability calculations
- **Problem 3**: Scaling analysis of M/M/1 systems with mathematical derivations
- **Problem 4**: Maximum arrival rate calculation for given service constraints
- **Problem 5**: Visualization of TCMP queueing model

## File Structure
- `project7_main.py`: Main script containing both parts of the project
- `README.md`: This documentation file
- *(Optional)* `lorenz_system.py`: Separate module for Part 1 (if split into multiple files)
- *(Optional)* `queueing_analysis.py`: Separate module for Part 2 (if split into multiple files)

## Dependencies
- Python 3.6+
- NumPy
- Matplotlib
- SciPy

## Team Member
- **Christian Nshuti Manzi**: Implemented both parts of the project

## Key Features
- Interactive Lorenz system visualization with parameter controls
- Mathematical derivations for all queueing theory problems
- Detailed hand calculations shown alongside code implementations
- Comprehensive visualizations for each analysis
- Clear documentation of algorithms and approaches

## References
- Lorenz, E. N. (1963). "Deterministic nonperiodic flow". *Journal of the Atmospheric Sciences*.
- Kleinrock, L. (1975). *Queueing Systems, Volume 1: Theory*. Wiley.
- Matplotlib and SciPy documentation
- Grand Canyon University course materials (CST-305)

## Notes
- The animation in Part 1 may be resource-intensive – uncomment in `main()` only when needed
- All mathematical derivations are shown in the console output when running the analysis
- The project fulfills all requirements specified in the CST-305 Project 7 assignment
