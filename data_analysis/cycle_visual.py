"""
cycle_visual is a CoolProp demo designed as a way to show an open source software that
leverages Thermodynamic Table data for cycle analysis purposes.
- The main function takes a desired net work output and generates
a basic Rankin Cycle to accomplish this goal.

Created for ME 401 @ Boise State Univerity
Author: Maxwell Hewes, Summer 2025
"""
import numpy as np
import matplotlib.pyplot as plt
from CoolProp.CoolProp import PropsSI

def rankine_cycle_analysis(desired_work_MW):
    """
    Analyze a simple Rankine cycle and generate PV and TS diagrams.

    Works on the assumption that mass flow rate is 1 kg / s    

    Parameters:
    desired_work_MW (float): Desired turbine work output in MW
    
    Returns:
    dict: Contains thermodynamic states, mass flow rate, and cycle efficiency
    """
    
    # Define cycle parameters (typical values for steam power plant)
    P_high = 8e6  # High pressure (Pa) - 80 bar
    P_low = 10e3  # Low pressure (Pa) - 0.1 bar (condenser pressure)
    T_superheat = 773.15  # Superheated steam temperature (K) - 500°C
    
    print(f"Rankine Cycle Analysis for {desired_work_MW} MW Turbine Work Output")
    print("="*60)

    """
    
    How coolProp works:

    PropsSI("Output_Property", "Input_Property_1", Value_1, "Input_Property_2", Value_2, "Fluid_Name")
    output = quantity that needs to be determined
    inputs 1 & 2 = independent states to find the output ("T" and "P", "P" and "H", "T" and "D")
    This is a very useful opensource library for thermal fluids research and simulation!
    
    ==The 'Q' below in the PropsSI function refers to Steam Quality not heat==
    """
    
    # State 1: Saturated liquid leaving condenser
    T1 = PropsSI('T', 'P', P_low, 'Q', 0, 'Water')  # Saturation temperature
    h1 = PropsSI('H', 'P', P_low, 'Q', 0, 'Water')  # Specific enthalpy
    s1 = PropsSI('S', 'P', P_low, 'Q', 0, 'Water')  # Specific entropy
    v1 = PropsSI('D', 'P', P_low, 'Q', 0, 'Water')**(-1)  # Specific volume
    
    # State 2: Compressed liquid leaving pump (isentropic compression)
    h2 = h1 + v1 * (P_high - P_low)  # Pump work approximation
    T2 = PropsSI('T', 'P', P_high, 'H', h2, 'Water')
    s2 = PropsSI('S', 'P', P_high, 'H', h2, 'Water')
    v2 = PropsSI('D', 'P', P_high, 'H', h2, 'Water')**(-1)
    
    # State 3: Superheated steam leaving boiler
    T3 = T_superheat
    h3 = PropsSI('H', 'P', P_high, 'T', T3, 'Water')
    s3 = PropsSI('S', 'P', P_high, 'T', T3, 'Water')
    v3 = PropsSI('D', 'P', P_high, 'T', T3, 'Water')**(-1)
    
    # State 4: Steam leaving turbine (isentropic expansion)
    s4 = s3  # Isentropic process
    h4 = PropsSI('H', 'P', P_low, 'S', s4, 'Water')
    T4 = PropsSI('T', 'P', P_low, 'S', s4, 'Water')
    v4 = PropsSI('D', 'P', P_low, 'S', s4, 'Water')**(-1)
    
    # Calculate specific work and heat transfer
    w_turbine = h3 - h4  # Turbine work (J/kg)
    w_pump = h2 - h1     # Pump work (J/kg)
    w_net = w_turbine - w_pump  # Net work (J/kg)
    q_in = h3 - h2       # Heat input (J/kg)
    q_out = h4 - h1      # Heat rejection (J/kg)
    
    # Calculate mass flow rate for desired work output
    desired_work_J_s = desired_work_MW * 1e6  # Convert MW to J/s
    mass_flow = desired_work_J_s / w_net  # kg/s
    
    # Calculate cycle efficiency
    efficiency = w_net / q_in * 100  # Percentage
    
    # Print thermodynamic states
    print("\nThermodynamic States:")
    print("-" * 80)
    print(f"{'State':<6} {'Description':<25} {'T(°C)':<8} {'P(bar)':<8} {'h(kJ/kg)':<10} {'s(kJ/kg·K)':<12} {'v(m³/kg)':<10}")
    print("-" * 80)
    
    states = [
        (1, "Saturated liquid (condenser)", T1-273.15, P_low/1e5, h1/1000, s1/1000, v1),
        (2, "Compressed liquid (pump)", T2-273.15, P_high/1e5, h2/1000, s2/1000, v2),
        (3, "Superheated steam (boiler)", T3-273.15, P_high/1e5, h3/1000, s3/1000, v3),
        (4, "Wet steam (turbine exit)", T4-273.15, P_low/1e5, h4/1000, s4/1000, v4)
    ]
    
    for state, desc, T, P, h, s, v in states:
        print(f"{state:<6} {desc:<25} {T:<8.1f} {P:<8.1f} {h:<10.1f} {s:<12.3f} {v:<10.4f}")
    
    print(f"\nCycle Performance:")
    print("-" * 40)
    print(f"Turbine work:     {w_turbine/1000:.1f} kJ/kg")
    print(f"Pump work:        {w_pump/1000:.1f} kJ/kg")
    print(f"Net work:         {w_net/1000:.1f} kJ/kg")
    print(f"Heat input:       {q_in/1000:.1f} kJ/kg")
    print(f"Cycle efficiency: {efficiency:.1f}%")
    print(f"Mass flow rate:   {mass_flow:.1f} kg/s")
    print(f"Steam flow rate:  {mass_flow*3.6:.0f} t/h")
    
    # Create arrays for plotting smooth curves
    n_points = 100
    
    # PV Diagram data
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Process 1-2: Pump (approximately vertical line due to low compressibility)
    v_12 = np.linspace(v1, v2, n_points)
    P_12 = np.linspace(P_low, P_high, n_points)
    
    # Process 2-3: Heating at constant pressure
    T_23 = np.linspace(T2, T3, n_points)
    v_23 = [PropsSI('D', 'P', P_high, 'T', T, 'Water')**(-1) for T in T_23]
    P_23 = np.full(n_points, P_high)
    
    # Process 3-4: Turbine (isentropic expansion)
    s_34 = np.full(n_points, s3)
    P_34 = np.linspace(P_high, P_low, n_points)
    v_34 = [PropsSI('D', 'P', P, 'S', s3, 'Water')**(-1) for P in P_34]
    
    # Process 4-1: Condenser (constant pressure)
    h_41 = np.linspace(h4, h1, n_points)
    v_41 = [PropsSI('D', 'P', P_low, 'H', h, 'Water')**(-1) for h in h_41]
    P_41 = np.full(n_points, P_low)
    
    # Plot PV diagram
    ax1.plot(v_12, P_12/1e5, 'b-', linewidth=2, label='1→2 Pump')
    ax1.plot(v_23, np.array(P_23)/1e5, 'r-', linewidth=2, label='2→3 Boiler')
    ax1.plot(v_34, np.array(P_34)/1e5, 'g-', linewidth=2, label='3→4 Turbine')
    ax1.plot(v_41, np.array(P_41)/1e5, 'orange', linewidth=2, label='4→1 Condenser')
    
    # Mark state points
    ax1.plot([v1, v2, v3, v4], [P_low/1e5, P_high/1e5, P_high/1e5, P_low/1e5], 
             'ko', markersize=8)
    
    # Add labels
    ax1.set_xlabel('Specific Volume (m³/kg)')
    ax1.set_ylabel('Pressure (bar)')
    ax1.set_title('P-V Diagram')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    v_min = min(v1, v2)
    v_max = max(v3, v4)
    margin = (v_max - v_min) * 0.1
    ax1.set_xlim(v_min - margin, v_max + margin)    
    ax1.set_yscale('log')  # Log scale better shows pressure range
    
    # TS Diagram data
    # Process 1-2: Pump
    s_12 = np.linspace(s1, s2, n_points)
    T_12 = [PropsSI('T', 'S', s, 'P', P_high if s > s1 + 0.7*(s2-s1) else P_low, 'Water') 
            for s in s_12]
    
    # Process 2-3: Heating
    T_23_ts = np.linspace(T2, T3, n_points)
    s_23_ts = [PropsSI('S', 'P', P_high, 'T', T, 'Water') for T in T_23_ts]
    
    # Process 3-4: Turbine
    s_34_ts = np.linspace(s3, s4, n_points)
    T_34_ts = [PropsSI('T', 'P', P, 'S', s, 'Water') for P in P_34]
    
    # Process 4-1: Condenser
    s_41_ts = np.linspace(s4, s1, n_points)
    T_41_ts = np.full(n_points, T1)
    
    # Plot TS diagram
    ax2.plot(np.array(s_12)/1000, np.array(T_12)-273.15, 'b-', linewidth=2, label='1→2 Pump')
    ax2.plot(np.array(s_23_ts)/1000, np.array(T_23_ts)-273.15, 'r-', linewidth=2, label='2→3 Boiler')
    ax2.plot(np.array(s_34_ts)/1000, np.array(T_34_ts)-273.15, 'g-', linewidth=2, label='3→4 Turbine')
    ax2.plot(np.array(s_41_ts)/1000, np.array(T_41_ts)-273.15, color = 'orange', linewidth=2, label='4→1 Condenser')
    ax2.plot([s3/1000, s4/1000], [T3-273.15, T4-273.15], 'g-', linewidth=2) # this is fixing a bug that needs to be resolved where T3->T4 is not plotting

    # Mark state points
    ax2.plot([s1/1000, s2/1000, s3/1000, s4/1000], 
             [T1-273.15, T2-273.15, T3-273.15, T4-273.15], 'ko', markersize=8)
    
    # Add labels
    ax2.set_xlabel('Specific Entropy (kJ/kg·K)')
    ax2.set_ylabel('Temperature (°C)') # alt 0176 for degree symbol
    ax2.set_title('T-S Diagram')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    plt.tight_layout()
    plt.show()
    
    # print("s_34_ts:", s_34_ts) #also for troubleshooting purposes
    # print("T_34_ts:", T_34_ts)

    
    return {
        'states': states,
        'mass_flow_kg_s': mass_flow,
        'mass_flow_t_h': mass_flow * 3.6,
        'efficiency_percent': efficiency,
        'net_work_kJ_kg': w_net/1000,
        'turbine_work_kJ_kg': w_turbine/1000,
        'pump_work_kJ_kg': w_pump/1000,
        'heat_input_kJ_kg': q_in/1000
    }

# Example usage
if __name__ == "__main__":
    # Analyze a 50 MW power plant
    results = rankine_cycle_analysis(1.2)
   
