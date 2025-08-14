import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')
from matplotlib.dates import DateFormatter, DayLocator

class HybridEnergyOptimizer:
    """    
    This model demonstrates the optimization of Battery Energy Storage Systems (BESS)
    in hybrid renewable energy systems with a thermodynamic cycle backup power.
    
    System Components:
    - PV Solar Generation
    - Wind Generation  
    - Battery Energy Storage System (2 MW fixed power, variable energy capacity)
    - Rankine Cycle Backup (biomass/hydrogen/methane fueled)
        
    Optimization Goal: Minimize loss of load probability while considering costs
    """
    
    def __init__(self):
        # System parameters
        self.bess_power_rating = 4.0  # MW (fixed)
        self.bess_efficiency = 0.85    # Round-trip efficiency
        self.bess_c_rate = 1.0         # Maximum C-rate (1C means full 
                                        #charge/discharge in 1 hour)
        self.reliability_target = 0.95  # 95% reliability requirement
        
        # Rankine cycle parameters
        self.rankine_efficiency = 0.30  # Thermal to electrical efficiency
        self.fuel_cost_per_mwh = 80.0   # $/MWh fuel cost
        self.rankine_max_power = 4.0    # MW maximum Rankine power
        
        # Cost parameters (simplified for education)
        self.bess_cost_per_mwh = 150000  # $/MWh battery cost
        self.discount_rate = 0.08        # Annual discount rate
        self.system_lifetime = 20        # Years
        
        # Time parameters
        self.hours_per_interval = 12     # 12-hour intervals
        self.intervals_per_day = 24 // self.hours_per_interval
        self.days_per_year = 365
        self.total_intervals = self.days_per_year * self.intervals_per_day
        
        # Data storage
        self.time_series = None
        self.results = None
    
    def visualize_data_patterns(self):
        """
        Create visualizations to help students understand the data before optimization.
        This builds intuition about renewable variability and load patterns.
        """
        if self.time_series is None:
            print("No data 'time_series' supplied.")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Hybrid Energy System Data Analysis', fontsize=16, fontweight='bold')
        
        # 1. Time series plot (first 30 days)
        subset_data = self.time_series.head(60)  # First 30 days (60 intervals)
        axes[0,0].plot(subset_data['datetime'], subset_data['pv_generation'], 
                      label='PV Generation', color='orange', linewidth=2)
        axes[0,0].plot(subset_data['datetime'], subset_data['wind_generation'], 
                      label='Wind Generation', color='blue', linewidth=2)
        axes[0,0].plot(subset_data['datetime'], subset_data['demand'], 
                      label='Demand', color='red', linewidth=2)
        axes[0,0].set_title('Generation and Demand')
        axes[0,0].set_ylabel('Power (MW)')
        axes[0,0].legend()
        axes[0,0].grid(True, alpha=0.3)
        
        #save figure for report

        plt.figure(figsize=(10, 6))
        plt.plot(subset_data['datetime'], subset_data['pv_generation'],
                label='PV Generation', color='orange', linewidth=2)
        plt.plot(subset_data['datetime'], subset_data['wind_generation'],
                label='Wind Generation', color='blue', linewidth=2)
        plt.plot(subset_data['datetime'], subset_data['demand'],
                label='Demand', color='red', linewidth=2)
        plt.title('Generation and Demand')
        plt.ylabel('Power (MW)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig('gen_demand1.png', dpi=300, bbox_inches='tight')
        plt.close()
                
        # 2. Load Duration Curve
        sorted_demand = np.sort(self.time_series['demand'])[::-1]
        sorted_renewable = np.sort(self.time_series['total_renewable'])[::-1]
        percentiles = np.arange(0, 100, 100/len(sorted_demand))
        
        axes[0,1].plot(percentiles, sorted_demand, label='Demand', color='red', linewidth=2)
        axes[0,1].plot(percentiles, sorted_renewable, label='Total Renewable', color='green', linewidth=2)
        axes[0,1].set_title('Load Duration Curve')
        axes[0,1].set_xlabel('Percentage of Time (%)')
        axes[0,1].set_ylabel('Power (MW)')
        axes[0,1].legend()
        axes[0,1].grid(True, alpha=0.3)

        #save figure for report

        plt.figure(figsize=(10, 6))
        plt.plot(percentiles, sorted_demand, label='Demand', color='red', linewidth=2)
        plt.plot(percentiles, sorted_renewable, label='Total Renewable', color='green', linewidth=2)
        plt.title('Load Duration Curve')
        plt.xlabel('Percentage of Time (%)')
        plt.ylabel('Power (MW)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig('LDC.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. Net Load Analysis
        axes[1,0].hist(self.time_series['net_load'], bins=50, alpha=0.7, color='purple', edgecolor='black')
        axes[1,0].axvline(0, color='red', linestyle='--', linewidth=2, label='Zero Net Load')
        axes[1,0].set_title('Net Load Distribution\n(Demand - Renewable Generation)')
        axes[1,0].set_xlabel('Net Load (MW)')
        axes[1,0].set_ylabel('Frequency')
        axes[1,0].legend()
        axes[1,0].grid(True, alpha=0.3)

        #save figure for report

        plt.figure(figsize=(10, 6))
        plt.hist(self.time_series['net_load'], bins=50, alpha=0.7, color='purple', edgecolor='black')
        plt.axvline(0, color='red', linestyle='--', linewidth=2, label='Zero Net Load')
        plt.title('Net Load Distribution\n(Demand - Renewable Generation)')
        plt.xlabel('Net Load (MW)')
        plt.ylabel('Frequency')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig('NLD.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 4. Monthly Energy Balance
        monthly_data = self.time_series.copy()
        monthly_data['month'] = monthly_data['datetime'].dt.month
        monthly_summary = monthly_data.groupby('month').agg({
            'total_renewable': 'sum',
            'demand': 'sum',
            'net_load': 'sum'
        }) * self.hours_per_interval  # Convert to MWh
        
        months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        
        x = np.arange(len(months))
        width = 0.35
        
        axes[1,1].bar(x - width/2, monthly_summary['total_renewable'], width, 
                     label='Renewable Generation', color='green', alpha=0.7)
        axes[1,1].bar(x + width/2, monthly_summary['demand'], width, 
                     label='Demand', color='red', alpha=0.7)
        axes[1,1].set_title('Monthly Energy Balance')
        axes[1,1].set_xlabel('Month')
        axes[1,1].set_ylabel('Energy (MWh)')
        axes[1,1].set_xticks(x)
        axes[1,1].set_xticklabels(months)
        axes[1,1].legend()
        axes[1,1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()

        #save figure for report

        plt.figure(figsize=(10, 6))
        plt.bar(x - width/2, monthly_summary['total_renewable'], width, 
                     label='Renewable Generation', color='green', alpha=0.7)
        plt.bar(x + width/2, monthly_summary['demand'], width, 
                     label='Demand', color='red', alpha=0.7)
        plt.xticks(x,months)
        plt.title('Monthly Energy Balance')
        plt.xlabel('Month')
        plt.ylabel('Energy (MWh)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig('energy_balance.png', dpi=300, bbox_inches='tight')
        plt.close()      
        
        # Print key statistics
        print("\n" + "="*60)
        print("SYSTEM DATA ANALYSIS SUMMARY")
        print("="*60)
        print(f"Peak Demand: {self.time_series['demand'].max():.1f} MW")
        print(f"Average Demand: {self.time_series['demand'].mean():.1f} MW")
        print(f"Peak Renewable Generation: {self.time_series['total_renewable'].max():.1f} MW")
        print(f"Average Renewable Generation: {self.time_series['total_renewable'].mean():.1f} MW")
        print(f"Maximum Net Load: {self.time_series['net_load'].max():.1f} MW")
        print(f"Minimum Net Load: {self.time_series['net_load'].min():.1f} MW")
        print(f"Periods with Excess Renewable: {(self.time_series['net_load'] < 0).sum()} / {len(self.time_series)} intervals")
        print(f"Renewable Penetration: {(self.time_series['total_renewable'].sum() / self.time_series['demand'].sum() * 100):.1f}%")
        
    def simulate_bess_operation(self, bess_capacity_mwh):
        """
        Simulate BESS operation using rule-based dispatch strategy.
        This is the core simulation that students will learn to optimize.
        
        Args:
            bess_capacity_mwh: Battery energy capacity in MWh
            
        Returns:
            Dictionary with simulation results
        """
        if self.time_series is None:
            raise ValueError("Please load data first")
        
        # Initialize arrays for simulation
        n_intervals = len(self.time_series)
        bess_soc = np.zeros(n_intervals)  # State of charge (MWh)
        bess_power = np.zeros(n_intervals)  # Power flow (MW, positive = discharge)
        rankine_power = np.zeros(n_intervals)  # Rankine cycle power (MW)
        load_not_served = np.zeros(n_intervals)  # Unmet demand (MW)
        
        # Initial battery state (start at 50% charge)
        current_soc = bess_capacity_mwh * 0.5 
        
        # Simulation loop
        for i in range(n_intervals):
            renewable_power = self.time_series.iloc[i]['total_renewable']
            demand = self.time_series.iloc[i]['demand']
            net_load = demand - renewable_power
            bess_discharge = 0
            # Determine required power from storage and backup
            if net_load > 0: 
                if current_soc > bess_capacity_mwh*0.2:  # Need additional power
                    # Try to discharge battery first
                    bess_discharge = min(
                        self.bess_power_rating,
                        net_load
                    ) 
                    net_load -= bess_discharge
                    bess_power[i] = bess_discharge
                # Use Rankine cycle for remaining load
                if net_load > 0:                                                        
                    net_load -= self.rankine_max_power 
                    if net_load < 0:
                        bess_charge = -net_load
                        energy_charged = bess_charge * self.hours_per_interval 
                        current_soc += energy_charged * self.bess_efficiency
                        bess_power[i] = -bess_charge  # Negative for charging

                    rankine_power[i] = self.rankine_max_power
                    
                
                # Record any unmet demand
                if net_load > 0:
                    load_not_served[i] = net_load
                
                # Update battery state
                current_soc -= bess_discharge * self.hours_per_interval 
                #bess_power[i] = bess_discharge
                
            else:  # Excess renewable power available
                # Try to charge battery
                excess_power = -net_load
                max_charge_power = min(
                    self.bess_power_rating,
                    (bess_capacity_mwh - current_soc) / self.hours_per_interval * self.bess_c_rate,
                    excess_power
                )
                
                bess_charge = max_charge_power
                energy_charged = bess_charge * self.hours_per_interval
                current_soc += energy_charged
                bess_power[i] = -bess_charge  # Negative for charging
            
            # Ensure SOC stays within bounds
            current_soc = np.clip(current_soc, 0, bess_capacity_mwh)
            bess_soc[i] = current_soc
        
        # Calculate performance metrics
        total_energy_not_served = np.sum(load_not_served) * self.hours_per_interval
        total_demand_energy = np.sum(self.time_series['demand']) * self.hours_per_interval
        reliability = 1 - (total_energy_not_served / total_demand_energy)
        
        # Loss of load events (continuous periods with unmet demand)
        lol_events = []
        current_event_duration = 0
        
        for i in range(n_intervals):
            if load_not_served[i] > 0:
                current_event_duration += 1
            else:
                if current_event_duration > 0:
                    lol_events.append(current_event_duration)
                    current_event_duration = 0
        
        # Don't forget the last event if it extends to the end
        if current_event_duration > 0:
            lol_events.append(current_event_duration)
        
        # Calculate loss of load penalty (squared duration)
        lol_penalty = sum(duration**2 for duration in lol_events)
        
        # Calculate costs
        fuel_cost = np.sum(rankine_power) * self.hours_per_interval * self.fuel_cost_per_mwh
        bess_capital_cost = bess_capacity_mwh * self.bess_cost_per_mwh
        
        return {
            'bess_soc': bess_soc,
            'bess_power': bess_power,
            'rankine_power': rankine_power,
            'load_not_served': load_not_served,
            'reliability': reliability,
            'lol_events': lol_events,
            'lol_penalty': lol_penalty,
            'fuel_cost': fuel_cost,
            'bess_capital_cost': bess_capital_cost,
            'total_energy_not_served': total_energy_not_served,
            'bess_capacity_mwh': bess_capacity_mwh
        }
    
    def objective_function(self, bess_capacity_mwh):
        """
        Objective function for optimization.
        Combines reliability penalty with economic costs.
        
        Args:
            bess_capacity_mwh: Battery capacity to evaluate (array for scipy.optimize)
            
        Returns:
            Total cost penalty (to be minimized)
        """
        # Extract scalar value (scipy.optimize passes array)
        if isinstance(bess_capacity_mwh, np.ndarray):
            capacity = bess_capacity_mwh[0]
        else:
            capacity = bess_capacity_mwh
        
        # Ensure positive capacity
        if capacity <= 0:
            return 100  # Large penalty for invalid solutions
        
        # Run simulation
        try:
            results = self.simulate_bess_operation(capacity)
        except Exception as e:
            print(f"Simulation failed at capacity {capacity}: {e}")
            return 100
        
        # Calculate penalty components
        reliability_penalty = 10
        if results['reliability'] < self.reliability_target:
            # Squared penalty for reliability shortfall
            reliability_shortfall = (self.reliability_target - results['reliability'])
            reliability_penalty = 100 * reliability_shortfall**2
        
        # Loss of load penalty (squared duration)
        lol_penalty = results['lol_penalty'] *1000  # Scale factor
        
        # Economic costs
        capital_cost = results['bess_capital_cost'] *0.1
        fuel_cost = results['fuel_cost'] * 100
        
        # Total objective (to minimize)
        total_cost = reliability_penalty + lol_penalty + capital_cost + fuel_cost
        
        return total_cost / 1e4 #scale cost to minimize gradient
    
    def optimize_bess_capacity(self, initial_guess=40.0, max_capacity=100.0):
        """
        Optimize BESS capacity using scipy.optimize.
        This demonstrates the power of automated optimization.
        
        Args:
            initial_guess: Starting point for optimization (MWh)
            max_capacity: Maximum allowable capacity (MWh)
            
        Returns:
            Optimization results
        """
        if self.time_series is None:
            raise ValueError("Please generate or load data first")
        
        print("\n" + "="*60)
        print("BATTERY ENERGY STORAGE SYSTEM OPTIMIZATION")
        print("="*60)
        print(f"Optimizing BESS capacity (MWh) with {self.bess_power_rating} MW power rating")
        print(f"Target reliability: {self.reliability_target*100:.1f}%")
        print(f"Using scipy.optimize.minimize with constraints...")
        print()
        
        # Set up optimization bounds and constraints
        bounds = [(1, max_capacity)]  # Capacity must be positive and reasonable
        
        # Optimization using scipy.optimize
        result = minimize(
            fun=self.objective_function,
            x0=[initial_guess],
            method='Powell',
            bounds=bounds,
            options={ 'disp': False,            
             'maxiter': 200 }
        )
        
        # Extract optimal capacity
        optimal_capacity = result.x[0]
        
        # Run final simulation with optimal capacity
        final_results = self.simulate_bess_operation(optimal_capacity)
        
        # Store results
        self.results = {
            'optimization_result': result,
            'optimal_capacity': optimal_capacity,
            'simulation_results': final_results
        }
        
        return self.results
    
    def calculate_performance_indicators(self):
        """
        Calculate key BESS performance indicators that system engineers need.
        These metrics help evaluate the effectiveness of the energy storage system.
        """
        if self.results is None:
            print("Please run optimization first using optimize_bess_capacity()")
            return
        
        results = self.results['simulation_results']
        
        # 1. Capacity Factor
        total_energy_discharged = np.sum(np.maximum(0, results['bess_power'])) * self.hours_per_interval
        theoretical_max_energy = self.bess_power_rating * 8760/2  # Full power for 1 year
        capacity_factor = total_energy_discharged / theoretical_max_energy
        
        # 2. Cycling Frequency
        # Count charge/discharge cycles (simplified: each direction change = 0.5 cycle)
        power_changes = np.diff(np.sign(results['bess_power']))
        cycle_count = np.sum(np.abs(power_changes)) / 4  # Approximate full cycles
        
        # 3. Energy Throughput Utilization
        total_energy_throughput = np.sum(np.abs(results['bess_power'])) * self.hours_per_interval
        max_theoretical_throughput = 48 * 365 * 2  # 4 MW per 12 hour period, 365 days per cycle
        throughput_utilization = total_energy_throughput / max_theoretical_throughput
        
        # 4. Peak Shaving Effectiveness
        peak_demand = self.time_series['demand'].max()
        net_peak_with_bess = (self.time_series['demand'] - 
                             pd.Series(results['bess_power'] * self.hours_per_interval)).max()
        peak_shaving_effectiveness = (peak_demand - net_peak_with_bess) / peak_demand
        
        # 5. Round-trip Efficiency Impact
        energy_charged = np.sum(np.minimum(0, results['bess_power'])) * self.hours_per_interval
        energy_discharged = np.sum(np.maximum(0, results['bess_power'])) * self.hours_per_interval
        actual_round_trip_efficiency = -energy_discharged / energy_charged if energy_charged < 0 else 0
        
        # 6. Economic Indicators
        levelized_cost_per_mwh = (results['bess_capital_cost'] * 
                                 (self.discount_rate * (1 + self.discount_rate)**self.system_lifetime) /
                                 ((1 + self.discount_rate)**self.system_lifetime - 1)) / total_energy_discharged
        
        performance_metrics = {
            'capacity_factor': capacity_factor,
            'annual_cycles': cycle_count,
            'throughput_utilization': throughput_utilization,
            'peak_shaving_effectiveness': peak_shaving_effectiveness,
            'actual_round_trip_efficiency': actual_round_trip_efficiency,
            'levelized_cost_per_mwh': levelized_cost_per_mwh,
            'total_energy_discharged': total_energy_discharged,
            'total_energy_charged': -energy_charged
        }
        
        return performance_metrics
    
    def display_results(self):
        """
        Display comprehensive optimization results and performance metrics.
        This helps students understand the value of the optimization process.
        """
        if self.results is None:
            print("Please run optimization first using optimize_bess_capacity()")
            return
        
        results = self.results['simulation_results']
        perf_metrics = self.calculate_performance_indicators()
        
        print("\n" + "="*70)
        print("OPTIMIZATION RESULTS & PERFORMANCE ANALYSIS")
        print("="*70)
        
        print(f"\nOPTIMAL BESS CONFIGURATION:")
        print(f"  Battery Power Rating: {self.bess_power_rating:.1f} MW (fixed)")
        print(f"  Optimal Energy Capacity: {results['bess_capacity_mwh']:.1f} MWh")
                
        print(f"\nSYSTEM RELIABILITY PERFORMANCE:")
        print(f"  Achieved Reliability: {results['reliability']*100:.2f}%")
        print(f"  Target Reliability: {self.reliability_target*100:.1f}%")
        print(f"  Total Energy Not Served: {results['total_energy_not_served']:.1f} MWh")
        print(f"  Number of Loss-of-Load Events: {len(results['lol_events'])}")
        if results['lol_events']:
            print(f"  Average Event Duration: {np.mean(results['lol_events']):.1f} intervals")
            print(f"  Maximum Event Duration: {max(results['lol_events'])} intervals")
        
        print(f"\nBESS PERFORMANCE INDICATORS:")
        print(f"  Capacity Factor: {perf_metrics['capacity_factor']*100:.1f}%")
        print(f"  Annual Equivalent Cycles: {perf_metrics['annual_cycles']:.0f}")
        print(f"  Throughput Utilization: {perf_metrics['throughput_utilization']*100:.1f}%")
        print(f"  Actual Round-trip Efficiency: {perf_metrics['actual_round_trip_efficiency']*100:.1f}%")
        print(f"  Energy Discharged (Annual): {perf_metrics['total_energy_discharged']:.0f} MWh")
        print(f"  Energy Charged (Annual): {perf_metrics['total_energy_charged']:.0f} MWh")
        
        print(f"\nECONOMIC ANALYSIS:")
        print(f"  BESS Capital Cost: ${results['bess_capital_cost']:,.0f}")
        print(f"  Annual Fuel Cost (Rankine): ${results['fuel_cost']:,.0f}")
        print(f"  Levelized Cost per MWh: ${perf_metrics['levelized_cost_per_mwh']:.2f}/MWh")
        
        print(f"\nBACKUP POWER UTILIZATION:")
        total_rankine_energy = np.sum(results['rankine_power']) * self.hours_per_interval
        print(f"  Total Rankine Energy: {total_rankine_energy:.0f} MWh")
        print(f"  Rankine Capacity Factor: {total_rankine_energy/(self.rankine_max_power*8760)*100:.1f}%")
        
        # Create visualization
        self.visualize_optimization_results()
        
    def visualize_optimization_results(self):
        """
        Create comprehensive visualizations of the optimization results.
        """
        if self.results is None:
            return
        
        results = self.results['simulation_results']
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f'BESS Optimization Results - {results["bess_capacity_mwh"]:.1f} MWh Capacity', 
                    fontsize=16, fontweight='bold')
        
        # Create time array for plotting with optional series subset of 120 days
        time_subset = self.time_series#.head(120)
        
        # 1. Energy balance and BESS operation
        axes[0,0].plot(time_subset['datetime'], time_subset['demand'], 
                      label='Demand', color='red', linewidth=2)
        axes[0,0].plot(time_subset['datetime'], time_subset['total_renewable'], 
                      label='Renewable Generation', color='green', linewidth=2)
        axes[0,0].plot(time_subset['datetime'], 
                      results['bess_power'] ,
                      label='Net Supply (with BESS)', color='blue', linewidth=2, linestyle='--')
        axes[0,0].set_title('Energy Balance')
        axes[0,0].set_ylabel('Power (MW)')
        axes[0,0].legend()
        axes[0,0].grid(True, alpha=0.3)
        axes[0,0].xaxis.set_major_locator(DayLocator(interval=30))  # or HourLocator(interval=6) for hours
        axes[0,0].xaxis.set_major_formatter(DateFormatter("%m-%d"))
        axes[0,0].tick_params(axis='x', rotation=45)

        #save figure for report

        plt.figure(figsize=(10, 6))
        plt.plot(time_subset['datetime'], time_subset['demand'], 
                      label='Demand', color='red', linewidth=2)
        plt.plot(time_subset['datetime'], time_subset['total_renewable'], 
                      label='Renewable Generation', color='green', linewidth=2)
        plt.plot(time_subset['datetime'], 
                      results['bess_power'] ,
                      label='Net Supply (with BESS)', color='blue', linewidth=2, linestyle='--')
        plt.title('Energy Balance')
        plt.ylabel('Power (MW)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.gca().xaxis.set_major_locator(DayLocator(interval=30))      
        plt.gca().xaxis.set_major_formatter(DateFormatter("%m-%d"))
        plt.tick_params(axis='x', rotation=45)
        plt.savefig('opt_en_bal.png', dpi=300, bbox_inches='tight')
        plt.close()      
        
        # 2. BESS State of Charge
        axes[0,1].plot(time_subset['datetime'], results['bess_soc'], 
                      color='purple', linewidth=2)
        axes[0,1].axhline(results['bess_capacity_mwh'], color='red', linestyle='--', 
                         label=f'Max Capacity ({results["bess_capacity_mwh"]:.1f} MWh)')
        axes[0,1].axhline(0, color='black', linestyle='-', alpha=0.5)
        axes[0,1].set_title('Battery State of Charge')
        axes[0,1].set_ylabel('Energy (MWh)')
        axes[0,1].legend()
        axes[0,1].grid(True, alpha=0.3)
        axes[0,1].xaxis.set_major_locator(DayLocator(interval=30))  # or HourLocator(interval=6) for hours
        axes[0,1].xaxis.set_major_formatter(DateFormatter("%m-%d"))
        axes[0,1].tick_params(axis='x', rotation=45)

        #save figure for report

        plt.figure(figsize=(10, 6))
        plt.plot(time_subset['datetime'], results['bess_soc'], 
                      color='purple', linewidth=2)
        plt.axhline(results['bess_capacity_mwh'], color='red', linestyle='--', 
                         label=f'Max Capacity ({results["bess_capacity_mwh"]:.1f} MWh)')
        plt.axhline(0, color='black', linestyle='-', alpha=0.5)
        plt.title('Battery State of Charge')
        plt.ylabel('Energy (MWh)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.gca().xaxis.set_major_locator(DayLocator(interval=30))      
        plt.gca().xaxis.set_major_formatter(DateFormatter("%m-%d"))
        plt.tick_params(axis='x', rotation=45)
        plt.savefig('SOC.png', dpi=300, bbox_inches='tight')
        plt.close()      
        
        # 3. Power flows
        axes[1,0].plot(time_subset['datetime'], results['bess_power'], 
                      label='BESS Power', color='blue', linewidth=2)
        axes[1,0].plot(time_subset['datetime'], results['rankine_power'],alpha=0.5,
                      label='Rankine Power', color='orange', linewidth=2)
        axes[1,0].plot(time_subset['datetime'], results['load_not_served'], 
                      label='Load Not Served', color='red', linewidth=2)
        axes[1,0].axhline(0, color='black', linestyle='-', alpha=0.5)
        axes[1,0].set_title('Power Flows (Positive = Discharge/Generation)')
        axes[1,0].set_ylabel('Power (MW)')
        axes[1,0].legend()
        axes[1,0].grid(True, alpha=0.3)
        axes[1,0].xaxis.set_major_locator(DayLocator(interval=30))  # or HourLocator(interval=6) for hours
        axes[1,0].xaxis.set_major_formatter(DateFormatter("%m-%d"))
        axes[1,0].tick_params(axis='x', rotation=45)

        #save figure for report

        plt.figure(figsize=(10, 6))
        plt.plot(time_subset['datetime'], results['bess_power'], 
                      label='BESS Power', color='blue', linewidth=2)
        plt.plot(time_subset['datetime'], results['rankine_power'],alpha=0.5,
                      label='Rankine Power', color='orange', linewidth=2)
        plt.plot(time_subset['datetime'], results['load_not_served'], 
                      label='Load Not Served', color='red', linewidth=2)
        plt.title('Power Flows (Positive = Discharge/Generation)')
        plt.ylabel('Power (MW)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.gca().xaxis.set_major_locator(DayLocator(interval=30))      
        plt.gca().xaxis.set_major_formatter(DateFormatter("%m-%d"))
        plt.tick_params(axis='x', rotation=45)
        plt.savefig('power_flow.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 4. Reliability metrics
        monthly_reliability = []
        for month in range(1, 13):
            month_mask = self.time_series['datetime'].dt.month == month
            month_demand = self.time_series.loc[month_mask, 'demand'].sum()
            month_not_served = np.sum(results['load_not_served'][month_mask])
            month_reliability = 1 - (month_not_served * self.hours_per_interval / 
                                   (month_demand * self.hours_per_interval))
            monthly_reliability.append(month_reliability * 100)
        
        months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        
        bars = axes[1,1].bar(months, monthly_reliability, color='skyblue', alpha=0.7, edgecolor='black')
        axes[1,1].axhline(self.reliability_target*100, color='red', linestyle='--', 
                         linewidth=2, label=f'Target ({self.reliability_target*100:.1f}%)')
        axes[1,1].set_title('Monthly Reliability Performance')
        axes[1,1].set_ylabel('Reliability (%)')
        axes[1,1].legend()
        axes[1,1].grid(True, alpha=0.3)

        #save figure for report

        plt.figure(figsize=(10, 6))
        plt.bar(months, monthly_reliability, color='skyblue', alpha=0.7, edgecolor='black')
        plt.axhline(self.reliability_target*100, color='red', linestyle='--', 
                         linewidth=2, label=f'Target ({self.reliability_target*100:.1f}%)')
        plt.title('Monthly Reliability Performance')
        plt.ylabel('Reliability (%)')
        plt.xticks(months)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig('reliability.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Add value labels on bars
        for bar, value in zip(bars, monthly_reliability):
            height = bar.get_height()
            axes[1,1].text(bar.get_x() + bar.get_width()/2., height + 0.1,
                          f'{value:.1f}%', ha='center', va='bottom', fontsize=8)
        plt.show()

    def progressive_optimization(self):
        tolerance_levels = [
            {'gtol': 1e-1, 'ftol': 1e-1, 'eps': 1e-3},  # Very relaxed
            {'gtol': 1e-2, 'ftol': 1e-2, 'eps': 1e-4},  # Relaxed  
            {'gtol': 1e-3, 'ftol': 1e-3, 'eps': 1e-5},  # Moderate
            {'gtol': 1e-5, 'ftol': 1e-9, 'eps': 1e-8},  # Default (strict)
        ]
        
        for i, opts in enumerate(tolerance_levels):
            print(f"\nTrying tolerance level {i+1}: {opts}")
            
            result = minimize(
                self.objective_function,
                x0=[50.0],  # Start away from problem point
                method='L-BFGS-B',
                options=opts
            )
            
            print(f"Success: {result.success}, Status: {result.status}, Iterations: {result.nit}")
            
            if result.success:
                print(f"Found solution with relaxed tolerances: {result.x[0]:.3f}")
                
        
        return None
    
    def compare_methods(self):
        methods = [
            ('L-BFGS-B', {}),
            ('L-BFGS-B', {'eps': 1e-3, 'gtol': 1e-2}),  # Modified L-BFGS-B
            ('SLSQP', {}),        # Different gradient-based method
            ('Powell', {}),       # Gradient-free
            ('Nelder-Mead', {}),  # Gradient-free, robust
        ]
        
        for method, opts in methods:
            try:
                result = minimize(self.objective_function, x0=[50.0], 
                                method=method, options=opts)
                print(f"{method}: success={result.success}, fun={result.fun:.2e}, x={result.x[0]:.2f}")
            except Exception as e:
                print(f"{method}: failed with {e}")