"""
This algorithm is designed to identify the best mix of solar and PV 
installed capacity to meet a determined % of given demand profile in two steps
first it formats the data (makes arrays of capacity wind / PV capacity profiles)
and then uses the scipy optimizer to find the best combination of solar and 
wind for a given site.

Created for ME 401 @ Boise State Univerity
Author: Maxwell Hewes, Summer 2025
"""

import numpy as np
from scipy.optimize import minimize

class SolarWindOptimizer:
    """    
    This model demonstrates the the use of numpy and SciPy to optimize a
    renewable energy generation portfolio given annual environmental conditions
    and load profile.

    Creates an 'optimizer' object which will hold the values for optimal wind
    and solar capacity
    
    Components:
        wind, solar, demand arrays generated in the data format object
                 
    Optimization Goal: 
        uses 'SLSQP', sequential least squares programming, algorithm to
        determine optimal mix of solar and wind capacity.
    """

    def __init__(self):
        self.opt_solar_ = 0
        self.opt_wind_ = 0

    def make_arrays(self, wind, solar, demand, target_generation = 0.7):
        """
        Builds object attributes for the optimization algorithm. Has a built in
        process to ensure all arrays are equivalent lengths. If solar / wind arrays
        do not match, a message is generated and the function exits. If demand / generation 
        arrays do not match arrays are trimmed and a message is produced detailing the operation.

        Requires: solar, wind, and demand arrays (from data formatting function)
        returns: object attributes -- .wind_array , .pv_array , .demand_array --
        """
        self.target_generation = target_generation
        if len(wind) != len(solar):
            print('energy generation mismatch wind = %d, solar = %d'% len(wind) %len(solar))
            return
        if len(demand) != len(solar):
            print('array mismatch demand / generation')
            diff = len(solar)-len(demand)
            if diff > 0:
                solar = solar[:-diff]
                wind = wind[:-diff]
                print('dropped %d values from generation arrays\n'% diff)
            if diff < 0:
                diff = diff *-1
                demand = demand[:-diff]
                print('dropped %d values from demand array\n'% diff)
        self.wind_array = wind
        self.pv_array = solar
        self.demand_array = demand 

    def objective_function(self,x): #This is an internal fuction used by the optimization algorithm
        pv_capacity, wind_capacity = x
        
        # Element-wise multiplication across all time steps
        #This is the process by which different generation capacities are created
        pv_generation = pv_capacity * self.pv_array      
        wind_generation = wind_capacity * self.wind_array 
        total_renewable = pv_generation + wind_generation  
        target_generation = self.target_generation * self.demand_array         
        
        # Sum of squared deviations across ALL time steps [np.sum is a vectorized function that is... 
            # ...much more efficient than a loop]
        return np.sum((total_renewable - target_generation)**2)

    def constraint(self, x): #another internal function called by scipy.minimize
        pv_capacity, wind_capacity = x
        avg_renewable = pv_capacity * np.mean(self.pv_array) + wind_capacity * np.mean(self.wind_array)
        return avg_renewable - self.target_generation * np.mean(self.demand_array)
    
    def optimize(self): # This function uses scipy.minimize to perform the optimization 
                            # more details in scipy.minimize documentation
        result = minimize(self.objective_function, 
                        x0=[500,500],  # Initial guess
                        method='SLSQP',
                        constraints={'type': 'eq', 'fun': self.constraint},
                        bounds=[(0, None), (0, None)])  # Non-negative capacities

        if result.success:
            pv_optimal, wind_optimal = result.x
            self.opt_solar_ = pv_optimal
            self.opt_wind_ = wind_optimal
            print(f"Optimization successful!")
            print(f"Optimal PV capacity: {pv_optimal:.2f} kW")
            print(f"Optimal Wind capacity: {wind_optimal:.2f} kW")
            print(f"PV/Wind ratio: {pv_optimal/wind_optimal:.2f}")
            
            # Calculate performance metrics
            total_gen = pv_optimal * self.pv_array + wind_optimal * self.wind_array
            target_gen = self.target_generation * self.demand_array
            
            print(f"Average renewable generation: {np.mean(total_gen):.2f} kW")
            print(f"Target generation: {np.mean(target_gen):.2f} kW")
            print(f"Root mean square error: {np.sqrt(result.fun/len(self.demand_array)):.2f} kW")
        else:
            print(f"Optimization failed: {result.message}")


   