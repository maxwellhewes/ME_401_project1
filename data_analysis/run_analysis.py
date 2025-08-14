"""
Run script designed to take three defined classes (
two optimization and one formatting)
with provided input data and generate performance metrics.

Created for ME 401 @ Boise State Univerity
Author: Maxwell Hewes, Summer 2025
"""
import bess_optimizer as heo # hybrid energy optimizer
import data_formatter as idf # input data formatter
import supply_optimizer as opt # wind / solar capacity optimizer

wind = 'data_sets/NOLA_wind.csv'
solar = 'data_sets/NOLA_solar.csv'
heat = 'data_sets/NOLA_heat.csv'
port_location = 'New Orleans'
port_file = 'data_sets/Port_demand_raw.csv'

data = idf.DataFormatter()
data.gather_port_demand(port_file,port_location)
data.gather_heat_demand(heat)
data.build_demand()
data.gather_solar(solar)
data.gather_wind(wind)
data.check_array_length()

optimizer = opt.SolarWindOptimizer()
optimizer.make_arrays(data.wind, data.solar, data.demand)
optimizer.optimize()

data.solar = data.solar * optimizer.opt_solar_
data.wind = data.wind * optimizer.opt_wind_

system = heo.HybridEnergyOptimizer()
system.time_series = data.produce_avg_data_frame()
#system.visualize_data_patterns()
results = system.simulate_bess_operation(10)
system.optimize_bess_capacity(initial_guess = 100.0)
print(system.results['optimal_capacity'])

pMetrics = system.calculate_performance_indicators()
system.display_results()

#system.progressive_optimization()
#system.compare_methods()

