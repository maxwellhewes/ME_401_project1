"""
Data formatting class designed to take in data sets pulled from
publicly available sources and generate a dataframe for further analysis.

Created for ME 401 @ Boise State Univerity
Author: Maxwell Hewes, Summer 2025
"""

import numpy as np
import pandas as pd

class DataFormatter:

    """    
    This model demonstrates the the use of numpy and pandas to take data sets and 
    format them for useful engineering analysis.
    
    Components:
    - Load Data
        - Thermal Load 
            (to simulate the heating / cooling of a residential development
            source: https://www.renewables.ninja/)
        - Port Load
            (to simulate the demands of electrifying the ground support infrastructure
             of a locations shipping port.
              source: https://www.pnnl.gov/projects/port-electrification-handbook)
    -Renewable Data
        - Wind Generation (source: https://www.renewables.ninja/)
        - PV Generation (source: https://www.renewables.ninja/) 
            
    Formatting Goal: Create an object which contains formatted data and 
        produces a dataframe which is ready for the optimization process.
    """

    def __init__(self): #Create arrays to store data in:
        self.solar = []
        self.wind = []
        self.demand = []
        self.port_demand = []
        self.heat_demand = []
        self.net_load = []
        self.date = []

    def gather_port_demand(self, port_demand_csv, port_location):
        """
        Read and store port demand data based on port location.

        Requires: port demand file (csv), port location(string)
        returns: an object attribute -- .port_demand --
        """
        df = pd.read_csv(port_demand_csv)
        mask = df['Port Name'] == port_location
        df = df[mask].copy()
        mask = df['Year'] == '100% eCHE'
        port_demand_df = df[mask].copy()
        port_demand_array = np.tile(port_demand_df['Demand (kW)'].values, 365)
        self.port_demand = np.array(port_demand_array)

    def gather_heat_demand(self, heat_demand_csv, housing_units = 500):
        """
        Read and store heat demand data based on port location. Also 
        pulls date information for time series from heat demand data.

        Requires: heat demand file (csv), optional housing units (int)
        returns: an object attribute -- .heat_demand --
                 object attribute -- .date --
        """
        heat_demand_df = pd.read_csv(heat_demand_csv) 
        heat_demand_df = heat_demand_df.iloc[9:] #drop first 9 rows
        heat_demand_df = heat_demand_df.iloc[:-15] #drop last 15 rows
        self.heat_demand = np.array(heat_demand_df['total_demand']) * housing_units
        self.date = np.array(heat_demand_df['time'])

    def build_demand(self):
        """
        Combines heat and port demand arrays to create 'demand' array.
        Checks to ensure the array lengths are equivalent and will trim
        arrays to fix mismatch.
        
        Requires: heat and port demand attributes (arrays)
        returns: object attribute -- .demand --
                 prints message detailing array mismatch 
        """
        if len(self.port_demand) != len(self.heat_demand):
            diff = len(self.port_demand) - len(self.heat_demand)
            if diff > 0:
                self.port_demand = self.port_demand[:-diff]
                print("demand arrays mismatched, trimmed %d values from port demand tail",diff)
            if diff < 0:
                diff = diff * -1
                self.heat_demand = self.heat_demand[:-diff] 
            print("demand arrays mismatched, trimmed %d values from heat demand tail",diff)
        self.demand = self.heat_demand + self.port_demand

    def gather_solar(self,solar_csv):
        """
        Read and store solar generation data based on port location.

        Requires: solar capacity file (csv)
        returns: an object attribute -- .solar --
        """        
        solar_generation_df = pd.read_csv(solar_csv)
        solar_generation_df = solar_generation_df.iloc[9:]
        self.solar = np.array(solar_generation_df['electricity'])

    def gather_wind(self, wind_csv):
        """
        Read and store wind generation data based on port location.

        Requires: wind capacity file (csv)
        returns: an object attribute -- .wind --
        """     
        wind_generation_df = pd.read_csv(wind_csv)
        wind_generation_df = wind_generation_df.iloc[9:]
        self.wind = np.array(wind_generation_df['electricity'])

    def produce_avg_data_frame(self, convert_to_MW = True):
        """
        Creates an averaged data set based on the stored
        attibute arrays. The function takes the mean of each
        twelve hour period and creates a single values.
        Each day has a two averaged values.
        The function also converts the kW values to MW values.
        This process makes the data representation more
        easily interpreted. 
        If the arrays are mismatched, tail values are dropped.

        Requires: completed attribute arrays 
                    (optional) convert to MW command (bool)
        returns: a dataframe with averaged values
                 a message indicating if arrays have been altered
        """     
        if convert_to_MW:
            self.wind = self.wind * 0.001
            self.solar = self.solar * 0.001
            self.demand = self.demand * 0.001
        data = {'datetime': self.date,
                'pv_generation': self.solar,
                'wind_generation': self.wind,
                'total_renewable': self.solar + self.wind,
                'demand': self.demand,
                'net_load':self.demand - (self.solar + self.wind)}
        avg_df = pd.DataFrame(data)
        avg_df['datetime'] = pd.to_datetime(avg_df['datetime'])  # ensure datetime
        avg_df.set_index('datetime', inplace=True)
        avg_df = avg_df.resample('12h').mean()
        avg_df = avg_df.round(2)
        avg_df.reset_index(inplace=True)

        return avg_df
    
    def check_array_length(self):

        wind = self.wind 
        solar = self.solar 
        demand = self.demand 

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
        self.wind = wind
        self.solar = solar
        self.demand = demand

    


    
    

