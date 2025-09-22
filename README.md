# ME 401 project 1 - Renewable Energy Infrastructure Optimization Toolkit

## Overview

This software demonstrates how Python can support engineering workflows in energy systems design. It is intended as an educational tool for engineering students working on renewable energy optimization projects.

## Purpose

The toolkit uses open-source data sets to evaluate solar and wind energy potential for a specific geographic region. It compares this potential against a modeled energy demand profile that includes:

- Heating and cooling loads of a small residential development  
- Base electrical load of an electrified regional shipping port  

The system is optimized by selecting and sizing:

- **Solar PV and Wind Turbines**  
- **Battery Energy Storage System (BESS)**    

The primary evaluation metric is **Levelized Cost of Energy (LCOE)**.

## Workflow Summary

1. **Data Ingestion**  
   - Pull and preprocess open data (renewables.ninja)

2. **Demand Modeling**  
   - Simulate residential and port electrical demand over a typical year

3. **Generation Modeling**  
   - Estimate solar and wind output potential for the region

4. **Storage and ORC Integration**  
   - Model BESS behavior  
   - Include ORC generation

5. **Optimization & Analysis**  
   - Size infrastructure to minimize LCOE and Loss of Load
   - Compare performance under different configurations

6. **Report Generation**  
   - Compile findings into a LaTeX report using a provided Overleaf template

## Requirements

- Python 3.x  
- pandas, numpy, matplotlib, scipy  
- COMSOL for ORC modeling  
- Overleaf account for final report 

## Intended Use

This toolkit supports student engineering teams in:

- Exploring the integration of renewables into complex energy systems  
- Applying computational tools for system design and cost analysis  
- Collaborating on technical documentation using LaTeX  

## Notes

- The ORC boiler and condenser are not simulated in Python — COMSOL must be used for this step.
- The code is modular and extensible for deeper analysis (e.g., hourly simulation, different demand scenarios).

---

### Download link:

To download the data-analysis package the following link can be used:

https://minhaskamal.github.io/DownGit/#/home?url=https://github.com/maxwellhewes/ME_401_project1/tree/main/data_analysis


This folder can be uploaded to kaggle, or any other notebook hosting service to run the package.

