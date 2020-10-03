# -*- coding: utf-8 -*-
"""
Created on Fri Aug 28 21:30:37 2020

@author: alext
"""

import numpy as np
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
from scipy.stats import gamma
from scipy.stats import pareto
from scipy.stats import lognorm

# =============================================================================
# asset expected values
# =============================================================================

def expect(mu, sigma, pos, cost):
    ev = lognorm.mean(sigma, scale=np.exp(mu))
    
    
    val = ((1-pos)*-cost) + (pos * (ev - cost))
    
    
    return(val)

#phase 1
expect(7.3, 1.5, .07, 0)

#phase 2
expect(7.3, 1.5, .15, 0)

#phase 3
expect(7.3, 1.5, .62, 0)

#launched
expect(7.3, 1.5, 1, 0)


# =============================================================================
# #simulation
# =============================================================================

sigma = 1.5
scale_factor=np.exp(7.3)

class portfolio:
    def __init__(self):
        
        #cost
        self.success_rates = []
        self.portfolio = []
        
        
        #simulation history
        self.sim_history = []

    def PoS_portfolio(self, PoS, size, scale=1):
        PoS_list = [PoS for i in range(size)]
        
        self.success_rates = [np.random.random() <= PoS for PoS in PoS_list]
        self.portfolio = [lognorm.rvs(sigma, scale=scale_factor) for prod in self.success_rates if prod == True]
        
        return(sum(self.portfolio), sum(self.portfolio)/size)

port = portfolio()

# =============================================================================
# Simulating a portfolio
# =============================================================================

p1_num = 44
p1_ports = [port.PoS_portfolio(0.07, size=p1_num, scale=scale_factor) for i in range(10000)]
    
p1_ports_df = pd.DataFrame(p1_ports, columns=['Total value', 'Value per asset'])

#minus ev
p1_ports_df['Net return'] = p1_ports_df['Total value'] - (lognorm.mean(sigma, scale=scale_factor)*p1_num*0.07)
#minus median
#p1_ports_df['Net return'] = p1_ports_df['Total value'] - (lognorm.median(sigma, scale=scale_factor)*p1_num*0.07)


p1_ports_df['Net return'].describe()



p2_num = 21
p2_ports = [port.PoS_portfolio(0.15, size=p2_num, scale=scale_factor) for i in range(10000)]
    
p2_ports_df = pd.DataFrame(p2_ports, columns=['Total value', 'Value per asset'])

#minus ev
p2_ports_df['Net return'] = p2_ports_df['Total value'] - (lognorm.mean(sigma, scale=scale_factor)*p2_num*0.15)
#minus median
#p2_ports_df['Net return'] = p2_ports_df['Total value'] - (lognorm.median(sigma, scale=scale_factor)*p2_num*0.15)

p2_ports_df['Net return'].describe()


p3_num = 5
p3_ports = [port.PoS_portfolio(0.62, size=p3_num, scale=scale_factor) for i in range(10000)]
    
p3_ports_df = pd.DataFrame(p3_ports, columns=['Total value', 'Value per asset'])

#minus ev
#p3_ports_df['Net return'] = p3_ports_df['Total value'] - (lognorm.mean(sigma, scale=scale_factor)*p3_num*0.62)
#minus median
p3_ports_df['Net return'] = p3_ports_df['Total value'] - (lognorm.median(sigma, scale=scale_factor)*p3_num*0.62)


p3_ports_df['Net return'].describe()
p3_ports_df['Net return'].median()
p3_ports_df['Total value'].median()
p3_ports_df['Value per asset'].median()

#plotting
plt.style.use('seaborn')
fig, ax = plt.subplots()

percentiles = [i/len(p3_ports_df) for i in range(1,len(p3_ports_df)+1)]

plt.plot(percentiles, p3_ports_df['Net return'].sort_values(ascending=True).values)
plt.plot(percentiles, p2_ports_df['Net return'].sort_values(ascending=True).values)
plt.plot(percentiles, p1_ports_df['Net return'].sort_values(ascending=True).values)




p2_ports = [port.PoS_portfolio(0.15, size=12, scale=scale_factor) for i in range(10000)]
    
p2_ports_df = pd.DataFrame(p2_ports, columns=['Total value', 'Value per asset'])

p2_ports_df['Total value'].describe()

p2_ports_df['Total value'].plot(kind='hist', bins=50)


p3_ports = [port.PoS_portfolio(0.62, size=3, scale=scale_factor) for i in range(10000)]
    
p3_ports_df = pd.DataFrame(p3_ports, columns=['Total value', 'Value per asset'])

p3_ports_df['Total value'].describe()

p3_ports_df['Total value'].plot(kind='hist', bins=50)


# =============================================================================
# Estimating the scale premium
# =============================================================================

medians = []

for i in range(1, 51):
    px_num = i
    px_ports = [port.PoS_portfolio(1, size=px_num, scale=scale_factor) for i in range(2000)]
        
    px_ports_df = pd.DataFrame(px_ports, columns=['Total value', 'Value per asset'])
    medians.append(px_ports_df['Total value'].median())
    
plt.plot(medians)

medians_per_asset = [medians[i-1]/i for i in range(1,51)]

from scipy.optimize import curve_fit

def sigmoid(x, L ,x0, k, b):
    y = L / (1 + np.exp(-k*(x-x0)))+b
    return (y)

p0 = [max(medians_per_asset), np.median([i for i in range(1,51)]),1,min(medians_per_asset)]

popt, pcov = curve_fit(sigmoid, [i for i in range(1,51)], medians_per_asset, p0)

plt.scatter([i for i in range(1,51)], medians_per_asset)

plt.plot([i for i in range(1,51)], sigmoid(np.array([i for i in range(1,51)]), *popt))

