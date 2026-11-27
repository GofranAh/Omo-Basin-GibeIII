# -*- coding: utf-8 -*-
"""
Created on Thu Dec 16 10:39:54 2021

@author: aow001
"""

# ===========================================================================
# Name        : main_Omo.py
# Author      : aow001, adapted from MarkW  (adapted JazminZ & MatteoG)
# Version     : 0.05
# Copyright   : Your copyright notice
# ===========================================================================
import csv
import logging
import numpy as np
import os
import pandas as pd
import random
import multiprocessing
cores = multiprocessing.cpu_count() #to specify the number of cpu cores used in running the model . 

#pip install platypus

from platypus import Problem, EpsNSGAII, Real, ProcessPoolEvaluator

import rbf_functions
from Omo_model import OmoModel

class TrackProgress:
    def __init__(self):
        self.nfe = []
        self.improvements = []
        self.objectives = {}

    def __call__(self, algorithm):
        self.nfe.append(algorithm.nfe)
        self.improvements.append(algorithm.archive.improvements)
        temp = {}
        for i, solution in enumerate(algorithm.archive):
            temp[i] = list(solution.objectives)
        self.objectives[algorithm.nfe] = pd.DataFrame.from_dict(temp,
                                                                orient='index')

    def to_dataframe(self):
        df_imp = pd.DataFrame.from_dict(dict(nfe=self.nfe,
                                             improvements=self.improvements))
        df_hv = pd.concat(self.objectives, axis=0)
        return df_imp, df_hv


def store_results(algorithm, track_progress, output_dir, rbf_name, seed_id):
    path_name = f"{output_dir}/{rbf_name}"
    try:
        os.makedirs(path_name, exist_ok=True)
    except OSError as e:
        print(f"Creation of the directory {path_name} failed:", e)

    header = ["hydropowerGibeIII","hydropower worse deficit", "irrigation down","Irrigation down worst Deficit", "environment", "floodcontrol", "swf", "Recession"]
    try:
        with open(f"{output_dir}/{rbf_name}/{seed_id}_solution.csv", "w", encoding="UTF8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(header)
            for solution in algorithm.result:
                writer.writerow(solution.objectives)
    except Exception as e:
        print("Error writing solution CSV file:", e)

    try:
        with open(f"{output_dir}/{rbf_name}/{seed_id}_variables.csv", "w", encoding="UTF8", newline="") as f:
            writer = csv.writer(f)
            for solution in algorithm.result:
                writer.writerow(solution.variables)
    except Exception as e:
        print("Error writing variables CSV file:", e)

    # save progress info
    df_conv, df_hv = track_progress.to_dataframe()
    df_conv.to_csv(f"{output_dir}/{rbf_name}/{seed_id}_convergence.csv")
    df_hv.to_csv(f"{output_dir}/{rbf_name}/{seed_id}_hypervolume.csv")


def main():
    seeds =  [10,20,30,40,50,60,70,80,90,100] ##Initial run # at least 5 seeds
    for entry in [
        #rbf_functions.original_rbf,
        rbf_functions.squared_exponential_rbf,
        #rbf_functions.inverse_multiquadric_rbf,
        #rbf_functions.inverse
        # _quadratic_rbf,
        #rbf_functions.exponential_rbf,
        #rbf_functions.matern32_rbf,
        #rbf_functions.matern52_rbf,
   ]:
        for seed in seeds:
            random.seed(seed)

            # RBF parameters
            n_inputs = 2  # (time, storage of GibeIII)
            n_outputs = 1 #(downstream release) #Jirriup, JirriWmup,
            n_rbfs = 4 #(n_inputs + 2)     
            rbf = rbf_functions.RBF(n_rbfs, n_inputs, n_outputs, rbf_function=entry)

            # Initialize model
            n_objectives = 8
            n_years = 38

            lowerOmo_river = OmoModel((11*(10**9)), 0, n_years, rbf,'swf') #adjust
            lowerOmo_river.set_log(True)

            # Lower and Upper Bound for problem.types
            #epsilons = [0.5, 0.05, 0.01, 0.05, 0.5]    #epsilon precision for each objective  
            #epsilons = [0.5, 0.05, 0.01, 0.05, 0.01] old
            #epsilons = [100, 0.01, 0.01, 0.05, 0.1]
            epsilons =  [50, 0.05,  0.1,  0.1, 0.1, 0.1, 0.1, 1] #epsilon precision for each objective #change 
            n_decision_vars = len(rbf.platypus_types)

            problem = Problem(n_decision_vars, n_objectives)
            problem.types[:] = rbf.platypus_types
            problem.function = lowerOmo_river.evaluate

            problem.directions[0] = Problem.MAXIMIZE  # annual hydropower_max- GibeIII
            problem.directions[1] = Problem.MINIMIZE  # montly hydropower worse deficit 
            problem.directions[2] = Problem.MAXIMIZE  # irrigation
            problem.directions[3] = Problem.MINIMIZE # irrigation 𝑃𝑒𝑟𝑐𝑒𝑛𝑡𝑖𝑙𝑒 𝑊𝑜𝑟𝑠𝑡 𝐷𝑒𝑚𝑎𝑛𝑑 𝐷𝑒𝑓𝑖𝑐𝑖𝑡
            #problem.directions[4] = Problem.MAXIMIZE  # irrigation upstream
            #problem.directions[5] = Problem.MINIMIZE # irrigation upstream 𝑃𝑒𝑟𝑐𝑒𝑛𝑡𝑖𝑙𝑒 𝑊𝑜𝑟𝑠𝑡 𝐷𝑒𝑚𝑎𝑛𝑑 𝐷𝑒𝑓𝑖𝑐𝑖𝑡            
            problem.directions[4] = Problem.MAXIMIZE  # environment
            #problem.directions[3] = Problem.MAXIMIZE  # flood events (level)
            problem.directions[5] = Problem.MINIMIZE  # flood events (flow release)
            problem.directions[6] = Problem.MAXIMIZE  # justice principles
            problem.directions[7] = Problem.MINIMIZE  # recession 

            # algorithm = EpsNSGAII(problem, epsilons=epsilons)
            # algorithm.run(1000)
            output_directory = os.path.join(".", "output")
            track_progress = TrackProgress()
            with ProcessPoolEvaluator(cores) as evaluator:
                algorithm = EpsNSGAII(problem, epsilons=epsilons,
                                      evaluator=evaluator)
                algorithm.run(1000000, track_progress) #trial run #should be 50000

            store_results(algorithm, track_progress, output_directory,
                          f"{entry.__name__}",
                          seed)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()