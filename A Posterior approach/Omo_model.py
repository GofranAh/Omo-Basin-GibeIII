"""
Created on Wed Nov 24 13:28:27 2024

@author: Gofan based on https://github.com/JazminZatarain/MUSEH2O/blob/main/susquehanna_model.py
"""
#Omo Gibe III River model

#import Python package
import os
import numpy as np
import math
import rbf_functions 
#Install using pip and restart kernel if unavailable
import utils 
from numba import njit #pip install numba #restart kernel (Ctrl + .)

# define path
def create_path(rest): 
    my_dir = os.path.dirname(os.path.realpath(__file__))
    return os.path.abspath(os.path.join(my_dir, rest))


class OmoModel:
    gammaH20 = 1000.0 #density of water- 1000 kg/m3
    GG = 9.81 #acceleration due to gravity- 9.81 m/s2
    n_months_in_year = 12 #months in a year
    
    #initial conditions
    def __init__(self, s0_GibeIII, d0, n_years, rbf, principle: str, historic_data=False):

        #principle: str,
        """

        Parameters
        ----------
        l0_GibeIII: float; initial condition (water level) at GibeIII
        d0 : int; initial start date 
        n_years : int
        rbf : callable
        historic_data : bool, optional; if true use historic data, 
                        if false use stochastic data
        """

        self.init_storage = s0_GibeIII  # initial water level @ start of simulation_ feet
        self.day0 = d0 # start day
        
        self.log_level_release = True                                  
        

        
        # log level / release
        self.blevel_GibeIII = [] # water level at GibeIII
        #self.rirri = []       # water released for downstream irrigation
        self.rirriUp = []     # water released for upstream irrigation
        self.renv = []      # downstream release
        self.rspill = []    # spill above full storage [m3/s] (logged for the release plots)
        
        # historical record (1961- 2005) and ?? year simulation horizon (using extended dataset)
        self.n_years = n_years # number of years of data
        self.time_horizon_H = self.n_months_in_year* self.n_years    #simulation horizon
        self.months_between_decisions = 1  # monthly time step 
        self.decisions_per_month = 3 #int(30 / self.days_between_decisions)
        self.n_months_one_year = 12
        self.days_in_a_month = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
        
        # Constraints for the reservoirs
        self.min_level_GibeIII =  800 #m  _ min level for hydropower generation 
        self.flood_warn_level = 892 #m #ft _ flood warning level where spilling starts (absolute max is 278ft)
      

        #GibeIII Characteristics
        self.lsv_rel = utils.loadMatrix(
            create_path("./Data/dataMC/lsvGibeIII.txt"),  # Path to the file 
            3,  # Number of rows: 3 (water levels, surface areas, storage volumes)
            1000  # Number of columns: number of data points in the file
            )
        
        self.turbines = utils.loadVector(
            create_path("./Data/dataMC/Turbines_GibeIII.txt"), 10
            )  # Max capacity - min capacity (cfs) - efficiency of each GibeIII plant turbine (10 identical turbines)
        

        self.spillways = utils.loadMatrix(
            create_path("./Data/dataMC/Spillways_GibeIII.txt"), 3, 76
            ) #number of rows 3 ( level (m) - max release (m3/s) - min release (m3/s) and number of columns 76(the number of data points).
        

                
        '''#Kpong Characteristics
        self.turbines_Kp = utils.loadVector(
            create_path("./Data/Kpong_xtics/2.Turbine/turbines_Kp.txt"), 3 
            ) #Max capacity (cfs) - min capacity (cfs) - efficiency of Kpong (4 identical turbines)
        '''
        
        self.historic_data = historic_data
        if historic_data:
            self.load_historic_data()
            self.evaluate = self.evaluate_historic
        else:
            self.load_stochastic_data()
            self.evaluate = self.evaluate_mc
           
     
        #Objective parameters  
        self.MonthlyHydropower = utils.loadVector(
            create_path("./Data/ObjPara/HyPGibeIII.txt"), self.n_months_one_year
        )   # annual hydropower target (GWh) 3,284.9 [GWh/year]
        self.annual_irri = utils.loadVector(
            create_path("./Data/ObjPara/IrrDemand.txt"), self.n_months_one_year
        )  # daily irrigation demand (cms) 
        self.annual_irriup = utils.loadVector(
            create_path("./Data/ObjPara/IrrDemandup.txt"), self.n_months_one_year
        )  # daily irrigation demand (cms        
        self.flood_protection = utils.loadVector(
            create_path("./Data/ObjPara/floodRGibeIII.txt"), self.n_months_one_year
        )  # q_flood: release above which flooding occurs (m3/s): 1300, upper end of the ESIA controlled flood
        self.clam_eflows_l = utils.loadVector(
            create_path("./Data/ObjPara/Eflow.txt"), self.n_months_one_year
        )  # lower bound of of e-flow required in November to March (cfs) (=50 m3/s)
        self.Min_irri = utils.loadVector(
            create_path("./Data/ObjPara/IrrMin.txt"), self.n_months_one_year
        )  # daily irrigation demand (cms)        
        self.Min_irri_up = utils.loadVector(
            create_path("./Data/ObjPara/IrrMinup.txt"), self.n_months_one_year
        )  # daily irrigation demand (cms)
        self.Min_Eflow = utils.loadVector(
            create_path("./Data/ObjPara/Eflow30.txt"), self.n_months_one_year
        )  # daily irrigation demand (cms)
        
        # standardization of the input-output of the RBF release curve 
        # variables for the normalization
        # Initialize as NumPy arrays
        self.input_min = np.array([0, 0])  # Initialize with the minimum values
        self.input_max = np.array([ self.n_months_in_year * self.decisions_per_month - 1,  # Max number of inputs
            892  # Max reservoir level in m (Gibe III crest level 896)
        ])
        self.output_max = np.array([
            11550  # Some maximum value for releases (e.g., power generation)
            #utils.computeMax(self.annual_irriup)  # Compute the max value for annual irrigation
        ])
        self.rbf = rbf     


        #self.output_max.append(utils.computeMax(self.annual_irri))
        # max release = total turbine capacity + spillways @ max storage (950+ 10600= 19020cms) #adjust 
        self.principle = principle


    def load_historic_data(self):   #ET, inflow, GibeIII tailwater, 
        self.losses = utils.loadMultiVector(
            create_path("./Data/data_historical/Vectors/losses.txt"),
            self.n_years, self.n_months_one_year
            ) #evaporation losses @ GibeIII 1984-2012 (mm per day)
        self.inflow_GibeIII = utils.loadMultiVector( 
            create_path("./Data/data_historical/Vectors/InflowtoGibeIIINew.txt"),
            #create_path("./Data/CC_data/vectors2/InflowGibeIII_cc5.txt"), #CC scenarios
            self.n_years, self.n_months_one_year
            )   # inflow, i.e. flows to GibeIII (cms) 1984-2012     
        self.tailwater_GibeIII = utils.loadMultiVector(
            create_path("./Data/data_historical/Vectors/TailwaterGibeIII_historical.txt"), 
            self.n_years, self.n_months_one_year
            ) # historical tailwater level @ GibeIII (m) 1984-2012

        
        
    def load_stochastic_data(self):   # historical data in matrix format
        self.losses = utils.loadArrangeMatrix(
            create_path("./Data/dataMC/losses.txt"),
            self.n_years, self.n_months_one_year
            ) #evaporation losses @ GibeIII_ stochastic data (inches per day)
        self.inflow_GibeIII = utils.loadArrangeMatrix(
            create_path("./Data/dataMC/InflowtoGibeIIINew.txt"),
            self.n_years, self.n_months_one_year
        )  # inflow, i.e. flows to GibeIII_stochastic data     
        self.tailwater_GibeIII = utils.loadArrangeMatrix(
            create_path("./Data/dataMC/TailwaterGibeIII_MC.txt"), 
            self.n_years, self.n_months_one_year
            ) # tailwater level @ GibeIII (ft) 

        
            
    def set_log(self, log_objectives):
        if log_objectives:
            self.log_objectives = True
        else:
            self.log_objectives = False

    def get_log(self):
        return self.blevel_GibeIII, self.rirriUp, self.renv

        
    def apply_rbf_policy(self, rbf_input):
        # Convert rbf_input to a NumPy array if it's not already one
        rbf_input = np.array(rbf_input)
        # normalize inputs 
        formatted_input = (rbf_input - self.input_min) / (self.input_max - self.input_min )
        #print(formatted_input)
        
        #apply rbf
        normalized_output = self.rbf.apply_rbfs(formatted_input)
        #print(normalized_output)
        
        # scale back normalized output
        scaled_output = normalized_output * self.output_max #uu
        #Dprint(f'RBfRelease(uu) = {scaled_output}')
        
        return scaled_output
    
    def evaluate_historic(self, var, opt_met=1):
        return self.simulate(var, self.inflow_GibeIII, self.losses, 
                             self.tailwater_GibeIII, opt_met)  #for running 1 year of data
    
    def evaluate_mc(self, var, opt_met=1):
        #Jirriup, JirriWmup,
        obj, Jhyd_GibeIII, Jhyd_deficit, Jirri,  JirrWm,  Jenv, Jflood, JRec = [], [], [], [], [], [], [], [],
        # MC simulations
        n_samples = 1        
        for i in range(0, n_samples): 
            #Jirrigationup, JirrWorstMup, JPrinciples,
            Jhydropower_GibeIII, Jhydropower_deficit, Jirrigation, JirrWorstM, Jenvironment ,\
                Jfloodcontrol, Jrecession = self.simulate(
                var,
                self.inflow_GibeIII,
                self.losses,
                self.tailwater_GibeIII, 
                #self.fh_Kpong
                opt_met,
            )
            #Dprint(f'hydropower={Jhydropower_GibeIII}')
            #Dprint(f'Irr={Jirrigation}')
            #Dprint(f'Env={Jenv}')
            #Dprint(f'flood={Jflood}')
            Jhyd_GibeIII.append(Jhydropower_GibeIII)
            Jhyd_deficit.append(Jhydropower_deficit)
            Jirri.append(Jirrigation)
            JirrWm.append(JirrWorstM)
            #Jirriup.append(Jirrigationup)
            #JirriWmup.append(JirrWorstMup)                      
            Jenv.append(Jenvironment)
            Jflood.append(Jfloodcontrol)
            #JJPrinciples.append(JPrinciples)
            JRec.append(Jrecession)


   
            

        
        #(99% reliability formulation: insert x-percentile )
        obj.insert(0, np.percentile(Jhyd_GibeIII, 99))
        obj.insert(1, np.percentile(Jhyd_deficit, 99))
        obj.insert(2, np.percentile(Jirri, 99))
        obj.insert(3, np.percentile(JirrWm, 99))
        #obj.insert(4, np.percentile(Jirriup, 99))
        #obj.insert(5, np.percentile(JirriWmup, 99))
        obj.insert(4, np.percentile(Jenv, 99))
        obj.insert(5, np.percentile(Jflood, 99))
        #obj.insert(6, np.percentile(JPrinciples, 99))
        obj.insert(6, np.percentile(JRec, 99))
        
        return obj
    
    #convert storage at current timestep to level and then level to surface area #adjust 


    def storage_to_level(self, st):
        # s : storage
        s = st/(10**6)
        if s <= 0:
            raise ValueError("Storage (s) must be greater than zero for the logarithm to be defined.")
        
        ln_s = math.log(s)  # Compute the natural logarithm of the storage value
        
        # Calculate level 'h' using the logarithmic-based equation
        h = (-10945947 + 1609083.3 * ln_s + 50334047 / ln_s
            - 158316.84 * ln_s**2 - 1.5388988e8 / ln_s**2
            + 10001.888 * ln_s**3 + 2.9792095e8 / ln_s**3
            - 367.06998 * ln_s**4 - 3.2748762e8 / ln_s**4
            + 5.9531645 * ln_s**5 + 1.5357409e8 / ln_s**5)
        
        return h
    
    def level_to_storage(self, h):
        s = utils.interpolate_linear(self.lsv_rel[0], self.lsv_rel[2], h)        
        return  s #utils.acreFeetToCubicFeet(s)

    def level_to_surface(self, h):
        s = utils.interpolate_linear(self.lsv_rel[0], self.lsv_rel[1], h)
        return s #utils.acreToSquaredFeet(s)
    

    def actual_release(self, uu, level_GibeIII, month_of_year):
        #uu = prescribed flow release policy
        Tcap = 950   # total turbine capacity at GibeIII (cms)
        qm_D = 102   # turbine flow corresponding to min flow for system stability

        # Two middle outlets: 725 m3/s each at full reservoir (Gibe III ESIA summary), sill at 750 m a.s.l.
        # They allow a controlled release above the turbine capacity (artificial flood).
        # Their capacity falls with the head above the outlets (orifice flow).
        head_ratio = (level_GibeIII - 750) / (self.flood_warn_level - 750)
        Ocap = 2 * 725 * math.sqrt(min(1.0, max(0.0, head_ratio)))
        qM_D = Tcap + Ocap

        # Spillway gates: capacity from the level-capacity curve (Spillways_GibeIII.txt).
        # Available at every level covered by the curve; below its lowest level the gates give nothing.
        if level_GibeIII >= min(self.spillways[0]):
            qM_D = qM_D + utils.interpolate_linear(self.spillways[0],
                                                   self.spillways[1],
                                                   level_GibeIII)

        # below the minimum operating level: no release
        if level_GibeIII < self.min_level_GibeIII:
            qM_D = 0.0
            qm_D = 0

        #actual release
        rr = []
        rr.append(min(qM_D, max(qm_D, uu[0])))

        return rr

        
    @staticmethod
    @njit
    def g_hyd_GibeIII(r, h, sim_step_days, hour0, GG, gammaH20, turbines,month_of_year):
         #hydropower @ GibeIII =f(prescribed release, water level
         # day of year, hour, gravitational acc,
         #water density, tailwater level, flow through turbines)
         
         Nturb = 10 #number of turbines at GibeIII
         g_hyd = [] #power generated in GWh each day
         pp = [] #power generated in GWh at each turbine per day
         c_hour = len(r) * hour0
         #print('r for hydropower=', r)
         for i in range(0, len(r)):
             eff = 0.8
             #print(i)
             deltaH = h[i] 
             q_split = r[i] 
             #print(q_split)
                 #cycling through turbines
             for j in range(0, Nturb):
                 ##if month_of_year == 5: #The turbines are shut down for maintenance in June
                     #qturb = 0.0
                 if q_split < turbines[1,j]: #turbine flow for system stability. if flow falls below this, the turbine is essentially shut off
                     qturb = 0.0
                 elif q_split >= turbines[0,j]:
                     qturb = turbines[0,j]
                 else:
                     qturb = q_split
                 q_split = q_split - qturb
                 #adjust pow(10,6)
                 p = (
                     eff
                     * GG
                     *gammaH20
                     * (qturb)
                     * (deltaH)
                     * 24*sim_step_days
                     / pow(10,9))  #GW
            
                 pp.append(p)
             g_hyd.append(np.sum(np.asarray(pp)))
             pp.clear()
             c_hour = c_hour +1
         #print('hydropwerarray=',g_hyd)
         Gp = np.sum(np.asarray(g_hyd))
         return Gp
             

    def res_transition_h(self, s0, uu, inflow, losess, month_of_year, sub_monthly_Index, tw_GibeIII):
        #HH = self.hours_between_decisions
        HH= self.months_between_decisions #self.n_months_in_year #Gofran

        
        #storages and levels
        shape = (HH+1, ) #+1 to account for initial time step
        storage_GibeIII = np.empty(shape)
        level_GibeIII = np.empty(shape)
        
        #Actual releases 
        shape = (HH, )
        release_I =np.empty(shape) #irrigation
        release_D = np.empty(shape) # downstream relaease including e-flows

        #initial conditions
        storage_GibeIII[0] = s0
        c_hour = HH * sub_monthly_Index

        # Adjust sim_step based on the sub_monthly_Index
        if sub_monthly_Index == 0:
            # First 10 days
            sim_step_days = 10
        elif sub_monthly_Index == 1:
            # Second 10 days
            sim_step_days = 10
        elif sub_monthly_Index == 2:
            # Remainder of the month
            if self.days_in_a_month[month_of_year] == 31:
                sim_step_days = 11
            elif self.days_in_a_month[month_of_year] == 30:
                sim_step_days = 10
            elif self.days_in_a_month[month_of_year] == 28:
                sim_step_days = 8
            else:
                sim_step_days = 10  # Default case, should not be needed with current logic

        # Converting days to seconds for sim_step
        sim_step = sim_step_days * 24 * 60 * 60
        #sim_step = days_in_a_month[month_of_year]*24*60*60 #converting month to seconds

        for i in range(0, HH):
            #Dprint(f"i_monthly index for waterbalance = {i}")
            level_GibeIII[i] = self.storage_to_level(storage_GibeIII[i])
            if level_GibeIII[i] > 892: #Gofran
                level_GibeIII[i] = 892

            #Compute actual release
            rr = self.actual_release(uu, level_GibeIII[i], month_of_year)
            #print(f'actual release ={rr}')
            release_I[i] = rr[0]
            release_D[i] = rr[0]

            #evaporation_losses_GibeIII = (0.001*ev*surface_GibeIII/(sim_step*3)) #from monthly to cms
            Reservoir_losses = losess
            h_GibeIII = level_GibeIII[i] - tw_GibeIII # head @GibeIII
            #print(f'head={h_GibeIII}') 
            #print(f'Eva={evaporation_losses_GibeIII }') 

            #system transition
            storage_GibeIII[i+1] = storage_GibeIII[i] + sim_step *(
                inflow - 
                release_D[i] - Reservoir_losses 
            ) 
            
            
        sto_GibeIII = storage_GibeIII[HH]

        rel_d = utils.computeMean(release_D)
        #print(rel_d)
        
        level_GibeIII = np.asarray(level_GibeIII) 
        #Dprint("water level = ", utils.computeMean(level_GibeIII)) #daily water level
        h_GibeIII= np.asarray(np.tile(h_GibeIII, int(len(level_GibeIII))))
        #print(h_GibeIII)
        #print(h_Kp)

        #print(r)
        #Dprint(f'release of year =', np.asarray(release_D + release_I))
        #decision timestep
        hp = OmoModel.g_hyd_GibeIII(
            np.asarray(release_D),
            h_GibeIII,
            sim_step_days,
            sub_monthly_Index,
            self.GG,
            self.gammaH20,
            self.turbines,
            month_of_year
            )
        
        
        
        return sto_GibeIII, rel_d, hp #hp_kp, rel_iup,
    
    
    #OBJECTIVE FUNCTIONS
        
    #Flood protection (GibeIII release > flood threshold in floodRGibeIII.txt, 1300 m3/s) -Minimization
    def q_flood_protectn_rel(self, q1, qTarget):
        #delta = 24 * 3600
        qTarget = np.tile(qTarget, int(len(q1) / self.n_months_one_year))
        maxarr = (q1) - (qTarget)
        maxarr[maxarr < 0] = 0
        gg = maxarr / (qTarget)
        #g = np.mean(np.square(gg)) # to penalize larger floods more than smaller floods
        #g = np.mean(gg) 
        return gg #target value = 0

 
    #Clam E-flows - Maximization #adjust   
    def g_eflows_index(self, q, lTarget):
        delta = 24*3600*30
        e = 0
        for i, q_i in np.ndenumerate(q):
            tt = i[0] % self.n_months_one_year
            if (q_i * delta) < (lTarget[tt] * delta):
                e += 1          
        
        G = 1 - (e / (self.n_years * np.sum(lTarget > 0)))
        return G  # target value = 1

    #E-flows on the 10-day steps - Maximization (target value = 1) #Gofran
    def g_eflows_index_10day(self, q_step, lTarget):
        """Share of 10-day steps in which the downstream flow meets the e-flow target.
        Same idea as g_eflows_index, but on the 10-day steps, because a monthly mean
        can hide short periods below the e-flow.
        q_step : downstream flow of every 10-day step [m3/s], shape (n_years*12, 3)
        lTarget: monthly e-flow target [m3/s], shape (12,)"""
        q_step = np.asarray(q_step, dtype=float).reshape(-1, self.decisions_per_month)
        n_months = q_step.shape[0]
        target = np.tile(np.asarray(lTarget, dtype=float), n_months // self.n_months_one_year)
        target = np.repeat(target[:, None], self.decisions_per_month, axis=1)   # same target in the 3 steps of a month
        active = target > 0                                                      # steps with an e-flow target
        if not np.any(active):
            return 1.0
        e = np.count_nonzero(q_step[active] < target[active])                   # 10-day steps below the e-flow
        G = 1 - e / np.count_nonzero(active)
        return G  # target value = 1

    #Irrigation - Maximization
    def g_vol_rel(self, q, qTarget):
        qTarget = np.tile(qTarget, int(len(q)/ self.n_months_one_year))
        g = np.minimum(1, (q)/ (qTarget)) #g = min(1,(q * delta)  / (qTarget * delta ))  #Gofran
        G = utils.computeMean(g)
        #print(q)
        return G #target value = 1
    
    #Irrigation - 𝑃𝑒𝑟𝑐𝑒𝑛𝑡𝑖𝑙𝑒 𝑊𝑜𝑟𝑠𝑡 𝐷𝑒𝑚𝑎𝑛𝑑 𝐷𝑒𝑓𝑖𝑐𝑖𝑡

    def Irr_worst_P(self, q, qTarget): #Gofran
        qTarget = np.tile(qTarget, int(len(q)/ self.n_months_one_year))  # Adjust tiling length to match q
        deficit = np.maximum(0, (qTarget - q) / qTarget)  # the deficit calculation
        P = np.percentile(deficit, 99)  # Use np.percentile for correct calculation
        return P


    def HydropowerDeficit(self, q, qTarget):  # Gofran
        # Adjust tiling length to match q
        qTarget = np.tile(qTarget, int(len(q) / self.n_months_one_year))
        
        # Create a safe version of qTarget to avoid division by zero
        epsilon = 1e-6
        qTarget_safe = np.where(qTarget == 0, epsilon, qTarget)  # Replace zero with epsilon only for division safety

        # Calculate the deficit using the following rules:
        # - If both q and qTarget are zero, set deficit to zero.
        # - Otherwise, calculate the deficit normally.
        deficit = np.where((qTarget == 0) & (q == 0), 0,  # Case when both are zero, no deficit
                        np.maximum(0, (qTarget_safe - q) / qTarget_safe))  # Normal deficit calculation
        
        # Use np.percentile for correct calculation
        #P = np.percentile(deficit, 99)
        #calculating the meandeficit
        P = utils.computeMean(deficit)
        
        return P


    
    #Hydropower- Maximization of hydropower generated
    def g_hydro_max(self, p):
        p1 = np.reshape(p, (self.n_years, self.n_months_one_year))
        p2 = np.sum(p1, axis=1)
        G = np.mean(p2)
        return G #target value = max  hydropower possible (approx 1.038GW x24 x 365 =9093GWh)
    

    def Justice_principle (self, j_hyd_GibeIII, release_d, J_Rec):  # , uncertainty_dict
        """Evaluate the KPI values based on the given input
        data and policy parameter configuration.

        Parameters
        ----------
        self : ModelZambezi object
        parameter_vector : np.array
            Parameter values for the reservoir control policy
            object (NN, RBF etc.)
        principle : str
            The principle with which the principle objective is calculated.

        Returns
        -------
        objective_values : list
            List of calculated objective values
        """

        # self = generate_input_data(self, **uncertainty_dict)
        self.Hydropowertarget = 6500 #5348.3 #6954.596 #5348.3*24*365 #adjusted to max #1.870*24*365*self.n_years 
        # Calculate hydropower aggregated deficit-to-target ratio over 20 years
        Hydropower_supply_ratio = j_hyd_GibeIII/self.Hydropowertarget

        Irr_Demand = np.tile(self.annual_irri, int(len(release_d)/12))
        #Irrup_Demand = np.tile(self.annual_irriup, int(len(release_iup)/12))
        Env_Demand = np.tile(self.clam_eflows_l, int(len(release_d)/12))

        # Calculate irrigation upstream and downstream  monthly supplied-to-target ratio #Gofran
        Irr_supply_ratio = max(0, np.mean(np.minimum(((release_d)/Irr_Demand), 1)))
        #Irrup_supply_ratio = max(0, np.mean(np.minimum(((release_iup) / Irrup_Demand), 1)))
        # Calculate environment  deficit-to-target ratio #Gofran
        Env_supply_ratio = max(0, np.mean(np.minimum(((release_d)/Env_Demand), 1)))
        objectivesP = [Hydropower_supply_ratio, Irr_supply_ratio, 
                       Env_supply_ratio]


        Min_hydropower = 3284.9 #5348  # #Gwh/year
        #Wmin_Hydro= np.tile(Min_hydropower, len(release_i))
        Wmin_Irr = np.tile(self.Min_irri, int(len(release_d)/12))
        #Wmin_Irrup = np.tile(self.Min_irri_up, int(len(release_iup)/12))
        Wmin_Eflow = np.tile(self.Min_Eflow, int(len(release_d)/12))


       
        if self.principle == "None":
            principle_result = None
        elif self.principle == "uwf":
            #modified_objectives = [1 - obj for obj in objectivesP]
            principle_result = sum(objectivesP)
            #Dprint(f'principle_resultUWF = {principle_result}')

        
        elif self.principle == "pwf":

            gamma = 3
            pwf_results = []
            # Calculate PWF values for each objective based on gamma and store them in pwf_results
            for obj in objectivesP:
                if (1 - obj) == 0:
                    pwf_obj = 0
                    pwf_results.append(pwf_obj)
                elif obj > 0:
                    pwf_obj = ((1-obj) ** (1 - gamma)) / (1 - gamma)
                    pwf_results.append(pwf_obj)
                elif obj < 0:
                    pwf_obj = ((1 + abs(obj)) ** (1 - gamma)) / (1 - gamma)
                    pwf_results.append(pwf_obj)
            
            # Calculate the total PWF
            principle_result = sum(pwf_results)
            #Dprint(f'principle_resultPwf = {principle_result}')

        #Sufficientarian #Gofran
        elif self.principle == "swf":

            def remaining_utility(release, wmin, demand):
                # Remaining utility above the minimum, bounded in [0, 1].
                # If demand <= minimum at a step, demand is already met, so utility = 1.
                release = np.asarray(release, dtype=float)
                wmin = np.asarray(wmin, dtype=float)
                demand = np.asarray(demand, dtype=float)
                gap = demand - wmin
                safe_gap = np.where(gap > 0, gap, 1.0)
                ratio = np.where(gap > 0, (release - wmin) / safe_gap, 1.0)
                return float(np.mean(np.clip(ratio, 0.0, 1.0)))

            swf_Irr = np.min(np.minimum(release_d / Wmin_Irr, 1))
            #swf_Irr_up = np.mean(np.minimum((release_iup)/Wmin_Irrup,1))
            swf_e_flow = np.min(np.minimum(release_d / Wmin_Eflow, 1))
            swf_hydropower = min(1, j_hyd_GibeIII / Min_hydropower)
            swf_recession = 1 if J_Rec[1] <= 8 else 0   # matches the /8 scale below
            min_objectives = [swf_Irr, swf_e_flow, swf_hydropower, swf_recession]

            if np.sum(min_objectives) < 4:
                swf_results = int(sum(1 for m in min_objectives if m == 1))
            else:
                Remaining_hydro_utility = max(0, (j_hyd_GibeIII - Min_hydropower) / (self.Hydropowertarget - Min_hydropower))   # original, no upper cap
                Remaining_Irr_utility = remaining_utility(release_d, Wmin_Irr, Irr_Demand)
                #Remaining_Irr_up_utility = remaining_utility(release_iup, Wmin_Irrup, Irrup_Demand)
                Remaining_Eflow_utility = remaining_utility(release_d, Wmin_Eflow, Env_Demand)
                Remaining_recession_utility = max(0, 1 - J_Rec[1] / 8)   # original scale
                Total_utility = [Remaining_hydro_utility, Remaining_Irr_utility,
                                 Remaining_Eflow_utility, Remaining_recession_utility]
                swf_results = np.sum(min_objectives) + np.sum(Total_utility)

            principle_result = swf_results

        elif self.principle == "gini": #Gofran
            #modified_objectives = [1 - obj for obj in objectivesP]
            n = len(objectivesP)
            mean_objectives = np.mean(objectivesP)
            diffs = np.abs(np.subtract.outer(objectivesP, objectivesP)).flatten()
            gini_index = np.sum(diffs) / (2.0 * (n**2) * mean_objectives) 
            principle_result = 1 - gini_index
        else:
            raise ValueError("Invalid principle. Please choose a valid principle: uwf, gini,or swf")
        #print(f' Hydropower_deficit_ratio = { Hydropower_deficit_ratio}')
        #print(f' Irr_deficit_ratio = { Irr_deficit_ratio}')
        #print(f' Env_deficit_ratio = { Env_deficit_ratio}')
        return (
            #Hydropower_deficit_ratio,
            #Irr_deficit_ratio,
            #Env_deficit_ratio,
            principle_result   
        )     
    
    def g_recession(self, q_step):
        """Flood recession objective on the 10-day steps.
        q_step: downstream flow of every 10-day step [m3/s], shape (n_years*12, 3).
        A year meets the target when at least one 10-day step of August or September
        reaches the required flow (10-day controlled flood of the Gibe III ESIA)."""

        required_flow = 1200

        yearly_deficits = []

        for y in range(self.n_years):

            # the 6 ten-day steps of August (month 7) and September (month 8)
            flood_season = np.asarray(q_step[y * 12 + 7 : y * 12 + 9]).ravel()
            step_deficits = np.maximum(0, (required_flow - flood_season) / required_flow)

            # only one 10-day flood required
            yearly_deficits.append(np.min(step_deficits))

        yearly_deficits = np.array(yearly_deficits)

        worst_deficit = np.max(yearly_deficits)
        failure_frequency = np.sum(yearly_deficits > 0)

        return worst_deficit,failure_frequency
    
    #Simulation
    #self.simulate(var, self.evap_GibeIII, self.inflow_GibeIII,
                         #self.tailwater_GibeIII, self.fh_Kpong, opt_met)
    def simulate(self, input_variable_list_var, inflow_GibeIII_n_sim, losses, 
             tailwater_GibeIII, opt_met):
        
        # Initializing daily variables: storages and levels
        shape = (self.time_horizon_H + 1,)
        storage_GibeIII = np.empty(shape)
        level_GibeIII = np.empty(shape)

        # GibeIII actual releases
        shape = (self.time_horizon_H,)
        #release_iup = np.empty(shape)  # irrigation upstream
        release_d = np.empty(shape)  # downstream release (hydro, e-flows, floods)
        release_spillway_total = np.empty(shape) #in m3
        release_step = np.empty((self.time_horizon_H, self.decisions_per_month))  # outflow of every 10-day step (m3/s)

        # Hydropower production
        hydropowerProduction_GibeIII = []  # Monthly energy production at GibeIII

        # Set decision variables for irrigation
        self.rbf.set_decision_vars(np.asarray(input_variable_list_var))

        # Initial condition
        storage_GibeIII[0] = self.init_storage
        level_GibeIII[0] = self.storage_to_level(storage_GibeIII[0])
        #print(f'level_GibeIII[0]={level_GibeIII[0]}')

        # Identification of the periodicity (365 x fdays)
        decision_steps_per_year = self.n_months_in_year * self.decisions_per_month
        year = 0

        # Run simulation
        for t in range(self.time_horizon_H):
            month_of_year = t % self.n_months_in_year
            if month_of_year % self.n_months_in_year == 0 and t != 0:
                year += 1

            shape = (self.decisions_per_month + 1,)
            monthly_storage_GibeIII = np.empty(shape)
            monthly_level_GibeIII = np.empty(shape)

            shape = (self.decisions_per_month,)
            #monthly_release_iup = np.empty(shape)
            monthly_release_d = np.empty(shape)
            monthly_release_spillways = np.empty(shape)

            # Initialization of monthly cycle
            monthly_level_GibeIII[0] = level_GibeIII[t]
            monthly_storage_GibeIII[0] = storage_GibeIII[t]

            # Initialize hydropower production accumulator for the month
            monthly_hydropower_GibeIII = 0

            # Sub-monthly cycle (10 days)
            for j in range(self.decisions_per_month):
                decision_step = (month_of_year * self.decisions_per_month) + j

                # Decision step i in a year
                jj = decision_step % decision_steps_per_year

                # Compute decision
                if opt_met == 0:  # Fixed release
                    uu.append(uu[0])  # Not used
                elif opt_met == 1:  # RBF-PSO
                    rbf_input = np.asarray([jj, monthly_level_GibeIII[j]])
                    uu = self.apply_rbf_policy(rbf_input)

                # System transition
                ss_rr_hp = self.res_transition_h(
                    monthly_storage_GibeIII[j],
                    uu,
                    inflow_GibeIII_n_sim[year][month_of_year],
                    losses[year][month_of_year],
                    month_of_year,
                    j,
                    tailwater_GibeIII[year][month_of_year]
                )


                # Update storage and level for the next decision step
                #assume spillways no release
                monthly_release_spillways [j] = 0 #m3
                monthly_storage_GibeIII[j+1] = ss_rr_hp[0]
                if monthly_storage_GibeIII[j+1] > 14.7*10**9:
                    monthly_release_spillways [j] = (monthly_storage_GibeIII[j+1] - (14.7*10**9))
                    monthly_storage_GibeIII[j+1]= 14.7*10**9                
                monthly_level_GibeIII[j+1] = self.storage_to_level(monthly_storage_GibeIII[j+1])

                # Update releases
                #monthly_release_iup[j] = ss_rr_hp[1]
                monthly_release_d[j] = ss_rr_hp[1]

                # total outflow of this 10-day step (release + spill) in m3/s, for the recession objective
                step_days = 10 if j < 2 else self.days_in_a_month[month_of_year] - 20
                release_step[t, j] = ss_rr_hp[1] + monthly_release_spillways[j] / (step_days * 24 * 60 * 60)

                # Accumulate hydropower production for the month
                monthly_hydropower_GibeIII += ss_rr_hp[2]

            # Append the total hydropower production for the month
            hydropowerProduction_GibeIII.append(monthly_hydropower_GibeIII)

            # Update monthly values
            level_GibeIII[t + 1] = monthly_level_GibeIII[self.decisions_per_month]
            storage_GibeIII[t + 1] = monthly_storage_GibeIII[self.decisions_per_month]
            
            #release_iup[t] = np.mean(monthly_release_iup)
            release_spillway_total[t] = np.sum(monthly_release_spillways)/(self.days_in_a_month[month_of_year]*24*60*60) #in m3/s  
            release_d[t] = np.mean(monthly_release_d)

        # Log level and release if necessary
        if self.log_objectives:
            self.blevel_GibeIII.append(level_GibeIII)
            #self.rirriUp.append(release_iup)
            self.renv.append(release_d)
            self.rspill.append(release_spillway_total)

        #print(f'Release={release_d}')

        # Compute objectives
        j_hyd_GibeIII = sum(hydropowerProduction_GibeIII) / self.n_years
        j_hyd_deficit = self.HydropowerDeficit(hydropowerProduction_GibeIII, self.MonthlyHydropower)
        j_irri = self.g_vol_rel((release_d + release_spillway_total), self.annual_irri)
        J_irri_worstM = self.Irr_worst_P((release_d + release_spillway_total), self.annual_irri)
        #j_irriup = self.g_vol_rel(release_iup, self.annual_irriup)
        #J_irri_worstMup = self.Irr_worst_P(release_iup, self.annual_irriup)
        #prepare releases for Downstream after Irrigation:
        Irr_Consumption = np.tile(self.annual_irri, int(len(release_d)/ self.n_months_one_year))
        DownstreamFlow = np.maximum(release_d + release_spillway_total- Irr_Consumption, 0)        
        DownstreamFlow_step = np.maximum(release_step - np.asarray(Irr_Consumption)[:, None], 0)  # 10-day steps
        #j_env = self.g_eflows_index(DownstreamFlow, self.clam_eflows_l)            # old: count of failed months
        j_env = self.g_eflows_index_10day(DownstreamFlow_step, self.clam_eflows_l)    # new: share of 10-day steps that meet the e-flow
        j_fldcntrl = self.q_flood_protectn_rel((release_d + release_spillway_total), self.flood_protection)
        J_Recession = self.g_recession(DownstreamFlow_step)
        #J_JPrinciples =  self.Justice_principle(j_hyd_GibeIII,(release_d + release_spillway_total),J_Recession)

        # Return objective values
        #print(f'hydropower={j_hyd_GibeIII}')
        #print(f'hydropowerDeficit={j_hyd_deficit}')
        #print(f'j_irri={j_irri}')
        #print(f'j_irriworstM={J_irri_worstM}')
        #print(f'j_env={j_env}')
        #print(f'flood={j_fldcntrl}')
        #print(f'Justice_principle ={J_JPrinciples}')
        #print(f'recession ={J_Recession}')
        # J_JPrinciples,

        return j_hyd_GibeIII, j_hyd_deficit, j_irri, J_irri_worstM, j_env, j_fldcntrl, J_Recession[1]