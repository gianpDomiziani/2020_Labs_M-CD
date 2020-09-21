#!/usr/bin/python3
import csv
import random
import uuid
import numpy as np
from queue import Queue, PriorityQueue
import math
import matplotlib.pyplot as plt
import pandas as pd
import scipy
import scipy.stats

# ******************************************************************************
# Constants
# ******************************************************************************
#BUFFER_SIZES = [1,2,3,4,9999999]
BUFFER_SIZES = [9999999]
N_SERVERS_POSSIBILITIES = [10,15]
ASSIGN_STRATEGIES = ["random"]
#ARRIVAL_RATES = np.arange(0.1, 3.1, 0.1)
INTER_ARRIVAL_TIMES_BY_HOUR = [30,30,30,30,30,30,30,15,15,15,15,20,20,20,15,15,20,20,15,15,20,20,20,20,20,20]
INTER_ARRIVAL_TIMES = [INTER_ARRIVAL_TIMES_BY_HOUR]
INTER_ARRIVAL_BTH_THRESHOLD = 15  #RATE -> High demand case, we allow to change batteries even if they are not fulled charged.
SIM_TIME = 23*60*0.9
W_MAX=5   #max waiting time for EV waiting for battery
B_THS = [1] 
#B_THS = [0.5,0.75,0.85,1]
CHARGING_TIME=180
CHARGE_RATE = 40000/(CHARGING_TIME/60)  #40kWh
TYPE1 = 1
N_RUNS = 10
N_PANELS = [100,200]
T_MAX_VALUES = [5,20]   # max postponed time
F_MAX_POSTPONED_VALUES = [0.2, 0.4] # max ratio of postponed batteries charging process
PV_POWER_WINTER = [0,0,0,0,0,0,0,0,9.645,80.677,172.659,243.886,269.087,242.486,173.262,84.253,15.09,0,0,0,0,0,0,0,0,0,0,0]
PV_POWER_SUMMER = [0,0,0,0,0,0,41.589,116.609,216.396,331.316,433.069,527.339,566.713,540.398,457.354,349.654,229.43,125.402,48.569,2.294,0,0,0,0,0,0,0,0]
EL_COST_SUMMER = [57.24847826,54.1075,52.35195652,52.21402174,53.5975,57.72945652,60.32108696,63.12086957,63.40347826,62.52836957,61.35467391,58.35054348,57.58119565,59.09793478,60.60402174,63.17336957,63.82576087,65.45532609,67.84032609,68.1751087,67.53445652,64.61956522,60.71923913,61.02456522,57.24847826,54.1075,52.35195652,52.21402174]
EL_COST_WINTER = [50.55277778,46.59133333,44.01511111,42.411,42.56033333,45.81966667,53.47866667,59.17088889,63.98255556,63.57522222,61.22622222,59.80977778,57.203,55.84822222,57.51522222,59.78033333,63.46488889,68.24822222,69.14044444,66.95288889,60.96277778,57.46877778,55.07033333,51.35622222,50.55277778,46.59133333,44.01511111,42.411]
PRICES = EL_COST_SUMMER
PV_POWER = PV_POWER_SUMMER

# ******************************************************************************
# To take the measurements
# ******************************************************************************
class Measure:
    def __init__(self,Narr,Ndep,NAveraegUser,OldTimeEvent,AverageDelay):
        self.arr = Narr
        self.dep = Ndep
        self.ut = NAveraegUser
        self.oldT = OldTimeEvent
        self.delay = AverageDelay
        self.rejects = 0
        self.server_measures = []
        self.waiting_time=0
        self.cost = 0
        self.n_charging_postponed = 0
        self.delayes = []
        self.departure_times = []
        self.rejects_times = []
        self.n_servers = 0
        self.n_panel = 0
        self.f_max = 0
        self.t_max = 0


class ServerMeasure:
    def __init__(self,server_index,busy_time,busy_perc,tot_assigned,tot_assigned_perc):
        self.server_index = server_index
        self.busy_time = busy_time
        self.busy_perc = busy_perc
        self.tot_assigned = tot_assigned
        self.tot_assigned_perc = tot_assigned_perc


# ******************************************************************************
# Client
# ******************************************************************************
class Car:
    def __init__(self,type,arrival_time):
        self.type = type
        self.arrival_time = arrival_time
        self.is_assigned = False


# ******************************************************************************
# Server
# ******************************************************************************
class SwitchingServer(object):

    # constructor
    def __init__(self, index):
        self.jobs_history = []
        # whether the server is idle or not
        self.index = index
        self.is_busy = False
        self.finish_time = 0

    def startJob(self, time):
        if self.is_busy:
            raise Exception("server " + str(self.index) + " is busy")
        self.jobs_history.append({"start":time, "end":time})
        self.is_busy = True

    def getWaitingTime(self, time):
        return max(self.finish_time - time, 0)

    def finishJob(self, time):
        cost = 0
        if not self.is_busy:
            raise Exception("server " + str(self.index) + " is not processing customers")
        self.jobs_history[-1]["end"] = time
        self.is_busy = False
        bth = 1
        if ARRIVAL_BY_HOUR[math.floor(time/60)] <= INTER_ARRIVAL_BTH_THRESHOLD:
            bth = B_TH
        self.finish_time = time + (CHARGING_TIME * bth)

        avg_charging_cost = np.mean(np.array(PRICES)/(10**6))*(CHARGING_TIME/60)*CHARGE_RATE*0.5*bth

        if (data.n_charging_postponed + 1)/data.dep <= F_MAX_POSTPONED:
            for shift in range(1, T_MAX):
                remaining = pv_grid.predictPowerUsage(CHARGE_RATE, time + shift, self.finish_time + shift)
                if self.getTotalCost(remaining) <= avg_charging_cost:
                    data.n_charging_postponed += 1
                    pv_grid.usePower(remaining, CHARGE_RATE, time + shift, self.finish_time + shift)
                    return self.getTotalCost(remaining)

        remaining = pv_grid.predictPowerUsage(CHARGE_RATE, time, self.finish_time)
        pv_grid.usePower(remaining, CHARGE_RATE, time, self.finish_time)
        return self.getTotalCost(remaining)


    def getTotalCost(self,remaining):
        cost = 0
        for min in range(0, len(remaining)):
            cost_per_minute = PRICES[math.floor(min/60)]/60
            cost += cost_per_minute / (10**6) * remaining[min]
        return cost

class PvGrid():
    def __init__(self, n_panels, kwps):
        self.nominal_kwp_capacity = 1*n_panels
        self.output_kwh_by_hour = np.array(kwps)*n_panels
        self.available_power_by_min = []
        for hour in range(0,len(kwps)):
            for min in range(0,60):
                self.available_power_by_min.append(self.output_kwh_by_hour[hour])

    def usePower(self, remaining, necessity, start, end):
        for i in range(math.floor(start), math.ceil(end) + 1,1):
            self.available_power_by_min[i] -= (necessity - remaining[i])
            a = 1

    def predictPowerUsage(self, necessity, start, end):
        remaining = np.zeros(28*60)
        available = self.available_power_by_min.copy()
        for min in range(math.floor(start), math.ceil(end) + 1,1):
            if (available[min] < necessity):
                remaining[min] = necessity - available[min]
                available[min] = 0
            else:
                available[min] = available[min] - necessity
        return remaining

class ChargingStation():
    def __init__(self, n_servers, pv_grid):
        self.servers = []
        self.pv_grid = pv_grid
        for i in range(n_servers):
            self.servers.append(SwitchingServer(i))

    def getServerMinimumWaitingTime(self):
        min = 999999
        fasterServer = None
        for server in self.servers:
            if server.getWaitingTime(time) < min and not server.is_busy:
                fasterServer = server
                min = server.getWaitingTime(time)
        return fasterServer

    def getServersAvailableSoonCount(self):
        return sum(1 for server in self.servers if server.getWaitingTime(time) <= W_MAX and not server.is_busy)

    def assignSwitchingServer(self, server_index, time):
        server = self.servers[server_index]
        server.startJob(time)
        return

    def freeServer(self, server_index, time):
        cost = self.servers[server_index].finishJob(time)
        return cost

# ******************************************************************************

# arrivals *********************************************************************
def arrival(time, FES, queue, strategy):
    global users
    global free_servers

    #print("Arrival no. ",data.arr+1," at time ",time," with ",users," users" )

    # cumulate statistics
    data.arr += 1
    data.ut += users*(time-data.oldT)
    data.oldT = time

    # sample the time until the next event
    ARRIVAL = ARRIVAL_BY_HOUR[math.floor(time/60)]
    inter_arrival = random.expovariate(lambd=1.0/ARRIVAL)

    # schedule the next arrival
    FES.put((time + inter_arrival, "arrival"))

    # create a record for the client
    car = Car(TYPE1, time)

    if charging_station.getServersAvailableSoonCount() > 0:
        server = charging_station.getServerMinimumWaitingTime()
        queue.append(car)
        users += 1
        waiting_time = server.getWaitingTime(time)
        data.waiting_time += waiting_time
        #faccio partire immediatamente il job e setto waiting time al tempo di ricarica completa
        charging_station.assignSwitchingServer(server.index, time)
        FES.put((time + waiting_time, "departure_"+str(server.index)))
    else:
        data.rejects += 1
        data.rejects_times.append(time)


# departures *******************************************************************
def departure(time, FES, queue, server_index, strategy):
    global users

    #print("Departure no. ",data.dep+1," at time ",time," with ",users," users" )

    # cumulate statistics
    data.dep += 1
    data.ut += users*(time-data.oldT)
    data.oldT = time

    # get the first element from the queue
    client = queue.pop(0)

    # do whatever we need to do when clients go away

    data.delay += (time-client.arrival_time)
    all_runs_departure_times.append(time)
    data.delayes.append(time-client.arrival_time)
    all_runs_delay_values.append(time-client.arrival_time)
    data.departure_times.append(time)
    users -= 1
    cost = charging_station.freeServer(server_index, time)
    data.cost += cost
    #print("cost for departure at time "+str(math.floor(time/60))+":"+str(math.floor(time%60))+" :" + str(cost))


def printMeasures(data):
    print("-------- SIMULATION RESULTS with " + str(data.n_servers) + " servers, buffer = " + str(data.buffer_size) + ", assign strategy = " + str(data.assign_strategy))
    print("No. of users in the queue:",data.users,"\nNo. of arrivals =",
          data.arr,"- No. of departures =",data.dep)

    print("Arrival rate: ",data.arrival_rate," - Departure rate: ",data.departure_rate)
    print("Rejects: ",data.rejects)
    print("Avg waiting time: ", data.avg_waiting_time)
    print("Average number of users: ",data.avg_users)

    print("Average delay: ",data.avg_delay)
    print("Total cost: ",data.cost)

    print("Actual queue size: ",data.final_queue_size)
    print("Loss probability: ",data.loss_prob)

    print("Postponend chargings: ", data.n_charging_postponed)

    for server_measure in data.server_measures:
        print("Busy time server # " + str(server_measure.server_index) + " " + str(server_measure.busy_time) + " % of sim time: " + str(server_measure.busy_perc))
        print("Total assigned server # " + str(server_measure.server_index) + " " + str(server_measure.tot_assigned) + " % " + str(server_measure.tot_assigned_perc))

def saveAllResults(measures):
    f = open("result" + str(uuid.uuid4().hex) + ".csv","w")
    #f.write("time,
    # nServers,buffer,load,assign strategy,users,arrival rate,departure rate,avg users,avg delay,final_queue_size,loss prob,busy 1,busy 1 perc,busy 2, busy 2 perc, busy 3, busy 3 perc");
    #f.write("\n")
    for data in measures:
        f.write(str(data.time)+';'+str(data.n_servers)+";"+str(data.buffer_size)+";"+data.assign_strategy+";"+str(data.users)+";"+str(data.arrival_rate)+";"+str(data.departure_rate)+";"+str(data.avg_users)+";"+str(data.avg_delay)+";"+str(data.final_queue_size)+";"+str(data.loss_prob))
        for server_measure in data.server_measures:
            f.write(";"+str(server_measure.busy_time)+";"+str(server_measure.busy_perc)+";"+str(server_measure.tot_assigned)+";"+str(server_measure.tot_assigned_perc))
        f.write("\n")
    f.close()

def plotTransientPhase():
    plt.plot(data.departure_times, data.delayes, ".-", label = "waiting delay")
    plt.plot(data.rejects_times, [0 for rej in data.rejects_times], ".", label="missed service")
    plt.legend()
    plt.xlabel("time (min)")
    plt.show()

def confidenceIntervalWidth(sample):
    sample = np.array(sample)
    confidence_level = 0.95
    degrees_freedom = sample.size - 1
    sample_mean = np.mean(sample)
    sample_standard_error = scipy.stats.sem(sample)
    confidence_interval = scipy.stats.t.interval(confidence_level, degrees_freedom, sample_mean, sample_standard_error)
    all_runs_ci.append(confidence_interval[1] - confidence_interval[0])
    return confidence_interval[1] - confidence_interval[0]

def plotFisherman():
    df = pd.DataFrame({"delay":all_runs_delay_values,"min":[math.floor(min) for min in all_runs_departure_times]})
    dfAggregated = df.groupby("min").mean()
    plt.plot(dfAggregated.index, dfAggregated, ".-", label = "waiting delay")
    plt.legend()
    plt.xlabel("time (min)")
    plt.show()

def missedServicesResults(measures):
    df = pd.DataFrame({"delay":[e.avg_delay for e in measures], "loss_prob": [e.loss_prob for e in measures], "b_th": [e.b_th for e in measures]}).groupby("b_th")
    for name, group in df:
        avg_delay = group["delay"].mean()
        avg_loss = group["loss_prob"].mean()
        print("missedServicesResults bth: ",name," delay: ",avg_delay," loss prob: ",avg_loss )

def totalCostResults(measures):
    df = pd.DataFrame({"cost":[e.cost for e in measures],"n_server": [e.n_servers for e in measures],"t_max": [e.t_max for e in measures],"f_max": [e.f_max for e in measures],"n_panel": [e.n_panel for e in measures],"perc_postponed": [e.perc_postponed for e in measures],"loss_prob": [e.loss_prob for e in measures]})\
        .groupby(["t_max","f_max","n_panel","n_server"])
    for name, group in df:
        avg_cost = group["cost"].mean()
        avg_perc_postponed = group["perc_postponed"].mean()
        avg_loss_prob = group["loss_prob"].mean()
        #print("totalCostResults cost ",avg_cost," t_max ",name[0]," f_max ",name[1]," n_panel ",name[2]," n_server ", name[3])
        print(str(avg_cost)+";"+str(name[0])+";"+str(name[1])+";"+str(name[2])+";"+str(name[3])+";"+str(avg_perc_postponed*100)+";"+str(avg_loss_prob))

# ******************************************************************************
# the "main" of the simulation
# ******************************************************************************
measures = []
all_runs_delay_values = []
all_runs_departure_times = []
all_runs_avg_delay = []
all_runs_ci = []
all_runs_loss_prob = []
all_runs_measures = []

for N_SERVERS in N_SERVERS_POSSIBILITIES:
    for ASSIGN_STRATEGY in ASSIGN_STRATEGIES:
        for ARRIVAL_BY_HOUR in INTER_ARRIVAL_TIMES:
            for BUFFER_SIZE in BUFFER_SIZES:
                for B_TH in B_THS:
                    for F_MAX_POSTPONED in F_MAX_POSTPONED_VALUES:
                            for T_MAX in T_MAX_VALUES:
                                    for N_PANEL in N_PANELS:
                                        for run in range(N_RUNS):
                                            #random.seed(42)

                                            arrivals=0
                                            users=0
                                            MM1=[]
                                            pv_grid = PvGrid(N_PANEL,PV_POWER)
                                            charging_station = ChargingStation(N_SERVERS,pv_grid)

                                            data = Measure(0,0,0,0,0)

                                            # the simulation time
                                            time = 0

                                            # the list of events in the form: (time, type)
                                            FES = PriorityQueue()

                                            # schedule the first arrival at t=0
                                            FES.put((0, "arrival"))

                                            # simulate until the simulated time reaches a constant
                                            while time < SIM_TIME:
                                                (time, event_type) = FES.get()

                                                # if event_type == "arrival_charged_battery":
                                                #     arrivalChargedBattery(time, FES, MM1, ASSIGN_STRATEGY)

                                                if event_type == "arrival":
                                                    arrival(time, FES, MM1, ASSIGN_STRATEGY)

                                                elif "departure" in event_type:
                                                    departure(time, FES, MM1, int(event_type.split("_")[1]), ASSIGN_STRATEGY)

                                            data.n_servers = N_SERVERS
                                            data.t_max = T_MAX
                                            data.f_max = F_MAX_POSTPONED
                                            data.n_panel = N_PANEL
                                            data.buffer_size = BUFFER_SIZE
                                            data.assign_strategy = ASSIGN_STRATEGY
                                            data.users = users
                                            data.arrival_rate = data.arr/time
                                            data.departure_rate = data.dep/time
                                            data.avg_users = data.ut/time
                                            data.avg_delay = data.delay/data.dep
                                            all_runs_avg_delay.append(data.avg_delay)
                                            data.avg_waiting_time = data.waiting_time/data.dep
                                            data.final_queue_size = len(MM1)
                                            data.loss_prob = data.rejects * 100 / data.arr
                                            all_runs_loss_prob.append(data.loss_prob)
                                            data.time = time
                                            data.avg_cost = data.cost/data.dep
                                            data.b_th = B_TH
                                            data.perc_postponed = data.n_charging_postponed/data.dep

                                            measures.append(data)
                                            all_runs_measures.append(data)
                                            #printMeasures(data)
                                            # if len(MM1)>0:
                                            #     print("Arrival time of the last element in the queue:",MM1[len(MM1)-1].arrival_time)


    #saveAllResults(measures)
    #plotTransientPhase()
    #plotFisherman()
    #width = confidenceIntervalWidth(all_runs_avg_delay)
    #missedServicesResults(all_runs_measures)
    totalCostResults(all_runs_measures)
