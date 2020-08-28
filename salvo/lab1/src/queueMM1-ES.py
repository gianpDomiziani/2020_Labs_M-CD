#!/usr/bin/python3

import random
import pandas as pd
import uuid
import numpy as np
from queue import Queue, PriorityQueue
import matplotlib.pyplot as plt

# ******************************************************************************
# Constants
# ******************************************************************************
# Scenarios: 
# Buffer sizes = 1, 500, 100000
# Server Possibilities 1 e 2
# assign strategies: random - less_cost

BUFFER_SIZES = [500]  # inf ~ 10e+5
N_SERVERS_POSSIBILITIES = [1]
ASSIGN_STRATEGIES = ["random"]
SERVER_COSTS = [1]
LOADS = np.arange(0.1, 3.1, 0.1)
SIM_TIME = 5000
SERVICES = [1, 2, 3]  # mu
ARRIVALS = [0.2, 0.4, 0.6, 0.8, 1, 1.5, 2, 2.5, 3]
TYPE1 = 1


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
class Client:
    def __init__(self,type,arrival_time):
        self.type = type
        self.arrival_time = arrival_time
        self.is_assigned = False


# ******************************************************************************
# Server
# ******************************************************************************
class Server(object):

    # constructor
    def __init__(self, index, cost):
        self.jobs_history = []
        # whether the server is idle or not
        self.index = index
        self.is_busy = False
        self.cost = cost

    def startJob(self, time):
        if self.is_busy:
            raise Exception("server " + str(self.index) + " is busy")
        # Why start == end ?? just for init
        self.jobs_history.append({"start": time, "end": time})
        self.is_busy = True

    def finishJob(self, time):
        if not self.is_busy:
            raise Exception("server " + str(self.index) + " is not processing customers")
        self.jobs_history[-1]["end"] = time
        self.is_busy = False

    def getTotalBusyTime(self):
        return sum([job["end"] - job["start"] for job in self.jobs_history])

    def getTotalAssignedJobs(self):
        return len(self.jobs_history)

# ******************************************************************************
# Cluster
# ******************************************************************************
class Cluster():
    def __init__(self, n_servers):
        self.servers = []
        for i in range(n_servers):
            self.servers.append(Server(i, SERVER_COSTS[i]))

    def freeServersNumber(self):
        return sum(1 for server in self.servers if not server.is_busy)

    def assignJob(self, strategy, time):
        server = None
        if strategy == "random":
            server = random.choice([server for server in self.servers if not server.is_busy])
        if strategy == "less_cost":
            server = self.servers[0]
            for candidate in self.servers:
                if candidate.cost < server.cost:
                    server = candidate
        server.startJob(time)
        return server

    def freeServer(self, server_index, time):
        self.servers[server_index].finishJob(time)

# ******************************************************************************

# arrivals *********************************************************************
def arrival(time, FES, queue, strategy):
    global users
    global free_servers

    #print("Arrival no. ",data.arr+1," at time ",time," with ",users," users" )

    # cumulate statistics
    data.arr += 1
    data.ut += users*(time-data.oldT) # ??
    data.oldT = time

    # sample the time until the next event
    inter_arrival = random.expovariate(lambd=ARRIVAL)

    # schedule the next arrival
    FES.put((time + inter_arrival, "arrival"))

    # create a record for the client
    client = Client(TYPE1, time)

    # insert the record in the queue
    if (len(queue) + 1) <= BUFFER_SIZE:
        queue.append(client)
        users += 1
    else:
        data.rejects += 1

    scheduleDepartures(queue, time, FES, strategy)


# ******************************************************************************

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
    users -= 1
    cluster.freeServer(server_index, time)

    scheduleDepartures(queue, time, FES, strategy)


def scheduleDepartures(queue, time, FES, strategy):
    unassigned_users = [user for user in queue if not user.is_assigned]
    for i in range(min(cluster.freeServersNumber(), len(unassigned_users))):
        unassigned_users[-1 + i].is_assigned = True
        server = cluster.assignJob(strategy, time)
        #service_time = random.expovariate(SERVICE)
        service_time = random.paretovariate(SERVICE)  #MG1
        #service_time = 1 + random.uniform(0, SEVICE_TIME)
        # schedule when the client will finish the server
        FES.put((time + service_time, "departure_"+str(server.index)))


def printMeasures(data):
    print("-------- SIMULATION RESULTS with " + str(data.n_servers) + " servers, buffer = " + str(data.buffer_size) + ", load = " + str(data.load) + ", assign strategy = " + str(data.assign_strategy))
    print("No. of users in the queue:",data.users,"\nNo. of arrivals =",
          data.arr,"- No. of departures =",data.dep)

    print("Load: ",data.load)

    print("Arrival rate: ",data.arrival_rate," - Departure rate: ",data.departure_rate)
    print("Rejects: ",data.rejects)

    print("Average number of users: ",data.avg_users)

    print("Average delay: ",data.avg_delay)
    print("Actual queue size: ",data.final_queue_size)
    print("Loss probability: ",data.loss_prob)

    for server_measure in data.server_measures:
        print("Busy time server # " + str(server_measure.server_index) + " " + str(server_measure.busy_time) + " % of sim time: " + str(server_measure.busy_perc))
        print("Total assigned server # " + str(server_measure.server_index) + " " + str(server_measure.tot_assigned) + " % " + str(server_measure.tot_assigned_perc))

def saveAllResults(measures):
    Uuid = uuid.uuid4().hex
    f = open("result" + str(Uuid) + ".csv","w")
    i = 1
    for data in measures:
        f.write(str(data.time)+';'+str(data.n_servers)+";"+str(data.buffer_size)+";"+str(data.load)+";"+data.assign_strategy+";"+str(data.users)+";"+str(data.arrival_rate)+";"+str(data.departure_rate)+";"+str(data.avg_users)+";"+str(data.avg_delay)+";"+str(data.final_queue_size)+";"+str(data.loss_prob))
        for server_measure in data.server_measures:
            f.write(";"+str(server_measure.busy_time)+str(i)+";"+str(server_measure.busy_perc)+str(i)+";"+str(server_measure.tot_assigned)+str(i)+";"+str(server_measure.tot_assigned_perc))
            i += 1
        f.write("\n")
    f.close()
    path = "result"+ str(Uuid)+ ".csv"
    return path

def createDF(file_path, nServers=1):

    if nServers == 1:
        columns = ['time', 'n_servers', 'buffer_size', 'load', 'strategy', 'users', 'arrival_rate', 'departure_rate',
               'avg_users', 'avg_delay', 'queue_size', 'loss_prob', 'Sbusy_time', 'Sbusy_perc', 'Stot_assigned', 'Stot_assiPerc']
    else:
        columns = ['time', 'n_servers', 'buffer_size', 'load', 'strategy', 'users', 'arrival_rate', 'departure_rate',
                   'avg_users', 'avg_delay', 'queue_size', 'loss_prob', 'Sbusy_time1', 'Sbusy_perc1', 'Stot_assigned1',
                   'Stot_assiPerc1', 'Sbusy_time2', 'Sbusy_perc2', 'Stot_assigned2',
                   'Stot_assiPerc2']


    df = pd.read_csv(file_path, sep=';', header=None)
    df.columns = columns
    df.to_csv(file_path)
    return df

def plot(df, case='MG1FiniteBufferSize', X='load', multi=False):

    x = df[X]
    avg_delay = df['avg_delay']
    avg_usr = df['avg_users']
    l_p = df['loss_prob']
    if multi:
        Sbusy_t1 = df['Sbusy_time1']
        Sbusy_t2 = df['Sbusy_time2']
    else:
        Sbusy_t = df['Sbusy_time']
    step = len(ARRIVALS)

    # =====================
    # Plots ArrivalRate
    # =====================
    figw, figh = 16.0, 8.0
    if not multi:
        fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(figw, figh))
    else:
        fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(5, 1, figsize=(figw, figh))

    fig.suptitle(f'{X} metrics analysis: {case}')
    ax1.plot(x[0:step], avg_delay[0:step], linewidth=1, marker='.', label=f'AvgD: {SERVICES[0]}')
    ax1.plot(x[step:2*step], avg_delay[step:2*step], linewidth=1, marker='^', label=f'AvgD: {SERVICES[1]}')
    ax1.plot(x[2*step:-1], avg_delay[2*step:-1], linewidth=1, marker='*', label=f'AvgD: {SERVICES[2]}')
    #ax1.set_title(f'{X}: Average Delay')
    ax1.set_ylabel('avg delay[s]')
    if X == 'arrival_rate':
        ax1.set_xlabel(f'{X} [1/s]')
    else:
        ax1.set_xlabel(f'{X}')

    #ax1.set_ylim(3, 60)
    ax1.legend(bbox_to_anchor=(0., 0.1, 1., .102), loc='center right')
    # ncol=4, mode="expand", borderaxespad=0.)
    ax1.grid(linestyle='--', linewidth=.4, which="both")

    # Loss Probability
    ax2.plot(x[0:step], l_p[0:step], linewidth=1, marker='.', label=f'LP:{SERVICES[0]}')
    ax2.plot(x[step:2*step], l_p[step:2*step], linewidth=1, marker='^', label=f'LP:{SERVICES[1]}')
    ax2.plot(x[2*step:-1], l_p[2*step:-1], linewidth=1, marker='*', label=f'LP:{SERVICES[2]}')
    #ax2.set_title('Loss Probability')
    ax2.set_ylabel('Loss Probability')
    if X == 'arrival_rate':
        ax2.set_xlabel('Arrival Rate [1/s]')
    else:
        ax2.set_xlabel(f'{X}')
    ax2.grid(linestyle='--', linewidth=.4, which="both")
    ax2.legend(bbox_to_anchor=(0, 0.1, 1., 0.1), loc='center right')
    #ax2.set_ylim(3, 60)

    #avg number of users
    ax3.plot(x[0:step], avg_usr[0:step], linewidth=1, marker='.', label=f'AvgU:{SERVICES[0]}')
    ax3.plot(x[step:2*step], avg_usr[step:2*step], linewidth=1, marker='^', label=f'AvgU:{SERVICES[1]}')
    ax3.plot(x[2*step:-1], avg_usr[2*step:-1], linewidth=1, marker='o', label=f'AvgU:{SERVICES[2]}')
    #ax3.set_title('Average Users')
    if X == 'arrival_rate':
        ax2.set_xlabel('Arrival Rate [1/s]')
    else:
        ax2.set_xlabel(f'{X}')
    ax3.set_ylabel('Avg users')
    ax3.grid(linestyle='--', linewidth=.4, which="both")
    ax3.legend(bbox_to_anchor=(0., 0.1, 1., .102), loc='center right')

    #SBusy_time
    if not multi:
        ax4.plot(x[0:step], Sbusy_t[0:step], linewidth=1, marker='8', label=f'SbusyT:{SERVICES[0]}')
        ax4.plot(x[step:2 * step], Sbusy_t[step:2 * step], linewidth=1, marker='o', label=f'SbusyT:{SERVICES[1]}')
        ax4.plot(x[2 * step:-1], Sbusy_t[2 * step:-1], linewidth=1, marker='v', label=f'SbusyT:{SERVICES[2]}')
        #ax4.set_title('Server Busy Time')
        if X == 'arrival_rate':
            ax4.set_xlabel('Arrival Rate [1/s]')
        else:
            ax4.set_xlabel(f'{X}')
        ax4.set_ylabel('SBusyTime [s]')
        ax4.grid(linestyle='--', linewidth=.4, which="both")
        ax4.legend(bbox_to_anchor=(0., 0.1, 1., .102), loc='center right')
    # ax3.set_ylim(3, 60)
        plt.subplots_adjust(left=1/figw, right=1-1/figw, bottom=1/figh, top=1-1/figh)
        plt.savefig(fname=f"../plots/arrivalRate{case}_{X}.png")
        plt.close()
    else:
        ax4.plot(x[0:step], Sbusy_t1[0:step], linewidth=1, marker='8', label=f'SbusyT1:{SERVICES[0]}')
        ax4.plot(x[step:2 * step], Sbusy_t1[step:2 * step], linewidth=1, marker='o', label=f'SbusyT1:{SERVICES[1]}')
        ax4.plot(x[2 * step:-1], Sbusy_t1[2 * step:-1], linewidth=1, marker='v', label=f'SbusyT1:{SERVICES[2]}')
        #ax4.set_title('Server Busy Time')
        if X == 'arrival_rate':
            ax5.set_xlabel('Arrival Rate [1/s]')
        else:
            ax5.set_xlabel(f'{X}')
        ax4.set_ylabel('SBusyTime [s]')
        ax4.grid(linestyle='--', linewidth=.4, which="both")
        ax4.legend(bbox_to_anchor=(0., 0.1, 1., .102), loc='center right')

        ax5.plot(x[0:step], Sbusy_t2[0:step], linewidth=1, marker='8', label=f'SbusyT2:{SERVICES[0]}')
        ax5.plot(x[step:2 * step], Sbusy_t2[step:2 * step], linewidth=1, marker='o', label=f'SbusyT2:{SERVICES[1]}')
        ax5.plot(x[2 * step:-1], Sbusy_t2[2 * step:-1], linewidth=1, marker='v', label=f'SbusyT2:{SERVICES[2]}')
        #ax5.set_title('Server Busy Time')
        if X == 'arrival_rate':
            ax5.set_xlabel('Arrival Rate [1/s]')
        else:
            ax5.set_xlabel(f'{X}')
        ax5.set_ylabel('SBusyTime [s]')
        ax5.grid(linestyle='--', linewidth=.4, which="both")
        ax5.legend(bbox_to_anchor=(0., 0.1 , 1., .102), loc='center right')
        # ax3.set_ylim(3, 60)
        plt.subplots_adjust(left=1 / figw, right=1 - 1 / figw, bottom=1 / figh, top=1 - 1 / figh)
        plt.savefig(fname=f"../plots/arrivalRate{case}_{X}.png")
        plt.close()



def state_distributionMM1(lambd, mu, N=10):
    """
    Compute the steady-state probability distribution for a M/M/1 queue: with assumption rho= LAMBDA/mu < 1.
    :param lambd: Interarrival time
    :param mu: Service Time
    :param N: number of states
    :return: steady-state probability distribution
    """

    rho = float(lambd/mu)
    L = rho/(1-rho)
    W = L/lambd
    p0 = 1-rho
    n = np.arange(N)
    pn = np.reshape(p0*(rho**n), (N, 1))
    return pn

def average_time_in_queue(lambd, mu):

    rho = lambd/mu
    L = rho/(1-rho)
    W = L/lambd

    return W
# ******************************************************************************
# the "main" of the simulation
# ******************************************************************************
measures = []
for N_SERVERS in N_SERVERS_POSSIBILITIES:
    for ASSIGN_STRATEGY in ASSIGN_STRATEGIES:
        for SERVICE in SERVICES:
            for BUFFER_SIZE in BUFFER_SIZES:
                for ARRIVAL in ARRIVALS:
                    LOAD = ARRIVAL/SERVICE  # ?? rho = lambda/mu
                    arrivals = 0
                    users = 0
                    MM1 = []
                    cluster = Cluster(N_SERVERS)

                    random.seed(42)

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

                        if event_type == "arrival":
                            arrival(time, FES, MM1, ASSIGN_STRATEGY)

                        elif "departure" in event_type:
                            departure(time, FES, MM1, int(event_type.split("_")[1]), ASSIGN_STRATEGY)

                    data.n_servers = N_SERVERS
                    data.buffer_size = BUFFER_SIZE
                    data.load = LOAD
                    data.assign_strategy = ASSIGN_STRATEGY
                    data.users = users
                    data.arrival_rate = data.arr/time
                    data.departure_rate = data.dep/time
                    data.avg_users = data.ut/time
                    data.avg_delay = data.delay/data.dep
                    data.final_queue_size = len(MM1)
                    data.loss_prob = data.rejects * 100 / data.arr
                    data.time = time

                    for server in cluster.servers:
                        server_measure = ServerMeasure(server.index, server.getTotalBusyTime(), server.getTotalBusyTime()*100/time, server.getTotalAssignedJobs(), server.getTotalAssignedJobs() * 100/data.dep)
                        data.server_measures.append(server_measure)

                    measures.append(data)
                    printMeasures(data)

path = saveAllResults(measures)
df = createDF(path, nServers=2)
plot(df, case='INFiniteBufferSizeMG1MS', X='arrival_rate', multi=True)

#THINGS TO DO TO RUN A SIMULATION 
'''
Set Number of servers 1 or 2 and also nServers in createDF
Set Buffer Size: 1 (no more), 500, 100000
If N° Server 2 --> set server_Costs[1, 2]
Remember to select the distribution: 
        #service_time = random.expovariate(SERVICE)   #MM1  
        service_time = random.paretovariate(SERVICE)  #MG1
in plot specify the case, the X and the multi bolean variable

#open in Terminal the "src" folder 
# python3 queueMM1-ES.py 
'''


