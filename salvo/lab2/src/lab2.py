#!/usr/bin/python3

import random
import uuid
import numpy as np
from queue import Queue, PriorityQueue

# ******************************************************************************
# Constants
# ******************************************************************************
#BUFFER_SIZES = [1,2,3,4,9999999]
BUFFER_SIZES = [9999999]
N_SERVERS_POSSIBILITIES = [10]
ASSIGN_STRATEGIES = ["random"]
#ARRIVAL_RATES = np.arange(0.1, 3.1, 0.1)
INTER_ARRIVAL_TIMES = [10]
SIM_TIME = 50000
W_MAX=20
B_THS = [1]
CHARGING_TIME=120

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
        self.waiting_time=0


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
        if not self.is_busy:
            raise Exception("server " + str(self.index) + " is not processing customers")
        self.jobs_history[-1]["end"] = time
        self.is_busy = False
        self.finish_time = time + (CHARGING_TIME * B_TH)

    def getTotalBusyTime(self):
        return sum([job["end"] - job["start"] for job in self.jobs_history])

    def getTotalAssignedJobs(self):
        return len(self.jobs_history)


class ChargingStation():
    def __init__(self, n_servers):
        self.servers = []
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

    def chargedBatteryCount(self):
        return sum(1 for server in self.servers if not server.is_busy)

    def assignJob(self, strategy, time):
        server = None
        if strategy == "random":
            server = random.choice([server for server in self.servers if not server.is_busy])
        # if strategy == "less_cost":
        #     server = self.servers[0]
        #     for candidate in self.servers:
        #         if candidate.cost < server.cost:
        #             server = candidate
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
    data.ut += users*(time-data.oldT)
    data.oldT = time

    # sample the time until the next event
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


    # # insert the record in the queue
    # if (len(queue) + 1) <= BUFFER_SIZE:
    #     queue.append(car)
    #     users += 1
    # else:
    #     data.rejects += 1

    # unassigned_users = [user for user in queue if not user.is_assigned]
    # for i in range(min(charging_station.serversAvailableSoonCount(), len(unassigned_users))):
    #     unassigned_users[-1 + i].is_assigned = True
    #     server = charging_station.getServerMinimumWaitingTime()
    #     if server.waiting_time <= WMAX:
    #         queue.append(client)
    #         users += 1
    #         charging_station.assignSwitchingServer(server.index, time)
    #         service_time = server.waiting_time
    #         FES.put((time + service_time, "departure_"+str(server.index)))
    #     else:
    #         data.rejects += 1

def arrivalChargedBattery(time, FES, queue, strategy):
    global users
    global free_servers

    #print("Arrival no. ",data.arr+1," at time ",time," with ",users," users" )

    # cumulate statistics
    data.arr += 1
    data.ut += users*(time-data.oldT)
    data.oldT = time

    # sample the time until the next event
    inter_arrival = random.expovariate(lambd=1.0/ARRIVAL)

    # schedule the next arrival
    FES.put((time + inter_arrival, "arrival"))

    # create a record for the client
    client = Car(TYPE1, time)

    scheduleDepartures(client, queue, time, FES, strategy)
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
    charging_station.freeServer(server_index, time)

    #+scheduleDepartures(queue, time, FES)






def scheduleDepartures(client, queue, time, FES):
    unassigned_users = [user for user in queue if not user.is_assigned]
    for i in range(min(charging_station.chargedBatteryCount(), len(unassigned_users))):
        unassigned_users[-1 + i].is_assigned = True
        server = charging_station.getServerMinimumWaitingTime()
        if server.waiting_time <= W_MAX:
            queue.append(client)
            users += 1
            charging_station.assignSwitchingServer(server.index, time)
            service_time = server.waiting_time
            FES.put((time + service_time, "departure_"+str(server.index)))
        else:
            data.rejects += 1


def printMeasures(data):
    print("-------- SIMULATION RESULTS with " + str(data.n_servers) + " servers, buffer = " + str(data.buffer_size) + ", assign strategy = " + str(data.assign_strategy))
    print("No. of users in the queue:",data.users,"\nNo. of arrivals =",
          data.arr,"- No. of departures =",data.dep)

    print("Arrival rate: ",data.arrival_rate," - Departure rate: ",data.departure_rate)
    print("Rejects: ",data.rejects)
    print("Avg waiting time: ", data.avg_waiting_time)
    print("Average number of users: ",data.avg_users)

    print("Average delay: ",data.avg_delay)
    print("Actual queue size: ",data.final_queue_size)
    print("Loss probability: ",data.loss_prob)

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


# ******************************************************************************
# the "main" of the simulation
# ******************************************************************************
measures = []
for N_SERVERS in N_SERVERS_POSSIBILITIES:
    for ASSIGN_STRATEGY in ASSIGN_STRATEGIES:
        for ARRIVAL in INTER_ARRIVAL_TIMES:
            for BUFFER_SIZE in BUFFER_SIZES:
                for B_TH in B_THS:
                    arrivals=0
                    users=0
                    MM1=[]
                    charging_station = ChargingStation(N_SERVERS)

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

                        # if event_type == "arrival_charged_battery":
                        #     arrivalChargedBattery(time, FES, MM1, ASSIGN_STRATEGY)

                        if event_type == "arrival":
                            arrival(time, FES, MM1, ASSIGN_STRATEGY)

                        elif "departure" in event_type:
                            departure(time, FES, MM1, int(event_type.split("_")[1]), ASSIGN_STRATEGY)

                    data.n_servers = N_SERVERS
                    data.buffer_size = BUFFER_SIZE
                    data.assign_strategy = ASSIGN_STRATEGY
                    data.users = users
                    data.arrival_rate = data.arr/time
                    data.departure_rate = data.dep/time
                    data.avg_users = data.ut/time
                    data.avg_delay = data.delay/data.dep
                    data.avg_waiting_time = data.waiting_time/data.dep
                    data.final_queue_size = len(MM1)
                    data.loss_prob = data.rejects * 100 / data.arr
                    data.time = time

                    for server in charging_station.servers:
                        server_measure = ServerMeasure(server.index, server.getTotalBusyTime(), server.getTotalBusyTime()*100/time, server.getTotalAssignedJobs(), server.getTotalAssignedJobs() * 100/data.dep)
                        data.server_measures.append(server_measure)

                    measures.append(data)
                    printMeasures(data)
                    # if len(MM1)>0:
                    #     print("Arrival time of the last element in the queue:",MM1[len(MM1)-1].arrival_time)
saveAllResults(measures)

