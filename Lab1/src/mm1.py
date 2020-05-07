import random
from queue import Queue, PriorityQueue
import matplotlib.pyplot as plt

# ******************************************************************************
# Constants
# ******************************************************************************
LOAD = 0.85
SERVICE = 10.0  # av service time
ARRIVAL = SERVICE / LOAD  # av inter-arrival time
TYPE1 = 1

SIM_TIME = 500000

users = 0
arrivals = 0
BusyServer = False  # True: server is currently busy; False: server is currently idle


Q = Queue()
FES = PriorityQueue()

# ******************************************************************************
# To take the measurements
# ******************************************************************************


class Measure:
    def __init__(self, Narr=0, Ndep=0, NAveraegUser=0, OldTimeEvent=0, AverageDelay=0):
        self.arr = Narr
        self.dep = Ndep
        self.ut = NAveraegUser
        self.oldT = OldTimeEvent
        self.delay = AverageDelay


# ******************************************************************************
# Client
# ******************************************************************************
class Client():
    def __init__(self, arrival_time):
       self.arrival_time = arrival_time


    # arrivals *********************************************************************
    def arrival(self, data):
        global users

        # print("Arrival no. ",data.arr+1," at time ",time," with ",users," users" )

        # cumulate statistics
        data.arr += 1
        data.ut += users * (self.arrival_time - data.oldT)
        data.oldT = self.arrival_time

        # sample the time until the next event
        inter_arrival = random.expovariate(lambd=1.0 / ARRIVAL)

        # schedule the next arrival
        FES.put((self.arrival_time + inter_arrival, "arrival"))

        users += 1

        # create a record for the client
        #client = Client(TYPE1, time)

        # insert the record in the queue
        Q.put(self)

        # if the server is idle start the service
        if users == 1:
            # sample the service time
            service_time = random.expovariate(1.0 / SERVICE)
            # service_time = 1 + random.uniform(0, SEVICE_TIME)

            # schedule when the client will finish the server
            FES.put((self.arrival_time + service_time, "departure"))


# ******************************************************************************
# Server
# ******************************************************************************
class Server():

    # constructor
    def __init__(self, dep_time):
        # whether the server is idle or not
        self.idle = True
        self.time = dep_time


    # departures *******************************************************************
    def departure(self, data):
        global users

        # print("Departure no. ",data.dep+1," at time ",time," with ",users," users" )

        # cumulate statistics
        data.dep += 1
        data.ut += users * (self.time - data.oldT)
        data.oldT = self.time

        # get the first element from the queue
        Client_obj = Q.get()

        # do whatever we need to do when clients go away

        data.delay += (self.time - Client_obj.arrival_time)
        users -= 1

        # see whether there are more clients to in the line
        if users > 0:
            # sample the service time
            service_time = random.expovariate(1.0 / SERVICE)

            # schedule when the client will finish the server
            FES.put((self.time + service_time, "departure"))



# ******************************************************************************
# the "main" of the simulation
# ******************************************************************************

def main():

    random.seed(42)
    data = Measure()


    # the simulation time
    time = 0

    # the list of events in the form: (time, type)
    #FES = PriorityQueue()

    # schedule the first arrival at t=0
    FES.put((0, "arrival"))

    # simulate until the simulated time reaches a constant
    while time <= SIM_TIME:
        (time, event_type) = FES.get()
        if event_type == "arrival":
            client = Client(arrival_time=time)
            client.arrival(data)
        elif event_type == "departure":
            server = Server(dep_time=time)
            server.departure(data)

    # print output data
    print("MEASUREMENTS \n\nNo. of users in the queue:", users, "\nNo. of arrivals =",
      data.arr, "- No. of departures =", data.dep)



    print("Load: ", SERVICE / ARRIVAL)
    print("\nArrival rate: ", data.arr / client.arrival_time, " - Departure rate: ", data.dep / server.time)

    print("\nAverage number of users: ", data.ut / client.arrival_time)

    if data.dep != 0:
        print("Average delay: ", data.delay / data.dep)
    print("Actual queue size: ", Q.qsize())

    if not Q.empty():
        print("Arrival time of the last element in the queue:", Q.get().arrival_time)

    return client, server

if __name__ == '__main__':
    client, server = main()

