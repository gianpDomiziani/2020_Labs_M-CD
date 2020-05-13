import simpy
import numpy as np
import random
from runstats import Statistics
import matplotlib.pyplot as plt

from queue import Queue

data_rt = Queue()
# *******************************************************************************
# Constants
# *******************************************************************************
RANDOM_SEED = 42
NUM_SERVERS = 1
SIM_TIME = 1000


# *******************************************************************************
# Arrival process
# *******************************************************************************
def arrival(environment, arrival_rate):
    # keep track of client number client id
    i = 0
    # arrival will continue forever
    while True:
        # sample the time to next arrival
        inter_arrival = random.expovariate(lambd=arrival_rate)

        # yield an event to the simulator
        yield environment.timeout(inter_arrival)

        # a new client arrived
        i += 1
        Client(environment, i)


# *******************************************************************************
# Client
# *******************************************************************************
class Client(object):

    def __init__(self, environment, i):
        self.env = environment
        self.number = i
        self.response_time = 0
        # the client is a "process"
        env.process(self.run())

    def run(self):
        # store the absolute arrival time
        time_arrival = self.env.now
        print("client", self.number, "has arrived at", time_arrival)

        # The client goes to the first server to be served ,now is changed
        # until env.process is complete
        yield env.process(env.servers.serve())

        self.response_time = self.env.now - time_arrival
        print("client", self.number, "response time ", self.response_time)
        stats.push(self.response_time)
        # data_rt.put(self.response_time)


# *******************************************************************************
# Servers
# *******************************************************************************
class Servers(object):

    # constructor
    def __init__(self, environment, num_servers, service_rate):
        self.env = environment
        self.service_rate = service_rate
        self.servers = simpy.Resource(env, num_servers)

    def serve(self):
        # request a server
        with self.servers.request() as request:  # create obj then destroy
            yield request

            # server is free, wait until service is finished
            service_time = random.expovariate(lambd=self.service_rate)

            # yield an event to the simulator
            yield self.env.timeout(service_time)


# *******************************************************************************
# main
# *******************************************************************************
if __name__ == '__main__':

    random.seed(RANDOM_SEED)  # same sequence each time

    mu = 20  # 2 customer on average per unit time (service time)
    lambd = 10  # one customer enter per time (arrival time)
    response_time = []

    # *********************************
    # setup and perform the simulation
    # *********************************

    # env = simpy.Environment()
    # stats = Statistics()

    # servers
    # env.servers = Servers(env, NUM_SERVERS, mu)  # service

    # start the arrival process
    for j in range(lambd):
        env = simpy.Environment()
        stats = Statistics()
        env.servers = Servers(env, NUM_SERVERS, mu)  # service
        env.process(arrival(environment=env, arrival_rate=j + 1))  # customers
        # simulate until SIM_TIME
        env.run(until=SIM_TIME)
        response_time.append(stats.mean())

    print(response_time)
    plt.figure()
    plt.title(f'M/M/{NUM_SERVERS}, service rate: {mu}')
    plt.plot(np.arange(1, lambd + 1), np.array(response_time))
    plt.xlabel("arrival rate")
    plt.ylabel("mean response time")
    plt.grid()
    plt.show()