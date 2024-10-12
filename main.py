import simpy
import random
import numpy as np
import matplotlib.pyplot as plt


Lambda1 = 1.5
Lambda2 = 1.0
X = 0.5
C = 1.0
t = 50
T = 200
N = 2
P = 0.5

queuing_mode = 'FIFO'


class Task:
    def __init__(self, id, priority, arrival_time):
        self.id = id
        self.priority = priority
        self.arrival_time = arrival_time
        self.start_time = None
        self.end_time = None


class Car:
    def __init__(self, env, id, controller, parked_car, X, P):
        self.env = env
        self.id = id
        self.controller = controller
        self.parked_car = parked_car
        self.X = X
        self.P = P
        self.action = env.process(self.run())

    def run(self):
        while True:
            interarrival_time = np.random.exponential(1.0 / self.X)
            yield self.env.timeout(interarrival_time)
            priority = random.randint(1, 3)
            task = Task(f"Car{self.id}_Task{self.env.now}", priority, self.env.now)
            self.controller.add_task(task)


class Controller:
    def __init__(self, env, Lambda1, parked_car, C, t, P, queuing_mode, N):
        self.env = env
        self.queue = []
        self.Lambda1 = Lambda1
        self.parked_car = parked_car
        self.C = C
        self.t = t
        self.P = P
        self.queuing_mode = queuing_mode
        self.N = N
        self.processors = [env.process(self.run_processor(i)) for i in range(N)]
        self.queue_length = []
        self.wait_times = []
        self.processing_times = {i: [] for i in range(N)}
        self.total_processed = 0
        self.times = []

    def add_task(self, task):
        self.queue.append(task)
        print(f"Task {task.id} added to queue at time {self.env.now}")

    def run_processor(self, processor_id):
        while True:
            if self.queue:
                task = self.next_task()
                task.start_time = self.env.now
                processing_time = float(np.random.exponential(1.0 / self.Lambda1))
                self.processing_times[processor_id].append(processing_time)
                yield self.env.timeout(processing_time)
                task.end_time = self.env.now
                self.wait_times.append(task.end_time - task.arrival_time)
                print(f"Task {task.id} started processing at {task.start_time} and ended at {task.end_time}")
                self.total_processed += 1
                if self.env.now >= self.t and random.random() < self.P:
                    print(f"Task {task.id} is being transferred to parked car with overhead {self.C}")
                    yield self.env.timeout(self.C)
                    self.parked_car.add_task(task)
                else:
                    self.finish_task(task)
            else:
                yield self.env.timeout(1)
            self.queue_length.append(len(self.queue))
            self.times.append(self.env.now)

    def next_task(self):
        if self.queuing_mode == 'FIFO':
            return self.queue.pop(0)
        elif self.queuing_mode == 'WRR':
            return self.weighted_round_robin()
        elif self.queuing_mode == 'NPPS':
            return self.non_preemptive_priority()
        else:
            raise ValueError("Invalid queuing mode")

    def weighted_round_robin(self):
        weights = {1: 3, 2: 2, 3: 1}
        priorities = [task.priority for task in self.queue]
        weighted_priorities = [weights[p] for p in priorities]
        total_weight = sum(weighted_priorities)
        random_choice = random.uniform(0, total_weight)
        cumulative_weight = 0
        for task in self.queue:
            cumulative_weight += weights[task.priority]
            if cumulative_weight >= random_choice:
                self.queue.remove(task)
                return task

    def non_preemptive_priority(self):
        self.queue.sort(key=lambda x: x.priority)
        return self.queue.pop(0)

    def finish_task(self, task):
        print(f"Task {task.id} completed at time {task.end_time}")


class ParkedCar:
    def __init__(self, env, Lambda2):
        self.env = env
        self.queue = []
        self.Lambda2 = Lambda2
        self.processor = env.process(self.run())
        self.queue_length = []
        self.wait_times = []
        self.processing_times = []
        self.total_processed = 0
        self.times = []

    def add_task(self, task):
        self.queue.append(task)
        print(f"Task {task.id} added to parked car at time {self.env.now}")

    def run(self):
        while True:
            self.queue_length.append(len(self.queue))
            self.times.append(self.env.now)
            if self.queue:
                task = self.non_preemptive_priority()
                task.start_time = self.env.now
                processing_time = float(np.random.exponential(1.0 / self.Lambda2))
                self.processing_times.append(processing_time)
                yield self.env.timeout(processing_time)
                task.end_time = self.env.now
                self.wait_times.append(task.end_time - task.arrival_time)
                print(f"Task {task.id} completed at time {task.end_time}")
                self.total_processed += 1
            else:
                yield self.env.timeout(1)

    def non_preemptive_priority(self):
        self.queue.sort(key=lambda x: x.priority)
        return self.queue.pop(0)


env = simpy.Environment()
parked_car = ParkedCar(env, Lambda2)
controller = Controller(env, Lambda1, parked_car, C, t, P, queuing_mode, N)
cars = [Car(env, i, controller, parked_car, X, P) for i in range(3)]
env.run(until=T)

average_queue_length_controller = np.mean(controller.queue_length)
average_queue_length_parked_car = np.mean(parked_car.queue_length)
average_wait_time_controller = np.mean(controller.wait_times)
average_wait_time_parked_car = np.mean(parked_car.wait_times)
efficiency_controller = {i: sum(times) / T for i, times in controller.processing_times.items()}
throughput_controller = {i: len(controller.processing_times[i]) / T for i in range(N)}
efficiency_parked_car = sum(parked_car.processing_times) / (T - t) if T > t else 0
throughput_parked_car = parked_car.total_processed / (T - t) if T > t else 0
all_wait_times = controller.wait_times + parked_car.wait_times
counts, bin_edges = np.histogram(all_wait_times, bins=100, density=True)
cdf = np.cumsum(counts)
cdf = cdf / cdf[-1]


plt.plot(bin_edges[1:], cdf)
plt.xlabel('Time spent in queue')
plt.ylabel('CDF')
plt.title('CDF of Time Spent in Queues')
plt.show()


def plot_queue_lengths(controller, parked_car):
    plt.figure(figsize=(12, 6))
    plt.plot(controller.times, controller.queue_length, label='Controller Queue Length')
    plt.plot(parked_car.times, parked_car.queue_length, label='Parked Car Queue Length')
    plt.xlabel('Time')
    plt.ylabel('Queue Length')
    plt.title('Queue Lengths Over Time')
    plt.xlim(0, T)
    plt.legend()
    plt.show()


def plot_wait_times(controller, parked_car):
    plt.figure(figsize=(12, 6))
    plt.hist(controller.wait_times, bins=50, alpha=0.5, label='Controller Wait Times')
    plt.hist(parked_car.wait_times, bins=50, alpha=0.5, label='Parked Car Wait Times')
    plt.xlabel('Wait Time')
    plt.ylabel('Frequency')
    plt.title('Histogram of Wait Times')
    plt.legend()
    plt.show()


plot_queue_lengths(controller, parked_car)
plot_wait_times(controller, parked_car)

print(f"Average queue length (Controller): {average_queue_length_controller}")
print(f"Average queue length (Parked Car): {average_queue_length_parked_car}")
print(f"Average wait time (Controller): {average_wait_time_controller}")
print(f"Average wait time (Parked Car): {average_wait_time_parked_car}")
for i, eff in efficiency_controller.items():
    print(f"Efficiency (Controller Processor {i}): {eff}")
    print(f"Throughput (Controller Processor {i}): {throughput_controller[i]}")
print(f"Efficiency (Parked Car): {efficiency_parked_car}")
print(f"Throughput (Parked Car): {throughput_parked_car}")
