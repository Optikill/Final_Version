import simpy
import numpy as np
from scipy import stats


#Global Variables
NumberofResources = 0
NumberofEntityTypes = 0
NumberofArrivals = 0
NumberofDepartures = 0
env = simpy.Environment()

#Class Dictionary
ResourceDict = {} # Dictionary of resources
EntityTypeDict = {} # Dictionary of entities


class Res(simpy.PriorityResource):
    """Resource class based off of :class:`simpy.PriorityResource` with additional functionality."""

    def __init__(self, input_dict, *args, **kwargs):
        super().__init__(env,*args, **kwargs)
        
        self.env = env
        """ Simulation Environment used"""
        self._name = "Res_" + str(NumberofResources)
        """Name of the resource"""
        self._id = NumberofResources
        """Unique identifier of the resource"""
        self._capacity = input_dict["_capacity"]
        """Maximum number of entities that can be assigned to the resource at any given time"""
        self._mu = input_dict["_average_service_time"]
        """Average service time of the resource"""
        self._defer_rate = input_dict["_defer_rate"]
        """Probability that an entity will be deferred after releasing the resource"""

        self._next_task_probability = input_dict["_next_task_probability"]
        """Probability that an entity will be assigned to a given task (None == sequential selection)"""
    
    def request(self, priority):
        """Request the *resource* with a given priority"""
        
        ResourceDict[self._id-1]["res_requests"] += 1
        ResourceDict[self._id-1]["res_utilization"].append(self.count/self._capacity)
        ResourceDict[self._id-1]["res_q_length"].append(len(self.queue))
        
        return super().request(priority)
    
    def release(self, req):
        """Release the *resource*"""

        ResourceDict[self._id-1]["res_releases"] += 1
        ResourceDict[self._id-1]["res_utilization"].append(self.count/self._capacity)
        ResourceDict[self._id-1]["res_q_length"].append(len(self.queue))
        return super().release(req)

class SetofResources(simpy.PriorityResource):
    """Resource class based off of :class:`simpy.PriorityResource` with additional functionality."""

    def __init__(self,listofres,next_task_probability,defer_rate,capacity,*args, **kwargs):
        super().__init__(env,*args, **kwargs)
        
        self.env = env
        """ Simulation Environment used"""

        self.list=listofres.copy()
        """ List of resource's id that make up the resource"""

        self._name="Set composed by "
        """Name of set of the resources"""
        for i in self.list:
            self._name+=str(i)+", "
        
        self._id = NumberofResources
        """Unique identifier of the resource"""

        self.resources=[]
        """ List of resources that make up the resource"""
        for i in listofres:
            self.resources.append(ResourceDict[i-1]['resource'])

        self._next_task_probability=next_task_probability
        """Probability that an entity will be assigned to a given task (None == sequential selection)"""

        self._defer_rate=defer_rate
        """Probability that an entity will be deferred after releasing the resource"""

        self._capacity=capacity
        """Maximum number of entities that can be assigned to the resource at any given time"""

        self.reqs=None
        """ Request to obtain the resources that make up the set"""

    def request(self,priority):
        """Request the set with a given priority """
        ResourceDict[self._id-1]["res_requests"] += 1
        ResourceDict[self._id-1]["res_utilization"].append(self.count/self._capacity)
        ResourceDict[self._id-1]["res_q_length"].append(len(self.queue))

        tasks=[]
        for i in self.list: # Recovery of "resources" objects of the subresources
            task= ResourceDict[i-1]['resource']
            tasks.append(task)
        
        
        self.reqs=tasks.copy()
        return super().request(priority)


    def release(self, req):
        """release the set """
        ResourceDict[self._id-1]["res_releases"] += 1
        ResourceDict[self._id-1]["res_utilization"].append(self.count/self._capacity)
        ResourceDict[self._id-1]["res_q_length"].append(len(self.queue))
        return super().release(req)
    
    def requestresources(self,entity):

        for i in self.reqs:
            yield i
            if entity._show_log == True:
                      print("Retrival: "+ str(entity._name) + ", " +  str(self.resources[self.reqs.index(i)]._name) + " at " + str(entity.env.now))
        
                    
        
    def service_time(self,entity):
        
        for i in self.list:
                yield self.env.timeout(np.random.exponential(1/(ResourceDict[i-1]['resource']._mu)) * 60) 
            


class Entity_Type:
    """An :class:`Entity` type class that share common attributes including arrival rate, starting task, and priority level."""

    def __init__(self, env, input_dict):
        self.env = env
        """ Simulation Environment used"""

        self._dict = input_dict
        """Dictionary of entity type attributes"""

        self._name = "Entity_Type_" + str(NumberofEntityTypes)
        """Name of the entity type"""
        self._id = NumberofEntityTypes
        
        self._arrival_rate = input_dict["_arrival_rate"]
        """Average arrival rate of the entity type"""
        self._start_task = input_dict["_start_task"]
        """Starting task of the entity type"""
        self._priority = input_dict["_priority"]
        """Priority level of the entity type"""
        self._start_delay = input_dict["_start_delay"]
        """Delay before the first entity of the entity type is created"""

        self._max_arrivals = input_dict["_max_arrivals"]
        """Maximum number of arrivals of the entity type"""
        self._max_tasks = input_dict["_max_tasks"]
        """Maximum number of tasks that an entity of the entity type can complete"""

        self._allow_repeats = input_dict["_allow_repeats"]
        """Boolean value indicating whether an entity can be assigned to the same task more than once"""
        self._show_log = input_dict["_show_log"]
        """Boolean value indicating whether entity details should be printed to the console"""

        self.env.process(self.create_entity())
    
    def create_entity(self):
        """Create an entity of the entity type"""
        if self._start_delay != None:
            yield self.env.timeout(self._start_delay) # Delay before first entity is created

        while True:
            if self._max_arrivals != None:
                if EntityTypeDict[self._id-1]["type_arrivals"] < self._max_arrivals:
                    EntityTypeDict[self._id-1]["type_arrivals"] += 1
                else:
                    break
            current_arrival = Entity(self) # Create entity
            if self._show_log:
                print("Arrival: " + current_arrival._name + " at time " + str(self.env.now))
            yield self.env.timeout(np.random.exponential(1/self._arrival_rate) * 60) # Wait for next arrival

class Entity:
    __meta__ = Entity_Type

    def __init__(self, entity_type):
        global NumberofArrivals
        NumberofArrivals += 1

        self.env = entity_type.env
        """ Simulation Environment used"""

        self._name = "Entity_" + str(NumberofArrivals)
        """Name of the entity"""
        self._id = NumberofArrivals
        """Unique identifier of the entity"""
        self._baseclass = entity_type
        """Name of the entity type"""

        self._priority = entity_type._dict["_priority"]
        """Priority level of the entity type"""
        self._deferred = False
        """Boolean value indicating whether the entity has been deferred"""
        self._max_tasks = entity_type._dict["_max_tasks"]
        
        self._arrival_time = self.env.now
        """Time entity arrived"""
        self._departure_time = None
        """Time entity departed"""
        self._task_request = None
        self._task_start = None
        """Time entity started a task"""
        self._task_end = None
        """Time entity ended a task"""
        self._queue_time = []
        """List of times entity spent in queues"""

        self._start_task = entity_type._dict["_start_task"]
        """Starting task of the entity type"""

        if entity_type._dict["_assigned_route"] !=None:
            self._assigned_route = entity_type._dict["_assigned_route"].copy()
        else:
            self._assigned_route = entity_type._dict["_assigned_route"]
        """List of tasks entity has been assigned to"""
        self._route = []
        self._potential_tasks = []


        """List of tasks entity has completed"""
        if entity_type._dict["_assigned_route"] != None:
            self._current_task = self._assigned_route.pop(0)
            """List of tasks entity has been assigned to"""
        else:
            if self._assigned_route != None: # If assigned route is specified, assign as first task
                self._current_task = self._assigned_route.pop(0)
                """Current task of entity"""
            elif self._start_task != None: # If start task is specified, assign as first task
                self._current_task = self._start_task
                """List of tasks entity has been assigned to"""
            else:# If not, start at task 1
                self._current_task = ResourceDict[0]['res_id']
                """List of tasks entity has been assigned to"""
        for i in range(len(ResourceDict)): # Create list of potential tasks
            self._potential_tasks.append(ResourceDict[i]['res_id'])
           
        
                
        self._request_status = False
        """Boolean value indicating whether the entity is currently requesting to a task.\n
            Bug Fix -> prevents entity from requesting multiple tasks at the same time"""
        self._departed = False
        """Boolean value indicating whether the entity has lefted the system\n
            Buf Fix -> prvents entity from exiting the system multiple times"""
        self._allow_repeats = entity_type._dict["_allow_repeats"]
        """Boolean value indicating whether an entity can be assigned to the same task more than once"""
        self._show_log = entity_type._dict["_show_log"]
        """Boolean value indicating whether entity details should be printed to the console"""
        self._show_summary = entity_type._dict["_show_summary"]
        """Boolean value indicating whether entity summary should be printed to the console after the simulation is complete"""

        self.current_task()
    
    def current_task(self):
        """Process the current task"""
        
        if self._request_status == False:
            self.env.process(request_resource(self))

def new_setofresource(listofresources,defer_rate = None, next_task_probability = None):
    global NumberofResources
    NumberofResources += 1

    resources=[]
    for i in listofresources: # Recovery of "resources" objects of the subresources
        resources.append(ResourceDict[i-1]['resource'])
    
    # Add resource record to dictionary
    ResourceDict[NumberofResources-1] = {"res_id": NumberofResources, 
        "resource": SetofResources(listofresources,defer_rate,next_task_probability,1), 
        "res_requests": 0, 
        "res_releases": 0,
        "res_utilization": [], 
        "res_q_length": [], 
        "res_q_time": [], 
        "res_p_time": []}

def new_resource(capacity, average_service_time, defer_rate = None, next_task_probability = None):
    """Function to create a new resource and add it to the ResourceList
    - :att:`capacity` -> number of entities that can be assigned to the resource at one time
    - :att:`average_service_time` -> average time it takes to complete a task
    - :att:`defer_rate` -> probability that an entity will be deferred from a task
    - :att:`next_task_probability` -> probability that an entity will select a given task (None == sequential selection)
    """
    global NumberofResources
    NumberofResources += 1

    input_dict = {"_capacity": capacity, 
                   "_average_service_time": average_service_time, 
                   "_defer_rate": defer_rate, 
                   "_next_task_probability": next_task_probability}
    
    # Add resource record to dictionary
    ResourceDict[NumberofResources-1] = {"res_id": NumberofResources, 
        "resource": Res(input_dict), 
        "res_requests": 0, 
        "res_releases": 0,
        "res_utilization": [], 
        "res_q_length": [], 
        "res_q_time": [], 
        "res_p_time": []}

def new_entity_type(arrival_rate, start_task = None, assigned_route = None, priority = 0, start_delay = None, max_arrivals = None, 
                    max_tasks = None, max_time = None, allow_repeats = False, show_log = False, show_summary= True):
    """Function to create a new entity type and add it to the EntityTypeDict
    - :att:`arrival_rate` -> average number of entities that arrive per hour
    - :att:`start_task` -> task entity starts at
    - :att:`assigned_route` -> list of tasks entity is assigned to
    - :att:`priority` -> priority level of entity
    - :att:`start_delay` -> time delay before first arrival
    - :att:`max_arrivals` -> maximum number of arrivals for specific entity type
    - :att:`max_tasks` -> maximum number of tasks entity can complete before being deferred
    - :att:`max_time` -> maximum time entity will spend in system before exiting
    - :att:`allow_repeats` -> boolean indicating whether entity can repeat tasks
    - :att:`show_log` -> boolean indicating whether entity details should be displayed
    - :att:`show_summary` -> boolean indicating whether entity final summary should be displayed
    """
    global NumberofEntityTypes
    NumberofEntityTypes += 1

    input_dict = {
        "env": env, 
        "_arrival_rate": arrival_rate, 
        "_start_task": start_task,
        "_assigned_route": assigned_route, 
        "_priority": priority, 
        "_start_delay": start_delay,
        "_max_arrivals": max_arrivals,
        "_max_tasks": max_tasks,
        "_max_time": max_time,
        "_allow_repeats": allow_repeats,
        "_show_log": show_log,
        "_show_summary": show_summary
    }
    
    EntityTypeDict[NumberofEntityTypes-1] = {
        "type_id": NumberofEntityTypes, 
        "type": Entity_Type(env, input_dict),
        "type_arrivals": 0,
        "type_departures": 0,
        "type_deferred": 0,
        "type_q_time": [],
        "type_task_time": [],
        "type_system_time": []}

def request_resource(entity):
    """Request a resource"""
    
    try:
        current_task = ResourceDict[entity._current_task-1]['resource'] # Get current task
        
    except KeyError:    
        return

    if not(isinstance(current_task,SetofResources)): # If the resource isn't a set of resources
        
        with current_task.request(entity._priority) as req: # Request of the resource
            
            entity._task_request = entity.env.now # Time entity requested task
            entity._request_status = True 

            if entity._show_log == True:
                    print("Type: "+str(entity._baseclass._id)+" Request: " + str(entity._name) + ", " + str(current_task._name) + " at " + str(entity.env.now))
            yield req # Wait until getting the resource
            
            entity._task_start = entity.env.now # Time entity started task
            entity._queue_time.append(entity._task_start - entity._task_request) # Time entity spent in queue
            EntityTypeDict[entity._baseclass._id-1]["type_q_time"].append(entity._task_start - entity._task_request) # Time entity spent in queue
            ResourceDict[entity._current_task-1]["res_q_time"].append(entity._task_start - entity._task_request) # Time entity spent in queue

            if entity._show_log == True:
                print("Type: "+str(entity._baseclass._id)+" Retrival: "+ str(entity._name) + ", " +  str(current_task._name) + " at " + str(entity._task_start))

            yield current_task.env.timeout(np.random.exponential(1/current_task._mu) * 60) # Wait for service time

            entity._task_end = entity.env.now # Time entity ended task
            EntityTypeDict[entity._baseclass._id-1]["type_task_time"].append(entity._task_end - entity._task_start) # Time entity spent in task
            ResourceDict[entity._current_task-1]["res_p_time"].append(entity._task_end - entity._task_start) # Time entity spent in task


            entity._route.append(entity._current_task) #Add the resource in the entitie's route 
            
            current_task.release(req) # Release the resource

            entity._request_status = False
            if entity._show_log == True:

                print("Type: "+str(entity._baseclass._id)+" Release: " + str(entity._name) + ", " + str(current_task._name) + " at " + str(entity._task_end))

            if current_task._defer_rate != None:
                if np.random.choice([True, False], p=[current_task._defer_rate, 1 - current_task._defer_rate]) == True:
                    entity._deferred = True
                    entity._departure_time = entity.env.now
                    
                    exit_system(entity)
                    return

            if entity._assigned_route != None:
                if len(entity._assigned_route) == 0:
                    entity._departure_time = entity.env.now
                    exit_system(entity)
                    return
                else:
                    entity._current_task = entity._assigned_route.pop(0)
                    entity.current_task()
                    return
            else: # if no assigned route, select next task based on transition probabilities
            
                if len(entity._route) == entity._max_tasks or len(entity._route) == len(ResourceDict):
                    # if entity has completed all tasks or has reached max tasks, exit system
                    entity._departure_time = entity.env.now
                    exit_system(entity)
                    return
                
                if current_task._next_task_probability != None:
                    if entity._allow_repeats == False:
                        p_next = []
                        
                        for i in range(1,len(ResourceDict)+1): # create temporary list of transition probabilities based on entity's potential tasks
                            if i in entity._potential_tasks:
                                p_next.append(current_task._next_task_probability[i-1])
                            elif i in entity._route:
                                p_next.append(0)
                        
                        if sum(p_next) == 0:
                            entity._departure_time = entity.env.now
                            exit_system(entity)
                            return
                        else:
                            entity._current_task = np.random.choice(entity._potential_tasks, p=p_next)
                            entity.current_task()
                else: # if no transition probabilities, go to next task in sequencial list (or loop to first task)
                    try:
                        entity._current_task = ResourceDict[current_task._id ]['res_id']
                    except KeyError:
                        entity._current_task = ResourceDict[1]['res_id']

                    if entity._current_task in entity._route and entity._allow_repeats == False: 
                        entity._departure_time = entity.env.now
                        exit_system(entity)
                        return
                    else:
                        entity.current_task()
    else:
        with current_task.request(entity._priority) as req:# Request of the set of resources
        
            entity._task_request = entity.env.now # Time entity requested task

            entity._request_status = True 
            if entity._show_log == True:
                    print("Type: "+str(entity._baseclass._id)+" Request: " + str(entity._name) + ", " + str(current_task._name) + " at " + str(entity.env.now))

            yield req # Wait until getting the set of resources
            
            entity._task_start = entity.env.now # Time entity started task
            entity._queue_time.append(entity._task_start - entity._task_request) # Time entity spent in queue
            EntityTypeDict[entity._baseclass._id-1]["type_q_time"].append(entity._task_start - entity._task_request) # Time entity spent in queue
            ResourceDict[entity._current_task-1]["res_q_time"].append(entity._task_start - entity._task_request) # Time entity spent in queue

            reqs_sub_resources=[sub_resource.request(entity._priority) for sub_resource in current_task.reqs] # Request of subresources
            time_sub_res=env.now # Time entity requested subtasks

            
            times_sub_start=[]
            for i in reqs_sub_resources:
                yield i# Wait until getting the subresource

                time_sub_res_start=env.now # Time entity started subtask
                entity._queue_time.append(time_sub_res_start - time_sub_res) # Time entity spent in queue's subresource
                EntityTypeDict[entity._baseclass._id-1]["type_q_time"].append(time_sub_res_start - time_sub_res) # Time entity spent in queue's subresource
                ResourceDict[(current_task.reqs[reqs_sub_resources.index(i)])._id-1]["res_q_time"].append(time_sub_res_start - time_sub_res) # Time entity spent in queue's subresource
                times_sub_start.append(time_sub_res_start)
            

            if entity._show_log == True: 
                print("Type: "+str(entity._baseclass._id)+" Retrival: "+ str(entity._name) + ", " +  str(current_task._name) + " at " + str(entity._task_start))
            
            time=0
            for i in current_task.list:
                time+=np.random.exponential(1/(ResourceDict[i-1]['resource']._mu))
                task_end=env.now # Time entity ended subtask
                
                EntityTypeDict[entity._baseclass._id-1]["type_task_time"].append(task_end - times_sub_start[current_task.list.index(i)]) # Time entity spent in task
                ResourceDict[entity._current_task-1]["res_p_time"].append(task_end - times_sub_start[current_task.list.index(i)]) # Time entity spent in task
            
            yield current_task.env.timeout(time * 60)    
            

            entity._task_end = entity.env.now
            EntityTypeDict[entity._baseclass._id-1]["type_task_time"].append(entity._task_end - entity._task_start) # Time entity spent in task
            ResourceDict[entity._current_task-1]["res_p_time"].append(entity._task_end - entity._task_start) # Time entity spent in task
            

            entity._route.append(entity._current_task)# Add the resource in the entitie's route

            for sub_resource in current_task.reqs:
                sub_resource.release(reqs_sub_resources[current_task.reqs.index(sub_resource)])

            current_task.release(req)
            
            entity._request_status = False

            if entity._show_log == True:

                print("Type: "+str(entity._baseclass._id)+" Release: " + str(entity._name) + ", " + str(current_task._name) + " at " + str(entity._task_end))

            if current_task._defer_rate != None:
                if np.random.choice([True, False], p=[current_task._defer_rate, 1 - current_task._defer_rate]) == True:
                    entity._deferred = True
                    entity._departure_time = entity.env.now
                    
                    exit_system(entity)
                    return

            if entity._assigned_route != None:
                if len(entity._assigned_route) == 0:
                    entity._departure_time = entity.env.now
                    exit_system(entity)
                    return
                else:
                    
                    entity._current_task = entity._assigned_route.pop(0)
                    entity.current_task()
                    return
            else: # if no assigned route, select next task based on transition probabilities
            
                if len(entity._route) == entity._max_tasks or len(entity._route) == len(ResourceDict):
                    # if entity has completed all tasks or has reached max tasks, exit system
                    entity._departure_time = entity.env.now
                    exit_system(entity)
                    return
                
                if current_task._next_task_probability != None:
                    if entity._allow_repeats == False:
                        p_next = []
                        
                        for i in range(1,len(ResourceDict)+1): # create temporary list of transition probabilities based on entity's potential tasks
                            if i in entity._potential_tasks:
                                p_next.append(current_task._next_task_probability[i-1])
                            elif i in entity._route:
                                p_next.append(0)
                        
                        if sum(p_next) == 0:
                            entity._departure_time = entity.env.now
                            exit_system(entity)
                            return
                        else:
                            entity._current_task = np.random.choice(entity._potential_tasks, p=p_next)
                            entity.current_task()
                else: # if no transition probabilities, go to next task in sequencial list (or loop to first task)
                    try:
                        entity._current_task = ResourceDict[current_task._id ]['res_id']
                    except KeyError:
                        entity._current_task = ResourceDict[1]['res_id']

                    if entity._current_task in entity._route and entity._allow_repeats == False: 
                        entity._departure_time = entity.env.now
                        exit_system(entity)
                        return
                    else:
                        entity.current_task()

def exit_system(entity):
    """Remove Entity from the system"""
    global NumberofDepartures

    if entity._departed == False:
        NumberofDepartures += 1
        EntityTypeDict[entity._baseclass._id-1]["type_departures"] += 1
        entity._departed = True

        if entity._show_log == True :
            print("Departure: " + str(entity._name) + " at time " + str(entity._departure_time))
        if entity._show_summary == True:
            entity._total_time = entity._departure_time - entity._arrival_time
            """Total time entity spent in the system"""
            entity._total_queue_time = sum(entity._queue_time)
            """Total time entity spent in queues"""
            entity._percent_queue_time = round(entity._total_queue_time/entity._total_time * 100, 2)
            """Percentage of time entity spent in queues"""
            entity._total_task_time = entity._total_time - entity._total_queue_time
            """Total time entity spent completing tasks"""
            entity._percent_task_time = round(entity._total_task_time/entity._total_time * 100, 2)
            """Percentage of time entity spent completing tasks"""

            print(f"{entity._name} -> Arrival Time: {entity._arrival_time}, Departure Time: {entity._departure_time}, Total Time: {entity._total_time}")
            print(f"Route: {entity._route}")
            print(f"Total Queue Time: {entity._total_queue_time} ({entity._percent_queue_time}%), Total Task Time: {entity._total_task_time}, ({entity._percent_task_time}%)\n")
            
    return
    
def calculate_correlation(data):
    """ Correlation calculation """

    x=data[:len(data)-1]
    y=data[1:]
    mean_x = np.mean(x)
    mean_y = np.mean(y)
    numerator = np.sum((x - mean_x) * (y - mean_y))
    denominator = np.sqrt(np.sum((x - mean_x)**2))*np.sqrt(np.sum((y - mean_y)**2))
    autocorrelation_coefficient = numerator / denominator

    return autocorrelation_coefficient

class Stastics_Simulation:
   
    def calc_stats(self,confidence):
        """ Calculates simulation statistics """
        number_batchs=self.nb_batchs
        
        dof = number_batchs - 1 
        """degrees of freedom for the t-distribution"""
        
        t_crit = np.abs(stats.t.ppf((1-confidence)/2,dof))
    
        batch_means = self.data_value
        """ means of batch"""
        average_batch_means = np.mean(batch_means,axis = 0) 
        """means of means"""
        standard_batch_means = np.std(batch_means, axis = 0) 
        """standard deviation"""
        
        inf_conf = average_batch_means-standard_batch_means*t_crit/np.sqrt(number_batchs) 
        sup_conf = average_batch_means+standard_batch_means*t_crit/np.sqrt(number_batchs)
        
        inf_conf = round(float(inf_conf),2)
        sup_conf = round(float(sup_conf),2)
        """Confidence Interval"""
        inf_pred = average_batch_means-standard_batch_means*t_crit/np.sqrt(1+(1/(number_batchs))) 
        sup_pred = average_batch_means+standard_batch_means*t_crit/np.sqrt(1+(1/(number_batchs)))
        
        inf_pred = round(float(inf_pred),2)
        sup_pred = round(float(sup_pred),2)
        """Prediction Interval"""

        print('')
        print(f'{self.name}')
        print('')
        print(f'Number of entity recs : {self.nb_recs}')
        print(f'{number_batchs} batches of {self.nb_recs_per_batch} records were used for calculations')
        print(f'Average : {average_batch_means}')
        print(f'Confidence Interval : [ {inf_conf} ; {sup_conf} ]')
        print(f'Prediction Interval : [ {inf_pred} ; {sup_pred} ]')
        print(f'Correlation Coefficient : {calculate_correlation(batch_means)}')
    def extract_values(self,attribute):
        """ Extraction of values in dictionaries """
        values=[]
        
        for value in self.data:
            if hasattr(value,attribute):
                values.append(getattr(value,attribute))
            else:
                print('ERROR ATTRIBUTE')
        return values



    def __init__(self,data,name,run,Level_Confidence=0):

        self.data=data
        self.nb_recs=len(data)
        self.nb_batchs=len(data)
        self.name=name
        self.run=run

        if run :# Statistics for a batch
            if(len(self.data)!=0):
                self.nb_recs=sum(self.extract_values('nb_recs'))
                self.data_value=self.extract_values('mean')
                self.nb_recs_per_batch=self.data[0].nb_recs
                self.mean=np.mean(self.data_value)   
                self.sum=np.sum(self.data_value)
                self.std=np.std(self.data_value)  
                self.calc_stats(Level_Confidence)
            else :
                print(self.name)
                print("No Data")

        else:
                self.mean=np.mean(data)   
                self.sum=np.sum(data)
                self.std=np.std(data)

def calc_batchs(Data,number_batchs,warm_up_p):
    """ Calculation of batchs """
    time=Data[warm_up_p:]
    """ Eliminating the warm-up period"""

    number_recs = len(time)
    recs_per_batch = int(number_recs/number_batchs)
    """ Number of recs per batch """

    # to guarantee equal number of records in each batch
    matrix_dim = number_batchs*recs_per_batch
    """ Desired total number of records across all batches """
    rows_to_eliminate = number_recs - matrix_dim
    """ Number of record to be removed to ensure all batchs have the same number of record """
    time=time[rows_to_eliminate:]
    """ Remove the extra record from the end to ensure all batchs have the same number of record """
        
    times=[]
    while time !=[]:
        times.append(time[:recs_per_batch])
        """adding the first batch to the matrix"""
        time=time[recs_per_batch:]
   
    return times

def create_objects_stats(batchs,name):
    """ Creation of stats object per batch """
    objetcs_stats=[]
    for batch in batchs:
        objetcs_stats.append(Stastics_Simulation(batch,name,False))
    return objetcs_stats

def Run_Sim(Simulation_time,warm_up,batch,Confidence):
    """Function to run the simulation
    - :att:`Simulation_time` -> Simulation Time
    - :att:`warm_up` -> Warm-up Time
    - :att:`batch` -> Number of Batch
    - :att:`Confidence` -> confidence interval for simulation statistics
    """
    
    env.run(until=Simulation_time)
    
    for i in range(len(ResourceDict)):
            print("===================RESOURCE "+str(i+1)+"====================================")
            Stastics_Simulation(create_objects_stats(calc_batchs(ResourceDict[i]["res_q_time"],batch,warm_up),"RES Q TIME"),"RES Q TIME",True,Confidence)
            Stastics_Simulation(create_objects_stats(calc_batchs(ResourceDict[i]["res_utilization"],batch,warm_up),"RES UTILIZATION"),"RES UTILIZATION",True,Confidence)


    for i in range(len(EntityTypeDict)):
            print("===================ENTITY "+str(i+1)+"====================================")
            Stastics_Simulation(create_objects_stats(calc_batchs(EntityTypeDict[i]["type_q_time"],batch,warm_up),"Q TIME"),"Q TIME",True,Confidence)
            Stastics_Simulation(create_objects_stats(calc_batchs(EntityTypeDict[i]["type_task_time"],batch,warm_up),"TASK TIME"),"TASK TIME",True,Confidence)

