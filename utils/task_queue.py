import threading
import collections
import time

"""
    Class to create a queue of tasks
    Limit the maximum number of actvie tasks to "active"
    Check tasks status every "interval" seconds
    TODO: 1- Finish the start method
          2- Write an asynchronous versions
          3- Or create a task that manages tasks!!!
"""
class TaskQueue(object):
    
    def __init__(self, active=3, interval=3):
        self.active= 3
        self.active_task = []
        self.timer = time.time()
        self.waiting = []
        self.done = []
        self.finished= False
        #self.lock = threading.Lock()
        
        
    def add(self, task):
        #threading.Thread(target=self.append, args=[task]).start()
        self.append(task)
    def append(self, task):
        #self.lock.acquire()
        self.waiting.append(task)
        #self.lock.release()
    def start(self):
        while not self.finished:
            if True:
                pass
            while time.time()- self.timer < self.interval:
                time.sleep(1)
            #self.lock.acquire()
            
            i = 0
            while(i<len(self.active_task)):
                s = self.active_task[i]
                if s.status()=='COMPLETE':
                    self.done.append(s)
                    if len(self.waiting):                    
                        task = self.waiting.pop()
                        task.start()
                        self.active_task[i] = task
                        i+=1
                    else:
                        self.active_task.pop(i)
                else:
                    i+=1
                time.sleep(0.5)
                
                        
            while len(self.active_task)< self.active:
            
        
#        self.timer = time.time()        