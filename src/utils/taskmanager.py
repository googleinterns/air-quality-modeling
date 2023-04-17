"""
Copyright 2020 Google LLC.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import threading
import time


class TaskManager:
    """Manages tasks by running a limited number of tasks at a time."""

    def __init__(self, max_active=3, max_waiting=7, interval=2.0,
                 verbose=False):
        """Initializes TaskManager.

        Parameters
        ----------
        max_active : int, optional
            Number of simultaneously active tasks. This is different from
            the maximum number of running tasks in Earth Engine.
            The default is 3.
        max_waiting : int, optional
            Maximum number of tasks in the waiting queue. The default is 7.
        interval : float, optional
            Waiting time between task status check. The default is 2.0.
        verbose : Boolean, optional
            Prints the class operations. The default is False.

        """
        self.active_tasks = []
        self.waiting_tasks = []
        self.max_active = max_active
        self.max_waiting = max_waiting
        self.verbose = verbose
        self.state = 0

        self.interval = interval
        self.lock = threading.Lock()
        self.running = True
        self.thread = threading.Thread(target=self.check_tasks, args=())
        self.thread.start()
        self.verbose = verbose

    def stop(self):
        """Stop TaskManager Thread and cancel all tasks."""
        self.lock.acquire()
        for task in self.active_tasks + self.waiting_tasks:
            task.cancel()
        self.running = False
        self.lock.release()
        self.thread.join()

    def is_busy(self):
        """Returns True if the active queue and the waiting queue are full.

        Returns
        -------
        busy : Boolean
            True if the task manager is busy, False otherwise

        """
        self.lock.acquire()
        busy = len(self.active_tasks) >= self.max_active
        busy = busy and len(self.waiting_tasks) >= self.max_waiting
        self.lock.release()
        return busy

    def submit(self, task):
        """Locks the task manager to add a task to the waiting queue.

        Parameters

        ----------
        task : EE Task
            Adds the task to the waiting queue
        Returns
        -------
        None.

        """
        while self.is_busy():
            time.sleep(self.interval / 2)
        self.lock.acquire()
        self.waiting_tasks.append(task)
        self.lock.release()

    def check_tasks(self):
        """Manages the submitted tasks asynchronously.

        Checks one active task every 2 seconds, if a task finishes running
        then it will start a task from the waiting queue (if there is any).
        Returns
        -------
        None.
        """
        while self.running:
            time.sleep(self.interval)
            self.lock.acquire()
            if len(self.waiting_tasks) == 0:
                continue
            if len(self.active_tasks) < self.max_active:
                #  Empty see  in the active queue
                task = self.waiting_tasks.pop(0)
                task.start()
                self.active_tasks.append(task)
            else:
                #  No empty seat, so check if a task has finished
                status = self.active_tasks[self.state].status()
                if status['state'].upper() in ['READY', 'RUNNING']:
                    continue
                if self.verbose:
                    print("Task %s finished with status: %s" % (
                        status['description'], status['state']))
                # take task from waiting queue
                task = self.waiting_tasks.pop(0)
                # start the task
                task.start()
                # repalce finished task with new task
                self.active_tasks[self.state] = task
                if self.verbose:
                    print("Started task %s" % task.status()['description'])
                # increment the state index
                self.state = (self.state + 1) % self.max_active
            self.lock.release()
