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


class TaskManager(object):
    """Class to manage tasks by running a limited number of tasks at a time."""

    def __init__(self, n_active=3, n_waiting=7, interval=2.0, verbose=False):
        """.

        Parameters
        ----------
        n_active : int, optional
            Number of simultaneously active tasks. This is different from
            the maximum number of running tasks in Earth Engine.
            The default is 3.
        n_waiting : int, optional
            Maximum number of tasks in the waiting queue. The default is 7.
        interval : float, optional
            Waiting time between task status check. The default is 2.0.
        verbose : Boolean, optional
            Prints the class operations. The default is False.

        """
        self.active_tasks = []
        self.waiting_tasks = []
        self.n_active = n_active
        self.n_waiting = n_waiting
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
        for t in (self.active_tasks+self.waiting_tasks):
            t.cancel()
        self.running = False
        self.lock.release()
        self.thread.join()

    def busy(self):
        """Return True if active queue the waiting queue are full.

        Returns
        -------
        busy : Boolean
            True is the task manager os busy, False otherwise

        """
        self.lock.acquire()
        busy = (len(self.active_tasks) >= self.n_active and
                len(self.waiting_tasks) >= self.n_waiting)
        self.lock.release()
        return busy

    def submit(self, task):
        """Block the tasks management to add a task.

        Parameters.

        ----------
        task : EE Task

            Adds the task to the waiting queue
        Returns
        -------
        None.

        """
        while(self.busy()):
            time.sleep(self.interval/2)
        self.lock.acquire()
        self.waiting_tasks.append(task)
        self.lock.release()

    def check_tasks(self):
        """Manage the submitted tasks asynchronously in the thread.

        Checks one active task every 2 seconds, if a task finishes running
        then it will start a task from the waiting queue (if there is any).
        Returns
        -------
        None.
        """
        while self.running:
            time.sleep(self.interval)
            self.lock.acquire()
            if len(self.waiting_tasks) > 0:
                if len(self.active_tasks) < self.n_active:
                    task = self.waiting_tasks.pop(0)
                    task.start()
                    self.active_tasks.append(task)
                else:
                    status = self.active_tasks[self.state].status()
                    if status['state'].upper() not in ['READY', 'RUNNING']:
                        if self.verbose:
                            print("Task %s finished with status: %s" % (
                                status['description'], status['state']))
                        task = self.waiting_tasks.pop(0)
                        task.start()
                        if self.verbose:
                            print("Started task %s" %
                                  task.status()['description'])

                        self.active_tasks[self.state] = task
                    self.state = (self.state+1) % self.n_active
            self.lock.release()
