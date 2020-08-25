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

    def __init__(self, n_active=3, verbose=False):
        """
        Parameters.

        ----------
        n_active : int, optional
            Number of simultaneously active tasks. This is different from
            the maximum number of running tasks in Earth Engine.
            The default is 3.
        verbose : Boolean, optional
            Prints the class operations. The default is False.

        Returns
        -------
        None.

        """
        self.active_tasks = []
        self.waiting_tasks = []
        self.timer = time.time()
        self.lock = threading.Lock()
        self.n_active = n_active
        self.state = 0
        self.verbose = True
        self.thread = threading.Thread(target=self.check_tasks, args=())
        self.thread.start()
        self.verbose = verbose

    def submit(self, task):
        """
        Parameters.

        ----------
        task : EE Task

            Adds the task to the waiting queue
        Returns
        -------
        None.

        """
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
        while True:
            time.sleep(2)
            self.lock.acquire()
            if len(self.waiting_tasks) > 0:
                if len(self.active_tasks) < self.n_active:
                    task = self.waiting_tasks.pop()
                    task.start()
                    self.active_tasks.append(task)
                else:
                    status = self.n_active[self.state].status()
                    if status['state'].uppder() != 'RUNNNING':
                        if self.verbose:
                            print("Task %s finished with status: %s" % (
                                status['ID'], status['state']))
                        task = self.waiting_tasks.pop()
                        task.start()
                        self.active_tasks[self.state] = task
                    self.state = (self.state+1) % self.n_active
            self.lock.release()
