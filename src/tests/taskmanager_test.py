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

import time
import sys
import os
import unittest
import random
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from utils import TaskManager
from dummytask import DummyTask

# run : python -m unittest taskmanager_test.py
class TestTaskManager(unittest.TestCase):
    """Unit testing the TaskManaget.
        Tasks are stills running at the end"""

    def test_added(self):
        """ Assert that total submitted tasks is consistent."""
        n_active = random.randint(3, 10)
        n_waiting = random.randint(5, 15)
        task_manager = TaskManager(n_active=n_active,
                                   n_waiting=n_waiting,
                                   interval=0.1)
        for _ in range(n_active):
            task_manager.submit(DummyTask())
        task_manager.lock.acquire()
        added_tasks = len(task_manager.active_tasks)
        added_tasks += len(task_manager.waiting_tasks)
        task_manager.lock.release()
        self.assertEqual(n_active, added_tasks)

    def test_active(self):
        """Assert that after n_active intervals, n_active tasks are active."""
        n_active = random.randint(3, 10)
        n_waiting = random.randint(5, 15)
        task_manager = TaskManager(n_active=n_active,
                                   n_waiting=n_waiting,
                                   interval=0.01)
        for _ in range(n_active):
            task_manager.submit(DummyTask())
        time.sleep(0.02*n_active)
        task_manager.lock.acquire()
        active_tasks = len(task_manager.active_tasks)
        task_manager.lock.release()
        self.assertEqual(n_active, active_tasks)

    def test_busy(self):
        """Test when TaskManager is busy."""
        n_active = random.randint(3, 6)
        n_waiting = random.randint(3, 6)
        task = DummyTask()  # Controlled task
        task_manager = TaskManager(n_active=n_active,
                                   n_waiting=n_waiting,
                                   interval=0.01)
        # All tasks take at least 5 seconds to finish
        task_manager.submit(task)
        for _ in range(n_active+n_waiting-1):
            task_manager.submit(DummyTask())
        # Enough time to start n_active tasks
        time.sleep(n_active*0.1)
        # task manager should be busy at this stage
        self.assertEqual(task_manager.busy(), True)
        # We force a task to finish
        task.lock.acquire()
        task.state = 2
        task.lock.release()
        time.sleep(0.1)
        # task manager shoudl NOT be busy now
        self.assertEqual(task_manager.busy(), False)
