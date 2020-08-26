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
import random
import string
import threading
LETTERS = string.ascii_lowercase
STATES = ["READY", "RUNNING", "COMPLETED"]


class DummyTask:
    """Dummy Task class to emualte EE tasks."""

    def __init__(self, interval=5.0):
        self.description = ''.join([random.choice(LETTERS)
                                    for i in range(10)]).upper()

        self.interval = interval
        self.state = 0
        self.lock = threading.Lock()
        self.running = True
        self.thread = threading.Thread(target=self.run, args=())

    def start(self):
        self.thread.start()

    def cancel(self):
        self.lock.acquire()
        self.running = False
        self.lock.release()
        self.thread.join()

    def run(self):
        while self.running:
            time.sleep((1+random.random())*self.interval)
            self.lock.acquire()
            self.state = (self.state+1) % len(STATES)
            self.lock.release()

    def status(self):
        self.lock.acquire()
        status_dict = {"state": STATES[self.state],
                       "description": self.description}
        self.lock.release()
        return status_dict