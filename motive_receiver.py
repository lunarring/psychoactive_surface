#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import subprocess
import time
import re
import threading

class MarkerTracker:
    def __init__(self, ip_address):
        self.ip_address = ip_address
        self.marker_data = []
        self.lock = threading.Lock()
        self.running = True
        self.sleep_time = 0.001

    def parse_marker_info(self, s):
        marker_id_pattern = r"MarkerID=(\d+)"
        pos_pattern = r"pos=(-?\d+\.\d+),(-?\d+\.\d+),(-?\d+\.\d+)"

        marker_id_match = re.search(marker_id_pattern, s)
        pos_match = re.search(pos_pattern, s)

        if marker_id_match and pos_match:
            marker_id = int(marker_id_match.group(1))
            x = float(pos_match.group(1))
            y = float(pos_match.group(2))
            z = float(pos_match.group(3))
            return {'marker_id': marker_id, 'x': x, 'y': y, 'z': z}
        return {}

    def start_process(self):
        self.process = subprocess.Popen(["./SampleClient", f"{self.ip_address}"],
                                        stdout=subprocess.PIPE, text=True)
        self.thread = threading.Thread(target=self.read_output)
        self.thread.start()

    def read_output(self):
        time.sleep(self.sleep_time)  # Allow time for process initialization
        try:
            while self.running:
                output = self.process.stdout.readline()
                if output == '':
                    continue
                
                output = output.strip()
                if 'Labeled Marker' in output:
                    marker_dict = self.parse_marker_info(output)
                    if marker_dict:
                        with self.lock:
                            self.marker_data.append(marker_dict)
        finally:
            self.process.kill()
            self.process.wait()

    def stop(self):
        self.running = False
        self.thread.join()

    def get_marker_data(self):
        with self.lock:
            return self.marker_data.copy()  # Return a copy to prevent modification

if __name__ == "__main__":
    tracker = MarkerTracker('192.168.50.64')
    try:
        tracker.start_process()
        while True:
            time.sleep(1)  # Main thread can perform other tasks or just sleep
    except KeyboardInterrupt:
        tracker.stop()
        print("Tracker stopped.")
        print(tracker.get_marker_data())  # Output the collected marker data
