#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import subprocess
import time
import re
import threading
import os
import numpy as np

os.chdir("/home/lugo/git/psychoactive_surface/")

class MarkerTracker:
    def __init__(self, ip_address, max_buffer_size=100000, start_process=True):
        self.ip_address = ip_address
        self.max_buffer_size = max_buffer_size
        self.marker_data = []
        self.lock = threading.Lock()
        self.running = True
        self.sleep_time = 0.000001
        self.list_raw_packets = []
        self.list_dict_packets = []
        if start_process:
            self.start_process()

        

    def get_marker_label(self, index):  
        # Check if the index is within the valid range
        if 0 <= index < len(self.marker_labels):
            return self.marker_labels[index]
        else:
            return "Invalid index"
        

    def parse_marker_info(self, s):
        # print(f"running parse_marker_info {len(self.marker_data)}")
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
        packet_content = []
        while self.running:
            time.sleep(self.sleep_time)
            output = self.process.stdout.readline()
            if output == '':
                if not self.running:  # Check if we are supposed to stop.
                    break  # Exit the loop if the output is empty and we're stopping.
                continue

            # self.list_raw_frames.append(output)
            output = output.strip()
            if output == "==========BEGINPACKET==========":
                # print("beginning packet!")
                packet_content = []
            elif output == "==========ENDPACKET==========":
                # print("end packet!")
                self.save_packet(packet_content)
                self.process_packet(packet_content)
            else:
                packet_content.append(output)
                
            # # if 'Timestamp :' in output:
            # #     self.list_raw_frames.append(output)
            # if 'Unlabeled Marker' in output:
            #     marker_dict = self.parse_marker_info(output)
            #     if marker_dict:
            #         with self.lock:
            #             # marker_dict['label'] = self.get_marker_label(marker_dict['marker_id'])
            #             self.marker_data.append(marker_dict)
            #             # Maintain the buffer size
            #             if len(self.marker_data) > self.max_buffer_size:
            #                 self.marker_data.pop(0)  # Remove the oldest element

    def save_packet(self, packet_content):
        self.list_raw_packets.append(packet_content)
        
    def process_packet(self, data):
        
        frame_id, timestamp = self.extract_timestamp(data)
        if frame_id is None:
            return 
        
        # Craete a new dict_package
        dict_package = {"frame_id" : frame_id, "timestamp" : timestamp}
        dict_package["labeled_markers"] = self.extract_labeled_markers(data)
        dict_package["unlabeled_markers"] = self.extract_unlabeled_markers(data)
        
        self.list_dict_packets.append(dict_package)

    def extract_timestamp(self, data):
        frame_id = None
        timestamp = None
        
        # Validate and extract FrameID
        if len(data) > 0 and "FrameID :" in data[0]:
            parts = data[0].split(" : ")
            if len(parts) == 2:
                frame_id = parts[1].strip()
        
        # Validate and extract Timestamp
        if len(data) > 1 and "Timestamp :" in data[1]:
            parts = data[1].split(" : ")
            if len(parts) == 2:
                timestamp = parts[1].strip()
        return frame_id, timestamp
        
    def extract_labeled_markers(self, data):
        labeled_markers = {}
        for line in data:
            if "Labeled Marker" in line:
                # Extract MarkerID and position more accurately
                marker_id_part = line.split('MarkerID=')[1]
                marker_id = int(marker_id_part.split(']')[0].strip())
    
                pos_part = line.split('pos=')[1]
                position_str = pos_part.split(']')[0].strip()
                position = [float(num) for num in position_str.split(',')]
    
                labeled_markers[marker_id] = position
        return labeled_markers    
        
    def extract_unlabeled_markers(self, data):
        unlabeled_markers = {}
        for line in data:
            if "Unlabeled Marker" in line:
                # Extract MarkerID and position more accurately
                marker_id_part = line.split('MarkerID=')[1]
                marker_id = int(marker_id_part.split(']')[0].strip())
    
                pos_part = line.split('pos=')[1]
                position_str = pos_part.split(']')[0].strip()
                position = [float(num) for num in position_str.split(',')]
    
                unlabeled_markers[marker_id] = position
        return unlabeled_markers    

    def stop(self):
        print("stopping process!")
        self.running = False
        self.process.kill()
        self.process.wait()
        self.thread.join()
        
    def get_last(self):
        return self.list_dict_packets[-1]


if __name__ == "__main__":
    motive = MarkerTracker('192.168.50.64')
    
    list_positions = []
    list_velocities = []
    maxlen = 1e6
    masses = np.abs(np.random.randn(4,1))
    masses = 70 * masses / masses.sum()
    total_mass = masses.sum()
    # masses = 1
    
    time.sleep(1)
    
    while True:
        
        # xm = motive.get_last()['labeled_markers']
        
        try:
            # list_p = [xm[k] for k in xm.keys()]
            # positions = np.array(list_p)
            positions = np.random.rand(4,3)

            
            
            
            # positions = coords
            list_positions.append(positions)
            try: 
                velocities = list_positions[-1] - list_positions[-2]
            except:
                velocities = positions * 0
            list_velocities.append(velocities)
            try:
                accelerations = list_velocities[-1] - list_velocities[-2]
            except:
                accelerations = velocities * 0
            forces = masses * accelerations
            center_of_mass = np.average(positions, axis=0, weights=masses[:,0])
            # center_of_mass = np.average(positions, axis=0)
            potential_energy = center_of_mass * total_mass # check xy
            rel_positions = positions - center_of_mass
            momenta = masses * velocities
            angular_momenta = np.cross(rel_positions, momenta) # gives scales for 2D!
            total_angular_momentum = angular_momenta.sum(axis=0)
            
            kinetic_energies = 0.5 * masses * np.linalg.norm(velocities, axis=1)**2
            total_kinetic_energy = kinetic_energies.sum(axis=0)
            
            print(f"total_kinetic_energy {total_kinetic_energy}")
        except Exception as E:
            print(f'fail {E}')
        
    
    # max_height = positions[:,1].max() - positions[:,1].min() # y extension (check!)
        

    
    
    
    # while True:
    #     time.sleep(1)  # Main thread can perform other tasks or just sleep
    #     print(len(marker_tracker.get_marker_data()))  # Output the current buffer size
    #     # print(len(marker_tracker.list_raw_frames))  # Output the current buffer size


"""
            
        
        # List of marker names in order
        self.marker_labels = [
            "HeadTop", "HeadFront", "HeadSide",
            "BackTop", "BackLeft", "BackRight",
            "Chest",
            "LShoulderTop", "LShoulderBack", "LUArmHigh",
            "LElbowOut", "LWristOut", "LWristIn",
            "LHandOut",
            "RShoulderTop", "RShoulderBack", "RUArmHigh",
            "RElbowOut", "RWristOut", "RWristIn",
            "RHandOut",
            "WaistLFront", "WaistLBack", "WaistRBack", "WaistRFront",
            "LThigh", "LKneeOut", "LShin", "LAnkleOut",
            "LHeel", "LToeIn", "LToeTip", "LToeOut",
            "RThigh", "RKneeOut", "RShin", "RAnkleOut",
            "RHeel", "RToeIn", "RToeTip", "RToeOut"
        ]

"""