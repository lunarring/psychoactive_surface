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
    def __init__(self, ip_address, max_buffer_size=100000, start_process=True, process_list=["unlabeled_markers"]):
        self.ip_address = ip_address
        self.max_buffer_size = max_buffer_size
        self.marker_data = []
        self.lock = threading.Lock()
        self.running = True
        self.sleep_time = 0.000001
        self.list_raw_packets = []
        self.list_dict_packets = []
        self.rigid_body_labels = ["left_hand", "right_hand", "head", "center", "right_foot", "left_foot"]
        
        # self.list_dict_unlabeled_markers = []
        
        self.process_list = ["rigid_bodies"]
        self.process_list = ["rigid_bodies", "unlabeled_markers", "velocities"]
        self.process_list = ["unlabeled_markers", "velocities"]
        self.process_list = process_list
        
        self.v_last_time = 0
        self.v_sampling_time = 0.01
        self.last_frame_id = 0
        self.list_labels = []
        self.dict_label_idx = {}
        self.set_labels = set()
        self.list_unlabeled = []
        self.list_timestamps = []
        
        self.max_nr_markers = 22
        self.positions = np.zeros([self.max_buffer_size, self.max_nr_markers, 3])*np.nan
        self.velocities = np.zeros([self.max_buffer_size, self.max_nr_markers, 3])
        self.pos_idx = 0
        self.last_timestamp = None
                        
        
        # self.rigid_body_positions = {label:[] for label in self.rigid_body_labels}
        if start_process:
            self.start_process()

    def compute_sq_distances(self, a, b):
        # Calculate differences using broadcasting
        diff = a[:, np.newaxis, :] - b[np.newaxis, :, :]
        # Calculate squared Euclidean distances
        dist_squared = np.sum(diff ** 2, axis=2)
        ## Take square root to get Euclidean distances
        # distances = np.sqrt(dist_squared)
        
        # Create dictionary to store dist_squared with index pairs
        distance_dict = {(i_a, i_b): dist_squared[i_a, i_b] for i_a in range(dist_squared.shape[0]) for i_b in range(dist_squared.shape[1])}
        distance_dict = dict(sorted(distance_dict.items(), key=lambda item: item[1]))
        return distance_dict


    def process_velocities(self):
        last_package = self.get_last()
        if last_package is not None:
            if not last_package['frame_id'] == self.last_frame_id:
                self.last_frame_id = last_package['frame_id'] 
                timestamp = int(last_package['frame_id'])
                self.list_timestamps.append(timestamp)
                dict_unlabeled = last_package['unlabeled_markers'] 
                current_labels = dict_unlabeled.keys()
                # import pdb; pdb.set_trace()
                if not self.list_labels:
                    self.list_labels = list(current_labels)[:self.max_nr_markers]
                    self.set_labels = set(self.list_labels)
                    new_positions = np.array([dict_unlabeled[k] for k in self.list_labels])
                    self.positions[self.pos_idx,:len(self.list_labels),:] = new_positions
                    self.dict_label_idx = dict(zip(self.list_labels,range(len(current_labels))))
                else:                   
                    set_current_labels = set(current_labels)
                    self.set_labels = set(self.list_labels)
                    set_missing_labels = self.set_labels - set_current_labels
                    list_missing_labels = list(set_missing_labels)
                    set_new_labels = set_current_labels - self.set_labels
                    list_new_labels = list(set_new_labels)
                    
                    list_known_labels = list(set_current_labels.intersection(self.set_labels))
                    if list_known_labels:
                        list_known_idx = [self.dict_label_idx[l] for l in list_known_labels]
                        list_known_idx.sort()
                        known_positions = np.array([dict_unlabeled[k] for k in list_known_labels])
                        self.positions[self.pos_idx, list_known_idx, :] = known_positions
                        assert self.pos_idx
                        dt = (timestamp - self.last_timestamp)/1000
                        self.velocities[self.pos_idx,list_known_idx,:] = (self.positions[self.pos_idx,list_known_idx,:] - self.positions[self.pos_idx-1,list_known_idx,:])/dt
                    
                    # print('uggggg')
                    dict_unlabeled_last = self.list_unlabeled[-1]
                    if list_missing_labels and list_new_labels:
                        list_missing_idx = [self.dict_label_idx[l] for l in list_missing_labels]
                        missing_positions = self.positions[self.pos_idx-1, list_missing_idx, :]
                        if np.isnan(missing_positions[0,0]):
                            import pdb; pdb.set_trace()
                        assert not np.isnan(missing_positions[0,0])
                        # missing_positions = np.array([dict_unlabeled_last[k] for i, k in enumerate(list_missing_labels)])
                        new_positions = np.array([dict_unlabeled[k] for i, k in enumerate(list_new_labels)])
                        sq_distances = self.compute_sq_distances(missing_positions, new_positions)
                        # xx
                        for i in sq_distances.keys():
                            missing_idx = i[0]
                            new_idx = i[1]
                            # missing_pos = missing_positions[missing_idx]
                            missing_label = list_missing_labels[missing_idx]
                            new_label = list_new_labels[new_idx]
                            if missing_label in set_missing_labels and new_label in set_new_labels:
                                new_pos = new_positions[new_idx]
                                marker_idx = self.dict_label_idx[missing_label]
                                self.positions[self.pos_idx, marker_idx, :] = new_pos
                                self.dict_label_idx[new_label] = self.dict_label_idx[missing_label]
                                del self.dict_label_idx[missing_label]
                                set_missing_labels.remove(missing_label)
                                set_new_labels.remove(new_label)
                                # print(set_missing_labels)
                                if not set_new_labels or not set_missing_labels:
                                    break
                    if set_missing_labels: # fill last values
                        list_missing_idx = [self.dict_label_idx[ml] for ml in set_missing_labels]
                        self.positions[self.pos_idx,list_missing_idx,:] = self.positions[self.pos_idx-1,list_missing_idx,:]
                        self.velocities[self.pos_idx,list_missing_idx,:] = 0
                            
                # dict_unlabeled_last = dict_unlabeled
                self.list_labels = list(self.dict_label_idx.keys())
                self.list_unlabeled.append(dict_unlabeled)
                self.pos_idx += 1
                self.last_timestamp = timestamp
                if self.pos_idx >= self.max_buffer_size:
                    print('cleaning buffer')
                    self.pos_idx = self.pos_idx // 2
                    self.list_unlabeled = self.list_unlabeled[self.pos_idx:]
                    self.list_timestamps = self.list_timestamps[self.pos_idx:]
                    positions_old = np.copy(self.positions)
                    velocities_old = np.copy(self.velocities)
                    self.positions = np.zeros([self.max_buffer_size, self.max_nr_markers, 3])*np.nan
                    self.velocities = np.zeros([self.max_buffer_size, self.max_nr_markers, 3])
                    self.positions[:self.pos_idx] = positions_old[-self.pos_idx:]
                    self.velocities[:self.pos_idx] = velocities_old[-self.pos_idx:]
                    
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
            # print(output)
            if output == "==========BEGINPACKET==========":
                # print("beginning packet!")
                packet_content = []
            elif output == "==========ENDPACKET==========":
                # print("end packet!")
                self.save_packet(packet_content)
                self.process_packet(packet_content)
                if "velocities" in self.process_list:
                    time_new = time.time()
                    dt = time_new - self.v_last_time
                    if dt > self.v_sampling_time:
                        self.v_last_time = time_new
                        self.process_velocities()
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
        if "labeled_markers" in self.process_list:
            dict_package["labeled_markers"] = self.extract_labeled_markers(data)
        if "unlabeled_markers" in self.process_list or "velocities" in self.process_list:
            dict_package["unlabeled_markers"] = self.extract_unlabeled_markers(data)
        if "rigid_bodies" in self.process_list:
            dict_package["rigid_bodies"] = self.extract_rigid_bodies(data)
        
        self.list_dict_packets.append(dict_package)
        if len(self.list_dict_packets) > self.max_buffer_size:
            self.list_dict_packets = self.list_dict_packets[-self.max_buffer_size//2:]

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
        
    def extract_rigid_bodies(self, data):
        rigid_bodies = {}
         # Labels to extract
        start_index = next((i for i, line in enumerate(data) if "Rigid Bodies" in line), None)
        
        if start_index is not None:
            i = start_index + 1  # Start after the "Rigid Bodies" line
            while i < len(data) and "----------" not in data[i]:  # End at the next separator
                line = data[i].strip()
                if any(label in line for label in self.rigid_body_labels):
                    label = line.split('[')[0].strip()
                    body_id = int(line.split('[ID=')[1].split(' ')[0])
                    error = float(line.split('Error(mm)=')[1].split(' ')[0])
                    tracked = int(line.split('Tracked=')[1].strip(']'))
                    
                    position_line = data[i + 2].strip()  # Position and orientation data line
                    pos_q = position_line.split()
                    position = list(map(float, pos_q[:3]))
                    orientation = list(map(float, pos_q[3:]))
                    
                    rigid_bodies[label] = {
                        "body_id": body_id,
                        "error": error,
                        "tracked": tracked,
                        "position": position,
                        "orientation": orientation
                    }
                    # self.rigid_body_positions[label].append(position)
                i += 3  # Move to the next rigid body entry (skipping position and orientation lines)
        
        return rigid_bodies

    
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
    
    # def get_key_measurements(self):
    #     xm = motive.get_last()['unlabeled_markers']
    #     positions = np.array(list(xm.values()))
        
    #     masses = np.abs(np.random.randn(len(positions),1))
    #     masses = 70 * masses / masses.sum()
    #     total_mass = masses.sum()
        
    #     dict_output = {}
        
    #     try:
    #         # positions = coords
    #         list_positions.append(positions)
    #         try: 
    #             velocities = list_positions[-1] - list_positions[-2]
    #         except:
    #             velocities = positions * 0
    #         list_velocities.append(velocities)
    #         try:
    #             accelerations = list_velocities[-1] - list_velocities[-2]
    #         except:
    #             accelerations = velocities * 0
    #         forces = masses * accelerations
    #         dict_output['center_of_mass'] = np.average(positions, axis=0, weights=masses[:,0])
    #         # center_of_mass = np.average(positions, axis=0)
    #         potential_energy = dict_output['center_of_mass'] * total_mass # check xy
    #         rel_positions = positions - dict_output['center_of_mass']
    #         momenta = masses * velocities
    #         angular_momenta = np.cross(rel_positions, momenta) # gives scales for 2D!
    #         dict_output['total_angular_momentum'] = angular_momenta.sum(axis=0)
            
    #         kinetic_energies = 0.5 * masses * np.linalg.norm(velocities, axis=1)**2
    #         dict_output['total_kinetic_energy'] = kinetic_energies.sum(axis=0)
            
    #         print(f"total_kinetic_energy {dict_output['total_kinetic_energy']}")
    #     except Exception as E:
    #         print(f'fail {E}')
            
    #     return dict_output

    def stop(self):
        print("stopping process!")
        self.running = False
        self.process.kill()
        self.process.wait()
        self.thread.join()
        
    def get_last(self, label=None):
        if len(self.list_dict_packets) == 0:
            return None
        else:
            if label is None:
                return self.list_dict_packets[-1]
            else:
                if label in self.list_dict_packets[-1].keys():
                    return self.list_dict_packets[-1][label]
                else:
                    return None
                
    def get_mean(self, label=0, mean_samples=1):
        if len(self.list_dict_packets) < mean_samples:
            return None
        else:
            if label is None:
                return self.list_dict_packets[-mean_samples -1:-1].mean()
            else:
                if label in self.list_dict_packets[-1].keys():
                    
                    return self.list_dict_packets[-mean_samples-1:-1][label].mean()
                else:
                    return None

    def get_at_index(self, index, label=None):
        if index < 0 or index >= len(self.list_dict_packets):
            return None
        else:
            if label is None:
                return self.list_dict_packets[index]
            else:
                if label in self.list_dict_packets[index].keys():
                    return self.list_dict_packets[index][label]
                else:
                    return None

    def get_range(self, start_index, end_index, label=None):
        if start_index < 0 or end_index >= len(self.list_dict_packets) or start_index > end_index:
            return None
        else:
            range_data = self.list_dict_packets[start_index:end_index + 1]
            if label is not None:
                mean_data = np.mean([packet[label] for packet in range_data if label in packet.keys()], axis=0)
                return mean_data
            
            if label is None:
                return range_data
            else:
                filtered_data = [packet[label] for packet in range_data if label in packet.keys()]
                return filtered_data if filtered_data else None



class RigidBody:
    def __init__(self, motive, label, mass=1000):
        self.motive = motive
        assert label in motive.rigid_body_labels, f"label has to be within: {motive.rigid_body_labels}"
        self.label = label
        self.orientations = []
        self.positions = []
        self.velocities = []
        self.accelerations = []
        self.forces = []
        self.kinetic_energies = []
        self.mass = mass
        self.pos = None
        self.buffer_size = 1000  # Define the buffer size for positions, velocities, accelerations, forces, and kinetic energies

    def updatexxx(self):
        try:
            self.pos = self.motive.list_dict_packets[-1]['rigid_bodies'][self.label]['position']
        except Exception as e:
            pass
        

    def update(self, mean_samples=1):
        
        if mean_samples == 1:
            rigid_bodies_data = self.motive.get_last("rigid_bodies")
        else:
            rigid_bodies_data = self.motive.get_mean("rigid_bodies", mean_samples)
            
            
        if rigid_bodies_data is None:
            return
        if self.label in rigid_bodies_data.keys():
            body_data = rigid_bodies_data[self.label]
            position = np.array(body_data['position'])
            self.positions.append(position)
            orientation = np.array(body_data['orientation'])
            self.orientations.append(orientation)

            if len(self.positions) > 1:
                velocity = self.positions[-1] - self.positions[-2]
                self.velocities.append(velocity)
                if len(self.velocities) > 1:
                    acceleration = self.velocities[-1] - self.velocities[-2]
                    self.accelerations.append(acceleration)
                    if self.mass is not None:
                        force = self.mass * acceleration
                        self.forces.append(force)
                        kinetic_energy = 0.5 * self.mass * np.linalg.norm(velocity)**2
                        self.kinetic_energies.append(kinetic_energy)
            else:
                try:
                    last_velocity = self.velocities[-1] if self.velocities else np.array([0, 0, 0])
                    last_acceleration = self.accelerations[-1] if self.accelerations else np.array([0, 0, 0])
                    last_force = self.forces[-1] if self.forces else np.array([0, 0, 0])
                    last_kinetic_energy = self.kinetic_energies[-1] if self.kinetic_energies else 0
                except IndexError:  # In case the lists are empty and accessing -1 fails
                    last_velocity = np.array([0, 0, 0])
                    last_acceleration = np.array([0, 0, 0])
                    last_force = np.array([0, 0, 0])
                    last_kinetic_energy = 0

                
                if len(self.positions) > self.buffer_size:
                    self.positions.pop(0)
                if len(self.velocities) > self.buffer_size:
                    self.velocities.pop(0)
                if len(self.accelerations) > self.buffer_size:
                    self.accelerations.pop(0)
                if len(self.forces) > self.buffer_size:
                    self.forces.pop(0)
                if len(self.kinetic_energies) > self.buffer_size:
                    self.kinetic_energies.pop(0)

                self.velocities.append(last_velocity)
                self.accelerations.append(last_acceleration)
                self.forces.append(last_force)
                self.kinetic_energies.append(last_kinetic_energy)

        # # Calculate angular velocity and angular acceleration from orientations
        # if len(self.orientations) > 1:
        #     # Calculate angular velocity (difference in orientation between consecutive frames)
        #     angular_velocity = self.orientations[-1] - self.orientations[-2]
        #     self.angular_velocities.append(angular_velocity)
            
        #     if len(self.angular_velocities) > 1:
        #         # Calculate angular acceleration (difference in angular velocity between consecutive frames)
        #         angular_acceleration = self.angular_velocities[-1] - self.angular_velocities[-2]
        #         self.angular_accelerations.append(angular_acceleration)
        # else:
        #     try:
        #         last_angular_velocity = self.angular_velocities[-1] if self.angular_velocities else np.array([0, 0, 0])
        #         last_angular_acceleration = self.angular_accelerations[-1] if self.angular_accelerations else np.array([0, 0, 0])
        #     except IndexError:  # In case the lists are empty and accessing -1 fails
        #         last_angular_velocity = np.array([0, 0, 0])
        #         last_angular_acceleration = np.array([0, 0, 0])

        #     if len(self.orientations) > self.buffer_size:
        #         self.orientations.pop(0)
        #     if len(self.angular_velocities) > self.buffer_size:
        #         self.angular_velocities.pop(0)
        #     if len(self.angular_accelerations) > self.buffer_size:
        #         self.angular_accelerations.pop(0)

        #     self.angular_velocities.append(last_angular_velocity)
        #     self.angular_accelerations.append(last_angular_acceleration)

# def compute_sq_distances(a, b):
#     # Calculate differences using broadcasting
#     diff = a[:, np.newaxis, :] - b[np.newaxis, :, :]
#     # Calculate squared Euclidean distances
#     dist_squared = np.sum(diff ** 2, axis=2)
#     ## Take square root to get Euclidean distances
#     # distances = np.sqrt(dist_squared)
    
#     # Create dictionary to store dist_squared with index pairs
#     distance_dict = {(i_a, i_b): dist_squared[i_a, i_b] for i_a in range(dist_squared.shape[0]) for i_b in range(dist_squared.shape[1])}
#     distance_dict = dict(sorted(distance_dict.items(), key=lambda item: item[1]))
#     return distance_dict

def compute_sq_distances(a, b):
    # Calculate differences using broadcasting
    diff = a[:, np.newaxis, :] - b[np.newaxis, :, :]
    # Calculate squared Euclidean distances
    dist_squared = np.sum(diff ** 2, axis=2)
    ## Take square root to get Euclidean distances
    # distances = np.sqrt(dist_squared)
    
    # Create dictionary to store dist_squared with index pairs
    distance_dict = {(i_a, i_b): dist_squared[i_a, i_b] for i_a in range(dist_squared.shape[0]) for i_b in range(dist_squared.shape[1])}
    distance_dict = dict(sorted(distance_dict.items(), key=lambda item: item[1]))
    return distance_dict

if __name__ == "__main__":
    motive = MarkerTracker('192.168.50.64')
    
    # last_frame_id = 0
    # list_labels = []
    # dict_label_idx = {}
    # set_labels = set()
    # list_unlabeled = []
    # list_timestamps = []
    
    # max_nr_markers = 10
    # max_buffer_size = 10000
    # positions = np.zeros([max_buffer_size, max_nr_markers, 3])*np.nan
    # velocities = np.zeros([max_buffer_size, max_nr_markers, 3])
    # pos_idx = 0
    
    # last_timestamp = None
    
    
    # for _ in range(10000):
    #     time.sleep(0.01)
    #     #%%
    #     last_package = motive.get_last()
    #     if last_package is not None:
    #         if not last_package['frame_id'] == last_frame_id:
    #             last_frame_id = last_package['frame_id'] 
    #             timestamp = int(last_package['frame_id'])
    #             list_timestamps.append(timestamp)
    #             dict_unlabeled = last_package['unlabeled_markers'] 
    #             current_labels = dict_unlabeled.keys()
                
    #             if not list_labels:
    #                 list_labels = list(current_labels)[:max_nr_markers]
    #                 set_labels = set(list_labels)
    #                 new_positions = np.array([dict_unlabeled[k] for k in list_labels])
    #                 positions[pos_idx,:len(list_labels),:] = new_positions
    #                 dict_label_idx = dict(zip(list_labels,range(len(current_labels))))
    #             else:                   
    #                 set_current_labels = set(current_labels)
    #                 set_labels = set(list_labels)
    #                 set_missing_labels = set_labels - set_current_labels
    #                 list_missing_labels = list(set_missing_labels)
    #                 set_new_labels = set_current_labels - set_labels
    #                 list_new_labels = list(set_new_labels)
                    
    #                 list_known_labels = list(set_current_labels.intersection(set_labels))
    #                 print(f'known {list_known_labels}')
    #                 print(f'missing {list_missing_labels}')
    #                 print(f'new {list_new_labels}')
    #                 if list_known_labels:
    #                     list_known_idx = [dict_label_idx[l] for l in list_known_labels]
    #                     list_known_idx.sort()
    #                     known_positions = np.array([dict_unlabeled[k] for k in list_known_labels])
    #                     positions[pos_idx, list_known_idx, :] = known_positions
    #                     assert pos_idx
    #                     dt = (timestamp - last_timestamp)/1000
    #                     velocities[pos_idx,list_known_idx,:] = (positions[pos_idx,list_known_idx,:] - positions[pos_idx-1,list_known_idx,:])/dt
                    
    #                 dict_unlabeled_last = list_unlabeled[-1]
    #                 if list_missing_labels and list_new_labels:
    #                     print('uggggg')
    #                     list_missing_idx = [dict_label_idx[l] for l in list_missing_labels]
    #                     missing_positions = positions[pos_idx-1, list_missing_idx, :]
    #                     assert not np.isnan(missing_positions[0,0])
    #                     # missing_positions = np.array([dict_unlabeled_last[k] for i, k in enumerate(list_missing_labels)])
    #                     new_positions = np.array([dict_unlabeled[k] for i, k in enumerate(list_new_labels)])
    #                     sq_distances = compute_sq_distances(missing_positions, new_positions)
    #                     # xx
    #                     for i in sq_distances.keys():
    #                         missing_idx = i[0]
    #                         new_idx = i[1]
    #                         # missing_pos = missing_positions[missing_idx]
    #                         missing_label = list_missing_labels[missing_idx]
    #                         new_label = list_new_labels[new_idx]
    #                         if missing_label in set_missing_labels and new_label in set_new_labels:
    #                             new_pos = new_positions[new_idx]
    #                             marker_idx = dict_label_idx[missing_label]
    #                             positions[pos_idx, marker_idx, :] = new_pos
    #                             dict_label_idx[new_label] = dict_label_idx[missing_label]
    #                             del dict_label_idx[missing_label]
    #                             set_missing_labels.remove(missing_label)
    #                             set_new_labels.remove(new_label)
    #                             print(set_missing_labels)
    #                             if not set_new_labels or not set_missing_labels:
    #                                 break
    #                 if set_missing_labels: # fill last values
    #                     print('here')
    #                     list_missing_idx = [dict_label_idx[ml] for ml in set_missing_labels]
    #                     positions[pos_idx,list_missing_idx,:] = positions[pos_idx-1,list_missing_idx,:]
                            
    #             # dict_unlabeled_last = dict_unlabeled
    #             list_labels = list(dict_label_idx.keys())
    #             list_unlabeled.append(dict_unlabeled)
    #             pos_idx += 1
    #             last_timestamp = timestamp



#%%
#                 # set_labels = set_labels.union(new_labels)
#                 # print(f'new: {len(new_labels)}, missing: {len(missing_labels)}')
#                 # print(dict_unlabeled)
#                 # if len(missing_labels):
#                 #     xx
#     #     left_hand.update()
        
    #     print(right_hand.positions)
    
        
    
# import numpy as np



    
#%%    
    
    # list_positions = []
    # list_velocities = []
    # maxlen = 1e6
    # masses = 1e3
    # time.sleep(1)
    
    # xm = motive.get_last()['labeled_markers']
    # positions = np.array(list(xm.values()))
    # nmb_markers = positions.shape[0]
    
    # params_rigid_bodies = motive.get_last()['rigid_bodies']
    
    # dict_list_positions = {}
    # dict_list_velocities = {}
    # dict_list_kinetic_energies = {}       
    # dict_masses = {}
    
    # rigid_body_labels = params_rigid_bodies.keys()
    # for body_label in rigid_body_labels:
    #     dict_list_positions[body_label] = []
    #     dict_list_velocities[body_label] = []    
    
        
    # marker_ids = []
    
    # while True:
    #     # continue
    #     # xm = np.array(list(motive.get_last()['rigid_bodies'].values())))
    #     xm = motive.get_last()['labeled_markers']
        
    #     marker_ids = list(xm.keys())
        
    #     params_rigid_bodies = motive.get_last()['rigid_bodies']
        
    #     # do rigid body by rigid body computation
    #     if True:
            
    #         for idx, body_label in enumerate(rigid_body_labels):
    #             positions = np.array(params_rigid_bodies[body_label]['position'])
                
    #             # positions = coords
    #             dict_list_positions[body_label].append(positions)
    #             if len(dict_list_positions[body_label]) > 2:
    #                 velocities = dict_list_positions[body_label][-1] - dict_list_positions[body_label][-2]
    #                 dict_list_velocities[body_label].append(velocities)
    #                 if len(dict_list_velocities[body_label]) > 2:
    #                     accelerations = dict_list_velocities[body_label][-1] - dict_list_velocities[body_label][-2]
                        
    #                 # forces = masses * accelerations
    #                 # center_of_mass = np.average(positions, axis=0, weights=masses[:,0])
    #                 # center_of_mass = np.average(positions, axis=0)
    #                 # potential_energy = center_of_mass * total_mass # check xy
    #                 # rel_positions = positions - center_of_mass
    #                 # momenta = masses * velocities
    #                 # angular_momenta = np.cross(rel_positions, momenta) # gives scales for 2D!
    #                 # total_angular_momentum = angular_momenta.sum(axis=0)
                    
    #                 kinetic_energies = 0.5 * masses * np.linalg.norm(velocities)**2
    #                 dict_list_kinetic_energies[body_label] = kinetic_energies.sum(axis=0)
                    
    #                 # print(f'positions {positions}')
    #                 print(f"total_kinetic_energy: {body_label} {dict_list_kinetic_energies[body_label]}")          
        
        
        
        
    #     time.sleep(0.1)
        
        # masses = np.abs(np.random.randn(len(positions),1))
        # masses = 70 * masses / masses.sum()
        # total_mass = masses.sum()
        
        # positions = coords
        # list_positions.append(positions.copy())
        
        # for marker_id in marker_ids:
        #     list_positions = dict_list_positions[marker_id]
        
        
        # if len(list_positions) > 2:
        #     velocities = list_positions[-1] - list_positions[-2]
        #     list_velocities.append(velocities)
        #     if len(list_velocities) > 2:
        #         accelerations = list_velocities[-1] - list_velocities[-2]
        #         forces = masses * accelerations
        #         center_of_mass = np.average(positions, axis=0, weights=masses[:,0])
        #         # center_of_mass = np.average(positions, axis=0)
        #         potential_energy = center_of_mass * total_mass # check xy
        #         rel_positions = positions - center_of_mass
        #         momenta = masses * velocities
        #         angular_momenta = np.cross(rel_positions, momenta) # gives scales for 2D!
        #         total_angular_momentum = angular_momenta.sum(axis=0)
                
        #         kinetic_energies = 0.5 * masses * np.linalg.norm(velocities, axis=1)**2
        #         total_kinetic_energy = kinetic_energies.sum(axis=0)
                
        #         # print(f"positions {positions}")
        #         print(f"total_kinetic_energy {np.mean(total_kinetic_energy)}")
            

        
      
            
    
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