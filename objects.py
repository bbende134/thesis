# -*- coding: utf-8 -*-
"""
Created on Fri Aug  5 09:01:41 2022

@author: BAR7BP
"""

import numpy as np
import functions as functions
import pickle

class Video:

    def __init__(self, landmark,name):
        self.landmark = [landmark]
        self.name = name
   
    def add_element(self, landmark):
        self.landmark.append(landmark)
        
    def display(self):
        print(self.name, "Hossza: ", len(self.landmark))
        
    def getData(self):
        return self.landmark
    
#%%
    
class PlotData:

    def __init__(self, name, y):
        self.y = [y]
        self.name = name
   
    def add_element(self, y):
        self.y.append(y)
        
    def display(self):
        print(self.name, "Hossza: ", len(self.y))
        
    def getData(self):
        return self.y
#%%

class Vector:

    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z
   
    def display(self):
        print("First number = " + str(self.x))
        print("Second number = " + str(self.y))
        print("Addition of two numbers = " + str(self.z))   
        
    def get_x(self):
        return self.x
    def get_y(self):
        return self.y
    def get_z(self):
        return self.z
    def get_vals(self):
        return [self.x,self.y,self.z]
    
#%%
    
class Vector_arr:

    def __init__(self, x, y, z, name):
        if x != '' or y != '' or z != '':
            self.x = [x]
            self.y = [y]
            self.z = [z]
        else:
            self.x = []
            self.y = []
            self.z = []
        self.name = name
        
    def add_element(self, x, y, z):
        if x != '' and y != '' and z != '':
            self.x.append(x)
            self.y.append(y)
            self.z.append(z)
            
   
    def display(self):
        print("x:" + str(self.x))
        print("y:" + str(self.y))
        print("z:" + str(self.z))   
        
    def mod_vals(self, mod, is_int):
        if is_int:
            for j in range(len(self.x)):
                self.x[j],self.y[j],self.z[j] = np.uint8(float(self.x[j])*mod),np.uint8(float(self.y[j])*mod),np.uint8(float(self.z[j])*mod)
        else:
            for j in range(len(self.x)):
                self.x[j],self.y[j],self.z[j] = (float(self.x[j])*mod),(float(self.y[j])*mod),(float(self.z[j])*mod)
        
    def get_x(self):
        return self.x
    def get_y(self):
        return self.y
    def get_z(self):
        return self.z
    def get_name(self):
        return self.name
    def get_vals(self):
        return [self.x,self.y,self.z]
    
#%% Exercises

class Exercise:
    
    def __init__(self, names, body_parts_array, time):
        self.names = names
        self.movement = {}
        for i,name in enumerate(names):
            self.movement[name] = [body_parts_array[i]]
        # self.name = names
        # self.movement = [body_parts_array]
        self.time = [functions.create_elapsed_time(time)]
        
    def new_set(self, body_parts_array, time):
        for i,name in enumerate(self.names):
            self.movement[name].append(body_parts_array[i].copy())
        # self.movement([body_parts_array])
        self.time.append(functions.create_elapsed_time(time.copy()))
        
    def get_data(self):
        return self.names, self.movement, self.time
    
#%% Person's object
from matplotlib import pyplot as plt

class Person(Exercise):
    
    def __init__(self):
        try:
            open('training_data.pkl', 'rb')
        except:
            open('training_data.pkl', 'wb')
        print("Person created")
        self.rep_start = False
        
        
    def get_name(self, name="new_name"):
        found = False
        with open('training_data.pkl', 'rb') as inp:
            while True:
                 try:
                     person = pickle.load(inp)
                     findname = person.name
                     if findname == name:
                         print("Welcome back, " + findname + "!")
                         self.max_length(person.dist_max_left, person.dist_max_right)
                         self.min_length(person.dist_min_left, person.dist_min_right)
                         found = True
                 except EOFError:
                     print("Name searching finished")
                     break 
        print("Name: " + name)
        if not found:
            self.name = name
            return False
        else:
            self.name = findname
            return True
        
        
    def max_length(self, dist_max_left, dist_max_right):
        self.dist_max_right = dist_max_right 
        self.dist_max_left = dist_max_left
        print("Max OK")
        
    def min_length(self, dist_min_left, dist_min_right):
        self.dist_min_left = dist_min_left
        self.dist_min_right = dist_min_right
        print("Min OK")
        
    def reps(self, name, body_parts_array, time):
        
        if not self.rep_start:
            super().__init__(name, body_parts_array, time)
            self.rep_start = True
        else:
            super().new_set(body_parts_array, time)
        return True
    
    def get_dists(self):
        return [self.dist_min_left, self.dist_min_right,self.dist_max_left, self.dist_max_right]
    
    def plot_data(self):
        names, movement, time = super().get_data()
        
        dists = self.get_dists()
        for i,name in enumerate(names):
            for j,rep in enumerate(movement[name]):
                time[j].pop()
                if len(rep) > len(time[j]):
                    while len(rep) > len(time[j]):
                        rep.pop()
                elif len(rep) < len(time[j]):
                    while len(rep) < len(time[j]):
                        time[j].pop()
                label = name + str(j+1)
                plt.plot(time[j], rep, label=label)
            #plt.ylim(0,1)
            plt.ylabel("Mediapipe's estimated distance between", size = 20)
            plt.xlabel("Time", size = 20)
            plt.rcParams["figure.figsize"] = (20,10)
            plt.legend()
            plt.title(label=name)
            plt.show()
        
                
                
        # for j,parts in enumerate(temp_parts_movement):
        #     part_name = name[j]
        #     for i,reps in enumerate(parts): 
        #         reps.pop()
        #         time[i].pop()
        #         label = part_name + str(i+1)
        #         plt.plot(time[i], reps, label=label)
        #     plt.legend()
        #     plt.title(label=part_name)
        #     plt.show()
        
    
        
    
    

            
            
            
            