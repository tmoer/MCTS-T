# -*- coding: utf-8 -*-
"""
Hash table

@author: thomas
"""
import numpy

class HashTable:
    ''' Hash table. Stores (hash(index) : Value) combinations '''
    
    def __init__(self):
        self.dict = {}
        
    def process(self,index,value):
        ''' Checks whether index exists in hash table
        If it does: returns the max of the hash table and new estimate
        If it doesn't: stores value in hash table and continues '''
        
        # hash index
        if type(index) is not numpy.ndarray:
            index = numpy.array(index)
        hashed_index = hash(index.tostring())
        
        # interact with dict
        if index in self.dict.keys():
            cur_value = self.dict[hashed_index]
            if value > cur_value:
                # replace entry with value
                self.dict[hashed_index] = value
                return value
            else:
                return cur_value
        else:
            # just store value
            self.dict[hashed_index] = value
            return value
