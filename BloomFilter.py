import math
import mmh3
from bitarray import bitarray

class BloomFilter(object):
    def __init__(self, items_count, fp_prob):
        ''' 
        items_count : int 
            Number of items expected to be stored in bloom filter 
        fp_prob : float 
            False Positive probability in decimal 
        '''
        # False posible probability in decimal 
        self.fp_prob = fp_prob 
  
        # Size of bit array to use 
        self.size = self.get_size(items_count,fp_prob) 
  
        # number of hash functions to use 
        self.hash_count = self.get_hash_count(self.size,items_count) 
  
        # Bit array of given size 
        self.bit_array = bitarray(self.size) 
  
        # initialize all bits as 0 
        self.bit_array.setall(0) 

    def add(self, item):
        '''
        Add an item in the filter
        '''
        digests = []
        for i in range(self.hash_count):
            # create digest
            # i-th number works as the seed to mmh3.hash() function
            # With a different seed, basically it's a new hash function
            digest = mmh3.hash(item, i) % self.size
            digests.append(digest)
            
            # Update the bit to true in bit_array
            self.bit_array[digest] = True

    def check(self, item):
        for i in range(self.hash_count):
            digest = mmh3.hash(item, i) % self.size
            if self.bit_array[digest] == False:
                # if any bit is false, immediately return that it's not in the set
                return False
        return True

    @classmethod
    def get_size(self, n, p):
        '''
        Return the size of the bit array to be used
        n: # of items expected to be stored in filter
        p: False Prositive probability in decimal
        '''
        m = -(n*math.log(p))/(math.log(2) ** 2)
        return int(m)
    
    @classmethod
    def get_hash_count(self, m, n):
        '''
        Return the hash function to be used
        m: size of bit array
        n: number of items expected to be stored in filter
        '''
        k = (m/n) * math.log(2)
        return int(k)

