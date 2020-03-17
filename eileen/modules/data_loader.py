import pyBigWig
from itertools import *
import numpy as np

def customSort(val):
      chrom_number = "".join(takewhile(str.isalpha,val[0]))
      start = val[0].replace(chrom_number,"")
      return int(start)

# Data loader that accepts multiple replicates and adds them together
    
class DataLoader(object):
    
    def __init__(self, replicates):
        """
        Args:
            replicates (list of bigwig files)
        """
        self.replicates = replicates
    
    def merge(self):
      bigWigs = []

      # opening bw files.
      for idx in range(len(self.replicates)):
        bigWigs.append(pyBigWig.open(self.replicates[idx]))
        
      chrms = ["chrI","chrII","chrIII","chrIV","chrV","chrVI","chrVII","chrVIII",
               "chrIX","chrX","chrXI","chrXII","chrXIII","chrXIV","chrXV","chrXVI","chrM"]
        
      # retrieve values. save to new array, then save that to new bw file.
      combined_dict = {}
      for c in chrms:
        for j in range(len(bigWigs)):
          intervals = bigWigs[j].intervals(c)
          for i in intervals:
            start = i[0]
            end = i[1]
            val = i[2]
            for k in range(start,end):
              if (c+str(start)) in combined_dict:
                combined_dict[c+str(start)]+=val
              else:
                combined_dict[c+str(start)]=val
      
      # sort combined_dict (sort each chrom, then add to dictionary)
      sorted_dict = {}
      flag = True
      chrom_list = []
      prev_chrom_num = ""

      for key in combined_dict:
        chrom_number = "".join(takewhile(str.isalpha,key))
        if (flag or chrom_number==prev_chrom_num):
          chrom_list.append((key,combined_dict[key]))
        else:
          chrom_list.sort(key=customSort)
          for entry in chrom_list:
            sorted_dict[entry[0]]=entry[1]
          chrom_list = []
          chrom_list.append((key,combined_dict[key]))
        flag = False
        prev_chrom_num=chrom_number
   
      # sorted_dict format
      """
      Key: chrom number + start, ex: "chrI324"
      Value: the sum of the values in the bigwig files
      """
      return sorted_dict 
    
    def getChromNums(self):
      # get an ordered list of the chromosome numbers
      ch_num = []
      for key in self.merge():
          chrom_number = "".join(takewhile(str.isalpha,key))
          ch_num.append(chrom_number)
      return np.array(ch_num)
    
    def getChromStart(self):
      # get an ordered list of the chromosome start positions
      ch_start = []
      for key in self.merge():
        chrom_number = "".join(takewhile(str.isalpha,key))
        start = key.replace(chrom_number,"")
        ch_start.append(int(start))
      return np.array(ch_start)
    
    def getChromEnd(self):
      # get an ordered list of the chromosome end positions. 1 base pair long so end pos is 1 more than start.
      ch_end = []
      for entry in self.getChromStart():
        ch_end.append(entry+1)
      return np.array(ch_end)
    
    def getChromVals(self):
      # get an ordered list of the chromosome values.
      ch_vals = []
      vals_dict = self.merge() 
      for key in vals_dict:
        ch_vals.append(vals_dict[key])
      return np.array(ch_vals)
