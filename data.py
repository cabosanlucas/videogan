'''
local Threads = require 'threads'
Threads.serialization('threads.sharedserialize')
'''
import threading
import os

#local unpack = unpack and unpack or table.unpack
class Data:
    def __init__(self, n, dataset_name, opt_):
       self.opt_ = opt_ or {}
       #self = {}
       #for k,v in data:
       #   self[k] = v

       self.randomize = opt_.randomize
       self.result = list()

       donkey_file = None
       if dataset_name == 'simple':
           donkey_file = 'donkey_simple.py'
       elif dataset_name == 'video2':
           donkey_file = 'donkey_video2.py'
       elif dataset_name == 'video3':
           donkey_file = 'donkey_video3.py'
       else:
           exit('Unknown dataset: ' + dataset_name)

       if n > 0:
          options = opt_

          #self.threads = Threads(n,
          #                       function() require 'torch' end,
          #                       def(idx):
          #                          opt = options
          #                          tid = idx
          #                          seed = (opt.manualSeed and opt.manualSeed or 0) + idx
          #                          #torch.manualSeed(seed)
          #                          #torch.setnumthreads(1)
          #                          print('Starting donkey with id: %d seed: %d'.format(tid, seed))
          #                          assert(options, 'options not found')
          #                          assert(opt, 'opt not given')
          #                          os.system('python ' + donkey_file)
          #)
       else:
          if donkey_file:
              os.system('python ' + donkey_file)
          self.threads = {}
          #def: self.threads.addjob(f1, f2) f2(f1())
          #function self.threads.dojob()
          #function self.threads.synchronize()

        n_samples = 0
        #self.threads.addjob(def(): return trainLoader.size(), def(c): nSamples = c )
        #self.threads.synchronize()
        #self._size = nSamples

        jobCount = 0
        for i in range(n):
          self.queueJob()
        #return self

    def queueJob(self):
      self.jobCount = self.jobCount + 1

      if self.randomize > 0:
        self.threads.addjob(#function()
                            #  return trainLoader:sample(opt.batchSize)
                            #end,
                            #self._pushResult
                            )
      else:
        indexStart = (self.jobCount-1) * self.opt['batchSize'] + 1
        indexEnd = (indexStart + self.opt['batchSize'] - 1)
        if indexEnd <= self.size():
            pass
          #self.threads.addjob(def(): return trainLoader.get(indexStart, indexEnd), self._pushResult)

    def _pushResult(self, *args):
       res = {args}
       if res == None:
           pass
          #self.threads:synchronize()
       self.result[0] = res

    def getBatch(self):
       #queue another job
       res = None
       while True:
          self.queueJob()
          #self.threads:dojob()
          res = self.result[1]
          self.result[1] = None
          if type(res) == 'list':
              break
       return tuple(res)

    def size(self):
       return self._size

#return data
