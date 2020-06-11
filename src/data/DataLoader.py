from __future__ import division
from __future__ import print_function



import numpy as np
from data import preproc as pp
#import preproc as pp
import pandas as pd
import os
import unicodedata
from itertools import groupby


class Sample:
    "single sample from the dataset"
    def __init__(self, gtText, filePath):
        self.gtText = gtText
        self.filePath = filePath
        
class Batch:
    "batch containing images and ground truth texts"
    def __init__(self, gtTexts, imgs):
        self.imgs = np.stack(imgs, axis=0)
        self.gtTexts = gtTexts

class DataLoader:
    "loads data which corresponds to IAM format"

    def __init__(self, filePath, batchSize, imgSize, maxTextLen):
        "loader for dataset at given location, preprocess images and text according to parameters"
        #will me augment the data in anyway?
        self.dataAugmentation = False
        #where does the index start - should always be 0
        self.currIdx = 0
        #self selected batch size
        self.batchSize = batchSize
        #X & Y coordinates of the png
        self.imgSize = imgSize
        #empty list of images to fill with the samples
        self.samples = []
        self.filePath = filePath
        self.maxTextLen = maxTextLen
        self.partitionNames = ['trainSample','validationSample']
        
        df = pd.read_csv('/home/austin/Documents/Github/handwritingnotebook/data/words_csv/2020-06-03 11:39:42.000901.csv')
        chars = set()
        for index, row in df.iterrows():
            # filename: part1-part2-part3 --> part1/part1-part2/part1-part2-part3.png
            fileName = row['file_name']
            # GT text are columns starting at 9
            gtText = row['truth']
            chars = chars.union(set(list(gtText)))
            # put sample into list
            self.samples.append(Sample(gtText, fileName))

        # split into training and validation set: 95% - 5%
        splitIdx = int(0.95 * len(self.samples))
        trainSamples = self.samples[:splitIdx]
        validationSamples = self.samples[splitIdx:]
        
        self.img_partitions = [trainSamples,validationSamples]


        # number of randomly chosen samples per epoch for training 
        self.numTrainSamplesPerEpoch = 25000 


        # list of all chars in dataset
        self.charList = sorted(list(chars))
        
        self.tokenizer = Tokenizer(self.charList,maxTextLen)
        
        self.train_steps = int(np.ceil(len(self.img_partitions[0]) / self.batchSize))
        self.valid_steps = int(np.ceil(len(self.img_partitions[1]) / self.batchSize))
       
    def truncateLabel(self, text):
        # ctc_loss can't compute loss if it cannot find a mapping between text label and input 
        # labels. Repeat letters cost double because of the blank symbol needing to be inserted.
        # If a too-long label is provided, ctc_loss returns an infinite gradient
        cost = 0
        for i in range(len(text)):
            if i != 0 and text[i] == text[i-1]:
                cost += 2
            else:
                cost += 1
            if cost > self.maxTextLen:
                return text[:i]
        return text


    def getIteratorInfo(self):
        "current batch index and overall number of batches"
        return (self.currIdx // self.batchSize + 1, len(self.samples) // self.batchSize)


    def getNext(self,train = True):
        "iterator"
        self.train = train
        if self.train == True:
            j = 0
        else:
            j = 1
        while True:
            if self.currIdx <= len(self.img_partitions[j]):
                index = self.currIdx
                until = self.currIdx + self.batchSize
            else:
                index = self.currIdx
                until = len(self.img_partitions[j])
            imgs = [pp.preprocess(os.path.join(self.filePath,self.img_partitions[j][i].filePath),self.imgSize) for i in range(index,until)]
            imgs = pp.augmentation(imgs,
                                   rotation_range=1.5,
                                   scale_range=0.05,
                                   height_shift_range=0.025,
                                   width_shift_range=0.05,
                                   erode_range=5,
                                   dilate_range=3)
            imgs = pp.normalization(imgs)
            gtTexts = [self.img_partitions[j][i].gtText for i in range(index,until)]
            gtTexts = [self.tokenizer.encode(gtTexts[i]) for i in range(len(gtTexts))]
            gtTexts = [np.pad(i, (0, self.tokenizer.maxlen - len(i))) for i in gtTexts]
            gtTexts = np.asarray(gtTexts, dtype=np.int16)
            yield(imgs,gtTexts)
       
class Tokenizer():
    """Manager tokens functions and charset/dictionary properties"""

    def __init__(self, chars, max_text_length=128):
        self.PAD_TK, self.UNK_TK = "¶", "¤"
        self.chars = str(chars)

        self.PAD = self.chars.find(self.PAD_TK)
        self.UNK = self.chars.find(self.UNK_TK)

        self.vocab_size = len(self.chars)
        self.maxlen = max_text_length

    def encode(self, text):
        """Encode text to vector"""

        text = unicodedata.normalize("NFKD", text).encode("ASCII", "ignore").decode("ASCII")
        text = " ".join(text.split())

        groups = ["".join(group) for _, group in groupby(text)]
        text = "".join([self.UNK_TK.join(list(x)) if len(x) > 1 else x for x in groups])
        encoded = []

        for item in text:
            index = self.chars.find(item)
            index = self.UNK if index == -1 else index
            encoded.append(index)

        return np.asarray(encoded)

    def decode(self, text):
        """Decode vector to text"""

        decoded = "".join([self.chars[int(x)] for x in text if x > -1])
        decoded = self.remove_tokens(decoded)
        decoded = pp.text_standardize(decoded)

        return decoded

    def remove_tokens(self, text):
        """Remove tokens (PAD) from text"""

        return text.replace(self.PAD_TK, "").replace(self.UNK_TK, "")


