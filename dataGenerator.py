import json
from sklearn.preprocessing import OneHotEncoder
import numpy as np
import gensim
import tensorflow as tf
import fasttext

class DataGenerator:

    def __init__(self, Data, embeddingPath,wordSymTablePath):
        """
        Initializes the generator.
        """
        self.i = 0
        self.Data = Data
        self.embeddingPath = embeddingPath
        self.wordSymTablePath = wordSymTablePath
        
        #extracting data
        self.sentences = [self.Data[i]['utterance'] for i in range(len(Data))]
        self.one_hot_encoding()
        self.PregeneratedEmbedding()
        
        pass
    
    def getSequence(self):
        
        sentences = np.array(self.sentences)
        self.words = [ gensim.utils.simple_preprocess(sentence.astype(str), deacc=False, min_len=1, max_len=50) 
                      for sentence in sentences]
        
        IDs = [self.word2int.get(word, -1) for word in self.words[self.i]]
        x = self.embeddings[IDs,...]
        y = self.labels[self.i] 
        return x,y
        
        
    def one_hot_encoding(self):
        
        #one hot encoding of the labels
        annotations = [self.Data[i]['intent'] for i in range(len(self.Data))]
        enc = OneHotEncoder(handle_unknown='ignore')
        annotations = np.array(annotations).reshape(-1,1)
        enc.fit(annotations)
        self.labels = enc.transform(annotations).toarray()
        
    def PregeneratedEmbedding(self):

        # Loading word symbol table.
        with open(self.wordSymTablePath, 'r') as f:
            lines = f.readlines()

        self.word2int = { line.split()[0].strip() : int(line.split()[1])
                            for line in lines }

        self.embeddings, mean, std = np.load(self.embeddingPath).values()

        # Adding a zero valued vector to the end for OOVs.
        embeddingDim = self.embeddings.shape[-1]
        self.embeddings = np.concatenate([
            self.embeddings, np.zeros((1, embeddingDim)),
        ], axis=0)

        
    def __len__(self):
        """
        Returns the length of the generator.
        """
        return len(self.sentences)
    
    def reset(self):
        self.i = 0

    def __iter__(self):
        """
        Resets any required values or parameters before
        the start of a iteration.
        """
        return self

    def __call__(self):
        return self

    def __next__(self):
        """
        Returns the next sample in the form (x, y) where x and y could be numpy
        arrays or dictionaries of numpy arrays.
        """
        if self.i >= len(self.sentences):
            raise StopIteration
            
        seq = self.getSequence()
        self.i+= 1
        return seq
    

    def outputSignature(self):
        
        #self.reset()
        #X, Y = self.getSequence()

        #xShape = [None] + list(x.shape[1:])
        #yShape = [None] + list(y.shape[1:])

        xSpec = tf.TensorSpec(shape=[None,300], dtype=tf.float32)
        ySpec = tf.TensorSpec(shape=[5], dtype=tf.float32)

        return xSpec, ySpec 
    
    def paddedShape(self):
        
        #self.reset()
        #X, Y = self.getSequence()

        xShape = [-1,300]
        yShape = [5]

        return xShape, yShape 
    

if __name__ == "__main__":
    
    datasetPath = "/home/rahim/exp/2022-04-10-call_summarization/output.json"
    embeddingPath = "/home/rahim/exp/2022-04-10-call_summarization/embeddings.npz"
    wordSymTablePath = "/home/rahim/exp/2022-04-10-call_summarization/words.txt"
    #trainFileIDPath = "/home/shahruk/exp/cow21.3/20211020-data-prep/prepared-20211030/normalized/tscd_train_file_id_list.txt"
    batchSize = 32

    Data  = [json.loads(line) for line in open(datasetPath, 'r')]

    gen = DataGenerator(Data,embeddingPath,wordSymTablePath)

    dataset = tf.data.Dataset.from_generator(
        gen, output_signature=gen.outputSignature(),
    ).padded_batch(
        batch_size=batchSize,
        padded_shapes = gen.paddedShape(),
    ).prefetch(batchSize)

    steps = 0
    for (x,y) in dataset:
        steps += 1
        print(x.shape)
        print(y.shape)


    print(f"generator length={len(gen)}")
    print(f"generator output signature={gen.outputSignature()}")
    print(f"generator length={len(gen)}")
    print(f"generator steps with batch_size@{batchSize}={steps}")