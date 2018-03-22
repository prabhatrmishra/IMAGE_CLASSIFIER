import random

documents = [('the movie is very good','positive'),('the movie is very bad','negative')]

random.shuffle(documents)

dataX=[]
dataY=[]
X=[]
Y=[]

for data,label in documents:
        dataX.append(data)
        dataY.append(label)

words=['movie','good','bad','is','very','the']

#create dictionary for mapping words and integers for input
word_to_int_input = dict((c, i+1) for i, c in enumerate(words))
int_to_word_input = dict((i+1, c) for i, c in enumerate(words))

output=['positive','negative']

#create dictionary for mapping words and integers for output
word_to_int_output = dict((c, i) for i, c in enumerate(output))
int_to_word_output = dict((i, c) for i, c in enumerate(output))


#encode words to integers for input
for sentence in dataX:

    sentence=word_tokenize(sentence)
    X.append([word_to_int_input[char] for char in sentence])

#encode words to integers for output
for sentence in dataY:

    Y.append(word_to_int_output[sentence])


#one hot encode
Y = np_utils.to_categorical(Y)

