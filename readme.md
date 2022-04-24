# Training data
- this is use to train the model to identify the type of intent of the utterance
- we cannot pass in the whole sentence into the model, thus we are using the concept of bags of word
- all the input are stored in the list of string

```sh 
["HI","How","are","you]
```

How are you = [0,1,1,1] <= this will be the features to learn
0 (greeting) <= this will be the label for the model 

# Tokenization
- splitting string into meaningful words and verbs

# Stemming
- generate the root form of the word
- connections, connected, connects -> “connect”
- there are different stemmers available which will results in different root form of the words


NLP Preprocessing Pipeline
1. Tokenization
2. Lower case + Stemming
3. Exclude punctuation, symbols
4. Convert to bag of words (X-vector - our training data)

Working with NLTK (Natural Language Toolkit)

