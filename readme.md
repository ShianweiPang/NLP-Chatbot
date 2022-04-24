# PyTorch NLTK Chatbot
## _A chabot with conversational function

This course was given [Python Engineer](https://www.youtube.com/channel/UCbXgNpp0jedKWcQiULLbDTA). Find out more from the link given. Additional codes are added to improve the efficiency and some aare the previous learning experience. 

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

# Results of Model Training 
![Epoch results](https://user-images.githubusercontent.com/63900253/164980537-06a1c46a-668f-4259-8c01-ed714c6cc809.png)
