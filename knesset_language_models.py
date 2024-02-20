#By:
#Faisal Omari - 325616894
#Mias Ghantous - 213461692
import pandas as pd 
import numpy as np
import heapq

class corpus:
    def __init__(self,type,corpus):
        try:
            self.type = type
            self.corpus = corpus
            # read the data
            df = pd.read_csv(corpus)

            frequncy_dictionary = {} #frequncy_dictionary[word] = how many did the token appear
            df = df.loc[df['protocol_type'] == type]
            for row_number,row in enumerate(df.itertuples()):
                all_words = str(row[5]).strip().split(" ")
                for word in all_words:
                    if word not in frequncy_dictionary.keys():
                        frequncy_dictionary[word]=1.0# if I dont have an intry make one
                    else:
                        frequncy_dictionary[word]+=1.0
            
            #getting the corpus size
            corpus_size = 0
            for word in frequncy_dictionary.keys():
                corpus_size+=frequncy_dictionary[word]

            # frequncy_dictionary_2_words[word1][word2] is how many did word 1 appear before word 2
            frequncy_dictionary_2_words = {}
            for row_number,row in enumerate(df.itertuples()):
                all_words = str(row[5]).strip().split(" ")
                first_word = None
                for word in all_words:
                    if first_word == None: #if it is the fist word in the sentnce then dont dont do any thing 
                        first_word = word 
                        continue

                    if first_word not in frequncy_dictionary_2_words.keys():#if the first word does not have an entry then make one
                        frequncy_dictionary_2_words[first_word] ={}

                    if word not in frequncy_dictionary_2_words[first_word].keys():# if we dont have an entry to the seond word then make one
                        frequncy_dictionary_2_words[first_word][word] = 0.0
            
                    frequncy_dictionary_2_words[first_word][word] +=1.0
                    first_word = word#change the first word in the bigram

            frequncy_dictionary_3_words = {}
            for row_number,row in enumerate(df.itertuples()):
                all_words = str(row[5]).strip().split(" ")
                first_word = None
                second_word = None

                for word in all_words:

                    if first_word == None:# if we are in the first word of the sentnce then we do not add anything
                        first_word = word
                        continue

                    if second_word == None:# if we are at the second then dont save anythig
                        second_word = word
                        continue
                    #starting from the third we begin to increce the counter
                    #make the entry if we dont have one yet and then increase the counter
                    if first_word not in frequncy_dictionary_3_words.keys(): 
                        frequncy_dictionary_3_words[first_word] ={}
                    if second_word not in frequncy_dictionary_3_words[first_word].keys():
                        frequncy_dictionary_3_words[first_word][second_word] ={}
                    if word not in frequncy_dictionary_3_words[first_word][second_word].keys():
                        frequncy_dictionary_3_words[first_word][second_word][word] = 0.0
                    frequncy_dictionary_3_words[first_word][second_word][word] +=1.0
                    
                    #for the next iteration change the word places in the current 3 words
                    first_word = second_word
                    second_word = word
            # save our computation
            self.frequncy_dictionary,self.frequncy_dictionary_2_words,self.frequncy_dictionary_3_words,self.corpus_size=frequncy_dictionary,frequncy_dictionary_2_words,frequncy_dictionary_3_words,corpus_size
        except Exception as e:
            print('error in __init__: '+str(e))

    def calculate_prob_of_sentence(self,sentence,smoothing = 'Linear'):
        try:
            frequncy_dictionary,frequncy_dictionary_2_words,frequncy_dictionary_3_words,corpus_size = self.frequncy_dictionary,self.frequncy_dictionary_2_words,self.frequncy_dictionary_3_words,self.corpus_size
            #notes
            '''sentence_prop calculation sentece will chane
            still need the progran the second type of smoothing'''
            lambda1 = 0.1
            lambda2 = 0.2
            lambda3 = 0.7
            sentence_token_temp = sentence.split(' ')
            sentence_token = []
            for token in sentence_token_temp:
                if token != '':
                    sentence_token.append(token)
            #smoothing = 'Laplace'

            if 'Laplace' == smoothing:
                V = len(frequncy_dictionary.keys())# number of diffrent words
                sentence_prop  = 0
                for token_number, token in enumerate(sentence_token):
                    if token_number ==0 : #if we are at the fist word in the sentence
                        
                        #claculate the prop of the word
                        sentence_prop = np.log((frequncy_dictionary.get(token,0)+1)/(corpus_size+V))
                    elif token_number == 1:# if we are at the second wrod of the sentnce
                        # then calcualte 
                        sentence_prop += np.log((frequncy_dictionary_2_words.get(sentence_token[0],{}).get(token,0)+1)/(frequncy_dictionary.get(sentence_token[0],0)+V))#may change
                    else:
                        #calcualte
                        first = sentence_token[token_number-2]
                        second = sentence_token[token_number-1]
                        third = token
                        sentence_prop += np.log((frequncy_dictionary_3_words.get(first,{}).get(second,{}).get(third,0)+1)/(frequncy_dictionary_2_words.get(first,{}).get(second,0)+V))#may change
                return float(format(sentence_prop, '.3f'))
            else:
                V = len(frequncy_dictionary.keys())# number of diffrent words
                sentence_prop  = 0
                for token_number, token in enumerate(sentence_token):
                    if token_number ==0 :# if we are at the first word
                        #caluclate
                        sentence_prop = np.log(lambda1*(frequncy_dictionary.get(token,0)+1)/(corpus_size+V))
                    elif token_number == 1:# if we are at the second word
                        #caluclate
                        bi_prob = (lambda2*(frequncy_dictionary_2_words.get(sentence_token[0],{}).get(token,0))/(frequncy_dictionary.get(sentence_token[0],1)))
                        Uni_prob = lambda1*(frequncy_dictionary.get(token,0)+1)/(corpus_size+V)
                        sentence_prop += np.log(bi_prob+Uni_prob)#may change
                    else:
                        first = sentence_token[token_number-2]
                        second = sentence_token[token_number-1]
                        third = token
                        tri_prob = (lambda3*(frequncy_dictionary_3_words.get(first,{}).get(second,{}).get(third,0))/(frequncy_dictionary_2_words.get(first,{}).get(second,1)))
                        bi_prob = (lambda2*(frequncy_dictionary_2_words.get(second,{}).get(third,0))/(frequncy_dictionary.get(second,1)))
                        Uni_prob = (lambda1*(frequncy_dictionary.get(third,0)+1)/(corpus_size+V))
                        sentence_prop += np.log(tri_prob+bi_prob+Uni_prob)#may change
                return float(format(sentence_prop, '.3f'))
        except Exception as e:
            print('error in calculate_prob_of_sentence: '+str(e))

    def get_next_token(self,sentence):
        try:
            frequncy_dictionary,frequncy_dictionary_2_words,frequncy_dictionary_3_words,corpus_size = self.frequncy_dictionary,self.frequncy_dictionary_2_words,self.frequncy_dictionary_3_words,self.corpus_size

            max_prop = -np.infty
            max_token = ''
            # for every possiable token make calcualte the prop of he sentce and return the token that have the highes prop
            sentnce_tokens = sentence.strip().split(' ')
            if len(sentnce_tokens) >2:
                sentnce_tokens = sentnce_tokens[-2:]
                sentence = sentnce_tokens[0]+' '+sentnce_tokens[1]

            for word in frequncy_dictionary.keys():
                if word == ' ' or word == '':
                    continue
                sentece_porp = self.calculate_prob_of_sentence(sentence=f'{sentence.strip()} {word}',smoothing='Linear')
                if sentece_porp >= max_prop: 
                    max_prop = sentece_porp
                    max_token = word

            return max_token
        except Exception as e:
            print('error in get_next_token: '+str(e))

    def  get_k_n_collocations(self,k,n):
        try:
            frequncy_dictionary,frequncy_dictionary_2_words,frequncy_dictionary_3_words,corpus_size = self.frequncy_dictionary,self.frequncy_dictionary_2_words,self.frequncy_dictionary_3_words,self.corpus_size
            collocations_dictionary = {} #the counter for all the possable collocations 
            # read the right type of data 
            df = pd.read_csv(self.corpus)
            df = df.loc[df['protocol_type'] == self.type]


            for row_number in range(len (df)):

                sentence_tokens = df.iloc[row_number]['sentence_text'].strip().split(' ') #all the tokensin the sentence
                sentece_length = len (sentence_tokens)
                #if we have less than n in the sentence then we have nothing to do
                if sentece_length<n:
                    continue

                first_collocation = sentence_tokens[0:n] # the firrst possable collocation is from 0 to (n-1)
                current_collocation_text = ''
                for word in first_collocation: # make them into a string
                    current_collocation_text+=word +' '
                current_collocation_text = current_collocation_text.strip()

                #for every extra word in the sentence
                for i in range(sentece_length-n+1):
                    # if we dont have he entry create one
                    if current_collocation_text not in collocations_dictionary.keys():
                        collocations_dictionary[current_collocation_text] = 0

                    collocations_dictionary [current_collocation_text]+=1 #increace the counter of the collocation 

                    current_collocation_text = current_collocation_text[current_collocation_text.find(' ')+1:]#delete teh first token
                    # if we dont have another word then go to the next sentence
                    if i+n == sentece_length:
                        continue

                    current_collocation_text += " " +sentence_tokens[i+n]#concatenate the next word
            # return the most common k in the dictionary
            return heapq.nlargest(k, collocations_dictionary, key=collocations_dictionary.get)
        except Exception as e:
            print('error in get_k_n_collocations: '+str(e))

def Q2_text (plenary_corpus,committee_corpus):
    try:
        text = 'Two-gram collocations:\n'
        text+= 'Committee corpus:\n'
        results = committee_corpus.get_k_n_collocations(10,2)
        for col in results:
            text+=col +'\n'
        text+='\n'
        text += 'Plenary corpus:\n'
        results = plenary_corpus.get_k_n_collocations(10,2)
        for col in results:
            text+=col +'\n'
        text+='\n'

        text += 'Three-gramcollocations:\n'
        text+= 'Committee corpus:\n'
        results = committee_corpus.get_k_n_collocations(10,3)
        for col in results:
            text+=col +'\n'
        text+='\n'
        text += 'Plenary corpus:\n'
        results = plenary_corpus.get_k_n_collocations(10,3)
        for col in results:
            text+=col +'\n'
        text+='\n'

        text += 'four-gram collocations:\n'
        text+= 'Committee corpus:\n'
        results = committee_corpus.get_k_n_collocations(10,4)
        for col in results:
            text+=col +'\n'
        text+='\n'
        text += 'Plenary corpus:\n'
        results = plenary_corpus.get_k_n_collocations(10,4)
        for col in results:
            text+=col +'\n'
        text+='\n'
        with open('knesset_collocations.txt','w',encoding='UTF-8') as file:
            file.write(text)
    except Exception as e:
        print('error in Q2_text: '+str(e))
        
    

def Q3_text (plenary_corpus,committee_corpus):
    try:
        with open('masked_sentences.txt', 'r',encoding='UTF-8') as file:
        # Read the contents of the file
            file_contents = file.read()
            sentences = file_contents.split('\n')
            text = ''#create the text
            for sentence in sentences:
                if sentence.strip() == '':
                    continue

                text+='Original sentence: '+sentence+ '\n'#print the sentnce 

                
                committee_sentence =''
                committee_tokens=[]
                committee_prop = 0

                plenary_sentence =""
                plenary_tokens=[]
                plenary_prop = 0 

                all_words = sentence.split(' ')
                for word in all_words:
                    if word != '[*]':# if it is a normal word then dont do anything
                        committee_sentence+= word+" "
                        plenary_sentence +=word + ' '
                    else:
                        #get the tokens and append them to the list and the sentnece
                        token_plenary = plenary_corpus.get_next_token(plenary_sentence)
                        token_committee = committee_corpus.get_next_token(committee_sentence)

                        committee_sentence += token_committee+' '
                        committee_tokens.append(token_committee)

                        plenary_sentence += token_plenary+' '
                        plenary_tokens.append(token_plenary)
                
                committee_sentence = committee_sentence.strip()
                plenary_sentence = plenary_sentence.strip()

                #print as the reqired
                text+='Committee sentence:'+committee_sentence+'\n'

                text += 'Committee tokens: ' #+str(committee_tokens) +'\n'
                for current_token in committee_tokens:
                    text+=f'{current_token},'
                text = text[:-1] + '\n'
                committee_prop = committee_corpus.calculate_prob_of_sentence(committee_sentence)
                plenary_prop = plenary_corpus.calculate_prob_of_sentence(committee_sentence)

                text+= 'Probability of committee sentence in committee corpus: '+str(committee_prop)+'\n'
                text += 'Probability of committee sentence in plenary corpus: '+str(plenary_prop)+'\n'
                text+= 'This sentence is more likely to appear in corpus: '
                if plenary_prop>committee_prop:
                    text+='plenary' +'\n'
                else:
                    text+='committee' +'\n'

                text+= 'Plenary sentence: '+plenary_sentence +'\n'
                text+= 'Plenary tokens: '#+str(plenary_tokens) +'\n'
                for current_token in plenary_tokens:
                    text+=f'{current_token},'
                text = text[:-1] + '\n'
                committee_prop = committee_corpus.calculate_prob_of_sentence(plenary_sentence)
                plenary_prop = plenary_corpus.calculate_prob_of_sentence(plenary_sentence)
                text+= 'Probability of plenary sentence in plenary corpus: '+str(plenary_prop) +'\n'
                text+='Probability of plenary sentence in committee corpus: '+str(committee_prop)+'\n'
                text+= 'This sentence is more likely to appear in corpus: '

                if plenary_prop>committee_prop:
                    text+='plenary' +'\n\n'
                else:
                    text+='committee' +'\n\n'
        #save the text
        with open('sentences_results.txt','w',encoding='utf-8') as file:
            file.write(text)
    except Exception as e:
        print('error in Q3_text: '+str(e))

if __name__ == '__main__':
    try:
        #we work on plenary
        plenary_corpus = corpus('plenary','knesset_corpus.csv')
        #Loading the corpus (using the one attached with the assignment 'example_knesset_corpus.csv')
        committee_corpus = corpus('committee','knesset_corpus.csv')
        #Step 2 - Collactions
        Q2_text(plenary_corpus=plenary_corpus,committee_corpus=committee_corpus)
        #Step 3 - Model Implementations
        Q3_text(plenary_corpus=plenary_corpus,committee_corpus=committee_corpus)
    except Exception as e:
        print('error in main: '+str(e))
