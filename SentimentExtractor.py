import pandas as pd
from nltk.tokenize import RegexpTokenizer
from nltk.tokenize import sent_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.util import ngrams
import numpy as np
from nltk.corpus import sentiwordnet as swn
from nltk.tag import pos_tag
import itertools
import re

class SentimentExtractor:
    def __init__(self, path, scoreMethod):
        self.text = ''
        self.path = path
        self.scoreMethod = scoreMethod
        self.NRC = pd.read_csv(self.path + "EMOLEX_NRC/NRC-Emotion-Lexicon-Wordlevel-v0.92.txt",
                  engine="python", header=None, sep="\t")
        self.NRC = self.NRC[(self.NRC != 0).all(1)].reset_index(drop=True)
        self.NRC_emotions = self.NRC.drop_duplicates(subset=1)[1].reset_index(drop=True)
        self.NRC_emotions = self.NRC_emotions.drop(self.NRC_emotions[(self.NRC_emotions=='positive')|(self.NRC_emotions=='negative')].index)
        self.negation_list = ['no','not','none','nobody','nothing','nowhere','never',
                              'hardly','scarcely','barely']
        self.negation_verb_list = ["doesn't","isn't","wasn't","shouldn't","wouldn't",
                                   "couldn't","won't","can't","don't"]
        self.grader_list = ['very', 'much', 'really', 'absolutely', 'extremely', 'quite',
                               'so', 'pretty', 'farily', 'somewhat']

        self.non_decimal = re.compile(r'[^\d.]+')
        self.tokenizer = RegexpTokenizer('[\w]+')
        self.lemmatizer = WordNetLemmatizer()

    def change_text(self, new_text):
        self.text = new_text.lower()

    def cleanse_negation(self):
        for negation_verb in self.negation_verb_list:
            self.text = self.text.replace(negation_verb,'not')

    def token_processing(self, token):
        t_token = ngrams(token,3)
        tok_list = []
        for tok in t_token: #check every n-gram tokens
            tk_list = []
            negation_count = 0
            grader_count = 0
            for tk in tok: #check every gram in n-gram
                negation_find = False
                grader_find = False
                for negation in self.negation_list:
                    if negation == tk: #if we have negation, count up
                        negation_count += 1
                        negation_find = True
                        break
                if negation_find :
                    continue
                for grader in self.grader_list: #if we have grader, count up
                    if grader == tk:
                        grader_count += 1
                        grader_find = True
                        break
                if grader_find :
                    continue
            for tk in tok:
                negation_find = False
                grader_find = False
                for negation in self.negation_list: #as negations have no sentiment, give XX tag
                    if negation == tk:
                        tk_list.append('XX'+tk)
                        negation_find = True
                        break
                if negation_find :
                    continue
                for grader in self.grader_list: #as graders have no sentiment, give XX tag
                    if grader == tk:
                        tk_list.append('XX'+tk)
                        grader_find = True
                        break
                if grader_find :
                    continue

                if (negation_count == 1) and (grader_count > 0) : #doubled negation is same as positive.
                    tk_list.append('NG'+tk) #NG for Negation + Grader
                elif (negation_count == 1):
                    tk_list.append('NX'+tk) #NX for Negation
                elif (grader_count > 0):
                    tk_list.append('XG'+tk) #XG for Grader
                else:
                    tk_list.append('XX'+tk) #XX for none

            tok_list.append(tk_list)
        return tok_list

    def lemmatize(self):
        data = self.text
        sent_token = sent_tokenize(data)
        lemm_token = []
        lemm_pos = []
        for sent in sent_token:
            tokens = self.tokenizer.tokenize(sent)
            #divide pos for accurate lemmatization
            pos_tokens = pd.DataFrame(pos_tag(tokens), columns=['word', 'pos'])
            pos_tokens['pos'] = pos_tokens['pos'].astype(str).str[0]
            pos_array = pos_tokens['pos'].copy()

            pos_array = pos_array.replace(['J'], 'A').replace().str.lower()
            pos_tokens['pos'] = pos_tokens['pos'].replace(['J'], 'A').replace(['N'], 'V').str.lower()

            pos_tokens = pos_tokens[pos_tokens.pos.isin(['a', 'n', 'r', 'v'])]
            pos_array = pos_array[pos_array.isin(['a', 'n', 'r', 'v'])]
            lemm_sent = []
            for token in pos_tokens.values:
                if len(token) > 1:
                    lemm_sent.append(self.lemmatizer.lemmatize(token[0], pos=token[1]))
            lemm_sent = self.token_processing(lemm_sent)

            try:
                final_lemm_sent = lemm_sent[0]
                del lemm_sent[0]
                for tri_tok in lemm_sent:
                    try:
                        final_lemm_sent.append(tri_tok[2])
                    except:
                        pass
                lemm_token.append(final_lemm_sent)
                lemm_pos.append(pos_array.values)
            except:
                pass

        return lemm_token, lemm_pos

    def switch_sentiment(self, sent_text, PorN, negation):
        if negation == True:
            if sent_text == 'trust':
                return 'fear'
            elif sent_text == 'fear':
                return 'trust'
            elif sent_text == 'sadness':
                return 'joy'
            elif sent_text == 'anger':
                return 'joy'
            elif sent_text == 'surprise':
                if PorN == True:
                    return 'shock'
                else:
                    return 'surprise'
            elif sent_text == 'disgust':
                return 'trust'
            elif sent_text == 'joy':
                return 'anger'
            elif sent_text == 'anticipation':
                return 'fear'
            else:
                return sent_text
        else:
            if sent_text == 'surprise':
                if PorN == True:
                    return 'surprise'
                else:
                    return 'shock'
            else:
                return sent_text

    def getSenti(self, array):
        token_N = array[0][0]
        token_G = array[0][1]
        token_value = array[0][2:]
        senti = list(swn.senti_synsets(token_value, array[1]))
        if len(senti) == 0:
            pos_avg = 0
            neg_avg = 0
        elif self.scoreMethod == 'single':
            pos_avg = senti[0].pos_score()
            neg_avg = senti[0].neg_score()
        else:
            pos = np.array([x.pos_score() for x in senti])
            neg = np.array([x.neg_score() for x in senti])
            pos_avg = np.average(pos, weights=np.arange((len(pos)) * 1, 0, -1))
            neg_avg = np.average(neg, weights=np.arange((len(neg)) * 1, 0, -1))

        pos_avg_temp = pos_avg
        neg_avg_temp = neg_avg
        if token_N == 'N' and token_G == 'G':
            pos_avg = neg_avg_temp * 1.5
            neg_avg = pos_avg_temp * 1.5
        elif token_N == 'N':
            pos_avg = neg_avg_temp
            neg_avg = pos_avg_temp
        elif token_G == 'G':
            pos_avg = pos_avg_temp * 1.5
            neg_avg = neg_avg_temp * 1.5
        else:
            pass

        return [token_value, pos_avg, neg_avg, array[1]]

    def sentiment_calculator(self):

        self.cleanse_negation()
        lemm_tokens, lemm_pos = self.lemmatize()
        #convert to 1d vector
        tokens = list(itertools.chain(*lemm_tokens))
        poss = list(itertools.chain(*lemm_pos))

        match_words = [x for x in tokens if x[2:] in list(self.NRC[0])]

        #process for EMOLEX
        emotion = []
        PorN = False
        for i in match_words:
            temp = list(self.NRC.iloc[np.where(self.NRC[0] == i[2:])[0], 1])
            PorN = ('positive' in temp)
            for j in temp:
                if i[0] == 'N' and i[1] == 'G':
                    emotion.append(self.switch_sentiment(j,PorN,True))
                    emotion.append(self.switch_sentiment(j,PorN, True))
                elif i[0] == 'N':
                    emotion.append(self.switch_sentiment(j,PorN,True))
                elif i[1] == 'G':
                    emotion.append(self.switch_sentiment(j,PorN,False))
                    emotion.append(self.switch_sentiment(j,PorN,False))
                else:
                    emotion.append(self.switch_sentiment(j,PorN,False))

        #use frequency method
        sentiment_result = []
        if self.scoreMethod == 'STD':
            sentiment_result = pd.Series(emotion).value_counts() / len(match_words)
        else:
            sentiment_result = pd.Series(emotion).value_counts()
        sentiment_result = sentiment_result.reindex(self.NRC_emotions)
        sentiment_result = sentiment_result.fillna(0)

        #process for SentiWordNet
        pos_tokens = pd.DataFrame(np.array([tokens,poss]).T, columns=['word', 'pos'])
        final_tokens = pd.DataFrame([self.getSenti(w) for w in pos_tokens.values], columns=['word','posit', 'negat', 'pos'])

        sentiment_tokens = final_tokens[(final_tokens[['posit','negat']].T != 0).any()]
        sentiment_sum = sentiment_tokens[['posit','negat']].sum().values
        sentiment_count = sentiment_tokens[['posit','negat']].count().values
        sentiment_score = sentiment_sum/sentiment_count
        sentiment_score = pd.Series(sentiment_score,index=['positive','negative'])

        #concatenate EMOLEX and SentiWordNEt results
        sentiment_result = sentiment_result.append(sentiment_score)

        return sentiment_result
