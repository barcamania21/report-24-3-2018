from __future__ import division
from collections import defaultdict as ddict
from collections import Counter
import re
from itertools import islice
from sklearn.metrics import accuracy_score

class HMM(object):
    INITIAL = 'Initial'
    FINAL = 'Final'

    def __init__(self):
        self._states = []
        self._transitions = ddict(lambda : ddict(lambda : 0.00001))
        self._emissions = ddict(lambda : ddict(lambda : 0.000005))

    def add(self, state, transition_dict = {}, emission_dict = {}):
        self._states.append(state)

        for target_state, prob in transition_dict.items():
            self._transitions[state][target_state] = prob
        for observation, prob in emission_dict.items():
            self._emissions[state][observation] = prob

    def predict(self, observations):
        probs = ddict(lambda : ddict(lambda : 0.0))
        probs[-1][self.INITIAL] = 1.0
        pointers = ddict(lambda : {})
        i = -1
        for i, observations in enumerate(observations):
            for state in self._states:
                path_probs = {}

                for prev_state in self._states:
                    path_probs[prev_state] = (probs[i-1][prev_state] *
                        (self._transitions[prev_state][state])*(self._emissions[state][observations]))

                best_state = max(path_probs, key = path_probs.get)
                probs[i][state] = path_probs[best_state]
                pointers[i][state] = best_state

        curr_state = max(probs[i], key = probs[i].get)
        states = []
        for i in range(i, -1, -1):
            states.append(curr_state)
            curr_state = pointers[i][curr_state]
        states.reverse()
        return states


    def transition_probs(self, tag, train_tags):
        transition_prob_dict = ddict(lambda : 0.0)
        tags = re.findall("\w+", train_tags)
        count_bigram_dict = dict(Counter(zip(tags, islice(tags, 1, None))))

        for next_tag in num_of_tag.keys():
            couple_tag = tag +' ' +next_tag
            if couple_tag in test_tags and (tag, next_tag) in count_bigram_dict.keys():
                transition_prob_dict[next_tag] = (count_bigram_dict[(tag, next_tag)]) / (num_of_tag[tag])
        return transition_prob_dict

    def observation_probs(self, tag):
        observation_prob_dict = ddict(lambda : 0.0)
        for sent in train_sent:
            for word_tag in sent.split():
                word_tag_broken = word_tag.split('/')
                if len(word_tag_broken) == 2 and len(word_tag_broken[0]) > 0 and len(word_tag_broken[1]) > 0:
                    word = word_tag_broken[0]
                word_tag_couple = word + '/' + tag
                if word_tag_couple in num_of_word_tag.keys():
                    observation_prob_dict[word] = (num_of_word_tag[word_tag_couple])  / (num_of_tag[tag])
        return observation_prob_dict

def filter_text(text):
    list  = []
    for word in text.split(" "):
        if ('/' in word ):
            list.append(word)
    return " ".join(list)

train_text = open('/Users/naduong1001/Desktop/data.txt').read()

sentences = train_text.split('\n')
size = int(len(sentences)*0.8)
train_sent = sentences[:size]
test_sent = sentences[size:]

test_words = ''
test_tags = ''
test_word = ''
test_tag = ''
for sent in test_sent:
    for word_tag in sent.split():
        word_tag_broken = word_tag.split('/')
        if len(word_tag_broken) == 2 and len(word_tag_broken[0]) > 0 and len(word_tag_broken[1]) > 0:
            wordBroken = word_tag_broken[0]
            tagBroken = word_tag_broken[1]
            test_word += wordBroken
            test_word += ' '
            test_tag += tagBroken
            test_tag += ' '
    test_words += test_word
    test_tags += test_tag
    test_word = ''
    test_tag = ''

train_tags = ''
train_tag = ''
train_words = ''
train_word = ''
for sent in train_sent:
    for word_tag in sent.split():
        word_tag_broken = word_tag.split('/')
        if len(word_tag_broken) == 2 and len(word_tag_broken[0]) > 0 and len(word_tag_broken[1]) > 0:
            wordBroken = word_tag_broken[0]
            tagBroken = word_tag_broken[1]
            train_tag += tagBroken
            train_tag += ' '
            train_word += wordBroken
            train_word += ' '
    train_tags += train_tag
    train_words += train_word
    train_tag = ''
    train_word = ''

rare_word = set()
word_count = ddict(int)
for word in train_words.split():
    word_count[word] += 1
for word in word_count.keys():
    if word_count[word] < 5:
        rare_word.add(word)

num_of_tag = ddict(lambda : 0)
num_of_word_tag = ddict(lambda : 0)

for sent in train_sent:
    for word_tag in sent.split():
        if word_tag not in num_of_word_tag.keys():
            num_of_word_tag[word_tag] = 1
        else:
            num_of_word_tag[word_tag] += 1
        word_tag_broken = word_tag.split('/')
        if len(word_tag_broken) == 2 and len(word_tag_broken[0]) > 0 and len(word_tag_broken[1]) > 0:
            word = word_tag_broken[0]
            tag = word_tag_broken[1]
        if tag not in num_of_tag.keys():
            num_of_tag[tag] = 1
        else:
            num_of_tag[tag] += 1
sum_of_no_tags = sum(num_of_tag.values())

num_of_sentences = 0
tag_in_the_first = ddict(lambda : 0)
for sent in train_sent:
    list_sent = list(sent.split())
    if(len(list_sent)):
        num_of_sentences += 1
        first_word_tag = list_sent[0]
        broken = first_word_tag.split('/')
        if len(broken) == 2 and len(broken[0]) > 0 and len(broken[1]) > 0:
            first_tag = broken[1]
        for tag in num_of_tag.keys():
            if tag == first_tag:
                tag_in_the_first[tag] += 1

for tag in tag_in_the_first.keys():
    tag_in_the_first[tag] /= (num_of_sentences)

hmm = HMM()
hmm.add(HMM.INITIAL, tag_in_the_first)
for tag in num_of_tag.keys():
    hmm.add(tag, hmm.transition_probs(tag, train_tags), hmm.observation_probs(tag))

y_pred = hmm.predict(test_words.split()[50:150])
y_true = test_tags.split()[50:150]

print("Accuracy:", accuracy_score(y_true, y_pred))

for i in range(len(y_pred)):
    if y_true[i] != y_pred[i]:
        print(y_pred[i], y_true[i], test_words.split()[i+50])

