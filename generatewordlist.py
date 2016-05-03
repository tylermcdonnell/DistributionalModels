######################################################################################
'''
@author: Tyler McDonnell / http://tylermcdonnell.com

Hooks for creating pairs of synonyms, antonyms, and unrelated words.

Initially conceived to evaluate synonym and antonym detection in distributional models.
'''
# Standard
import sys
import thesaurus
import random
from collections import namedtuple, Set
######################################################################################  

Pair = namedtuple('Pair', ['relationship', 'w1', 'w2'])

def generate_pairs(seed, pairs):
    for word in seed:
        result = thesaurus.search(word)
        # Only use words of higher relevance rating.
        synonyms = [s for s in result.synonyms if s.relevance == 3]
        antonyms = [a for a in result.antonyms if a.relevance == 3]
        # Randomly sample; else in alphabetical.
        synonyms = random.sample(synonyms, min(pairs, len(synonyms)))
        antonyms = random.sample(antonyms, min(pairs, len(antonyms)))
        for s in synonyms:
            yield Pair(relationship="SYNONYMS", w1=word, w2=s.synonym)
        for a in antonyms:
            yield Pair(relationship="ANTONYMS", w1=word, w2=a.antonym)

def generate_unrelated_pairs(seed, picks, pairs):
    for word in random.sample(seed, picks):
        count = 0
        while (count < pairs):
            rword    = random.sample(seed, 1)[0]
            result   = thesaurus.search(word)
            synonyms = [s.synonym for s in result.synonyms]
            antonyms = [a.antonym for a in result.antonyms]
            if (rword != word) and (rword not in synonyms) and (rword not in antonyms):
                count += 1
                yield Pair(relationship="UNRELATED", w1=word, w2=rword)
            else:
                print ("Attempted match failed.")

if __name__ == "__main__":
    pairs = list(generate_pairs(['beautiful'], 10))
    useed = [line.rstrip('\n') for line in open('WordLists/unrelated_seed.txt')]
    pairs.extend(list(generate_unrelated_pairs(useed, 15, 1)))
    all_words = set()
    for pair in pairs:
        print (pair)
        all_words.add(pair.w1)
        all_words.add(pair.w2)
    f = open('target_words_beautiful.txt', 'w')
    for word in all_words:
        # Python 2
        # f.write('%s' % (word))
        # Python 3
        print(("{}").format(word), file=f)

    f = open('training_pairs_beautiful.txt', 'w')
    for pair in pairs:
        # Python 2
        f.write('%s %s %s' % (pair.relationship, pair.w1, pair.w2))
        # Python 3
        print(("{} {} {}").format(pair.relationship, pair.w1, pair.w2), file=f)


    '''
    seed  = [line.rstrip('\n') for line in open('seed.txt')]
    useed = [line.rstrip('\n') for line in open('unrelated_seed.txt')]
    all_words = set()
    count = 0
    pairs = []
    pairs.extend(list(generate_unrelated_pairs(useed, 500, 1)))
    pairs.extend(list(generate_pairs(seed, 10)))
    for pair in pairs:
        print (pair)
        all_words.add(pair.w1)
        all_words.add(pair.w2)

    f = open('target_words.txt', 'w')
    for word in all_words:
        # Python 2
        # f.write('%s' % (word))
        # Python 3
        print(("{}").format(word), file=f)

    f = open('training_pairs.txt', 'w')
    for pair in pairs:
        # Python 2
        f.write('%s %s %s' % (pair.relationship, pair.w1, pair.w2))
        # Python 3
        print(("{} {} {}").format(pair.relationship, pair.w1, pair.w2), file=f)
    '''