######################################################################################
'''
@author: Tyler McDonnell / http://tylermcdonnell.com

Scrapes synonyms and antonyms from Dictionary.com. 

As of the time of writing, the public API for Dictionary.com is no longer available.

Requirements: BeautifulSoup, requests
'''
# Standard
import re
from collections import namedtuple
# Extra
import requests
from bs4 import BeautifulSoup
######################################################################################  

# This is the base URL for the Thesaurus. Words replace the %s. 
base_url = 'http://www.thesaurus.com/browse/%s/adjective' 

def fetch_thesaurus_entry(word):
    try:
        return requests.get((base_url % word), timeout=3).text
    except:
        return ("")

def search(word):
    '''
    Returns a namedtuple:

    .synonyms -- Iterable of namedtuple:
                 .synonym   -- A synonym of the word.
                 .relevance -- Relevance score on scale [1,3] of word, where 3 is a
                               highly relevant synonym and 1 is least relevant.
    .antonyms -- Iterable of namedtuple:
                 .antonym   -- An antonym of the word.
                 .relevance -- Relevance score on scale [1,3] of word, where 3 is a
                               highly relevant antonym and 1 is least relevant.
    '''
    ThesaurusResult = namedtuple('ThesaurusResult', ['synonyms', 'antonyms'])
    soup = BeautifulSoup(fetch_thesaurus_entry(word))
    return ThesaurusResult(synonyms=list(_synonyms(soup, word)), 
                           antonyms=list(_antonyms(soup, word)))

def _synonyms(soup, word):
    SynonymsResult = namedtuple('SynonymsResult', ['synonym', 'relevance'])
    for container in soup.findAll('div', { 'class' : 'filters', 'id' : 'filters-0' }):
        for synonym_container in container.findAll('a', { 'class' : 'common-word' }):
            # Hidden "star" strings are included for their UI.
            synonym   = re.sub(r"star$", "", synonym_container.text)
            relevance = re.findall(r'relevant-[0-3]', str(synonym_container))[0]
            relevance = int(relevance.replace("relevant-",""))
            yield SynonymsResult(synonym=synonym, relevance=relevance)

def _antonyms(soup, word):
    AntonymsResult = namedtuple('AntonymsResult', ['antonym', 'relevance'])
    for container in soup.findAll('section', { 'class' : 'container-info antonyms' }):
        for antonym_container in container.findAll('a', { 'class' : 'common-word' }):
            # Hidden "star" strings are included for their UI.
            antonym  = re.sub(r"star$", "", antonym_container.text) 
            relevance = re.findall(r'relevant--[0-3]', str(antonym_container))[0]
            relevance = int(relevance.replace('relevant--',''))
            yield AntonymsResult(antonym=antonym, relevance=relevance)
