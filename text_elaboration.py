'''
Code updated at 12/12/2019
Useful for babycenter data
'''

from numpy import *
import re

# chech if there are non latin characters
def isEnglish(s):
    '''
    Check if there are strange characters. Return True or False
    '''
    try:
        s.encode(encoding='utf-8').decode('cp1250')
    except UnicodeDecodeError:
        return False
    else:
        return True


def load_contractions(file):
    '''
    Load a dictionary of expansion of contractions (ex. he'll: he will / he shall). This function will load this file as a dictionary where contractions are keys and the first expansion is the value.
    
    ATTENTION TO APOSTROPHE CHARACTER!!
    
    Input:
    file : str, the file where there are the contractions
    
    Returns:
    CONTRACTIONS : dict of strings (contr:expa), couple contraction and expansion
    '''
    
    CONTRACTIONS = []
    with open(file, 'rt') as rr:
        for line in rr:
            line = line.split(':')
            CONTRACTIONS.append({line[0].strip().replace("’", "'").lower():line[1].split('/')[0].strip().lower()})

    CONTRACTIONS = {cont:exp for dic in CONTRACTIONS for cont,exp in dic.items()}
    
    return CONTRACTIONS


def load_acronyms_BBC(file):
    '''
    Load the acronyms of BabyCenter and return a dictionary
    '''
    
    ACRONYMS = []

    with open(file, 'rt') as rr:
        for line in rr:
            line = line.split('\t')
            ACRONYMS.append(tuple(line))

    # report all the keys as upper: no problem for that because replace_acronyms search case insensitive
    ACRONYMS = {acr.upper():exp.strip() for acr,exp in ACRONYMS}
    
    return ACRONYMS

def expand_contractions(text, contraction_dict):
    '''
    Given a text, expand all its contracted form. All the substitutions are in lower case!
    
    WARNING: changes in this function! contraction_in_text finds contractions in the text when lowered! When was done for BBC it was not lowered! It is lowerded for Reddit
    
    WARNING: changes in the function in 22/01/20: now the function does not return a lowered text! Only expansions are lowered, so keep attention for the pronoun "I" because if it is in a contraction is is converted do 'i'
    
    Inputs:
    text: str, a text
    contraction_dict : dict of strings, (contraction:expansion)
    
    Returns:
    text : str, the text with expanded contractions
    '''
    # find potential contractions and check if these are in the dict, then take them as expansions
    # the string 'no_cont' is needed to skip this no-contraction during the substitution
    # the second list needs to find contractions in the dict which are in lower case
    contraction_in_text = re.findall("\w+[']\w+", text)
    contraction_in_text_lowered = [c.lower() for c in contraction_in_text]

    expansions = [contraction_dict[cont] if cont in contraction_dict.keys() else 'no_cont' 
                                                                      for cont in contraction_in_text_lowered]

    # substitute the contraction founded in the text, whether if they are in upper or lower case
    for con, exp in zip(contraction_in_text, expansions):
        if exp=='no_cont':
            continue

        text = text.replace(con, exp)
        
    return text

def replace_acronyms(text, acronym_dict):
    '''
    Replace the acronyms of BBC with their expansion. Usually, in BBC acronyms are followed by their meaning 
    (ex. My LO (little one)) and this funcion take in consideration also that.
    
    Returns:
    text : str, the text with the substitutions
    '''
    
    # select all upper characters long more that 2 character and the following thing between parentesis if is there
    possible_acronyms_extended = [upp for upp in re.findall('[A-Z]+(?:\s+\([^)]+\))?', text) if len(upp)>1]
    #pattern = r'(?i)\b(?:'+r'|'.join(acronym_dict.keys())+r')\b(?:\s+\([^)]+\))?'
    #possible_acronyms_extended = [upp for upp in re.findall(pattern, text)]
    
    for possib_acron in possible_acronyms_extended:
        
        if ' ' in possib_acron: # if it matches also the parentesis, remove acronym and take only the parentesis content
            
            # in the dictionary, the keys are upper case! So needed to transform as upper
            # but the original (acron_in_text) is needed to replace in the text
            acron_in_text = possib_acron.split(' ')[0]
            
            
            acron = acron_in_text.upper()
            
            if acron not in acronym_dict.keys(): # if not in the dictionary, it is not an acronym
                continue
            
            thing = re.findall('(?<=\().+(?=\))', possib_acron)[0]
            
            
            # check if the content inside the parenthesis is similar with the meaning of the acronym
            # acronyms are managed to contain at least one word of the original meaning of BBC
            if acronym_dict[acron] in thing:
                text = text.replace(possib_acron, acronym_dict[acron])
      
            else: # if parentesis content is not the extension of the acronym
                text = text.replace(acron_in_text, acronym_dict[acron])
            
        else: # if no parentesis
                
            acron = possib_acron.upper()
            sub = acronym_dict[acron] if acron in acronym_dict.keys() else possib_acron
            text = text.replace(possib_acron, sub)
           
    return text


def get_wordnet_pos(word):
    """
    Map POS tag to first character lemmatize() accepts
    
    It is about 5 times slow than lemma_with_pos
    """ 
    from nltk.corpus import wordnet
    from nltk import pos_tag
    
    tag = pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ, "N": wordnet.NOUN, "V": wordnet.VERB, "R": wordnet.ADV}
    return tag_dict.get(tag, wordnet.NOUN)


def ngram_tokens(tokens, stop_words, ng):
    '''
    Take list of tokens and return a list of ngrams. No manipulation of words such as lowering, remove things and so on.
    
    Inputs:
    tokens : list of strings. Each element is a token. F
    
    stop_words : list of strings, a list of stopwords to remove 
    
    ng : int, n-grams (default 3)
    
    Returns:
    ngrams : list of strings, the ngrams 
    '''
    
    # Tokenizer and stemmer
    from nltk import ngrams
   
    ngmss = []
         
    for n_ in range(1, ng + 1):

        ngmss.extend([' '.join(i) for i in ngrams(tokens, n_)])
        
    return ngmss



def lemma_with_pos(pos_tag, lang='eng'):
    '''
    Given a tuple (word, POS), it returns the lemma of the word according to the POS tag using WordNet.
    This function is used with the map builtin function, so beware for generalization with more parameters!!!
    
    Inputs:
    pos_tag : tuple of strings, (word, POS_TAG) pos tag obtained with nltk pos_tag
    lang : str, default 'eng
    
    Returns:
    lemma : str, the lemma of the word according to the tag
    '''
    
    from nltk.corpus import wordnet
    
    # lemmatizer
    if lang == 'eng':
        from nltk.stem import WordNetLemmatizer
        lemmatizer = WordNetLemmatizer()
    
    w = pos_tag[0]  # the word
    pos = pos_tag[1][0]  # the first letter of the tag
    
    tag_dict = {"J": wordnet.ADJ, "N": wordnet.NOUN, "V": wordnet.VERB, "R": wordnet.ADV}
    
    return lemmatizer.lemmatize(w, tag_dict.get(pos, wordnet.NOUN))



def lemmatize(text, lang, tokenizer, stopwords, ng=0, min_len=3, exceptions=['!', '?', '.', ','], return_as_text=True):
    '''
    Given a text, the function returns the lemmatized version of each token and return ngrams if required
    
    Inputs:
    text : str, the text
    lang : str, the language like 'eng'
    tokenizer : str, tokenize_words or tokenize_words_punctuation. The former take as tokens only words while the latter conseders also punctuation as token and words like "mother-in-law" as a single token
    stopwords : list of str, the list of stopwords. Attention! This function prunes stopwords after lemmatization!!
    ng : int, the size of the ngram. If ng=0, only monogram
    min_len : int, the minimum lenght for a word to be considered
    exceptions : list of strings, special characters not to be pruned because of the lenght threshold. Default exceptions are '!', '?', '.', ',' (they have lenght 1, so pruned if min_len>1). Punctuation may be useful sometimes
    return_as_text : bool, return as a frat text or false to return as a list of tokens (useful when ngrams)
    
    Return:
    text_tokenized : str, a text where tokens are spaced by space character (also punctuation)
    '''
    
    from nltk import pos_tag
    from nltk.tokenize import RegexpTokenizer
        
    # Tokenizer and stemmer
    if tokenizer == 'tokenize_words':
        tokenize = RegexpTokenizer(r'\w+') 
        
    elif tokenizer == 'tokenize_words_punctuation':
        tokenize = RegexpTokenizer(r'(?<!-)\b\w+\b(?!-)|[^\w\s-]|(?:\w+-)+\w+') 
    
    
    # the commented row does the same thing as the following row, but slower
    #text_tokenized = [lemmatizer.lemmatize(t, get_wordnet_pos(t)) for t in tokenize.tokenize(text)]
    filt = lambda x: (len(x)>=min_len and isEnglish(x) and x not in stopwords) or (x in exceptions)
    text_tokenized = list(filter(filt, map(lemma_with_pos, pos_tag(tokenize.tokenize(text)))))
    
    #text_tokenized = [w for w in text_tokenized if w not in stopwords and len(w)>=min_len and isEnglish(w)]
    
    if ng!=0:

        text_tokenized = ngram_tokens(text_tokenized, stopwords, ng)
        
    if return_as_text:
        text_tokenized = ' '.join(text_tokenized)
    
    return text_tokenized





def extraxt_tokens(text, lang, contractions=None, acronyms=None, stopwords=None, remove_pattern=None, tokenize_sentences=True, min_len=3, tokenizer='tokenize_words', n_gram=2, exceptions=['!', '?', '.', ','], return_as_text=True):
    '''
    Take a text as an input and after basic pre-process steps (remove things with pattern 'remove_pattern', remove single characters, under-score_ and excessive white space, remove accents, lower case) divide the text in sentences and stemm each word. The processes are the following:
    
    1) extend contractions
    2) extend acronyms
    3) lower case, remove pattern, remove superfluous spaces (ex. more than 1 or at the beginning/eng of the text
    4) tokenization and lemmatization: tokenization per sentence and ngram or tokenization simple
    
    Input:
    text : a string, the text
    lang : str, the language of the text to import the right lemmatizer. At the moment only english ('eng')
    contractions : dict of strings, dictionary with (contraction:expansion) used to transform contractions
    acronyms : dict of strings, dictionary with (acronym:expansion) used to transform acronyms
    stopwords : list of strings, list of stopwords
    remove_pattern : string, regex pattern which identify strings or characters to be removed
    tokenize_sentences : bool, whether to tokenize per sentence or not. If True, it is also possible to take ngrams
    min_len : int, if len(w)<min_len the word is removed
    tokenizer : str, tokenize_words or tokenize_words_punctuation. The former take as tokens only words while the latter conseders also punctuation as token
    ng : int, the size of the ngram if tokenize_sentences=True. If ng=0, only monogram
    exceptions : list of strings, special characters not to be pruned because of the lenght threshold. Default exceptions are '!', '?', '.', ',' (they have lenght 1, so pruned if min_len>1). Punctuation may be useful sometimes
    return_as_text : bool, return as a frat text or false to return as a list of tokens (useful when ngrams)
        
    Returns:
    words : str, a text tokenized. 
    
    Example:
    'The Cat is on the Table. Oh yeah' ---> [[the, cat, is, on, the, table], [oh, yeah]]
    '''
    import unicodedata
    from nltk.tokenize import sent_tokenize
    from nltk import ngrams, pos_tag
    from nltk.corpus import wordnet
    #from nltk.corpus import stopwords
    
    import re
    
    # remove accents
    remove_acc = lambda st: ''.join((c for c in unicodedata.normalize('NFD', st) if unicodedata.category(c) != 'Mn'))
    
    text = text.replace("’", "'")
       
    # replace acronyms   
    if acronyms:
        text = replace_acronyms(text, acronyms)
        
        
    # from here the text is lowered!!  
    text = text.lower()   
        
    #expand contractions 
    if contractions:  
        text = expand_contractions(text, contractions)
        
  
    
    if remove_pattern:
        #rx = "@\w+|http\S+|\d+|\&\w{1,3}(?!\w)"  #remove mentions, urls, number and large white spaces sobstituted with a space
        rx = remove_pattern
        ### FIRST preprocess: remove things depending on remove pattern
        text = re.sub(rx, " ", text)  #sustitute a token structure with a space and transform in lower case
        

    text = re.sub('^\s{1,}|\s{1,}$', "", text)  # drop all spaces at the beginning/end of the text
    text = re.sub('\s{2,}', ' ', text)  # sobstitute each space character longer than one with only one space

        
    
    ## Second preprocess: stemming and tokenization of sentence
    
    # posso essere interessato sia a fare il tutto frase per frase (tipo quando voglio fare ngrams) o a fare tutto senza dividere in frasi (quando mi interessano solo i token
    
    if tokenize_sentences: # split the text in sentences. It returns a list of strings, one for each sentence
                           # here also ngrams!
            
        text_sentences = sent_tokenize(text)
        
        tokens = [lemmatize(text=sent, lang=lang, tokenizer='tokenize_words', stopwords=stopwords, ng=n_gram, min_len=min_len, exceptions=exceptions, return_as_text=return_as_text) for sent in text_sentences]
        
        # return as a flat text
        if return_as_text:
            tokens = ' '.join(tokens)
        else: # but if ngrams, frat text rise count of each word. So return as a list of tokens
            tokens = [item for sublist in tokens for item in sublist]

        
        
    else:
        # it returns a string with each token separated by a space (also punctuation and !?@# are characters)
        tokens = lemmatize(text, lang, tokenizer, stopwords, ng=0, min_len=min_len, exceptions=exceptions)
        
    
    return tokens
    
    
    
    
    
    
    