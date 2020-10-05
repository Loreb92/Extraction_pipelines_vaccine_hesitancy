import spacy
import re
from numpy import nan as NAN
from os import listdir

from sys import path
path.append('/home/lorenzobetti/Lagrange/Babycenter/Babycenter_US_definitive_paper/coding_work_in_progress')

import text_elaboration as te
import Dependency_tree_functions as DepTree

nlp = spacy.load("en_core_web_sm")

# pattern to remove urls
remove_pattern = "(?:https?:\/\/)?(?:[^@\/\n]+@)?(?:www\.)[^)\s]+|https?:\/\/(?!www\.)[^)\s]+"
# load contractions and acronyms
CONTRACTIONS = te.load_contractions('utils/contractions_eng.txt')
ACRONYMS = te.load_acronyms_BBC('utils/acronyms_BBC_eng.txt')

# US (ultra sound) conflicts with US (united states)
del ACRONYMS['US']

def preprocess_sentences(sent, kind_of_pattern):
    '''
    Preprocessing of the sentences. In order do:
    1) replace things like 'delayed/spreaded' with 'delayed and spreaded'
    2) remove strange characters and pattern define above
    3) replace numbers with words : 'one' --> 1
    4) replace 'my husband and I' with 'we'
    5) remove smiles
    6) replace acronyms and expand contractions
    7) take sentences between parenthesis
    8) remove superfluous spaces and replace "dr." with "dr"
    
    Input:
    sent : str, a sentence
    kind_of_pattern : str, one of ['schedule', 'delay_verbs']
    
    Returns:
    sent : str, the processed sentence
    '''
    
    # substitute things like "select/delay"  with "select and delay"
    matches = [match for match in re.findall(r"(?:\w+\/)+\w+", sent)] 
    if len(matches) != 0:
        for match in matches:
            try:
                sent = sent.replace(match, ' and '.join(match.split('/')))
            except:
                continue
    
    # remove strange characters
    sent = sent.replace("’", "'")
    sent = sent.replace('\xa0', ' ')
    sent = sent.replace("\n", ' ')
    sent = sent.replace("*", '')
    sent = re.sub(remove_pattern, " ", sent)  #sustitute a token structure with a space
    
    # split numbers and words if no space between them (e.g., "my daughter2" --> "my daughter two")
    '''
    for num, num_str in zip(['1', '2', '3'], [' one', ' two', ' three']):
        if num in sent:
            sent = sent.replace(num, num_str)
    '''       
    # if a number occurs at the end of a word, split it and make it in characters (no change h1n1)
    # "My DD2 is vaccinated" --> "My DD two is vaccinated"
    for num, num_str in zip(['1', '2', '3'], [' one', ' two', ' three']):        
        sent = re.sub('(?i)(?<=\w)(?<!h1n)%s(?=\s)'%num, num_str, sent)
    
            
    if 'my husband and I' in sent:
        sent = sent.replace('my husband and I', 'we')
            
    sent = sent.replace(':)', '')
    sent = sent.replace(':-)', '')
    sent = sent.replace(':(', '')

    # Substitute BabyCenter acronyms (like DD) with their extended forms (dear daughter) and do the same with contractions ("I'm" --> "I am")
    sent = te.replace_acronyms(sent, ACRONYMS)
    sent = te.expand_contractions(sent, CONTRACTIONS)

    # search things between parentesis: if the keyword in the parenthesis, then analyze the parenthesis
    # else replace parenthesis with white space
    sent_between_parenthesis = re.findall('(?<=\().+?(?=\))', sent)
    if kind_of_pattern == 'schedule':
        with_keywords_parenthesis = list(filter(lambda txt: 'schedule' in txt, sent_between_parenthesis))
    elif kind_of_pattern == 'delay_verbs':
        with_keywords_parenthesis = list(filter(lambda txt: 'spac' in str(txt) or 'dela' in str(txt) or 'split' in str(txt)
                                , sent_between_parenthesis))
    else:
        print('WRONG KIND OF PATTERN!')

    if len(with_keywords_parenthesis)>0: 
        sent = with_keywords_parenthesis[0]  # consider only the first if more
    else:
        # remove parentheses
        for parent in sent_between_parenthesis:
            sent = sent.replace('('+parent+')', '')

    # substitute multiple spaces with single spaces and remove spaces at the start or end of the sentence
    sent = re.sub('^\s{1,}|\s{1,}$', "", sent)  # drop all spaces at the beginning/end of the text
    sent = re.sub('\s{2,}', ' ', sent)  # sobstitute each space character longer than one with only one space

    sent = sent.lower() # lower case to avoid problems with dep tree matching upper case things
    sent = sent.replace("dr.", 'dr') # useful because sometimes the dot makes the dependency parser a mess
    
    return sent


def filter_out_non_verbs(G, lemma):
    '''
    This function searches all nodes of the dependency tree G corresponding to a certain lemma. Then, it filters out all nodes not corresponding to verbs like "delayed schedule", where delayed is an adjective).
    
    Input: 
    G : nx.DiGraph, the dependency tree
    lemma : str, the lemma to search among the nodes of G
    
    Returns:
    idxs_verb : list of int, list containing the indices of the verbs whose lemma is the one required  
    '''
    # search the node names corresponding to the word delay
    idxs_verb = DepTree.get_nodes_by_attribute(G, 'lemma', lemma)
    # take delay which is a Verb and not amod, compound or possessive
    idxs_verb = [i for i in idxs_verb if G.nodes(data='pos')[i] =='VERB']
    idxs_verb = [source for i in idxs_verb for source, target, dep in G.edges(i, data='dep') 
                                                              if dep not in ['amod', 'compound', 'poss']]
    return idxs_verb

def first_filter_keywords_syntactic(text):
    '''
    This first filter is used to filter in sentences with keywords in certain dependency rules.
    
    1. is "schedule", "spac", "dela" or "split" in text? Is there a question mark?
    2. if "schedule" in text:
        - discard all "schedule" occurring as subjects or verbs
    3. if "spac", "dela" or "split" in text:
        - take only when occurring as verbs
    
    Input:
    text : str, the sentence to be inspected
    
    Returns:
    idxs_keywords : dict, it is a summary of the key features of the sentence, given the rule matched. The first key of the dictionay corresponds to the name of the pattern matched ("schedule_noun" or "delay_verbs"). Then, it is composed by the indiced of the keywords matched ("keywords_idxs"), the text processed ("text"), and the dependency tree of that sentence ("G"). 
    
    Example:
    
    >>> text = 'I delay this vaccine and follow this schedule'
    >>> idxs_keywords = SP.first_filter_keywords_syntactic(text)
    >>> idxs_keywords
    {'schedule_noun': {'keywords_idxs': [37],
                       'text': 'i delay this vaccine and follow this schedule',
                       'G': <networkx.classes.digraph.DiGraph at 0x7f0850b63d90>},
     'delay_verbs': {'keywords_idxs': [2],
                     'text': 'i delay this vaccine and follow this schedule',
                     'G': <networkx.classes.digraph.DiGraph at 0x7f0850b63ed0>}}
 
    '''
    # if no keywords in text, return empty dict. I used the stems of the words because building the DepTree for each sentence of the dataset is very time consuming. These words ("schedule", "dela") are common strings of all the inflections of these words
    if ("schedule" not in text and 'spac' not in text and 'dela' not in text and 'split' not in text):
        return {}
    
    idxs_keywords = {}
    if "schedule" in text:
        text_schedule = preprocess_sentences(text, 'schedule')
        if '?' not in text_schedule: # check if question mark
            G = DepTree.make_dep_tree(nlp(text_schedule))

            # search the node names corresponding to the word schedule
            idxs_schedule = DepTree.get_nodes_by_attribute(G, 'lower', 'schedule')

            # take "schedule" which is not the subject and not a Verb
            idxs_schedule = [source for i in idxs_schedule 
                             for source, target, dep in G.edges(i, data='dep') if dep not in ['nsubj', 'nsubjpass']]
            idxs_schedule = [i for i in idxs_schedule if G.nodes(data='pos')[i] !='VERB']

            if len(idxs_schedule) > 0:
                idxs_keywords['schedule_noun'] = {'keywords_idxs' : idxs_schedule,
                                                  'text' : text_schedule,
                                                  'G' : G}
            
    if 'spac' in text or 'dela' in text or 'split' in text:
        text_delay = preprocess_sentences(text, 'delay_verbs')
        if '?' not in text_delay: # check if question mark
            G = DepTree.make_dep_tree(nlp(text_delay))
            # search the node names corresponding to the word delay, space and split occurring as a verb
            idxs_delay = filter_out_non_verbs(G, 'delay')
            idxs_space = filter_out_non_verbs(G, 'space')
            idxs_split = filter_out_non_verbs(G, 'split')
            idxs_delay = idxs_delay + idxs_space + idxs_split

            if len(idxs_delay) > 0:
                idxs_keywords['delay_verbs'] = {'keywords_idxs' : idxs_delay,
                                                'text' : text_delay,
                                                'G' : G}
    
    return idxs_keywords
    
    
def inspect_principal_clause(G, verb):
    '''
    Given a verb, this function searches for the head clause connected through "xcomp" dependency and returns the subject (and modifiers) and the verb of the principal clause
    
    Input:
    G : nx.DiGraph, the dependency tree
    verb : int, index of a verb
    
    Returns:
    amod, compound, pos, det, subject_xcomp, verb_xcomp, verb_phrase_xcomp, verb_tense_xcomp : lists of int. If no element matched, the lists are empty
    
    Example:
    
    >>> text = 'My husband thinks to delay her vaccines'
    >>> idxs_keywords = SP.first_filter_keywords_syntactic(text)
    >>> G, delay_verbs = idxs_keywords['delay_verbs']['G'], idxs_keywords['delay_verbs']['keywords_idxs']
    >>> print(G.nodes('lower'))
    [(0, 'my'), (3, 'husband'), (11, 'thinks'), (18, 'to'), (21, 'delay'), (27, 'her'), (31, 'vaccines')]
    
    >>> SP.inspect_principal_clause(G, delay_verbs[0])
    ([], [], [0], [], [3], [11], 'thinks', 'PresentSimple')
    '''
    # search xcomp dependency of the verb
    verb_xcomp = [tar for sour, tar, dep in G.edges(verb, data='dep') if dep=='xcomp']
    
    if len(verb_xcomp)>0:
        verb_xcomp = [verb_xcomp[0]]
    else:
        return [], [], [], [], [], [], [], []
    
    verb_phrase_xcomp, verb_tense_xcomp = DepTree.get_verb_phrase(G, verb_xcomp[0])
    # find subject (no passive because there are molto pochi)
    subject_xcomp = DepTree.find_subject(G, verb_xcomp[0], passive=False)
    
    if len(subject_xcomp)>0:
        amod, compound, pos, det = DepTree.find_modifiers(G, subject_xcomp[0])
    else:
        return [], [], [], [], [], [], [], []
    
    return amod, compound, pos, det, subject_xcomp, verb_xcomp, verb_phrase_xcomp, verb_tense_xcomp


# ---------------------------------------------------------------------------------------- #

def Structured_representation(idxs_keywords):
    '''
    This function returns the structured representation of a sentence by means of its main syntactic constituents and it allows a better comprehension of the context around the matched keywords. It takes as input the dictionary containing the matched keywords and the dependency tree and returns a list of dictionaries. Each dictionary correspond to the structured representation of the sentence containing the specific keyword matched. The outputs correspond to the indices of the words on the dependency tree.
    
    Input: 
    
    idxs_keywords : dict, it is the output of the function "first_filter_keywords_syntactic"
    
    Returns:
    reports : list of dict, each report is a dictionary whose keys correspond to different elements of the sentence and information about the keyword matched. 
        
    '''
    
    # loop over all reports
    reports = []
    for group_of_keyword, keyword_report in idxs_keywords.items():
        G, sent = keyword_report['G'], keyword_report['text']
        
        for idx_keyword in keyword_report['keywords_idxs']:
            report = {'G':G, 'sent':sent, 'pattern_matched':group_of_keyword}
            if group_of_keyword == 'schedule_noun':
                # search closest verb and modifiers
                dobjs = [idx_keyword]
                verb = DepTree.find_closest_verb_of_word(G, dobjs[0])
                verb_phrase, verb_tense = DepTree.get_verb_phrase(G, verb)

                amod_dobj, compound_dobj, pos_dobj, det_dobj = DepTree.find_modifiers(G, dobjs[0])
                report['dobj_amod'] = amod_dobj
                report['compound_dobj'] = compound_dobj
                report['pos_dobj'] = pos_dobj


            elif group_of_keyword == 'delay_verbs':
                # search dobj and its modifiers
                verb = idx_keyword
                verb_phrase, verb_tense = DepTree.get_verb_phrase(G, verb)
                dobjs = DepTree.find_objects(G, verb)
                
                if len(dobjs)>0:
                    amod_dobj, compound_dobj, pos_dobj, det_dobj = DepTree.find_modifiers(G, dobjs[0])
                    for dobj in dobjs[1:]:
                        amod_dobj_, compound_dobj_, pos_dobj_, det_dobj_ = DepTree.find_modifiers(G, dobj)
                        amod_dobj.extend(amod_dobj_)
                        compound_dobj.extend(compound_dobj_)
                        pos_dobj.extend(pos_dobj_)
                        det_dobj.extend(det_dobj_)
                      
                    report['dobj_amod'] = amod_dobj
                    report['compound_dobj'] = compound_dobj
                    report['pos_dobj'] = pos_dobj

            else:
                print('PROBLEM!!')
                
            # fill report
            report['dobj'] = dobjs
            report['verb'] = [verb]
            report['verb_phrase'] = verb_phrase
            report['verb_tense'] = verb_tense
            
            # find subject
            in_dep_verb = [dep for source, target, dep in G.in_edges(verb, data='dep')]
            out_dep_verb = [dep for source, target, dep in G.edges(verb, data='dep')]
            if 'nsubj' in in_dep_verb or 'nsubjpass' in in_dep_verb or 'conj' in out_dep_verb:
                subject, active = DepTree.find_subject(G, verb), 'ACTIVE'
                if subject==[]:
                    subject, active = DepTree.find_subject(G, verb, passive=True), 'PASSIVE'          
              
                report['subject'] = subject
                report['subject_active'] = active
                
                amod_subj, compound_subj, pos_subj, det_subj = DepTree.find_modifiers(G, subject)
                report['amod_subj'] = amod_subj
                report['compound_subj'] = compound_subj
                report['pos_subj'] = pos_subj
                
            else:
                # search in the principal clause. (principal clause only active forms)
                amod_subj_xcomp, compound_subj_xcomp, pos_subj_xcomp, det_subj_xcomp, subject_xcomp, \
                verb_xcomp, verb_phrase_xcomp, verb_tense_xcomp = inspect_principal_clause(G, verb)
                
                # if here is no subjects, it means that there is no subject in the matched sentence, so continue
                if subject_xcomp == []:
                    continue

                report['amod_subj_xcomp'] = amod_subj_xcomp
                report['compound_subj_xcomp'] = compound_subj_xcomp
                report['pos_subj_xcomp'] = pos_subj_xcomp
                report['subject_xcomp'] = subject_xcomp
                report['verb_xcomp'] = verb_xcomp
                report['verb_phrase_xcomp'] = verb_phrase_xcomp
                report['verb_tense_xcomp'] = verb_tense_xcomp
                
                    
            # we need to find negations inside the sentence we are analizing. Search only in the string containing the words of interest we are studying and not in the whole sent (it may be long and it may take negations that do not change the meaning of the clause of interest) the following takes the idx of the subjects and adjectives to obtain the first element of the shorter sentence
            indices_words = [idx[0] for idx in report.values() if type(idx)==list and len(idx)>0]
            indices_words = [G.nodes()[idx]['position'] for idx in indices_words]
            sent_short = nlp(sent)[min(indices_words) : max(indices_words)+1]
            G_short = DepTree.make_dep_tree(sent_short)
            
            # search all neg in the short sentence
            negations = [-1 for sour, tar, dep in G_short.edges(data='dep') if dep=='neg']
            negations = 1 if len(negations)%2==0 else -1
            
            report['text_short'] = sent_short.text
            report['negations'] = negations
    
            reports.append(report)
    
    return reports
    

def translate_response(response, c_id, comment_author, thread_id, comment_date):
    '''
    The name of the nodes of the dependency tree corresponds to an index, so all the previous functions works considering indices. This function translates node indices to the corresponding word and add information of the sentence like the id of the comment and the author. 
    
    Input:
    response : dict, one of the items of the output of the function "Structured_representation"
    c_id : str, the identifier of the comment containing the sentence under analysis
    comment_author : str, the author of the comment
    thread_id : str, the identifier of the thread in which the comment was submitter
    comment_date : str, the date of the comment
    
    Returns:
    response_new : dict, same as response (see function "Structured_representation") but with strings instead of node indices. The following is the list of all the possible keys. Note that it is not guaranteed that all these keywords are present in the responce (depending on if the element is present in the sentence) and their values can also be np.nan:
    - Comment Info - 
    sent : refers to the processed sentence containing the keyword
    c_id : it is the identifier of the comment containing the sentence
    comment_author : the author of the comments
    thread_id : the identifier of the thread in which the comment was submitter
    comment_date : the date of the comment
    pattern_matched : one of "schedule_noun" or "delay_verbs", which is the specific pattern matched in the sentence
    text_short : the specific clause containing the relevant information 
    negations : +1 or -1 depending on whether there are negations in text_short
    
    - XCOMP clause (if any) -
    amod_subj_xcomp_lemma, amod_subj_xcomp_lower : the adjective modifier (lemma and lower case) of the subject of the principal clause
    compound_subj_xcomp_lemma, compound_subj_xcomp_lower : the compound (lemma and lower case) of the subject of the principal clause
    pos_subj_xcomp_lemma, pos_subj_xcomp_lower : the possessive (lemma and lower case) of the subject of the principal clause
    subject_xcomp_lemma, subject_xcomp_lower : the subject (lemma and lower case) of the principal clause
    verb_xcomp_lemma, verb_xcomp_lower : the verb (lemma and lower case) of the principal clause
    verb_phrase_xcomp : verb phrase of the principal clause
    verb_tense_xcomp : tense of the verb of the principal clause
    
    - matched clause - 
    amod_subj_lemma, amod_subj_lower : the adjective modifier (lemma and lower case) of the subject
    compound_subj_lemma, compound_subj_lower : the compound (lemma and lower case) of the subject
    pos_subj_lemma, pos_subj_lower : the possessive (lemma and lower case) of the subject
    subject_lemma, subject_lower : the subject (lemma and lower case) 
    subject_active : one of "ATTIVE" or "PASSIVE", depending on the kind of structure
    verb_lemma, verb_lower : the verb (lemma and lower case) 
    verb_phrase : verb phrase
    verb_tense : tense of the verb
    dobj_amod_lemma, dobj_amod_lower : the adjective modifier (lemma and lower case) of the direct object of the verb
    compound_dobj_lemma,  compound_dobj_lower: the compound (lemma and lower case) of the direct object of the verb
    pos_dobj_lemma, pos_dobj_lower : the possessive (lemma and lower case) of the direct object of the verb
    dobj_lemma, dobj_lower : the direct object (lemma and lower case) of the verb
    
    Example:
    
    >>> text = 'my child is vaccinated and is not on a delayed schedule.'
    >>> idxs_keywords = first_filter_keywords_syntactic(text)
    >>> r = Structured_representation(idxs_keywords)
    >>> [SP.translate_response(i, 'c123456', 'Mommy_2002') for i in r]
    
    [{'sent': 'my child is vaccinated and is not on a delayed schedule.',
      'pattern_matched': 'schedule_noun',
      'dobj_lemma': 'schedule',
      'dobj_lower': 'schedule',
      'verb_lemma': 'be',
      'verb_lower': 'is',
      'verb_phrase': 'is',
      'verb_tense': 'PresentSimple',
      'dobj_amod_lemma': 'delay',
      'dobj_amod_lower': 'delayed',
      'compound_dobj_lemma': nan,
      'compound_dobj_lower': nan,
      'pos_dobj_lemma': nan,
      'pos_dobj_lower': nan,
      'subject_lemma': 'child',
      'subject_lower': 'child',
      'subject_active': 'PASSIVE',
      'amod_subj_lemma': nan,
      'amod_subj_lower': nan,
      'compound_subj_lemma': nan,
      'compound_subj_lower': nan,
      'pos_subj_lemma': 'my',
      'pos_subj_lower': 'my',
      'c_id': 'c123456',
      'comment_author': 'Mommy_2002',
      'text_short': 'my child is vaccinated and is not on a delayed schedule',
      'negations': -1}]
    '''

    G = response['G']
    del response['G']

    response_new = {}
    for key in response.keys():
        if key in ['sent', 'verb_phrase_xcomp', 'verb_tense_xcomp', 'verb_phrase', 'verb_tense', 'pattern_matched', 'subject_active', 'text_short', 'negations']:
            response_new[key] = response[key]
            continue

        translation_lemma = [G.nodes()[idx]['lemma'] if '-' not in G.nodes()[idx]['lemma'] else G.nodes()[idx]['lower'] for idx in response[key]]
        translation_lower = [G.nodes()[idx]['lower'] for idx in response[key]]

        response_new[key+'_lemma'] = ', '.join(translation_lemma) if translation_lemma!=[] else NAN
        response_new[key+'_lower'] = ', '.join(translation_lower) if translation_lower!=[] else NAN

    response_new['c_id'], response_new['comment_author'] = c_id, comment_author
    response_new['thread_id'], response_new['comment_date'] = thread_id, comment_date

    return response_new

# ---------------------------------------------------------------------------------------- #

'''
Now that the structured representation is obtained, it can be read as a Pandas DataFrame

>>> sentences_representation_df = []
>>> for text in ["I think to follow this schedule and I delay vaccines", "I delay this vaccine", "I am happy", "modified schedule!"]: # these are example sentences
>>>    idxs_keywords = SP.first_filter_keywords_syntactic(text)
>>>    sentences_representation_df.extend([SP.translate_response(i, 'c_id', 'Ugo') for i in SP.Structured_representation(idxs_keywords)])
>>> sentences_representation_df = pd.DataFrame.from_dict(sentences_representation_df)
'''

# useful function to get values from the DF
split_objects = lambda string: string.split(', ') if type(string)==str else [NAN]

# load keywords to be filtered with certain depencencies
files_schedule_noun_pattern = listdir('Schedule_noun_pattern_keywords')
KEYWORDS_schedule_noun = {}
for file in [i for i in files_schedule_noun_pattern if '.txt' in i]:
    with open('Schedule_noun_pattern_keywords/'+file) as f:
        KEYWORDS_schedule_noun[file[:-4]] = set(f.read().splitlines())
    
files_delay_verbs_pattern = listdir('Delay_verbs_pattern_keywords')
KEYWORDS_delay_verbs = {}
for file in [i for i in files_delay_verbs_pattern if '.txt' in i]:
    with open('Delay_verbs_pattern_keywords/'+file) as f:
        KEYWORDS_delay_verbs[file[:-4]] = set(f.read().splitlines())

modifiers_stance = {
    'delayed':-1,
    'recommended':+1,
    'alternative':-1,
    'regular':+1,
    'alternate':-1,
    'modified':-1,
    'normal':+1,
    'selective':-1,
    'own':-1,
    'current':+1,
    'different':-1,
    'standard':+1,
    'same':0,
    'traditional':+1,
    'slower':-1,
    'spread':-1,
    'routine':+1,
    'full':+1,
    'typical':+1,
    'strict':0,
    'whole':0,
    'similar':0,
    'suggested':0,
    'entire':0,
    'extended':-1,
    'new':0,
    'vaxing':0,
    'select':-1,
    'spaced':-1,
    'usual':+1,
    'staggered':-1,
    'adjusted':-1,
    'good':0,
    'altered':-1,
    'select':-1       
}

compounds_stance = {
    'cdc' : +1,
    'sears' : -1,
    'delay' : -1,
    'aap' : +1,
    'alt' : -1,
    'reg' : +1,
    'drsears' : -1,  
    'alternative' : -1,
    'custom' : -1,
    'diff' : -1,
    'sear' : -1
}


possessives_schedule_stance = {
    'sears' : -1,
    'searss' : -1,
    'sear' : -1,
    'cdc' : +1,
    'aap' : +1,
    'bob' : -1,
    'cave' : -1,
    'drsears' : -1
    
}

def Classifier_schedule_noun_pattern(row):
    '''
    This function classifies the sentences matched with the "schedule_noun" pattern. It takes into account both the modifiers of the word "schedule" and negations in the sentence.
    
    Input:
    row : a DataFrame row corresponding to the structured representation of the sentence
    
    Returns:
    classification : int, +1 if "recommended" and -1 if "alternative"
    '''
    classification = 1 if row.verb_lemma not in ['delay', 'space', 'spread'] else -1

    if classification == +1:
        # search if at least a modifier with -1
        schedule_modifiers = []
        # stance of the modifiers
        if type(row.dobj_amod_lower) == str:
            class_modif = [modifiers_stance[m] for m in row.dobj_amod_lower.split(', ') if m in modifiers_stance.keys()]
            class_modif = [c for c in class_modif if c!=0] #remove neutral modifiers
            # class modif is -1 if at least one modif with label -1
            class_modif = -1 if -1 in class_modif else +1
            schedule_modifiers.append(class_modif)

        # stance of the compounds
        if type(row.compound_dobj_lower) == str:
            class_compounds = [compounds_stance[m] for m in row.compound_dobj_lower.split(', ') 
                                                           if m in compounds_stance.keys()]
            class_compounds = [c for c in class_compounds if c!=0] #remove neutral modifiers
            # class modif is -1 if at least one modif with label -1
            class_compounds = -1 if -1 in class_compounds else +1
            schedule_modifiers.append(class_compounds)

        # stance of the possessives
        if type(row.pos_dobj_lower) == str:
            class_possessives = [possessives_schedule_stance[m] for m in row.pos_dobj_lower.split(', ') 
                                                 if m in possessives_schedule_stance.keys()]
            class_possessives = [c for c in class_possessives if c!=0] #remove neutral modifiers
            # class modif is -1 if at least one modif with label -1
            class_possessives = -1 if -1 in class_possessives else +1
            schedule_modifiers.append(class_possessives)

        # if at least a negative modifier, change stance
        schedule_modifiers = -1 if -1 in schedule_modifiers else +1 

        classification *= schedule_modifiers   # prod([])=1 , so don't change stance 
        
    # check if negations in the sentence
    classification *= row.negations
    
    return classification
    
    
def Classifier_delay_verbs_pattern(row):
    '''
    This function classifies the sentences matched with the "delay_verbs" pattern. As the verb indicate an altarnative way of vaccinating by itself, here only negations are taken into account.
    
    Input:
    row : a DataFrame row corresponding to the structured representation of the sentence
    
    Returns:
    classification : int, +1 if "recommended" and -1 if "alternative"
    '''
    classification = -1
    classification *= row.negations
    
    return classification


def filter_schedule_noun(row):
    '''
    This function applies the filter to the sentences matched with the 'schedule_noun' pattern to identify sentences not related to the vaccination schedule behavior of the author. It compares the structure representation of the sentence with a set of words in order to filter out unrelevant sentences. For the sentences identified as schedule-related, this function calls the function "Classifier_schedule_noun_pattern" to assign the final label.
    
    Input:
    row : a DataFrame row corresponding to the structured representation of the sentence
    
    Returns:
    classification : 'FILTERED_OUT' if the sentence is not relevant, else +1 or -1 depending on the kind of schedule followed by the author
    '''
    
    # if the modifiers are the ones to avoid skipp
    #if len(set(row.compound_dobj.split(', ')).intersection(set(COMPOUNDS_TO_AVOID)))>0 or \
    #            len(set(row.pos_dobj.split(', ')).intersection(set(POSSESSIVE_SCHEDULE_TO_AVOID)))>0:
    flag = False
    if len(set(split_objects(row.compound_dobj_lower)).intersection(KEYWORDS_schedule_noun['COMPOUNDS_TO_AVOID']))>0 or \
                len(set(split_objects(row.pos_dobj_lower)).intersection(KEYWORDS_schedule_noun['POSSESSIVE_SCHEDULE_TO_AVOID']))>0:
        flag = True 

    # if the verb not among the ones to use
    if row.verb_lemma not in KEYWORDS_schedule_noun['VERBS_lemma_TO_USE']:
        flag = True

    # if subject not nan and not among the ones to use
    if type(row.subject_lower) == str and row.subject_lower not in KEYWORDS_schedule_noun['SUBJECTS_TO_USE']:
        flag = True#1

    # if xcomp verb not among the ones to use
    if type(row.verb_xcomp_lemma) == str and row.verb_xcomp_lemma not in KEYWORDS_schedule_noun['XCOMP_VERB_lemma_TO_USE']:
        flag = True#2

    # if xcomp subject not among the ones to use
    if type(row.subject_xcomp_lower) == str and row.subject_xcomp_lower not in KEYWORDS_schedule_noun['SUBJECTS_TO_USE']:
        flag = True#3

    # if possessive of subject not among the ones to be used
    if (row.pos_subj_xcomp_lower not in KEYWORDS_schedule_noun['POSSESSIVES_SUBJECT_TO_USE'] and type(row.pos_subj_xcomp_lower)==str) or  \
                    (row.pos_subj_lower not in KEYWORDS_schedule_noun['POSSESSIVES_SUBJECT_TO_USE'] and type(row.pos_subj_lower)==str):
        flag = True

    # if no subjects
    if type(row.subject_xcomp_lower) != str and type(row.subject_lower) != str:
        flag = True


    if flag:
        return 'FILTERED_OUT'
    else:
        classification = Classifier_schedule_noun_pattern(row)
        return classification
    
def filter_delay_verbs(row):
    
    '''
    This function applies the filter to the sentences matched with the 'delay_verbs' pattern to identify sentences not related to the vaccination schedule behavior of the author. It compares the structure representation of the sentence with a set of words in order to filter out unrelevant sentences. For the sentences identified as schedule-related, this function calls the function "Classifier_schedule_noun_pattern" to assign the final label.
    
    Input:
    row : a DataFrame row corresponding to the structured representation of the sentence
    
    Returns:
    classification : 'FILTERED_OUT' if the sentence is not relevant, else +1 or -1 depending on the kind of schedule followed by the author
    '''
    
    flag = False
    # if the objects is among the ones not to take
    if len(set(split_objects(row.dobj_lower)).intersection(KEYWORDS_delay_verbs['OBJECTS_TO_AVOID']))>0:
        flag = True

    # if the modifiers of the object are the ones to avoid skipp
    if len(set(split_objects(row.compound_dobj_lower)).intersection(KEYWORDS_delay_verbs['COMPOUNDS_TO_AVOID']))>0 or \
                 len(set(split_objects(row.pos_dobj_lower)).intersection(KEYWORDS_delay_verbs['POSSESSIVES_TO_AVOID']))>0:
        flag = True 

    # if subject not among the ones to use
    if type(row.subject_lower) == str and row.subject_lower not in KEYWORDS_delay_verbs['SUBJECT_TO_USE'] and row.subject_active == 'ACTIVE':
        flag = True

    # if subject not among the ones to use
    if type(row.subject_lower) == str and row.subject_active == 'PASSIVE' and row.subject_lower not in KEYWORDS_delay_verbs['SUBJECT_PASSIVE_TO_USE']:
        flag = True

    # if xcomp verb not among the ones to use
    if type(row.verb_xcomp_lemma) == str and row.verb_xcomp_lemma not in KEYWORDS_delay_verbs['XCOMP_VERBS_lemma_TO_USE']:
        flag = True  

    # if xcomp subject not among the ones to use
    if type(row.subject_xcomp_lower) == str and row.subject_xcomp_lower not in KEYWORDS_delay_verbs['SUBJECT_TO_USE']:
        flag = True   

    # if no subjects
    if type(row.subject_xcomp_lower) != str and type(row.subject_lower) != str:
        flag = True

    # store filtered out results
    if flag:
        return 'FILTERED_OUT'
    else:
        classification = Classifier_delay_verbs_pattern(row)
        return classification

    
def Filter(row):
    '''
    This function applies the filter to the sentences matched with the patterns "schedule_noun" and "delay_verbs" and applies the classifier to schedule-related sentences.
    
    Input:
    row : a DataFrame row corresponding to the structured representation of the sentence
    
    Returns:
    classification : 'FILTERED_OUT' if the sentence is not relevant, else +1 or -1 depending on the kind of schedule followed by the author
    '''
    
    flag = False
    which_pattern = row.pattern_matched
    
    if which_pattern == 'schedule_noun':
        
        return filter_schedule_noun(row)
        
    elif which_pattern == 'delay_verbs':
        
        return filter_delay_verbs(row)