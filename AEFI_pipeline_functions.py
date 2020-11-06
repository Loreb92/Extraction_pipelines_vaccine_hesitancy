import pandas as pd
import spacy
import re
from nltk.tokenize import sent_tokenize
from numpy import nan as NAN
import text_elaboration as te
import Dependency_tree_functions as DepTree

nlp = spacy.load("en_core_web_sm")

remove_pattern = "(?:https?:\/\/)?(?:[^@\/\n]+@)?(?:www\.)[^)\s]+|https?:\/\/(?!www\.)[^)\s]+"

# load contractions and acronyms
CONTRACTIONS = te.load_contractions('utils/contractions_eng.txt')
ACRONYMS = te.load_acronyms_BBC('utils/acronyms_BBC_eng.txt')
        
# US (ultra sound) conflicts with US (united states)
del ACRONYMS['US']

# get posts and tokenize them (both title and body)
folder_data = 'data/'
posts = pd.read_csv(folder_data+'sample_of_post_titles.csv', dtype=str)
posts = posts[['thread_title', 'thread_id']]

# load stopwords
stopw = []
with open('Experiences_AEFI/stopwords_post_titles.txt', 'rt') as rr:
    for line in rr:
        stopw.append(line.strip())
        
remove_pattern = "(?:https?:\/\/)?(?:[^@\/\n]+@)?(?:www\.)[^)\s]+|https?:\/\/(?!www\.)[^)\s]+"

# tekenize post titles
posts['token_title'] = posts.thread_title.apply(lambda txt: te.extraxt_tokens(re.sub('\d+', '', str(txt)), 'eng', 
                  [], None, stopwords=stopw, remove_pattern=remove_pattern, 
                  tokenize_sentences=True, min_len=3, tokenizer='tokenize_words', n_gram=4,
                  exceptions=[], return_as_text=False))


def preprocess_sentence(sent):
    '''
    Preprocessing of the sentences. In order do:
    2) remove strange characters and pattern define above (remove urls)
    3) replace numbers with words : 'one' --> 1
    4) replace 'my husband and I' with 'we'
    5) remove smiles
    6) replace acronyms and expand contractions
    7) take sentences between parenthesis
    8) remove superfluous spaces and replace "dr." with "dr"
    
    Input:
    sent : str, a sentence
    remove_pattern : str, pattern to be removed from the sentence
    
    Returns:
    sents : list of str, a list containing the text without parenthesis and the texts between parentheses
    '''
    
    sent = sent.replace("’", "'")
    sent = sent.replace('\xa0', ' ')
    sent = sent.replace("\n", ' ')
    sent = sent.replace("*", '')
    sent = re.sub(remove_pattern, " ", sent)  #sustitute a token structure with a space and transform in lower case
    
    
    # split numbers and words if no space between them (e.g., "my daughter2" --> "my daughter two")
    # if a number occurs at the end of a word, split it and make it in characters (no change h1n1)
    # "My DD2 is vaccinated" --> "My DD two is vaccinated"
    for num, num_str in zip(['1', '2', '3'], [' one', ' two', ' three']):        
        sent = re.sub('(?i)(?<=\w)(?<!h1n)%s(?=\s)'%num, num_str, sent)
            
    if 'my husband and I' in sent:
        sent = sent.replace('my husband and I', 'we')
            
    sent = sent.replace(':)', '')
    sent = sent.replace(':-)', '')
    sent = sent.replace(':(', '')

    sent = te.replace_acronyms(sent, ACRONYMS)
    sent = te.expand_contractions(sent, CONTRACTIONS)

    # match strings between parentheses
    sent_between_parenthesis = re.findall('(?<=\().+?(?=\))', sent)
    
    for parent in sent_between_parenthesis:
        sent = sent.replace('('+parent+')', ' ')

    sent = re.sub('^\s{1,}|\s{1,}$', "", sent)  # drop all spaces at the beginning/end of the text
    sent_between_parenthesis = [re.sub('^\s{1,}|\s{1,}$', "", s) for s in sent_between_parenthesis]
    
    sent = re.sub('\s{2,}', ' ', sent)  # sobstitute each space character longer than one with only one space
    sent_between_parenthesis = [re.sub('\s{2,}', ' ', s) for s in sent_between_parenthesis]

    sent = sent.lower() # lower case to avoid problems with dep tree matching upper case things
    sent_between_parenthesis = [s.lower() for s in sent_between_parenthesis]
    
    sent = sent.replace("dr.", 'dr')
    sent_between_parenthesis = [s.replace("dr.", 'dr') for s in sent_between_parenthesis]
    
    sentences = [sent]+sent_between_parenthesis
    
    return sentences


def filter1_couples_of_words(text, list1, list2):
    '''
    Given two lists of words, search in the text the words contained in the two lists (disregarding case)
    
    Input:
    text : str, the text to be inspected
    list1, list2 : lists of strings, the words to be searched in text
    
    Returns:
    list_1_matches, list_2_matches : lists of strings, the matches found in the text for the two lists respectively. If no matches, the list is empty
    
    Example:
    >>> text = "Hi there, I think to be sick as Daniel"
    >>> filter1_couples_of_words(text, ['hotdog'], ['daniel', 'there'])
    ([], ['Daniel', 'there'])
    '''
    
    list_1_matches = list(set(re.findall(r"(?i)\b(?:%s)\b"%('|'.join(list1)), text)))
    list_1_matches = [] if list_1_matches==[''] else list_1_matches
    
    list_2_matches = list(set(re.findall(r"(?i)\b(?:%s)\b"%('|'.join(list2)), text)))
    list_2_matches = [] if list_2_matches==[''] else list_2_matches
    
    return list_1_matches, list_2_matches

def basic_filter(text, verbs_pattern1, objects_pattern1, verbs_pattern2, objects_pattern2, verbs_pattern3):
    '''
    Perform the first basic filter in order to check if the sets of words related to different patterns are present in the text.
    
    Input:
    text : str, the text to be inspected
    verbs_pattern1, objects_pattern1 : list of strings, the list of words (verbs and objects) used for pattern 1
    verbs_pattern2, objects_pattern2 : list of strings, the list of words (verbs and objects) used for pattern 2
    verbs_pattern3 : list of string, verbs for pattern 3
    
    Returns:
    results : dict, the response with the list of keywods for each pattern
    
    Example:
    >>> text = 'My son had an high fever and it was not funny.'
    >>> basic_filter(text, verbs_pattern1, objects_pattern1, verbs_pattern2, objects_pattern2, verbs_pattern3)
    {'pattern1': {'verbs': ['had'], 'objects': ['fever']}}
    '''
    
    results = {}
    
    list_verbs1, list_objects1 = filter1_couples_of_words(text, verbs_pattern1, objects_pattern1)
    if len(list_verbs1)>0 and len(list_objects1)>0:
        results['pattern1'] = {'verbs' : list_verbs1, 'objects' : list_objects1}
    
    list_verbs2, list_objects2 = filter1_couples_of_words(text, verbs_pattern2, objects_pattern2)
    if len(list_verbs2)>0 and len(list_objects2)>0:
        results['pattern2'] = {'verbs' : list_verbs2, 'objects' : list_objects2}
    
    list_verbs3, list_objects3 = filter1_couples_of_words(text, verbs_pattern3, [])
    if len(list_verbs3)>0:
        list_verbs3 = [v.split()[0] for v in list_verbs3]
        results['pattern3'] = {'verbs' : list_verbs3}
        
    return results

# load keywords (determined through manual inspection)
# PATTERN 1
verbs_pattern1 = []
with open('Experiences_AEFI/pattern1_verb_and_dobj/VERBS_TO_TAKE_INFLECT.txt', 'rt') as f:
    for line in f:
        verbs_pattern1.append(line.strip())
        
objects_pattern1 = []
with open('Experiences_AEFI/pattern1_verb_and_dobj/OBJECTS_TO_KEEP_INFLECTIONS.txt', 'rt') as f:
    for line in f:
        objects_pattern1.append(line.strip())
        
objects_pattern1_with_vax = []
with open('Experiences_AEFI/pattern1_verb_and_dobj/OBJECTS_TO_KEEP_WITH_VAX.txt', 'rt') as f:
    for line in f:
        objects_pattern1_with_vax.append(line.strip())
        
objects_support_pattern1 = ['signs', 'cases', 'bit', 'type', 'form', 'sort']
        
# PATTERN 2
verbs_pattern2 = []
with open('Experiences_AEFI/pattern2_be_and_acomp/VERBS_TO_TAKE_INFLECT.txt', 'rt') as f:
    for line in f:
        verbs_pattern2.append(line.strip())
        
objects_pattern2 = []
with open('Experiences_AEFI/pattern2_be_and_acomp/ACOMP_TO_TAKE.txt', 'rt') as f:
    for line in f:
        objects_pattern2.append(line.strip())
        
objects_pattern2_with_vax = []
with open('Experiences_AEFI/pattern2_be_and_acomp/ACOMP_WITH_VAX.txt', 'rt') as f:
    for line in f:
        objects_pattern2_with_vax.append(line.strip())
        
# PATTERN 3
verbs_pattern3 = ['react', 'reacted', 'reacting', 'reacts']
verbs_pattern3 = [r+' to' for r in verbs_pattern3]

objects_pattern3_not_to_take = []
with open('Experiences_AEFI/pattern3_react_to/OBJ_TO_REMOVE.txt', 'rt') as f:
    for line in f:
        objects_pattern3_not_to_take.append(line.strip())
        

# VACCINE VERBS AND WORDS 
vaccine_words = []
with open('Experiences_AEFI/WORDS_VAX_INFLECTIONS.txt', 'rt') as f:
    for line in f:
        vaccine_words.append(line.strip())
        
vaccine_verbs = []
with open('Experiences_AEFI/VERB_VAX_INFLECTIONS.txt', 'rt') as f:
    for line in f:
        vaccine_verbs.append(line.strip())

##### CONTEXTUAL PATTERNS
def replace_shot_with_vaccine(sent):
    '''
    Given a sentence, replace the word 'shot' with the word 'vaccine'. It is useful because the parser ofter considers 'shot' as a verb. This function is used for the contextual patterns.
    '''
    sent = re.sub(r'(?i)\b(?:shot)\b', 'vaccine', sent)
    sent = re.sub(r'(?i)\b(?:shots)\b', 'vaccines', sent)
    
    return sent


def after_vaccine(G, vaccine_word):
    '''
    Contextual pattern
    This function returns a flag, indicating if in the dependency tree there is a link between an adverb of 
    time (e.g., "after") and a word related to vaccines.
    
    Input:
    G : nx.DiGraph, the dependency tree
    vaccine_word : str, a word related to vaccine
    
    Returns:
    flag : bool, True if the link exists in the dependency tree, else False
    '''
    
    # find the indices of the "get vax" expression
    matches = [True for sour, tar, dep in G.edges(data='dep') 
                                               if G.nodes()[sour]['lower'] in vaccine_word and
                                                  #G.nodes()[tar]['lower'] in ['after', 'whenever', 'since', 'from']]
                                                   G.nodes()[tar]['lower'] in ['after', 'whenever']]
        
    return any(matches) 


def someone_get_vaccine(G, vaccine_word, vaccine_verb):
    '''
    Contextual pattern
    Determine if one of the following patterns is present in the dependency tree: 
    <someone> -- nsubj --> < get > <-- dobj -- < vaccine > 
    <after/whenever> + <someone> -- nsubj --> < get > <-- dobj -- < vaccine >
    after/whenever + < get > <-- dobj -- < vaccine > 
    
    Input: 
    G : nx.DiGraph, the dependency tree
    vaccine_word : str, one of the words identifying the vaccine (e.g., 'vaccine', 'vax', 'mmr')
    vaccine_verb : str, one of the verbs inflected related to receiving the vaccine (e.g., 'had', 'got')
    
    Returns:
    flag : bool, True if the pattern exists in the dependency tree, else False
    '''
    
    # find the indices of the "get vax" expression
    matches = [(sour, tar) for sour, tar, dep in G.edges(data='dep') 
                                                                   if G.nodes()[sour]['lower'] in vaccine_word and
                                                                      G.nodes()[tar]['lower'] in vaccine_verb]   
    for match in matches:
        vax_word_idx, vax_verb_idx = match
        
        # search subjects
        subjs = DepTree.find_subjects(G, vax_verb_idx, passive=False)
        
        # is 'after/whenever' an out_neig of the verb?
        after_idx = [tar for sour, tar, dep in G.out_edges(vax_verb_idx, data='dep') 
                                                 if G.nodes()[tar]['lower'] in ['after', 'whenever', 'since', 'from']]
        # if no subj or 'after' continue
        if len(subjs)>0 or len(after_idx)>0:
            return True
            
    return False


def is_reaction_to_vaccine(G, verb_idx, object_idx, vaccine_words):
    '''
    Contextual pattern
    This contextual pattern is applied when a retrieval pattern is matched. It searches a pattern like:
    <retrieval pattern> --prep--> <preposition> <--pobj-- <vaccine> (e.g., "From my understanding, they can get a rash of the shots")
    <retrieval pattern> --prep--> <preposition> <--pobj-- <noun> <--prep-- <preposition> <--pobj-- <vaccine> (e.g., "From my understanding they can get a rash from a reaction of the shots")
    
    Inputs:
    G : nx.DiGraph, the dependency tree
    verb_idx : int, the index of the verb matched with the retrieval patter
    object_idx : int, the index of the direct object matched with the retrieval patter
    vaccine_words : iterable of strings, words identifying vaccine (like 'vaccine', 'vax')
    
    Returns:
    flag : bool, True if the contextual pattern is matched, False elsewhere
    '''
    
    # chech the prepositions linked with the verb or the obj
    preps_idx = [sour for sour, tar, dep in G.edges(data='dep') if dep=='prep' and (tar==verb_idx or tar==object_idx)]
    
    # search among in_edges of prep (pobj dependency)
    preps_objs = [sour for prep_idx in preps_idx 
                                  for sour, tar, dep in G.in_edges(prep_idx, data='dep') if dep=='pobj']
    preps_objs_name = [G.nodes()[prep_obj]['lower'] for prep_obj in preps_objs]
    
    # if no matches, search the next preposition (<prep>+<whatever>+<prep>+<vaccine>)
    matches = set(preps_objs_name).intersection(set(vaccine_words))
    if len(matches)==0:
        second_preps_idx = [sour for prep_idx in preps_objs 
                                  for sour, tar, dep in G.in_edges(prep_idx, data='dep') if dep=='prep']
        
        preps_objs = [sour for prep_idx in second_preps_idx 
                                  for sour, tar, dep in G.in_edges(prep_idx, data='dep') if dep=='pobj']
        preps_objs_name = [G.nodes()[prep_obj]['lower'] for prep_obj in preps_objs]
        
        matches = set(preps_objs_name).intersection(set(vaccine_words))
    
    flag = True if len(matches)>0 else False
    
    
    return flag


def post_about_reactions(post_title_tokens):
    '''
    Contextual pattern
    This function check whether there are certain tokens in the main post. If none of the previous contextual pattern is matched, this function check if the title of the posts is related to adverse reactions through n-grams
    
    Input:
    post_title_tokens : list of strings, the ngrams found in the post title
    
    Returns:
    flag : bool, True if the title of the post contain at least one of the ngrams considered
    '''
    
    # these ngrams were determined by inspecting the most frequent ones among the titles of the posts
    ngrams_post_title = set(['reaction', 'after vaccine', 'side effect', 'vaccine reaction', 'after vaccination', 
                         'fever after', 'after shot',
'vaccine side effect', 'reaction vaccine', 'after month vaccine', 'fever after vaccine', 'after month shot',
'bad reaction', 'after mmr', 'vaccination reaction', 'fever after vaccination', 'after tdap',
'after month vaccination', 'mmr vaccine reaction', 'mmr reaction', 'vaccination side effect'])
    
    flag = False
    if len(set(post_title_tokens).intersection(ngrams_post_title))!=0:
        flag = True
    
    return flag
#####


def filter_reactions(G, verb_idx, reaction_idxs, objects_list, objects_with_vax_list, objects_support_list=[]):
    '''
    This functions take as input the verb and the candidate reactions and filter them accrding to the lists of relevant/unrelevant words.
    It returns the filtered list of reactions.
    
    Input:
    G : nx.DiGraph, the dependency tree
    verb_idx : int, the index of the verb matched
    reaction_idxs : list of ints, the indices in the dependency tree of the candidate reactions
    objects_list : list of str, the list of reaction words to consider
    objects_with_vax_list : list of str, the list of reaction words to consider in combination with the contextual pattern "is_reaction_to_vaccine"
    objects_support_list : list of str, list of words of support (e.g., "sort" for "She had a sort of fever")
    
    Returns : 
    reactions_filtered : list of ints, the indices of the reactions filtered in
    is_reaction_to_vaccine_flag : bool, the result of the contextual pattern "is_reaction_to_vaccine"
    '''
    # check "is_reaction_to_vaccine"
    is_reaction_to_vaccine_flag = any([is_reaction_to_vaccine(G, verb_idx, reaction_idx, vaccine_words) 
                                                                       for reaction_idx in reaction_idxs])
    
    # filter out the candidate reactions not in list
    reactions_filtered = []
    for reaction_idx in reaction_idxs:
        
        reactions_lower = G.nodes()[reaction_idx]['lower']
        
        if reactions_lower in objects_list: # if in objects_list do not filter out
            reactions_filtered.append(reaction_idx)
            
        elif reactions_lower in objects_with_vax_list: # check if "is_reaction_to_vaccine" pattern
            
            if is_reaction_to_vaccine_flag:
                #reactions_filtered.append(reaction_idx)
                None

        elif reactions_lower in objects_support_list: # check if ".. kind of reaction" pattern
            ofs_idx = [sour for sour, tar, dep in G.in_edges(reaction_idx, data='dep') 
                                          if G.nodes()[sour]['lower']=='of']
            
            if len(ofs_idx)>0:

                objs_of_idxs = DepTree.find_objects(G, ofs_idx[0], depend='pobj')
                for objs_of_idx in objs_of_idxs:
                    objs_of_lower = G.nodes()[objs_of_idx]['lower']

                    if objs_of_lower in objects_list: # if in objects_list do not filter out
                        reactions_filtered.append(objs_of_idx)
                        
                    elif objs_of_lower in objects_with_vax_list: # check if "is_reaction_to_vaccine" pattern
                        if is_reaction_to_vaccine_flag:
                            #reactions_filtered.append(objs_of_idx)
                            None
            
        else:
            None
                        
    return reactions_filtered, is_reaction_to_vaccine_flag



def first_filter_keywords_syntactic(text):
    '''
    Given a text, search for the retrieval patterns and return the indices of the corresponding words as a dictionary.
    
    Input:
    text : str, the sentence to inspect
    
    Returns:
    idxs_keywords : dict, it contains a report with the kind of pattern matched and the indices matched. Each match has its own key in the dictionary.
    
    Example:
    
    >>> text = 'My son had an high fever and it was not funny, however he was also fussy.'
    >>> first_filter_keywords_syntactic(text)
    {'pattern1_1': {'verb_idx': 7,
                    'reaction_idxs': [19],
                    'is_reaction_to_vaccine_flag': False},
     'pattern2_1': {'verb_idx': 58,
                    'reaction_idxs': [67],
                    'is_reaction_to_vaccine_flag': False},
     'G': <networkx.classes.digraph.DiGraph at 0x7f3444c1a790>,
     'after_vaccine_flag': False,
     'someone_get_vaccine_flag': False,
     'text': 'My son had an high fever and it was not funny, however he was also fussy.'}
    
    '''
    
    # first search of keywords belonging to the three patterns
    first_filter_results = basic_filter(text, verbs_pattern1, objects_pattern1+objects_pattern1_with_vax,
                                              verbs_pattern2, objects_pattern2+objects_pattern2_with_vax,
                                              verbs_pattern3)
    
    idxs_keywords = {}
    if len(first_filter_results)>0:
        G = DepTree.make_dep_tree(nlp(text))
        
        counter = {'pattern1' : 0, 'pattern2' : 0, 'pattern3' : 0}
        
        for kind_of_pattern, candidate_keywords in first_filter_results.items():
            
            if kind_of_pattern != 'pattern3':
                # take the words matched
                verbs_name, reactions_name = candidate_keywords['verbs'], candidate_keywords['objects']
                
                for verb_name in set(verbs_name):
                    # search the indices of the verbs in the dep tree (exclude auxiliaries)
                    idxs_keyword_verb = DepTree.get_nodes_by_attribute(G, 'lower', verb_name.lower())
                    idxs_keyword_verb = [sour for idx_verbs in idxs_keyword_verb 
                                  for sour, tar, dep in G.out_edges(idx_verbs, data='dep') 
                                                                          if dep not in ['aux', 'auxpass']]
                    
                    for idx_keyword_verb in idxs_keyword_verb:
                            
                        if kind_of_pattern=='pattern1':
                            
                            idxs_candidate_reactions = DepTree.find_objects(G, idx_keyword_verb, ['dobj'])
                            idxs_candidate_reactions, is_reaction_to_vaccine_flag = filter_reactions(G, idx_keyword_verb, idxs_candidate_reactions, 
                                                                        objects_pattern1, 
                                                                        objects_pattern1_with_vax, 
                                                                        objects_support_list=objects_support_pattern1)
                        
                        else:
                            idxs_candidate_reactions = DepTree.find_objects(G, idx_keyword_verb, ['attr', 'acomp'])    
                            idxs_candidate_reactions, is_reaction_to_vaccine_flag = filter_reactions(G, idx_keyword_verb, idxs_candidate_reactions, 
                                                                        objects_pattern2, 
                                                                        objects_pattern2_with_vax, 
                                                                        objects_support_list=[])
                            
                        if len(idxs_candidate_reactions)>0:
                            counter[kind_of_pattern] += 1
                            # check after vaccine pattern
                            idxs_keywords[kind_of_pattern+'_'+str(counter[kind_of_pattern])] = {'verb_idx':idx_keyword_verb,
                                                                   'reaction_idxs':idxs_candidate_reactions,
                                                                   'is_reaction_to_vaccine_flag':is_reaction_to_vaccine_flag} 
                            
                            
            else: # it is pattern3
                
                verbs_name = candidate_keywords['verbs']
                
                for verb_name in set(verbs_name):
                    # search the indices of the verbs in the dep tree (exclude auxiliaries)
                    idxs_keyword_verb = DepTree.get_nodes_by_attribute(G, 'lower', verb_name.lower())
                    
                    for idx_keyword_verb in idxs_keyword_verb:
                        
                        # search index of 'to'
                        to_idxs = [sour for sour, tar, dep in G.in_edges(idx_keyword_verb, data='dep') 
                              if G.nodes()[sour]['lower']=='to' and dep=='prep']
                        
                        if len(to_idxs)==0: 
                            continue
                            
                        to_idx = to_idxs[0]
                        pobj_to_idxs = DepTree.find_objects(G, to_idxs, depend=['pobj'])
                        
                        #pobj_to_idxs = [pobj_to_idx for pobj_to_idx in pobj_to_idxs if G.nodes()[pobj_to_idx]['lower'] in vaccine_words]
                        pobj_to_idxs = [pobj_to_idx for pobj_to_idx in pobj_to_idxs if G.nodes()[pobj_to_idx]['lower'] not in objects_pattern3_not_to_take]
                        
                        
                        
                        if len(pobj_to_idxs)>0:
                            counter[kind_of_pattern] += 1
                            idxs_keywords[kind_of_pattern+'_'+str(counter[kind_of_pattern])] = {'verb_idx':idx_keyword_verb,
                                                                   'reaction_idxs':pobj_to_idxs,
                                                                   'is_reaction_to_vaccine_flag':True} # here the flag is True by definition 
    # if at least one match, fill the report          
    if len(idxs_keywords)>0:
        idxs_keywords['G'] = G
        
        after_vaccine_flag = after_vaccine(G, vaccine_words)
        someone_get_vaccine_flag = someone_get_vaccine(G, vaccine_words, vaccine_verbs)
        
        idxs_keywords['after_vaccine_flag'] = after_vaccine_flag
        idxs_keywords['someone_get_vaccine_flag'] = someone_get_vaccine_flag
        idxs_keywords['text'] = text
        
    
    return idxs_keywords


def check_contextual_patterns_whole_comment(comment, comment_info, thread_id):
    '''
    Search matches of the contextual patterns in the whole comment.
    
    Input:
    comment : str, the comment inspected
    commen_info : list, it contains information abou the sentences in which a match was found
    thread_id : str, the index of the thread under which the comment was submitted
    
    Returns:
    
    is_related, is_reaction_to_vaccine_flag, after_vaccine_flag, someone_got_vaccine_flag, post_related_flag : bool
    '''   
    # check the title of the post
    token_title = posts[posts.thread_id==thread_id].token_title.values[0]
    post_related = post_about_reactions(token_title)
    
    if post_related:
        return True, False, NAN, NAN, True

    # for each sentence, search the contextual patterns
    for sentence in sent_tokenize(comment):
        for sent in preprocess_sentence(sentence):

            sent = replace_shot_with_vaccine(sent)
            G = DepTree.make_dep_tree(nlp(sent))
            after_vaccine_flag = after_vaccine(G, vaccine_words)
            someone_get_vaccine_flag = someone_get_vaccine(G, vaccine_words, vaccine_verbs)
            
            if after_vaccine_flag or someone_get_vaccine_flag:
                return True, False, after_vaccine_flag, someone_get_vaccine_flag, False

    return False, False, False, False, False


               
# ---------------------------------------------------------------------------------------- #
def Structured_representation(idxs_keywords):
    '''
    Given the words matched by each kind of pattern, extract the structured representation by searching for noun modifiers and subjects.
    
    Input:
    idxs_keywords : dict, the output of "first_filter_keywords_syntactic"
    
    Returns:
    reports : list, contains the relevant elements associated to each match
    '''
    
    # loop over all reports
    if len(idxs_keywords)==0:
        return []
    
    G, sent = idxs_keywords['G'], idxs_keywords['text']
    
    reports = []
    for group_of_keyword, keyword_report in idxs_keywords.items():
        if group_of_keyword in ['G', 'text', 'after_vaccine_flag', 'someone_get_vaccine_flag']:
            continue
        
        verb_idx = keyword_report['verb_idx']
        reaction_idxs = keyword_report['reaction_idxs']
        is_reaction_to_vaccine_flag = keyword_report['is_reaction_to_vaccine_flag']
        
        report = {'G':G, 'sent':sent, 'pattern_matched':group_of_keyword}
        #group_of_keyword = group_of_keyword.split('_')[0]
        verb_phrase, verb_tense = DepTree.get_verb_phrase(G, verb_idx)
        neg_verb = any([True for sour, tar, dep in G.in_edges(verb_idx, data='dep') if dep=='neg'])
        
        subjects_idxs = DepTree.find_subjects(G, verb_idx)
        subjects_idxs = [DepTree.iterate_over_conj(G, subjects_idx) for subjects_idx in subjects_idxs]
        subjects_idxs = [item for items in subjects_idxs for item in items]
        subjects = {}
        for n, subjects_idx in enumerate(subjects_idxs):
            subjects['subject_'+str(n+1)] = {}
            subjects['subject_'+str(n+1)]['subject'] = subjects_idx
                        
            amod_subj, compound_subj, pos_subj, det_subj = DepTree.find_modifiers(G, subjects_idx)
            negations_subj = [sour for sour, tar, dep in G.in_edges(subjects_idx, data='dep') 
                                                 if G.nodes()[sour]['lower'] in ['none', 'no', 'neither',
                                                                                        'zero']]
            
            subjects['subject_'+str(n+1)]['amod_subj'] = amod_subj
            subjects['subject_'+str(n+1)]['compound_subj'] = compound_subj
            subjects['subject_'+str(n+1)]['pos_subj'] = pos_subj
            subjects['subject_'+str(n+1)]['det_subj'] = det_subj
            subjects['subject_'+str(n+1)]['negations_subj'] = negations_subj
        
        if len(subjects)==0: continue
            
        reactions = {}
        for n, reaction_idx in enumerate(reaction_idxs):
            reactions['reaction_'+str(n+1)] = {'reaction':reaction_idx}

            amod_react, compound_react, pos_react, det_react = DepTree.find_modifiers(G, reaction_idx)
            negations_react = [sour for sour, tar, dep in G.in_edges(reaction_idx, data='dep') 
                                                 if G.nodes()[sour]['lower'] in ['none', 'no', 'neither',
                                                                                        'zero']]
            
            reactions['reaction_'+str(n+1)]['amod_react'] = amod_react
            reactions['reaction_'+str(n+1)]['compound_react'] = compound_react
            reactions['reaction_'+str(n+1)]['pos_react'] = pos_react
            reactions['reaction_'+str(n+1)]['det_react'] = det_react
            reactions['reaction_'+str(n+1)]['negations_react'] = negations_react
            reactions['reaction_'+str(n+1)]['is_reaction_to_vaccine_flag'] = is_reaction_to_vaccine_flag
            
        
        report['subjects'] = subjects
        
        report['verb'] = verb_idx
        report['verb_phrase'] = verb_phrase 
        report['verb_tense'] = verb_tense
        report['verb_negation'] = neg_verb
        
        report['reactions'] = reactions
        
        report['after_vaccine_flag'] = idxs_keywords['after_vaccine_flag']
        report['someone_get_vaccine_flag'] = idxs_keywords['someone_get_vaccine_flag']
        
        reports.append(report)
    
        
    return reports

def translation_subjects_and_reactions(G, dict_indices):
    '''
    This function translate the "subjects" and "reactions" dict.
    
    Input:
    G : 
    dict_indices : dict, the "subjects" or "reactions" dict
    
    Returns:
    dict_indices_new : dict, equal to dict_indices but with words instead of indices
    '''
    dict_indices_new = {}
    for item_n, value_object in dict_indices.items():
        dict_indices_new[item_n] = {}
        for key, value in value_object.items():
            
            if key == 'is_reaction_to_vaccine_flag':
                dict_indices_new[item_n][key] = value
            else:
        
                dict_indices_new[item_n][key+'_lower'] = [G.nodes()[v]['lower'] for v in value] if type(value)==list \
                                                                    else G.nodes()[value]['lower']

                dict_indices_new[item_n][key+'_lemma'] = [G.nodes()[v]['lemma'] if '-' not in G.nodes()[v]['lemma'] else G.nodes()[v]['lower'] for v in value] if type(value)==list \
                                                                    else G.nodes()[value]['lemma'] if '-' not in G.nodes()[value]['lemma'] else G.nodes()[value]['lower']

    return dict_indices_new
            
        
def translate_response(response, c_id, comment_author, thread_id, comment_date):        
    '''
    Thranslate the structured representation replacing indices with the corresponding words.
    In addition, add the information of the comment like c_id, thread_id
    
    Input:
    response : dict, the output of "Structured_representation"
    c_id : str, the identifier of the comment
    comment_author : str, the author. ofthe comment
    thread_id : str, the identifier of the thread under which the comment was submitted
    comment_date : str, the date of the comment
    
    Returns:
    response_new : dict, equal to response but with words instead of indices
    '''
    
    G = response['G']
    del response['G']
    
    response_new = {}
    for key_response, values_response in response.items():
        
        if key_response in ['subjects', 'reactions']:
            
            response_new[key_response] = translation_subjects_and_reactions(G, values_response)
                
        elif key_response == 'verb':
            response_new[key_response+'_lower'] = G.nodes()[values_response]['lower']
            response_new[key_response+'_lemma'] = G.nodes()[values_response]['lemma'] if '-' not in G.nodes()[values_response]['lemma'] else G.nodes()[values_response]['lower']
            
        else:
            response_new[key_response] = values_response
            
    response_new['c_id'], response_new['comment_author'] = c_id, comment_author
    response_new['thread_id'], response_new['comment_date'] = thread_id, comment_date
                
    return response_new
                
                
# ---------------------------------------------------------------------------------------- #

# LOAD BACKLISTS
# load subjects to keep Pattern1
pattern1_SUBJECTS_TO_KEEP = []
with open('Experiences_AEFI/pattern1_verb_and_dobj/SUBJECTS_TO_KEEP.txt', 'rt') as rr:
    for line in rr:
        pattern1_SUBJECTS_TO_KEEP.append(line.strip())
        
pattern1_SUBJECTS_TO_KEEP_WITH_POSS = []
with open('Experiences_AEFI/pattern1_verb_and_dobj/SUBJECTS_TO_KEEP_WITH_POSS.txt', 'rt') as rr:
    for line in rr:
        pattern1_SUBJECTS_TO_KEEP_WITH_POSS.append(line.strip())
                
# load subjects to keep Pattern2 
pattern2_SUBJECTS_TO_KEEP = []
with open('Experiences_AEFI/pattern2_be_and_acomp/SUBJECTS_TO_KEEP.txt', 'rt') as rr:
    for line in rr:
        pattern2_SUBJECTS_TO_KEEP.append(line.strip())
        
pattern2_SUBJECTS_TO_KEEP_WITH_POSS = []
with open('Experiences_AEFI/pattern2_be_and_acomp/SUBJECTS_TO_KEEP_WITH_POSS.txt', 'rt') as rr:
    for line in rr:
        pattern2_SUBJECTS_TO_KEEP_WITH_POSS.append(line.strip())
        
pattern2_SUBJECTS_TO_REMOVE = []
with open('Experiences_AEFI/pattern2_be_and_acomp/SUBJECTS_TO_REMOVE.txt', 'rt') as rr:
    for line in rr:
        pattern2_SUBJECTS_TO_REMOVE.append(line.strip())

pattern2_BODY_PARTS_TO_TAKE = []      
with open('Experiences_AEFI/pattern2_be_and_acomp/BODY_PARTS_TO_TAKE.txt', 'rt') as rr:
    for line in rr:
        pattern2_BODY_PARTS_TO_TAKE.append(line.strip())
        
# load subjects to keep Pattern3 (these are the same as pattern1)
pattern3_SUBJECTS_TO_KEEP = pattern1_SUBJECTS_TO_KEEP.copy()
pattern3_SUBJECTS_TO_KEEP_WITH_POSS = pattern1_SUBJECTS_TO_KEEP_WITH_POSS.copy()


def cathegory_of_acquaintance(subject):
    '''
    Assign to the subject given in input one of the cathegories 'author', 'author_child', 'others' or 'unclear'.
    It is done by comparing the lemma related to the subject with different list of nouns.
    
    Input:
    subject : dict, containing the information related to the matched subject
    
    Returns:
    subject_cathegory : str, one of 'author', 'author_child', 'others' or 'unclear'
    '''
    
    name = subject['subject_lemma']
    poss = subject['pos_subj_lower']
    amod = subject['amod_subj_lower']
    
    if name in ['son', 'daughter', 'kid', 'baby', 'child', 'guy', 'girl', 'boy', 'twin', 'kiddo',
               'lo', 'dd', 'ods', 'old', 'oldest', 'ds', 'yds', 'ydd', 'odd', 'mdd', 'yo', 'yr',
               'first', 'youngest'] and 'my' in poss:
        return 'author_child'

    elif name=='one' and 'little' in amod and 'my' in poss:
        return 'author_child'

    elif name == 'mine':
        return 'author_child'

    elif name in ['he', 'she']:
        return 'author_child'

    elif name in ['i', 'dh', 'husband']:
        return 'author'
    
    elif name in ['arm', 'leg'] and 'my' in poss:
        return 'author'
    
    elif name in ['arm', 'leg'] and ('his' in poss or 'her' in poss):
        return 'author_child'

    elif name in ['sister', 'brother', 'cousin', 'nephew', 'mom', 'friend', 'niece', 'mother', 'dad',
                 'father', 'uncle', 'aunt', 'grandma'] and 'my' in poss:
        return 'others'
    
    else:
        return 'unclear'
    

    
def process_reaction(effect):
    '''
    Assign to the adverse reaction given in input one of the cathegories of adverse reaction.
    It is done by comparing the lemma related to the advere reaction with different list of nouns.
    
    Input:
    effect : dict, containing the information related to the matched adverse reaction
    
    Returns:
    effect_cathegory : str
    
    The cathegories are:
    - 'allergic reaction'
    - 'regression'
    - 'skin reaction'
    - 'seizure'
    - 'reaction'
    - 'side effect'
    - 'fever'
    - 'pain'
    - 'lethargy'
    - 'lump'
    - 'fussy'
    - 'runny nose'
    - 'bump'
    '''
    
          
    eff, amods = effect['reaction_lemma'], effect['amod_react_lower']
    compou, poss, det = effect['compound_react_lower'], effect['pos_react_lower'], effect['det_react_lower']
    
    
    if eff in ['reaction', 'reactions', 'damage', 'damages', 'disorder', 'disorders', 'trouble',
              'troubles']:

        modifs = set(amods).union(set(compou))

        if modifs.intersection(set(['anaphylactic', 'allergic', 'allergy'])):
            kind_of_effect = 'allergic reaction'

        elif modifs.intersection(set(['neurological', 'brain', 'developemental', 'sensory'])):
            kind_of_effect = 'regression'

        elif modifs.intersection(set(['skin'])):
            kind_of_effect = 'skin reaction'

        elif modifs.intersection(set(['fever'])):
            kind_of_effect = 'skin reaction'

        elif modifs.intersection(set(['seizure'])):
            kind_of_effect = 'seizure'

        else:
            kind_of_effect = 'reaction'

    elif eff in ['effect', 'effects']:
        kind_of_effect = 'side effect'

    elif eff in ['fever', 'fevers', 'temp', 'temperature']:
        kind_of_effect = 'fever'

    elif eff in ['arm', 'sore', 'pain']:
        kind_of_effect = 'pain'

    elif eff in ['sleepy', 'lethargic', 'sleepier']:
        kind_of_effect = 'lethargy'

    elif eff in ['swollen', 'lump', 'bump', 'bumps']:
        kind_of_effect = 'lump'

    elif eff in ['cranky', 'fussy', 'grumpy', 'irritable', 'crabby', 'fussier', 'crankier', 
                  'grumpier']:
        kind_of_effect = 'fussy'

    elif eff in ['nose']:
        kind_of_effect = 'runny nose'


    elif eff in ['regression', 'regressions', 'change', 'changes']:
        kind_of_effect = 'regression'

    elif eff in ['lump', 'lumps']:
        kind_of_effect = 'bump'
    else:
        kind_of_effect = eff
                    
    return kind_of_effect


def classify_pattern(row):
    '''
    Given the structured representation of a match, obtain the classification respect to the experience with adverse event following immunization.
    Get also the kind of person mentioned and the kind of reaction.
    
    Input: 
    row : pd.Series, the structured representation of the match
    
    Returns:
    is_reaction : bool, True if a reaction is reported, False if absence of reaction reported
    subjects : dict, the subjects identified in the pattern
    kind_of_persons : str, kind of persons mentioned (separated by ", ")
    reactions : dict, the reactions identified in the pattern
    kind_of_reactions : str, kind of reactions mentioned (separated by ", ")
    '''
    
    subjects, reactions = row.subjects, row.reactions
    negations_subjs = any([True for subj_n, subj in subjects.items() if subj['negations_subj_lower']!=[]])
    negations_subjs_name = any([True for subj_n, subj in subjects.items() 
                                if subj['subject_lower'] in ['none', 'neither']])
    negations_subjs = negations_subjs or negations_subjs_name
    
    reactions_not_negated = [react for react_n, react in reactions.items() 
                                                     if react['negations_react_lower']==[]]
    
    verb_negation = row.verb_negation
    
    if verb_negation or negations_subjs or len(reactions_not_negated)==0: # it is reported a non reaction
        is_reaction = False
        return is_reaction, NAN, NAN, NAN, NAN
        
    else: # it is a reaction. Identify both person and reaction
        is_reaction = True
        
        kind_of_reactions = [process_reaction(react) for react in reactions_not_negated]
        reactions_not_negated_raw = [react['reaction_lemma'] for react in reactions_not_negated]
        kind_of_persons = [cathegory_of_acquaintance(subject) for n_subj, subject in subjects.items()]
        
        return is_reaction, subjects, ', '.join(kind_of_persons), reactions, ', '.join(kind_of_reactions)
    
    
def filter_pattern1(row):
    '''
    Apply the filter to matches retrieved with pattern1
    '''
    subjects = row.subjects
    subjects_new = {}
    after_vaccine_flag = row.after_vaccine_flag  
    someone_get_vaccine_flag = row.someone_get_vaccine_flag
    
    for subj_n, subject in subjects.items():
        
        subj_lower, amods_subj_lower = subject['subject_lower'], subject['amod_subj_lower']
        compounds_subj_lower, poss_subj_lower = subject['compound_subj_lower'], subject['pos_subj_lower']
        dets_subj_lower, negations_subj_lower = subject['det_subj_lower'], subject['negations_subj_lower']
        
        #print(subj_lower, poss_subj_lower)
        
        if subj_lower in pattern1_SUBJECTS_TO_KEEP and 'your' not in poss_subj_lower:
            subjects_new[subj_n] = subject
            
        elif subj_lower in pattern1_SUBJECTS_TO_KEEP_WITH_POSS and len(set(poss_subj_lower).intersection(set(['my', 'his', 
                                                                                  'her', 'our', 'their'])))>0:
            subjects_new[subj_n] = subject
            
        elif subj_lower in ['one', 'ones']:
            if 'little' in amods_subj_lower or 'no' in dets_subj_lower or len(set(poss_subj_lower).intersection(set(['my', 'his', 
                                                                                  'her', 'our', 'their'])))>0:
                subjects_new[subj_n] = subject
                
    reactions = row.reactions
    reactions_new = {}
    
    for react_n, reaction in reactions.items():
        
        react_lower, amods_react_lower = reaction['reaction_lower'], reaction['amod_react_lower']
        is_reaction_to_vaccine = reaction['is_reaction_to_vaccine_flag']
        
        if react_lower in objects_pattern1_with_vax:
            if after_vaccine_flag or someone_get_vaccine_flag or is_reaction_to_vaccine:
                if len(set(amods_react_lower).intersection(set(['immune', 'autoimmune', 'other', 'cervical', 'chronic'])))==0:
                    reactions_new[react_n] = reaction
        else: 
            if len(set(amods_react_lower).intersection(set(['immune', 'autoimmune', 'other', 'cervical', 'chronic'])))==0:
                reactions_new[react_n] = reaction
                
    if len(subjects_new)>0 and len(reactions_new)>0:
        
        row.subjects = subjects_new
        row.reactions = reactions_new
        
        is_reaction, subjects_raw, kind_of_persons, reactions_raw, kind_of_reactions = classify_pattern(row)
        return is_reaction, subjects_raw, kind_of_persons, reactions_raw, kind_of_reactions
        
    else:
        return 'Not_related', NAN, NAN, NAN, NAN
    
def filter_pattern2(row):
    '''
    Apply the filter to matches retrieved with pattern2
    '''
    subjects = row.subjects
    subjects_new = {}
    after_vaccine_flag = row.after_vaccine_flag  
    someone_get_vaccine_flag = row.someone_get_vaccine_flag
    
    for subj_n, subject in subjects.items():
        
        subj_lower, amods_subj_lower = subject['subject_lower'], subject['amod_subj_lower']
        compounds_subj_lower, poss_subj_lower = subject['compound_subj_lower'], subject['pos_subj_lower']
        dets_subj_lower, negations_subj_lower = subject['det_subj_lower'], subject['negations_subj_lower']
        
        if subj_lower in pattern2_SUBJECTS_TO_REMOVE:
            continue
        
        elif subj_lower in pattern2_SUBJECTS_TO_KEEP_WITH_POSS and len(set(poss_subj_lower).intersection(set(['my', 'his', 
                                                                                  'her', 'our', 'their'])))==0:
            continue
            
        elif subj_lower in ['one', 'ones']:
            if 'little' in amods_subj_lower or 'no' in dets_subj_lower or len(set(poss_subj_lower).intersection(set(['my', 'his', 
                                                                                  'her', 'our', 'their'])))>0:
                subjects_new[subj_n] = subject
                
            else:
                continue
            
        elif subj_lower in pattern2_BODY_PARTS_TO_TAKE:
            subjects_new[subj_n] = subject
            
        else:
            subjects_new[subj_n] = subject
            
         
    reactions = row.reactions
    reactions_new = {}
    
    for react_n, reaction in reactions.items():
        
        react_lower, amods_react_lower = reaction['reaction_lower'], reaction['amod_react_lower']
        is_reaction_to_vaccine = reaction['is_reaction_to_vaccine_flag']
        
        if react_lower in objects_pattern1_with_vax:
            if after_vaccine_flag or someone_get_vaccine_flag or is_reaction_to_vaccine:
                if len(set(amods_react_lower).intersection(set(['immune', 'autoimmune', 'other', 'cervical', 'chronic'])))==0:
                    reactions_new[react_n] = reaction
        else: 
            reactions_new[react_n] = reaction

        
    if len(subjects_new)>0 and len(reactions_new)>0:
        
        row.subjects = subjects_new
        row.reactions = reactions_new
        
        is_reaction, subjects_raw, kind_of_persons, reactions_raw, kind_of_reactions = classify_pattern(row)
        return is_reaction, subjects_raw, kind_of_persons, reactions_raw, kind_of_reactions
        
    else:
        return 'Not_related', NAN, NAN, NAN, NAN
    
    
def filter_pattern3(row):
    '''
    Apply the filter to matches retrieved with pattern3
    '''
    
    subjects = row.subjects
    subjects_new = {}
    
    for subj_n, subject in subjects.items():
        
        subj_lower, amods_subj_lower = subject['subject_lower'], subject['amod_subj_lower']
        compounds_subj_lower, poss_subj_lower = subject['compound_subj_lower'], subject['pos_subj_lower']
        dets_subj_lower, negations_subj_lower = subject['det_subj_lower'], subject['negations_subj_lower']
        
        if subj_lower in pattern1_SUBJECTS_TO_KEEP and 'your' not in poss_subj_lower:
            subjects_new[subj_n] = subject
            
        elif subj_lower in pattern1_SUBJECTS_TO_KEEP_WITH_POSS and len(set(poss_subj_lower).intersection(set(['my', 'his', 
                                                                                  'her', 'our', 'their'])))>0:
            subjects_new[subj_n] = subject
            
        elif subj_lower in ['one', 'ones']:
            if 'little' in amods_subj_lower or 'no' in dets_subj_lower or len(set(poss_subj_lower).intersection(set(['my', 'his', 
                                                                                  'her', 'our', 'their'])))>0:
                subjects_new[subj_n] = subject
                
    reactions_new = row.reactions
    
    if len(subjects_new)>0 and len(reactions_new)>0:
        
        row.subjects = subjects_new
        row.reactions = reactions_new
        
        is_reaction, subjects_raw, kind_of_persons, reactions_raw, kind_of_reactions = classify_pattern(row)
        return is_reaction, subjects_raw, kind_of_persons, reactions_raw, kind_of_reactions
        
    else:
        return 'Not_related', NAN, NAN, NAN, NAN


def Filter(row):
    '''
    Apply the filter to the structured representation of a match.
    
    Input:
    row : pd.Series, the structured representation of the match
    
    Returns:
    pd.Series : contains the classification and the kinds of reaction and subject identified. 
    
    Example:
    
    '''
    
    flag = False
    which_pattern = row.pattern_matched.split('_')[0]
    
    #c_id = row.c_id
    #other_sentences_comment_info = comment_info_non_matches_sentences.loc[c_id]
    #is_related_contextual_comment = other_sentences_comment_info.is_related_to_AEFI.any()
    
    if which_pattern=='pattern1':
        is_reaction, subjects_raw, kind_of_persons, reactions_raw, kind_of_reactions = filter_pattern1(row)
        
    elif which_pattern=='pattern2':
        is_reaction, subjects_raw, kind_of_persons, reactions_raw, kind_of_reactions = filter_pattern2(row)
        
    elif which_pattern=='pattern3':
        is_reaction, subjects_raw, kind_of_persons, reactions_raw, kind_of_reactions = filter_pattern3(row)
        
    return pd.Series({'is_reaction':is_reaction, 
                          'subjects_raw':subjects_raw, 'kind_of_persons':kind_of_persons,
                          'reactions_raw':reactions_raw, 'kind_of_reactions':kind_of_reactions,
                     #'is_related_contextual_comment':is_related_contextual_comment
                     })