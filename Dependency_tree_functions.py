'''
21/09/2010

This file contains the functions used to i) construct a dependency tree from a sentence and ii) retrieve information from the tree.

Dependency trees are obtained through the SpaCy library and represented using the Networkx library:
- nodes are words, each word have a proper identifier (int), and node labels are information of the word like pos, tag, lemma, lower case
- links represents dependecies between words, and the label of the link correspond to the name of the dependency

Because of it was more useful for me, the directed links indicating syntactic relationships have opposite directions respect to the usual ones, so from all the nodes there always exists a path from the node to the root of the tree.

See "Dependency_tree_functions_debugging.ipynb" for examples


NOTES:
19/10/20  
changed function "find_objects" : now it takes the argument "depend" as a list in order to collect more than one dependency

21/10/20  
changed function "find_subject"-->"find_subjects" : now now it returns also more than one subject
'''

import networkx as nx
from numpy import argmin

def make_dep_tree(doc):
    '''
    Make a dependency tree with all the properties for each node using networkx. The name of nodes are the idx of the 
    tokens obtained with spacy nlp (this avoids confusion between words appearing more times).
    
    The graph is not a proper tree, because in this case the root is reachable from all nodes and not the opposite.
    The returned graph is directed.
    
    The name of each node is given by the idx of the token of doc and the attributes are:
    'text' : the corresponding string on the original text
    'lower' : the lower case of "text"
    'lemma' : the lemma of the word (lowered form)
    'pos' : part of speach
    'tag' : part of speach tag
    'shape' : 
    'is_alpha' : is alphanumeric
    'is_stop' : is stopword
    'position' : int, corresponds to the position of the word in the sentence, starting from 0
    
    Input:
    doc : spacy doc object, nlp(text) where text is a string
    
    Returns:
    G : nx.DiGraph object, a tree where nodes are tokens and links are dipendencies among nodes. 
    
    
    Example:
    
    >>> import Dependency_tree_functions as DepTree
    >>> import spacy
    >>> nlp = spacy.load("en_core_web_sm")
    
    >>> text = "We are on a delayed schedule and we chose not to vaccinate"
    >>> doc = nlp(text)
    >>> G = DepTree.make_dep_tree(doc)
    >>> G
    <networkx.classes.digraph.DiGraph at 0x7ff703bee990>
    '''
    # create graph object
    G = nx.DiGraph()
    
    # loop over token
    for n, token in enumerate(doc):
        
        # save node attribute
        G.add_node(token.idx, text=token.text, lower=token.text.lower(), lemma=token.lemma_.lower(), pos=token.pos_, 
                               tag=token.tag_, shape=token.shape_, is_alpha=token.is_alpha, 
                               is_stop=token.is_stop, position=n)
        # make edge with the head of the token
        G.add_edge(token.idx, token.head.idx, dep=token.dep_)
        
    return G


def get_nodes_by_attribute(G, att, att_name):
    '''
    Given a dependency tree, search the name of the nodes by the name of the attribute. Attention, it can return a
    list of nodes if they share the same attribute. For example, if two nodes of the tree have the same name.
    
    Input:
    G : nx.DiGraph, the dependency tree
    att : str, one among the attributes of a node ('lower', 'pos', 'tag', ..)
    att_name : str, the name of the attribute
    
    Returns:
    nodes : list of idxs (int), the idx of the nodes corresponding to the name of the attribute searched
    
    Example:
    
    >>> import Dependency_tree_functions as DepTree
    >>> import spacy
    >>> nlp = spacy.load("en_core_web_sm")
    
    >>> text = "We are on a delayed schedule and we chose not to vaccinate"
    >>> doc = nlp(text)
    >>> G = DepTree.make_dep_tree(doc)
    >>> idxs_choose = DepTree.get_nodes_by_attribute(G, 'lemma', 'choose')
    >>> idxs_choose, G.nodes()[idxs_choose[0]]['lower']
    ([36], 'chose')
    '''
    
    nodes = [k for k,token in G.nodes.data(att) if token==att_name]
    return nodes



def find_links_with_dependency(G, dep):
    '''
    Extract all the edges with attribute corresponding to a certain dependency.
    
    Input:
    G : nx.DiGraph, the dependency tree
    dep : str, one of the possible dependency
    
    Returns:
    links : list of tuples of int like (token, head), where dep is the attribute of the link (token, head)
    
    Example:
    
    >>> import Dependency_tree_functions as DepTree
    >>> import spacy
    >>> nlp = spacy.load("en_core_web_sm")
    
    >>> text = "We are on a delayed schedule and I chose not to vaccinate"
    >>> doc = npl(text)
    >>> G = DepTree.make_dep_tree(doc)
    >>> G.nodes(data='lower')
    NodeDataView({0: 'we', 3: 'are', 7: 'on', 10: 'a', 20: 'schedule', 12: 'delayed', 29: 'and', 33: 'we', 36: 'chose', 42: 'not', 49: 'vaccinate', 46: 'to'}, data='lower')
    >>> DepTree.find_links_with_dependency(G, 'nsubj')
    [(0, 3), (33, 36)]   # 0 and 33 corresponds to "we" and "I", and 3 and 36 to "are" and "choose"
    '''
    
    links = [(source, target) for source,target,attr in G.edges(data=True) if attr['dep']==dep]
    return links



def iterate_over_conj(G, idx_word):
    '''
    Given a word by its index on a Dependency Tree, find all words linked with dep='conj'. This function is useful to take for
    example two subjects: "My son and I live in LA". The nsubj of live is 'son'. By applying this function to 'son', it returns
    ['son', 'i']. In other words, the function retrieves words which have the same dependency on the sentence but that are
    linked by a conjunction.
    
    Input: 
    G : nx.DiGraph, the dependency tree
    idx_word : int, the index (name in the graph) of the word
    
    Returns:
    words : list of ints, list with the indices of the words linked by a conjunction
    
    Example:
    
    >>> import Dependency_tree_functions as DepTree
    >>> import spacy
    >>> nlp = spacy.load("en_core_web_sm")
    
    >>> text = "My son and I live in LA"
    >>> doc = nlp(text)    
    >>> G = DepTree.make_dep_tree(doc)
    >>> G.nodes(data='lower')
    [(0, 'my'), (3, 'son'), (13, 'live'), (7, 'and'), (11, 'i'), (18, 'in'), (21, 'la')]
    >>> idxs_dobj_link = DepTree.find_links_with_dependency(G, 'dobj') # suppose to find idxs of (son, love)
    >>> idx_dobj = idxs_dobj_link[0][0] # get the idx of the direct object (son)
    >>> DepTree.iterate_over_conj(G, idx_dobj)
    [3, 11] # it is the index corresponding to the word "son" and "I"
    '''
    
    words = [idx_word]
    while True:
        
        new_word_ = [sour for sour, tar, dep in G.in_edges(idx_word, data='dep') if dep=='conj']
        
        if len(new_word_)==1:
            words.append(new_word_[0])
            idx_word = new_word_[0]
        else:
            break
            
    return words



def get_shortest_path(G, source, target, as_undirected=True):
    '''
    Find the shortest path between two tokens given the dependency tree. By default it searches on an undirected 
    graph but undirected can be considered.
    
    Input:
    G : nx.DiGraph, the dependency tree
    source : idxs (int), the source node
    target : idxs (int), the target node
    as_undirected : bool, def True, whether to compute the path on a directed or indirected graph.
    
    Returns:
    path : list of idxs (int), the idx of the nodes defining the path, from source to target (both included)
    
    Example:
    
    >>> import Dependency_tree_functions as DepTree
    >>> import spacy
    >>> nlp = spacy.load("en_core_web_sm")
    
    >>> text = "My son is on a delayed schedule"
    >>> doc = nlp(text)    
    >>> G = DepTree.make_dep_tree(doc)
    >>> print(G.nodes(data='lower'))
    [(0, 'my'), (3, 'son'), (7, 'is'), (10, 'on'), (13, 'a'), (23, 'schedule'), (15, 'delayed')]
    >>> idxs_schedule = DepTree.get_nodes_by_attribute(G, 'lower', 'schedule')
    >>> idxs_be = DepTree.get_nodes_by_attribute(G, 'lemma', 'be')
    >>> DepTree.get_shortest_path(G, idxs_schedule[0], idxs_be[0], as_undirected=False)
    [23, 10, 7]
    >>> DepTree.get_shortest_path(G, idxs_be[0], idxs_schedule[0], as_undirected=False) # note when inverting source and target
    []
    >>> DepTree.get_shortest_path(G, idxs_be[0], idxs_schedule[0], as_undirected=True) # when considering the tree as undirected
    [7, 10, 23]
    '''
    
    G = G.to_undirected() if as_undirected else G
    
    try:
        path = [i for i in nx.shortest_path(G, source, target)]
    except:
        path = []
    
    return path


def find_closest_verb_of_word(G, idx_word):
    '''
    Given the idx of a word, find its closest verb in the dependency tree. In this case the shortest path has to be taken on the DIRECT graph. Starting from idx_word, walk on the tree searching for the closest verb, AUX or root. (Remember: following the direction of the links means to walk through the dependency tree in the opposite direction, going to the root!)
    
    TO DO:
    - check whether it is equal to search for "attr['tag'][0] == 'V' "
    
    Input: 
    G : nx.DiGraph, the dependency tree
    idx_word : int, the index (name in the graph) of the word you want to find the closest verb
    
    Returns:
    closest_verb : int, the index (name in the graph) of the verb (or root). It always returns something because it is possible to reach the root from every node
    
    Example:
    
    >>> import Dependency_tree_functions as DepTree
    >>> import spacy
    >>> nlp = spacy.load("en_core_web_sm")
    
    >>> text = "We are on a delayed schedule and we chose not to vaccinate"
    >>> doc = nlp(text)
    >>> G = DepTree.make_dep_tree(doc)
    >>> idxs_schedule = DepTree.get_nodes_by_attribute(G, 'lemma', 'schedule')
    >>> idx_closest_verb = DepTree.find_closest_verb_of_word(G, idxs_schedule[0])
    >>> G.nodes()[idx_closest_verb]['lower']
    'are'
    '''
    
    # find verbs or AUX or roots 
    verbs = [idx_node for idx_node, attr in G.nodes(data=True) if attr['pos']=='VERB' or attr['pos']=='AUX' or
             idx_node in dict(G[idx_node]).keys()] # this means that is a selfloop (so a Root)
    
    verbs = [verb for verb in verbs if 'prep' not in [dep for sour, tar, dep in G.out_edges(verb, data='dep')]] # why?
        
    
    # search the path NOT AS UNDIRECTED! you want to go above on the tree when searching for verbs
    word_verb_paths = [get_shortest_path(G, idx_word, verb, as_undirected=False) for verb in verbs]

    # take the verb given by the index of the closest path. If a path p as len(p)=0 (verb not conncected with 
    # idx_word) then let len(p)=100. 
    closest_verb = verbs[argmin([len(p) if len(p)>0 else 100 for p in word_verb_paths])]
    
    return closest_verb


def find_neighbors_with_dependencies(G, idx_word, dependencies, out_neig=False):
    
    '''
    Given the idx of a word, find the neighbor having that dependency.
    
    Input:
    G : nx.DiGraph, the dependency tree
    idx_verb : int, the index (name in the graph) of the word
    dependencies : list of str, the dependencies to inspect.
    out_neig : bool, whether to search among out neighbors
    
    Returns:
    word : list of int, the indexes (name in the graph) of the source of the edge having the dependency with idx_word
    if no matches an empty list is returned
    
    Example:
    
    >>> import Dependency_tree_functions as DepTree
    >>> import spacy
    >>> nlp = spacy.load("en_core_web_sm")
    
    >>> text = "We are on a delayed schedule and we chose not to vaccinate"
    >>> doc = nlp(text)
    >>> G = DepTree.make_dep_tree(doc)
    >>> G.nodes('lower')
    [(0, 'we'), (3, 'are'), (7, 'on'), (10, 'a'), (20, 'schedule'), (12, 'delayed'), (29, 'and'), (33, 'we'), (36, 'chose'), (42, 'not'), (49, 'vaccinate'), (46, 'to')]
    >>> idxs_be = DepTree.get_nodes_by_attribute(G, 'lemma', 'be')
    >>> DepTree.find_neighbors_with_dependencies(G, 3, ['nsubj'], out_neig=False)
    [0]  # the index of the work "we"
    '''
    
    if out_neig:
        word = [tar for sour, tar, dep in G.out_edges(idx_word, data='dep') if dep in dependencies]
    else:
        word = [sour for sour, tar, dep in G.in_edges(idx_word, data='dep') if dep in dependencies]
   
    return word

    
def get_verb_phrase(G, idx_verb):
    '''
    Given the dependency tree and the index of the verb, extract the verb phrase and estimate the tense of the verb.
    
    TO DO:
    - At the moment, modal verbs are not taken into account (like 'I would like to eat something' consider only 'to eat' and 'would like' separately).
    - Maybe the tense can be estimated better
    
    
    Input:
    G : nx.DiGraph, the dependency tree
    idx_verb : int, the index (name in the graph) of the word
    
    Returns: 
    verb_phrase : str, the verb phrase (with auxiliars for example)
    tense : str, the tense of the verb. ("Undetermined" when no rules matched for the tense detection)
    
    Example:
    
    >>> import Dependency_tree_functions as DepTree
    >>> import spacy
    >>> nlp = spacy.load("en_core_web_sm")
    
    >>> text = "We have been on a delayed schedule"
    >>> doc = nlp(text)
    >>> G = DepTree.make_dep_tree(doc)
    >>> idxs_be = DepTree.get_nodes_by_attribute(G, 'lemma', 'be')
    >>> DepTree.get_verb_phrase(G, idxs_be[0])
    ('have been', 'PresentPerfect')  
    '''
    
    tense, passive = 'Undetermined', False
    # search auxiliaries and passive auxiliaries
    auxs = [sour for sour, tar, dep in G.in_edges(idx_verb, data='dep') if dep=='aux']
    auxs_pass = [sour for sour, tar, dep in G.in_edges(idx_verb, data='dep') if dep=='auxpass']

    # if an aux passive were foung, notify it changing the value of the flag "passive"
    if len(auxs_pass)>0:
        auxs = auxs+auxs_pass
        passive = True

    # reconstruct the verb phrase preserving the order in the sentence (like "would have had"), and lower the characters
    verb_phrase = [(v, G.nodes()[v]['position']) for v in auxs+[idx_verb]]
    verb_phrase = [i for i,j in sorted(verb_phrase, key=lambda item: item[1])]
    verb_phrase_lower = [G.nodes()[v]['lower'] for v in verb_phrase]

    # Estimate the tense
    if passive:
        if G.nodes()[verb_phrase[0]]['tag'] in ['VBD']:
            tense = 'PastPassive'
        else:
            tense = 'PresentPassive'
        
    if len(verb_phrase) == 1:
        
        tag = G.nodes()[verb_phrase[0]]['tag']
        
        if tag=='VBP' or tag=='VBZ':
            tense = 'PresentSimple'
        elif tag == 'VB':
            tense = 'Infinite'
        elif tag == 'VBD':
            tense = 'PastSimple'
        elif tag == 'VBG':
            tense = 'Gerundive'
        elif tag == 'VBN':
            tense = 'PastParticipe'
            
    elif 'will' in verb_phrase_lower:
        tense = 'Future'
    
    elif 'VBG' in [G.nodes()[v]['tag'] for v in verb_phrase]:
        if len(verb_phrase) == 2 and G.nodes()[verb_phrase[0]]['lemma']=='be':
            if G.nodes()[verb_phrase[0]]['tag'] in ['VBP', 'VBZ']:
                tense = 'PresentContinuous'
            else:
                tense = 'PastContinuous'
        elif G.nodes()[verb_phrase[0]]['lemma'] == 'have':
            tense = 'PerfectContinuous'
            
    elif G.nodes()[verb_phrase[0]]['lemma'] == 'do':
        if G.nodes()[verb_phrase[0]]['tag'] in ['VBP', 'VBZ']:
            tense = 'PresentSimple'
        else:
            tense = 'PastSimple'
                
    elif G.nodes()[verb_phrase[0]]['lemma'] == 'have':
        if G.nodes()[verb_phrase[0]]['tag'] in ['VBP', 'VBZ']:
            tense = 'PresentPerfect'
        else:
            tense = 'PastPerfect'

    elif 'to' in verb_phrase_lower:
        tense = 'Infinite'
        
    elif 'should' in verb_phrase_lower or 'could' in verb_phrase_lower or 'might' in verb_phrase_lower \
                                    or 'would' in verb_phrase_lower or 'may' in verb_phrase_lower or 'can' in verb_phrase_lower:
        tense = 'Conditional'
         
    return ' '.join(verb_phrase_lower), tense


def find_modifiers(G, idx_word):
    '''
    Given a word, search the source of the links with label "amod", "compound", "pos" and "det".
    
    Input:
    G : nx.DiGraph, the dependency tree
    idx_word : int, the index (name in the graph) of the word
    
    Returns:
    result : four list. Each list is a list of indices
    
    Example:
    
    >>> import Dependency_tree_functions as DepTree
    >>> import spacy
    >>> nlp = spacy.load("en_core_web_sm")
    
    >>> text = "I follow a delayed vaccination schedule"
    >>> doc = nlp(text)
    >>> G = DepTree.make_dep_tree(doc)
    >>> print(G.nodes(data='lower'))
    [(0, 'i'), (2, 'follow'), (9, 'a'), (31, 'schedule'), (11, 'delayed'), (19, 'vaccination')]
    >>> idxs_schedule = DepTree.get_nodes_by_attribute(G, 'lemma', 'schedule')
    >>> DepTree.find_modifiers(G, idxs_schedule[0])
    ([11], [19], [], [9])
    '''
    
    flatten_list_and_unique = lambda l: list(set([item for sublist in l for item in sublist]))
    
    modifs = find_neighbors_with_dependencies(G, idx_word, ['amod'])
    #modif.extend([sour for mod in modif for sour, tar, dep in G.in_edges(mod, data='dep') if dep=='conj'])
    modifs = flatten_list_and_unique([iterate_over_conj(G, modif) for modif in modifs])
    
    compounds = find_neighbors_with_dependencies(G, idx_word, ['compound'])
    compounds = flatten_list_and_unique([iterate_over_conj(G, compound) for compound in compounds])
    
    posss = find_neighbors_with_dependencies(G, idx_word, ['poss'])
    posss = flatten_list_and_unique([iterate_over_conj(G, poss) for poss in posss])

    det = find_neighbors_with_dependencies(G, idx_word, ['det'])

    return modifs, compounds, posss, det

def iterate_over_conj(G, idx_word):
    '''
    Given a word by its index on a Dependency Tree, find all words linked with dep='conj'. This function is useful to take for
    example two subjects: "My son and I live in LA". The nsubj of live is 'son'. By applying this function to 'son', it returns
    ['son', 'i']. In other words, the function retrieves words which have the same dependency on the sentence but that are
    linked by a conjunction.
    
    Input: 
    G : nx.DiGraph, the dependency tree
    idx_word : int, the index (name in the graph) of the word
    
    Returns:
    words : list of ints, list with the indices of the words linked by a conjunction
    
    Example:
    
    >>> import Dependency_tree_functions as DepTree
    >>> import spacy
    >>> nlp = spacy.load("en_core_web_sm")
    
    >>> text = text = "I follow a delayed and spaced vaccination schedule"
    >>> doc = nlp(text)    
    >>> G = DepTree.make_dep_tree(doc)
    >>> print(G.nodes(data='lower'))
    [(0, 'i'), (2, 'follow'), (9, 'a'), (42, 'schedule'), (11, 'delayed'), (19, 'and'), (23, 'spaced'), (30, 'vaccination')]
    >>> idxs_schedule = DepTree.get_nodes_by_attribute(G, 'lemma', 'schedule')
    >>> amod, compound, poss, det = DepTree.find_modifiers(G, idxs_schedule[0])
    >>> amod[0], DepTree.iterate_over_conj(G, amod[0])
    (11, [11, 23])
    '''
    
    words = [idx_word]
    while True:
        
        new_word_ = [sour for sour, tar, dep in G.in_edges(idx_word, data='dep') if dep=='conj']
        
        if len(new_word_)==1:
            words.append(new_word_[0])
            idx_word = new_word_[0]
        else:
            break
            
    return words



def find_objects(G, idx_word, depend=['dobj'], out_neig=False):
    '''
    Given a word, search the source of the links with label "dobj", including conjunctions.
    
    Input:
    G : nx.DiGraph, the dependency tree
    idx_word : int, the index (name in the graph) of the word
    depend : list of str, kind of dependencies to search for
    out_neig : bool, whether to search among out neighbors
    
    TO DO:
    - if more dependencies and searching for conj too, these are mixed together. Maybe ok
    
    Returns:
    objs : list of ints, idxs of the objects
    
    Example:
    
    >>> import Dependency_tree_functions as DepTree
    >>> import spacy
    >>> nlp = spacy.load("en_core_web_sm")
    
    >>> text = "I love my son and my daughter"
    >>> doc = nlp(text)
    >>> G = DepTree.make_dep_tree(doc)
    >>> print(G.nodes(data='lower'))
    [(0, 'i'), (2, 'love'), (7, 'my'), (10, 'son'), (14, 'and'), (18, 'my'), (21, 'daughter')]
    >>> idxs_love = DepTree.get_nodes_by_attribute(G, 'lemma', 'love')
    >>> DepTree.find_objects(G, idxs_love[0], ['dobj])
    [10, 21] # respectively, the index of "son" and "daughter"  
    
    '''
    
    objs = find_neighbors_with_dependencies(G, idx_word, depend, out_neig=out_neig)
    
    if objs!=[]:
        objs_ = []
        
        for obj in objs:
            objs_.extend(iterate_over_conj(G, obj))
    else:
        objs_ = []
            
        
    return objs_



def find_subjects(G, idx_verb, passive=False):
    '''
    Find the subject of a verb. If the subject is not explicit, search among 'conj'
    
    Input:
    G : nx.DiGraph, the dependency tree
    idx_verb : int, the index (name in the graph) of the verb
    passive : bool, def False. If True search 'nsubjpass' without searching the principal clause
    
    Return:
    result : int, the index of the subject
    
    Example:
    
    >>> import Dependency_tree_functions as DepTree
    >>> import spacy
    >>> nlp = spacy.load("en_core_web_sm")
    
    >>> text = "I love my son and my daughter"
    >>> doc = nlp(text)
    >>> G = DepTree.make_dep_tree(doc)
    >>> print(G.nodes(data='lower'))
    [(0, 'i'), (2, 'love'), (7, 'my'), (10, 'son'), (14, 'and'), (18, 'my'), (21, 'daughter')]
    >>> idxs_love = DepTree.get_nodes_by_attribute(G, 'lemma', 'love')
    >>> DepTree.find_subject(G, idxs_love[0], passive=False)
    [0]
    
    # in this case, the verb "following" is not directly linked to its subject "I"
    >>> text = "I am doing and following almost all of the shots."
    >>> doc = nlp(text)
    >>> G = DepTree.make_dep_tree(doc)
    >>> print(G.nodes(data='lower'))
    [(0, 'i'), (5, 'doing'), (2, 'am'), (11, 'and'), (15, 'following'), (25, 'almost'), (32, 'all'), (36, 'of'), (39, 'the'), (43, 'shots'), (48, '.')]
    >>> idxs_love = DepTree.get_nodes_by_attribute(G, 'lemma', 'follow')
    >>> DepTree.find_subject(G, idxs_love[0], passive=False)
    [0]
    
    '''
    # flag indicating whether to search for active or passive
    dep = 'nsubj' if not passive else 'nsubjpass'
    dep_ = 'nsubjpass' if not passive else 'nsubj'
    
    # search direct subject of the verb
    subject = find_neighbors_with_dependencies(G, idx_verb, [dep])

    if len(subject)>0: 
        # take the last subject
        #subject = [subject[-1]]
        None
      
    # if there is the passive (active) don't search with and!
    elif len(find_neighbors_with_dependencies(G, idx_verb, [dep_]))>0:
        subject = []
        
    # search the subject of the principal clause connected with a 'conj'
    else:
        # check if there is 'conj'
        # 'i am doing and following almost all of the shots, but on a somewhat delayed and spaced out schedule.'       
        
        verb_conj = find_neighbors_with_dependencies(G, idx_verb, ['conj'], out_neig=True)
        if len(verb_conj)>0:
            subject = find_neighbors_with_dependencies(G, verb_conj, [dep])
            if len(subject)>0:
                #subject = [subject[-1]]
                None
            else:
                subject = []     
        else:
            subject = []

    return subject


def is_equal_dep_tree(G1, G2):
    '''
    Check if the two dependency trees are equal.
    
    Input:
    G1 : nx.DiGraph, the dependency tree
    G2 : nx.DiGraph, the dependency tree
    
    Returns:
    are_equal : bool, True if the two graphs are equal.
    '''
    
    df_G1 = nx.to_pandas_edgelist(G1)
    df_G2 = nx.to_pandas_edgelist(G2)
    
    df_G1.loc[: ,'source_lower'] = df_G1.source.apply(lambda s: G1.nodes()[s]['lower'])
    df_G1.loc[: ,'target_lower'] = df_G1.target.apply(lambda t: G1.nodes()[t]['lower'])
    df_G1 = df_G1.sort_values('dep').reset_index()
    
    df_G2.loc[: ,'source_lower'] = df_G2.source.apply(lambda s: G2.nodes()[s]['lower'])
    df_G2.loc[: ,'target_lower'] = df_G2.target.apply(lambda t: G2.nodes()[t]['lower'])
    df_G2 = df_G2.sort_values('dep').reset_index()
    
    are_equal = all(df_G1[['dep', 'source_lower', 'target_lower']] == df_G2[['dep', 'source_lower', 'target_lower']])
    
    return are_equal



