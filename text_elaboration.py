'''
Code updated at 12/12/2019
Useful for babycenter data
'''

from numpy import *
import re

def load_contractions(file):
    '''
    Load a dictionary of expansion of contractions (ex. he'll: he will / he shall). This function will load this file as a dictionary where contractions are keys and the first expansion is the value.
    
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
    Load the acronyms of BabyCenter and return a dictionary of acronyms (e.g., "DD" : 'dear daughter')
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
    Given a text, expand all its contracted form. All the substitutions are in lower case.
    
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
    (e.g., "My LO (little one)") and this funcion take in consideration also that.
    
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


    
    
    
    
    
    