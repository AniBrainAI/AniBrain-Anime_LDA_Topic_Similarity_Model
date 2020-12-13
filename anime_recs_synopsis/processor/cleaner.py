import spacy
from spacy.lang.en import English
import string
import unidecode

def __remove_people(text):
    """
    Removes character names from synopsis (identified through Named Entity Recognition).
    
    Parameters
    ----------
    text: string
        The synopsis to filter out the character names.
    """
    
    text_nlp = nlp(text)
    
    for ent in text_nlp.ents:
        if(ent.label_ == 'PERSON'):
            text = text.replace(ent.text, '')
            
    return text

def __remove_non_alphabetical_characters(text):
    """
    Cleans text, keeping only alphabetical characters
        
    Parameters
    ----------
    token: str
        Word being processed

    Returns
    -------
    str:
        Cleaned string
    """
    
    text = str(text).translate(str.maketrans(string.punctuation, ' '*len(string.punctuation)))
    
    split_text = [
        ''.join([letter for letter in word if letter.isalpha()])
        for word in text.split()
    ]
    
    cleaned_text = ' '.join(split_text)
    
    return cleaned_text

def prepare_text(text):
    """
    Lemmatizes text using spaCy
        
    Parameters
    ----------
    text: str
        Text being processed

    Returns
    -------
    str:
        Lemmatized text
    """
    
    if text == '':
        return text
    
    # Remove people
    cleaned_text = __remove_people(text)
    
    # Clean  
    cleaned_text = __remove_non_alphabetical_characters(cleaned_text)
    
    # Tokenize
    tokens = nlp(cleaned_text.lower())
    
    custom_stop_list = [
    'episode',
    'series',
    'short',
    'anime',
    'story',
    'film',
    'feature',
    'character',
    'special',
    'movie',
    'animation',
    'original',
    'release',
    'manga',
    'ova',
    'scene',
    'animate',
    'show',
    'volume',
    'adaptation',
    'video',
    'manga',
    'crossover',
    'season'
    ]
    
    # Lemmatize
    lemmatized_text = ' '.join([token.lemma_ for token in tokens if not token.is_stop and not token.is_punct and (token.pos_ == 'NOUN' or token.pos_ == 'ADJ' or token.pos_ == 'VERB' or token.pos_ == 'ADV') and token.lemma_ not in custom_stop_list])
    
    # Remove short words (<= 2)
    removed_short_words = [x for x in lemmatized_text.split() if len(x) > 2]
    
    full = ' '.join(removed_short_words)
    
    return unidecode.unidecode(full)