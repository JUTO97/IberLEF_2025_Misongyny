import spacy
import re
import sqlite3
import datetime
import asyncio
import deepl
import pandas as pd
import os
import html
import random

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from googletrans import Translator
from google.cloud import translate_v2 as gtranslate  # Oficial Google Translate API

HOMOGLYPHS = {
    'е': 'e', 'Е': 'E',  # Cyrillic e
    'а': 'a', 'А': 'A',
    'о': 'o', 'О': 'O',
    'с': 'c', 'С': 'C',
    'р': 'p', 'Р': 'P',
    'і': 'i', 'І': 'I',
    'ё': 'e', 'Ё': 'E',
    'ђ': 'd', 'Ћ': 'D',
    'њ': 'n', 'Њ': 'N',
    'μ': 'u',
    'π': 'p',
}

PUNCTUATIONS = [".", "?", "!", ",","'"]

# set  environment FOR OFFICIAL GOOGLE TRANSLATE API
# os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "your_google_credentials"


auth_key = "your_deepl_auth_key"  


deepl_client = deepl.DeepLClient(auth_key)
SEED = 42
random.seed(SEED)


def load_spacy_model(lang_code):

    spacy_models = {
    'eng': None,
    'spa': None
    }

    if lang_code == 'eng' and spacy_models['eng'] is None:
        spacy_models['eng'] = spacy.load('en_core_web_trf')
    elif lang_code == 'spa' and spacy_models['spa'] is None:
        spacy_models['spa'] = spacy.load('es_dep_news_trf')
    return spacy_models[lang_code]


def normalize_text(text, log=False):
    """
    Normalize text by replacing special characters.
    """
    replaced_chars = {}  # for logging what was replaced

    # Replace homoglyphs
    for char, replacement in HOMOGLYPHS.items():
        if char in text:
            count = text.count(char)
            replaced_chars[char] = {'replacement': replacement, 'count': count}
            text = text.replace(char, replacement)
    
    # logreplacement info
    if replaced_chars and log:
        print("Homoglyph replacements detected:")
        for char, info in replaced_chars.items():
            print(f"   '{char}' → '{info['replacement']}' (replaced {info['count']} times)")

    return text


def remove_bracketed(text, partial_remove=False):
    """
    Remove content inside square brackets.

    If partial_remove=True:
      Keep bracketed expressions only if they contain `"` or `:`
      Remove any stopword (sw) from inside the brackets, no matter the position.
    """
    sw = ['coro', 'chorusx', 'chorus', 'prechorus', 'precoro', 'verso', 'verse', 'bridge',
          'estribillo', 'postestribillo', 'puente' 'instrumental' , 'intro', 'outro',
          'interludio', 'interlude', 'instrumental', 'solo', 'break', 'breakdown', 'refrain', 'refrán', 'postcoro',
          'coda', 'codetta', 'prelude', 'prologue', 'epilogue', 'prologo', 'epilogo']

    def clean_bracket(match):
        inside = match.group(0)[1:-1].replace('-', '')  # Remove the brackets
        for word in sw:
            pattern = r'\b' + re.escape(word) + r'\b'
            inside = re.sub(pattern, '', inside, flags=re.IGNORECASE)
        inside = re.sub(r'\s+', ' ', inside).strip()  # Clean extra spaces
        return '['+inside+']'

    if partial_remove:
        def conditional_keep(match):
            inside = match.group(0)[1:-1]
            if '"' in inside or ':' in inside:
                return clean_bracket(match)
            return ''
        return re.sub(r'\[.*?\]', conditional_keep, text)
    else:
        return re.sub(r'\[.*?\]', '', text)


def remove_duplicate_sentences(text):
    """
    Removes duplicate lines (sentences) from a block of text where each sentence is separated by '\n'.
    Preserves the original order.
    """
    seen = set()
    unique_lines = []
    sentences = text.split('\n')
    original_length = len(sentences)
    for line in sentences:
        line = line.strip()
        if line and line not in seen:
            seen.add(line)
            unique_lines.append(line)

    num_duplicates = original_length - len(unique_lines)

    return ('\n'.join(unique_lines), num_duplicates)

def remove_special_characters(text, lang='spa'):
    """
    Remove special characters from text, keeping Spanish letters, digits, basic punctuation, and spaces. FOR DL models.
    """
    if lang == 'spa':
        return re.sub(r'[^a-zA-ZáéíóúÁÉÍÓÚñÑüÜ.,;¡!¿?&()\[\]"\'\- \n]', "", text)
    else:
        return re.sub(r'[^a-zA-Z.,;!?&()\[\]"\'\- \n]', "", text)

def basic_cleaning(text, partial_remove_brackets=True):
    text = normalize_text(text.strip())
    text = remove_bracketed(text, partial_remove=partial_remove_brackets)
    return remove_special_characters(text)


def save_eval_results(db_path, db_file_name, task_val, model_name, eval_type,  acc, pre, rec, f1_, study_name, params, lang):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    date = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    insert_query = """
    INSERT INTO eval (model, database, task, date, accuracy, precision, recall, f1, type, study, params, lang)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
    """

    cursor.execute(
        insert_query,
        (model_name, db_file_name, task_val, date, acc, pre, rec, f1_, eval_type, study_name, params, lang)
    )

    conn.commit()
    conn.close()

    print("\nInserted CV results into 'eval' table in tfm.db.")


def split_into_chunks(sentences, n_chunks=2):
    k, m = divmod(len(sentences), n_chunks)
    chunks = []
    start = 0
    for i in range(n_chunks):
        end = start + k + (1 if i < m else 0)
        chunks.append(sentences[start:end])
        start = end
    return chunks

# Google Translate (async wrapper)
async def async_translate_google(text, source_lang='es', target_lang='en'):
    translator = Translator()
    result = await translator.translate(text, src=source_lang, dest=target_lang)
    return result.text

def sync_google_translate(text, source_lang='es', target_lang='en'):
    try:
        return asyncio.run(async_translate_google(text, source_lang, target_lang))
    except Exception as e:
        print(f"Google Translate failed:\n{text[:200]}...\nError: {e}")
        return ""

def translate(text, target_lang="es", source_lang='en', translator="deepl"):
    """
    Translates a text from source_lang to target_lang using DeepL or Google Translate (unofficial API) and Google Cloud Translate (official API).
    """
    text = basic_cleaning(text, partial_remove_brackets=True)
    sentences = [s.strip() for s in text.split('\n') if len(s.strip()) > 1]
    chunks = split_into_chunks(sentences, n_chunks=4)

    try:
        if translator == "deepl":
            results = [
                deepl_client.translate_text('. '.join(chunk), target_lang=target_lang, source_lang=source_lang)
                for chunk in chunks
            ]
            text_translated = '. '.join([r.text for r in results])

        elif translator == "google":
            text_translated = sync_google_translate(text, source_lang, target_lang)

        elif translator == "google_official":
            # Initialize the Google Cloud Translation client
            translate_client = gtranslate.Client()

            joined_text = '. '.join(sentences)
            result = translate_client.translate(
                joined_text,
                source_language=source_lang,
                target_language=target_lang
            )
            text_translated = html.unescape(result['translatedText'])
            print(f"Translated {len(joined_text)} characters to\n\n {len(text_translated)} characters.")
            print(f"Original: {joined_text}...\n\nTranslated: {text_translated}...")

        else:
            raise ValueError(f"Unknown translator option: {translator}")


        return text_translated.replace('. ', '\n')

    except Exception as e:
        print(f"⚠️ {translator} failed, falling back to Google Translate...\n{text[:200]}...\nError: {e}")
        return sync_google_translate(text, source_lang, target_lang).replace('. ', '\n')
    

def augment_lyrics(lyrics, aug):
    # Remove bracketed info and split into non-empty lines
    text = remove_special_characters(remove_bracketed(lyrics.strip()))
    sentences = [s.strip() for s in text.split('\n') if len(s.strip()) > 1]
    chunks = split_into_chunks(sentences, n_chunks=4)
    
    augmented_sentences = []

    for chunk in chunks:
        try:
            augmented = aug.augment('. '.join(chunk), n=1)  # Ensures a single result
            if isinstance(augmented, list):
                augmented = augmented[0]
        except Exception as e:
            print('ERROR!')
            augmented = 'ERROR_DATAAUG_ContextualWordEmbsAug'
        augmented_sentences.append(augmented)

    augmented_chunks = ". ".join(augmented_sentences)
    
    return augmented_chunks.replace('. ', '\n')


def insert_punctuation_marks(sentence, punc_ratio=0.3):
    sentence = basic_cleaning(sentence, partial_remove_brackets = True) 
    # Split the sentence into words
    words = sentence.split(' ')
    # Determine how many punctuations to insert (at least one)
    num_punc = random.randint(1, max(1, int(punc_ratio * len(words))))
    # Randomly choose positions in the list to insert punctuation
    positions = random.sample(range(len(words)), num_punc)
    
    new_words = []
    for idx, word in enumerate(words):
        # If the current index is selected, insert a random punctuation before the word
        # if idx in positions:
        #     new_words.append(random.choice(PUNCTUATIONS))
        # new_words.append(word)
        if idx in positions:
            word = word + random.choice(PUNCTUATIONS)
        new_words.append(word)
    
    # Reconstruct the sentence with inserted punctuation
    augmented_sentence = ' '.join(new_words)
    # print(f"Original: {sentence}\nAugmented: {augmented_sentence}")
    return augmented_sentence


def text_preprocess(text, lang='eng', model_type='ml', stem=False, lemmatize=False, remove_duplicates = False, cased = False, return_str=False):

    text = normalize_text(text.strip()) 

    if model_type == 'dl':
        text = remove_bracketed(text, partial_remove=True)
    else: # ml model
        text = remove_bracketed(text, partial_remove=False)
    
    if not cased:
        text = text.lower()

    if remove_duplicates:
        text = remove_duplicate_sentences(text)[0]

    # Light preprocessing for DL (transformers)
    if model_type == 'dl':
        text = remove_special_characters(text, lang=lang) 
        lyric = '\n'.join([s.strip() for s in text.split('\n') if len(s.strip()) > 1]) 
        return lyric

    # For ML preprocessing (traditional models):
    # Keep accents, remove digits/punctuations (keep letters and accented characters)
    clean_text = re.sub(r'[^a-zA-ZáéíóúüñÁÉÍÓÚÜÑ\s]', '', text)

    # Tokenization
    tokens = word_tokenize(clean_text, language='spanish' if lang == 'spa' else 'english')

    # Stopword removal
    base_stopwords = set(stopwords.words('spanish' if lang == 'spa' else 'english'))
    extra_sw = {'coro', 'chorusx', 'chorus',  'prechorus','verso', 'verse', 'bridge', 'puente', 'estribillo', 'postestribillo', 'puente instrumental', 
                'intro', 'outro', 'interludio', 'interlude', 'instrumental', 'solo', 'break', 'breakdown', 'refrain', 'coda', 'codetta', 'prelude', 
                'prologue', 'epilogue', 'prologo', 'epilogo'}
    stop_words = base_stopwords | extra_sw  # Union of both sets
    
    # Filter out stopwords and words with <= 2 characters
    tokens = [word for word in tokens if word not in stop_words and len(word) > 2]

    # Optional lemmatization
    if lemmatize:
        nlp = load_spacy_model(lang)
        doc = nlp(" ".join(tokens))
        tokens = [token.lemma_ for token in doc if token.lemma_ not in stop_words]

    # Optional stemming (applies after lemmatization if both are True)
    if stem:
        stemmer = SnowballStemmer('spanish' if lang == 'spa' else 'english')
        tokens = [stemmer.stem(word) for word in tokens]

    return ' '.join(tokens) if return_str else tokens


