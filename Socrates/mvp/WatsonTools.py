from ibm_watson import NaturalLanguageUnderstandingV1
from ibm_watson.natural_language_understanding_v1 import *
import json
from ibm_watson import ToneAnalyzerV3
from ibm_cloud_sdk_core.authenticators import IAMAuthenticator
from ibm_watson import ApiException
from ibm_watson import *

with open("config.json", 'r') as f:
        config = json.load(f)

keyedAuthenticator = IAMAuthenticator(config["tone_analyzer"]["apikey"])
version = config["tone_analyzer"]["version"]

## Create Tone Analyzer instance with API key
tone_analyzer = ToneAnalyzerV3(
    version= f'{version}',
    authenticator = keyedAuthenticator
)

keyedAuthenticator = IAMAuthenticator(config["natural_language_understanding"]["apikey"])

nlu = NaturalLanguageUnderstandingV1(
    version=version,
    authenticator= keyedAuthenticator
)


##############################################################################



## Takes in some string text and
## make call to IBM Watson Tone Analyzer with the inputted text
## Returns a tuple of 3 outputs directly from the api call
## 1. document_tone and sentence_tone
##    - document tone is a json string that includes the tones that have over .5/1 tone_scores
##    - sentence_tone is a json string that tone analyzes each sentence from the `inputtedTxt`
## 2. headers is metadata
##    - most important is timestamp of api call(can be linked to when question is asked)
## 3. Status code over HTTP
def analyze_tone(inputtedTxt):
    try:
        # Invoke a Tone Analyzer method
        #         inputtedTxt = "Team, I know that times are tough! "

        if inputtedTxt in analyze_tone.memo:
            return analyze_tone.memo[inputtedTxt]

        j = tone_analyzer.tone({'text': inputtedTxt}, content_type='application/json')

        result = j.get_result(), j.get_headers(), j.get_status_code()

        analyze_tone.memo[inputtedTxt] = result

        return result

    except ApiException as ex:
        print("Method failed with status code " + str(ex.code) + ": " + ex.message)
        raise ex


analyze_tone.memo = {}






def to_sentiment_dict(ret_res):
    ## Default sentiment vector of all possible emotions returned from IBM Watson Tone Analyzer
    ## THIS IS TEMPLATE
    sentiment_vector_zeroed = {"anger": 0.0, "fear": 0.0, "joy": 0.0, "sadness": 0.0, "analytical": 0.0,
                               "confident": 0.0, "sentative": 0.0}

    # ret_res['document_tone']['tones']
    ## Create (tone, score) pairs from ret_res
    sents = [(tone['tone_id'], tone['score']) for tone in ret_res['document_tone']['tones']]

    ## Create copy of sentiment_vector_zeroed to have zeroed values
    # for each of the sentiments found from Tone Analyzer, set the corresponding value into our sentiment vector
    sents_vector = sentiment_vector_zeroed
    for tone in sents:
        tone_id = tone[0]
        tone_score = float(tone[1])
        sents_vector[tone_id] = tone_score

    return sents_vector


####################################################################3

def get_categories(text_str, limit=3):
    key = (text_str, limit)
    if key in get_categories.memo:
        return get_categories.memo[key]

    response = nlu.analyze(
        text=text_str,
        features=Features(categories=CategoriesOptions(limit=limit))
    )

    result = response.get_result(), response.get_headers(), response.get_status_code()
    get_categories.memo[(text_str, limit)] = result

get_categories.memo = {}


########################################################################

def get_concepts(text_str, limit=3):
    key = (text_str, limit)
    if key in get_concepts.memo:
        return get_concepts.memo[key]

    response = nlu.analyze(
        text=text_str,
        features=Features(concepts=ConceptsOptions(limit=limit))
    )

    result = response.get_result(), response.get_headers(), response.get_status_code()
    get_concepts.memo[(text_str, limit)] = result

    return result


get_concepts.memo = {}


########################################################

def get_targeted_emotion(text_str, target_words_and_phrase_list):
    key = (text_str, target_words_and_phrase_list)
    if key in get_targeted_emotion.memo:
        return get_targeted_emotion.memo[key]

    response = nlu.analyze(
        text=text_str,
        features=Features(emotion=EmotionOptions(targets=target_words_and_phrase_list))
    )

    result = response.get_result(), response.get_headers(), response.get_status_code()

    get_targeted_emotion.memo[(text_str, target_words_and_phrase_list)] = result

    return result


get_targeted_emotion.memo = {}


################################################################################


def get_entity_info(text_str, limit):
    key = (text_str, limit)
    if key in get_entity_info.memo:
        return get_entity_info.memo[key]

    response = nlu.analyze(
        text=text_str,
        features=Features(entities=EntitiesOptions(sentiment=True, mentions=True, emotion=True, limit=limit))
    )

    result = response.get_result(), response.get_headers(), response.get_status_code()
    get_entity_info.memo[(text_str, limit)] = result

    return result


get_entity_info.memo = {}


####################################################################


def get_keyword_info(text_str, limit):
    key = (text_str, limit)
    if key in get_keyword_info.memo:
        return get_keyword_info.memo[key]


    response = nlu.analyze(
        text=text_str,
        features=Features(keywords=KeywordsOptions(sentiment=True,
                                                   emotion=True,
                                                   limit=limit
                                                   )
                          )
    )

    result = response.get_result(), response.get_headers(), response.get_status_code()
    get_keyword_info.memo[(text_str, limit)] = result

    return result


get_keyword_info.memo = {}


####################################################

def get_relational_info(text_str):
    key = text_str
    if key in get_relational_info.memo:
        return get_relational_info.memo[key]

    response = nlu.analyze(
        text=text_str,
        features=Features(relations=RelationsOptions())
    )

    result =  response.get_result(), response.get_headers(), response.get_status_code()

    get_relational_info.memo[text_str] = result

    return result


get_relational_info.memo = {}



#################################################################


def get_semantic_roles(text_str, limit):
    key = (text_str, limit)
    if key in get_semantic_roles.memo:
        return get_semantic_roles.memo[key]

    response = nlu.analyze(
        text=text_str,
        features=Features(semantic_roles=SemanticRolesOptions(keywords=True, entities=True, limit=limit))
    )

    result = response.get_result(), response.get_headers(), response.get_status_code()

    get_semantic_roles.memo[(text_str, limit)] = result

    return result


get_semantic_roles.memo = {}


##################################################################


def get_targeted_sentiment(text_str, target_words_phrases_list):
    key = (text_str, target_words_phrases_list)
    if key in get_targeted_sentiment.memo:
        return get_targeted_sentiment.memo[key]

    response = nlu.analyze(
        text=text_str,
        features=Features(sentiment=SentimentOptions(document=True, targets=target_words_phrases_list))
    )

    result = response.get_result(), response.get_headers(), response.get_status_code()

    get_targeted_sentiment.memo[(text_str, target_words_phrases_list)] = result
    return result


get_targeted_sentiment.memo = {}


#######################################################################


def get_syntax_info(text_str):
    key = text_str
    if key in get_syntax_info.memo:
        return get_syntax_info.memo[key]

    response = nlu.analyze(
        text=text_str,
        features=Features(syntax=SyntaxOptions(
            sentences=True,
            tokens=SyntaxOptionsTokens(lemma=True,
                                       part_of_speech=True)))
    )

    result = response.get_result(), response.get_headers(), response.get_status_code()

    get_syntax_infot.memo[text_str] = result
    return result


get_syntax_info.memo = {}
