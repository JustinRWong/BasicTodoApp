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
        j = tone_analyzer.tone({'text': inputtedTxt}, content_type='application/json')

        result = j.get_result(), j.get_headers(), j.get_status_code()

        self.memo[inputtedTxt] = result

        return result

    except ApiException as ex:
        print("Method failed with status code " + str(ex.code) + ": " + ex.message)
        raise ex


analyze_tone.memo = {}






def to_sentiment_dict(ret_res):
    ## Default sentiment vector of all possible emotions returned from IBM Watson Tone Analyzer
    ## THIS IS TEMPLATE
    sentiment_vector_zeroed = {"anger": 0.0, "fear": 0.0, "joy": 0.0, "sadness": 0.0, "snalytical": 0.0,
                               "sonfident": 0.0, "sentative": 0.0}

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


to_sentiment_dict.memo = {}





def get_categories(text_str, limit=3):
    response = nlu.analyze(
        text=text_str,
        features=Features(categories=CategoriesOptions(limit=limit))
    )

    return response.get_result(), response.get_headers(), response.get_status_code()


get_categories.memo = {}







def get_concepts(text_str, limit=3):
    if (text_str, limit) in self.memo.keys():
        return self.memo[(text_str, limit)]

    response = nlu.analyze(
        text=text_str,
        features=Features(concepts=ConceptsOptions(limit=limit))
    )

    result = response.get_result(), response.get_headers(), response.get_status_code()
    self.memo[(text_str, limit)] = result

    return result


get_concepts.memo = {}






def get_targeted_emotion(text_str, target_words_and_phrase_list):
    response = nlu.analyze(
        text=text_str,
        features=Features(emotion=EmotionOptions(targets=target_words_and_phrase_list))
    )

    result = response.get_result(), response.get_headers(), response.get_status_code()

    self.memo[(text_str, target_words_and_phrase_list)] = result

    return result


get_targeted_emotion.memo = {}



def get_entity_info(text_str, limit):
    response = nlu.analyze(
        text=text_str,
        features=Features(entities=EntitiesOptions(sentiment=True, mentions=True, emotion=True, limit=limit))
    )

    result = response.get_result(), response.get_headers(), response.get_status_code()
    self.memo[(text_str, limit)] = result

    return result


get_entity_info.memo = {}








def get_keyword_info(text_str, limit):
    response = nlu.analyze(
        text=text_str,
        features=Features(keywords=KeywordsOptions(sentiment=True,
                                                   emotion=True,
                                                   limit=limit
                                                   )
                          )
    )

    result = response.get_result(), response.get_headers(), response.get_status_code()
    self.memo[(text_str, limit)] = result

    return result


get_keyword_info.memo = {}



def get_relational_info(text_str):
    response = nlu.analyze(
        text=text_str,
        features=Features(relations=RelationsOptions())
    )

    result =  response.get_result(), response.get_headers(), response.get_status_code()

    self.memo[text_str] = result

    return result


get_relational_info.memo = {}








def get_semantic_roles(text_str, limit):
    response = nlu.analyze(
        text=text_str,
        features=Features(semantic_roles=SemanticRolesOptions(keywords=True, entities=True, limit=limit))
    )

    result = response.get_result(), response.get_headers(), response.get_status_code()

    self.memo[(text_str, limit)] = result

    return result


get_semantic_roles.memo = {}









def get_targeted_sentiment(text_str, target_words_phrases_list):
    response = nlu.analyze(
        text=text_str,
        features=Features(sentiment=SentimentOptions(document=True, targets=target_words_phrases_list))
    )

    result = response.get_result(), response.get_headers(), response.get_status_code()

    self.memo[(text_str, target_words_phrases_list)] = result
    return result


get_targeted_sentiment.memo = {}







def get_syntax_info(text_str):
    response = nlu.analyze(
        text=text_str,
        features=Features(syntax=SyntaxOptions(
            sentences=True,
            tokens=SyntaxOptionsTokens(lemma=True,
                                       part_of_speech=True)))
    )

    result = response.get_result(), response.get_headers(), response.get_status_code()

    self.memo[text_str] = result
    return result


get_syntax_info.memo = {}
