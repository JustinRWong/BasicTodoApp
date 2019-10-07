import json
from ibm_watson import ToneAnalyzerV3
from ibm_cloud_sdk_core.authenticators import IAMAuthenticator
from ibm_watson import ApiException
from ibm_watson.natural_language_understanding_v1 import Features, SyntaxOptions, SyntaxOptionsTokens


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
        return j.get_result(), j.get_headers(), j.get_status_code()
    except ApiException as ex:
        print("Method failed with status code " + str(ex.code) + ": " + ex.message)
        raise ex


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

