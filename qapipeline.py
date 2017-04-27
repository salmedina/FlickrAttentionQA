from __future__ import division
import requests
import json
import spacy
import fasttext
import pysolr
import dateutil.parser
from squad.demo_prepro import prepro
from basic.demo_cli import Demo
import collections

Flick = collections.namedtuple('Flick', 'id, userid, username, dt, title, desc, tags, url, mediaurl')


class BiDAFServer(object):
    def __init__(self, url):
        self.url = url

    def get_answer(self, question, text):
        params = {'question': question, 'paragraph': text}
        req = requests.post(self.url, params=params)
        return json.loads(req.text)['result']

class BiDAF(object):
    def __init__(self):
        self.core = Demo()

    def get_answer(self, paragraph, question):
        pq_prepro = prepro(paragraph, question)
        if len(pq_prepro['x']) > 1000:
            return "[Error] Sorry, the number of words in paragraph cannot be more than 1000."
        if len(pq_prepro['q']) > 100:
            return "[Error] Sorry, the number of words in question cannot be more than 100."
        return self.core.run(pq_prepro)

class QAPipeline(object):
    def __init__(self, qclf_path, index_url, bidaf_url='', index_timeout=20, get_answer=None):
        '''
        Initializes the constants and instantiates the required objects
        @qclf_path: path to the question classifier binary file
        @index_url: url of the Flickr index
        @bidaf_url: url to the BiDAF service, if empty then BiDAF is run locally
        @index_timeout: index response timeout in secs
        @returns: None
        '''

        #TODO: validate the connections - add try/catch
        # Store params
        self.qclf_path = qclf_path
        self.bidaf_url = bidaf_url
        self.index_url = index_url
        self.num_answers = 10
        # Declare constants
        self.index_fieldnames = [u'id', u'userid_s', u'username_s', u'title_t', u'desc_t', u'url_s', u'mediaurl_s', u'feedback_s', u'datetime_dt']
        self.multimodal_question_types = [u'how_many', u'what', u'when', u'when_and_where', u'where', u'show_me', u'yes/no', u'who']
        self.uninformative_verbs = [u'be', u'do', u'have']
        self.show_me_verbs = [u'show', u'display', u'play', u'find', u'look', u'search']
        self.polite_phrases = [u'please', u'Please', u'could you please', u'Could you please']
        self.timesort_phrases = [u'first time',u'last time']
        # Init auxiliary objects
        print('Loading classifier')
        self.qclf = fasttext.load_model(qclf_path)
        print('Loading Bi-DAF')
        self.bidaf = BiDAF() if len(bidaf_url) < 1 else BiDAFServer(self.bidaf_url)
        print('Loading Indexer')
        self.solr_flickr = pysolr.Solr(index_url, timeout=20)
        print('Loading NLP pipeline')
        self.nlp = spacy.load('en')
        self.get_answer = get_answer if get_answer is not None else self.bidaf.get_answer

    def remove_politeness(self, text):
        '''Removes the phrases related to polite requests'''
        for phrase in self.polite_phrases:
            text = text.replace(phrase, '')
        return text.strip()

    def is_command(self, text):
        '''Verifies whether the question is a command or a question'''
        q_doc = self.nlp(text)
        for np in q_doc.noun_chunks:
            if np.root.head.lemma_ in self.show_me_verbs:
                return True
        return False

    def is_dt_str(self, text):
        try:
            dateutil.parser.parse(text)
            return True
        except Exception:
            return False

    def beautify_datetime(self, text):
        try:
            text_dt = dateutil.parser.parse(text)
            return text_dt.strftime('%B %d, %Y')
        except Exception:
            return ''

    def classify_question(self, question):
        '''Classifies the question based on the given classifier during the init'''
        # TODO: improve the classifier and incorporate yes/no label
        # 1. Detect if it is a command
        if self.is_command(question):
            return 'show_me'

        # 2. Heuristics for classification
        q_doc = self.nlp(question)
        first_token = q_doc[0].lemma_
        if first_token == u'where':
            return u'where'
        elif first_token in [u'how many', u'how much']:
            return u'how_many'
        elif first_token == u'when':
            return u'when'
        elif first_token == u'who':
            return u'who'
        elif first_token == u'which':
            return u'what'
        elif first_token in self.uninformative_verbs:
            return 'yes/no'

        # 3. Fasttext classification if heuristic fails
        label, prob = self.qclf.predict_proba([question])[0][0]
        label = label.replace('__label__', '')

        # 4. Bin the questions into target question for demo
        norm_label = self.normalize_question_class(label)

        return norm_label

    def extract_keyterms(self, question, q_class):
        '''Extracts the keyterms from the question'''
        keyterms = []

        q_doc = self.nlp(question)
        # Extract verbs
        for word in q_doc:
            if word.pos_ == u'VERB' and \
                (word.lemma_ not in self.uninformative_verbs) and \
                (word.lemma_ not in self.show_me_verbs):
                keyterms.append(word.text)

        # Extract NE
        for ent in q_doc.ents:
            keyterms.append(ent.text)

        # Extract NP
        for np in q_doc.noun_chunks:
            if np.lemma_ not in ['i', 'me', 'we']:
                keyterms.append(np.text)

        # TODO: verify if NE's must be added as a single element
        keyterms = list(set(keyterms))
        print(keyterms)
        return keyterms  # Remove repeated terms

    def solr_res_to_list(self, response):
        '''Converts a solr response into a list of dictionaries that represent the answer'''
        res_list = []
        for r in response:
            res_dict = {}
            for fieldname in self.index_fieldnames:
                if fieldname in r:
                    res_dict[fieldname] = r[fieldname]
                else:
                    res_dict[fieldname] = ''
            res_dict['votes'] = 0
            res_list.append(res_dict)

        return res_list

    def retrieve_user_posts(self, userid, keyterms):
        '''Retrieves from the index the user's post with userid and based on the keyterms'''
        keyterms_s = ' '.join(keyterms)
        query_s = 'userid_s:"%s" AND (title_t:%s OR desc_t:%s)' % (userid, keyterms_s, keyterms_s)
        res = self.solr_flickr.search(query_s, rows=25)
        res_list = self.solr_res_to_list(res)
        print(query_s)
        return res_list

    def retrieve_user_posts_date(self, userid, kyeterm, date_str):
        pass

    def get_index_field_val(self, field_val):
        if type(field_val) is list:
            return field_val[0]
        if type(field_val) is str:
            return field_val
        return field_val

    def build_video_url(self, vid):
        video_url = u"https://www.flickr.com/video_download.gne?id={}".format(vid)
        return video_url

    def extract_bidaf_answer(self, q, text):
        return self.get_answer(text, q)

    def extract_ner_answer(self, q_class, text):
        ''' Heuristics to extract the related named entities according to question type '''
        ners = []
        doc = self.nlp(text)
        if q_class == u'how_many':
            ners = [w.text for w in doc.ents if w.label_ in ['ORDINAL', 'CARDINAL', 'QUANTITY', 'MONEY', 'PERCENT']]
        elif q_class == u'when':
            ners = [w.text for w in doc.ents if w.label_ in ['CARDINAL', 'ORDINAL', 'DATE', 'TIME']]
        elif q_class == u'where':
            ners = [w.text for w in doc.ents if w.label_ in ['FACILITY', 'ORG', 'GPE', 'LOC']]
        elif q_class == u'when_and_where':
            ners = [w.text for w in doc.ents if w.label_ in ['FACILITY', 'ORG', 'GPE', 'LOC', 'ORDINAL', 'CARDINAL', 'DATE', 'TIME']]
        elif q_class == u'who':
            ners = [w.text for w in doc.ents if w.label_ in ['PERSON', 'ORG', 'EVENT', 'WORK_OF_ART']]
        else:
            ners = [w.text for w in doc.ents]

        ners_str = ' '.join(ners)

        return ners_str

    def extract_answers(self, question, q_class, q_results):
        '''
        Extracts the answer from the document retrieval phase
        @question: querying question
        @q_class: type of answer expected
        @q_results: index querying results
        @returns: list of answers
        '''

        '''
        According to type of question is what will be done, q_class should be the binned question
        1. Go through each of the retrieved docs
        2. Extract bidaf answer from title then desc, add to answer and increase vote
        3. According to the question class:   
            - Extract corresponding ner, add to answer dict and add vote
        '''
        # TODO: This needs to be further improved by making an analysis based on the type of question
        answers = []
        for rank, res in enumerate(q_results):
            # build answer
            answer = {}
            answer['url'] = self.build_video_url(res['id'])
            answer['vid'] = res['id']
            answer['bidaf'] = {'title':'', 'desc':''}
            answer['ner'] = {'title':'', 'desc':''}
            answer['evidence'] = ''
            answer['snippets'] = ''
            answer['votes'] = 0
            answer['default'] = ''

            '''
            1. Annotate the text for re-ranking
                Priorities for answer extraction
                1. BIDAF
                    a) description
                    b) title
                2. NER
                    a) description
                    b) title
            2. Build evidence, either title or description
            3. Get snippet from annotation
            '''
            # Get the text
            res_title = self.get_index_field_val(res['title_t'])
            res_desc =  self.get_index_field_val(res['desc_t'])
            # TITLE
            if res_title:
                answer['ner']['title'] = self.extract_ner_answer(q_class, res_title)
                if answer['ner']['title']:
                    answer['votes'] += 1
                    if q_class in ['when', 'where_and_when'] and self.is_dt_str(answer['ner']['title']):
                        answer['votes'] += 1

                answer['bidaf']['title'] = self.extract_bidaf_answer(question, res_title)
                if answer['bidaf']['title']:
                    answer['votes'] += 1

            # DESCRIPTION
            if res_desc:
                answer['ner']['desc'] = self.extract_ner_answer(q_class, res_desc)
                if answer['ner']['desc']:
                    answer['votes'] += 1
                    if q_class in ['when', 'where_and_when'] and self.is_dt_str(answer['ner']['desc']):
                        answer['votes'] += 1

                answer['bidaf']['desc'] = self.extract_bidaf_answer(question, res_desc)
                if answer['bidaf']['desc']:
                    answer['votes'] += 1

            # Snippet and evidence
            if answer['bidaf']['desc']:
                answer['evidence'] = res_desc
                answer['snippets'] = answer['bidaf']['desc']
            elif answer['bidaf']['title']:
                answer['evidence'] = res_title
                answer['snippets'] = answer['bidaf']['title']
            elif answer['ner']['desc']:
                answer['evidence'] = res_desc
                answer['snippets'] = answer['ner']['desc']
            elif answer['ner']['title']:
                answer['evidence'] = res_title
                answer['snippets'] = answer['ner']['title']

            # default according to type
            if q_class in ['when', 'where_and_when']:
                answer['default'] = res['datetime_dt']
            elif q_class in ['yes/no']:
                if answer['evidence'] or answer['snippet']:
                    answer['default'] = 'yes'
                else:
                    answer['default'] = 'no'

            answers.append(answer)

        # Sort according to extraction votes
        answers = sorted(answers, key=lambda x:x['votes'], reverse=True)

        # Remove the fields that will not be shown in final answer
        rank = 0
        for a in answers:
            a['rank'] = rank
            rank += 1
            a.pop('bidaf', None)
            a.pop('ner', None)
            a.pop('votes', None)

        if q_class in ['when', 'where_and_when']:
            answers = self.build_when_answers(answers)

        return answers

    def normalize_question_class(self, q_class):
        '''Bins the questions into the demo classes'''
        # TODO: needs a major cleanup and more pythonic way of doing it
        '''multimodal_question_types = ['how_many', 'what', 'when', 'when_and_where', 'where', 'show_me', 'yes/no']'''

        # TODO: need to handle when_and_where, show_me, yes/no
        bin_class = ''
        if q_class in ['NUM:count']:
            bin_class = 'how_many'
        elif q_class in ['NUM:dist', 'NUM:money', 'NUM:perc', 'NUM:speed', 'NUM:temp', 'NUM:weight', 'NUM:volsize']:
            bin_class = 'how_much'
        elif q_class in ['NUM:date']:
            bin_class = 'when'
        elif q_class in ['LOC:city', 'LOC:country', 'LOC:mount', 'LOC:other', 'LOC:state']:
            bin_class = 'where'
        elif q_class in ['show_me:all']:
            bin_class = 'show_me'
        elif q_class in ['HUM:desc', 'HUM:gr', 'HUM:ind','HUM:title']:
            bin_class = 'who'
        else:
            bin_class = 'what'

        return bin_class

    def build_when_answers(self, q_answers):
        '''
        Android app is reading the evidence into the cards, 
        that is why we beautify the answer in evidence field
        :param q_answers: 
        :return: 
        '''
        formatted_answers = []
        for answer in q_answers:
            summary = self.beautify_datetime(answer['snippets'])
            if not summary:
                summary = self.beautify_datetime(answer['default'])
            answer['evidence'] = summary
            formatted_answers.append(answer)
        return formatted_answers

    def summarize_answers(self, question, q_class, q_answers):
        ''' Generates the final answer according to the evidence '''
        summary = ''
        if q_answers:
            if q_class in ['when', 'where_and_when']:
                summary = self.beautify_datetime(q_answers[0]['snippets'])
                if not summary:
                    summary = self.beautify_datetime(q_answers[0]['default'])
            else:
                if q_answers[0]['snippets']:
                    summary = q_answers[0]['snippets']
                elif q_answers[0]['evidence']:
                    summary = q_answers[0]['evidence']


        return summary

    def build_reponse(self, question, q_class, q_answers):
        '''Formats the answer to the API format'''
        res = {}

        #TODO: improve the answer summary based on type of answer
        res['answer_summary'] = self.summarize_answers(question, q_class, q_answers)
        res['answers'] = q_answers[:self.num_answers]
        res['highlighted_keyword'] = [a['snippets'] for a in res['answers'] if a['snippets']]
        res['question_type'] = q_class
        res['user_question'] = question

        return res

    def build_error_response(self, msg):
        res = {}
        res['Error'] = {}
        res['Error']['message'] = msg

        return res

    def is_valid_user(self, userid):
        #TODO: validate if the user is within the database
        if userid is None:
            return False
        return True

    def is_valid_question(self, question):
        if len(question) < 2 or question is None:
            return False
        return True

    def answer_user_question(self, userid, question):
        '''Obtains the answer to the question according to the index, this runs the full qa pipeline'''
        if not self.is_valid_user(userid):
            return self.build_error_response('Not a valid userid')
        if not self.is_valid_question(question):
            return self.build_error_response('Not a valid question')

        question = self.remove_politeness(question)
        q_class =  self.classify_question(question)
        q_keyterms = self.extract_keyterms(question, q_class)
        q_results = self.retrieve_user_posts(userid, q_keyterms)
        q_answers = self.extract_answers(question, q_class, q_results)
        #TODO: add user reranking based on history
        response = self.build_reponse(question, q_class, q_answers)
        return response

def run_test():
    config = json.load(open('qaconfig.json'))
    qap = QAPipeline(config['qclf_path'], config['index_url'], config['bidaf_url'])
    userid = '88008488@N00'
    question = 'show me a video from Tokyo'
    res = qap.answer_user_question(userid, question)
    print(res)

if __name__ == '__main__':
    run_test()