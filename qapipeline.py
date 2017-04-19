from __future__ import division
import requests
import json
import spacy
import fasttext
import pysolr
from squad.demo_prepro import prepro
from basic.demo_cli import Demo


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
        # Declare constants
        self.index_fieldnames = [u'userid_s', u'username_s', u'title_t', u'desc_t', u'url_s', u'datetime_dt']
        self.multimodal_question_types = [u'how_many', u'what', u'when', u'when_and_where', u'where', u'show_me', u'yes/no']
        self.uninformative_verbs = [u'be', u'do', u'have']
        self.show_me_verbs = [u'show', u'display', u'play', u'find', u'look', u'search']
        self.polite_phrases = [u'please', u'Please', u'could you please', u'Could you please']
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

    def classify_question(self, question):
        '''Classifies the question based on the given classifier during the init'''
        # TODO: incorporate into the classifier
        # TODO: add yes/no label into training of classifier
        if self.is_command(question):
            return 'show_me:all', 1.0

        label, prob = self.qclf.predict_proba([question])[0][0]
        label = label.replace('__label__', '')
        return label, prob

    def extract_keyterms(self, question, q_class):
        '''Extracts the keyterms from the question'''
        label_root, label_sub = q_class.split(':')
        keyterms = []

        q_doc = self.nlp(question)
        # Extract verbs
        for word in q_doc:
            if word.pos_ == u'VERB' and \
                (word.lemma_ not in self.uninformative_verbs) and \
                (word.lemma_ not in self.show_me_verbs):
                keyterms.append(word.text)

        # Extract NP
        for np in q_doc.noun_chunks:
            keyterms.append(np.text)

        # Extract NE
        for ent in q_doc.ents:
            keyterms.append(ent.text)

        # If no keyterms found, default to the full question words
        # if len(keyterms) < 1:
        #     for word in q_doc:
        #         keyterms.append(word.text)

        # TODO: verify if NE's must be added as a single element
        return list(set(keyterms))  # Remove repeated terms

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
            res_list.append(res_dict)

        return res_list

    def retrieve_user_posts(self, userid, keyterms):
        '''Retrieves from the index the user's post with userid and based on the keyterms'''
        keyterms_s = ' '.join(keyterms)
        query_s = 'userid_s="%s" AND (title_t: %s OR desc_t: %s)' % (userid, keyterms_s, keyterms_s)

        res = self.solr_flickr.search(query_s, rows=25)
        res_list = self.solr_res_to_list(res)

        return res_list

    def get_index_field_val(self, field_val):
        if type(field_val) is list:
            return field_val[0]
        if type(field_val) is str:
            return field_val
        return field_val

    def extract_answers(self, question, q_class, q_results):
        '''Filters out answers through BiDAF'''
        # TODO: This needs to be further improved by making an analysis based on the type of question
        answers = []
        for rank, res in enumerate(q_results):
            # Extract answers from text fields
            # TODO: move the period addition into the Bi-DAF service
            snippet = ''
            evidence = ''
            text = self.get_index_field_val(res['desc_t'])
            if len(text) > 0:
                print(text)
                bidaf_ans = self.get_answer(question, text)
                snippet = bidaf_ans
                evidence = text
            # if not get it from title
            text = self.get_index_field_val(res['title_t'])
            if len(evidence) < 1 and len(text) > 0:
                bidaf_ans = self.get_answer(question, text)
                snippet = bidaf_ans
                evidence = text

            # build answer dictionary
            answer = {}
            answer['rank'] = rank
            answer['url'] = res['url_s']
            answer['evidence'] = evidence
            answer['snippets'] = snippet
            answer['vid'] = ''

            answers.append(answer)
        # TODO: need to rerank based on snippet extraction

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
        else:
            bin_class = 'what'

        return bin_class

    def build_reponse(self, question, q_class, q_answers):
        '''Formats the answer to the API format'''
        res = {}

        # TODO: build a smarter answer summary
        res['answer_summary'] = q_answers[0]['snippets']
        res['answers'] = q_answers
        res['highlighted_keyword'] = [a['snippets'] for a in q_answers if len(a['snippets']) > 0]
        res['question_type'] = self.normalize_question_class(q_class)
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
        q_class, q_class_prob = self.classify_question(question)
        q_keyterms = self.extract_keyterms(question, q_class)
        q_results = self.retrieve_user_posts(userid, q_keyterms)
        q_answers = self.extract_answers(question, q_class, q_results)
        # TODO: add user reranking
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