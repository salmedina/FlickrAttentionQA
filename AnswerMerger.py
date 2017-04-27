from __future__ import print_function
import json
import re
import itertools
try:
    # Python 3
    from itertools import zip_longest
except ImportError:
    # Python 2
    from itertools import izip_longest as zip_longest

class AnswerMerger(object):
    def __init__(self):
        pass

    def beautify_text_evidence(self, text_answer):
        for i in range(len(text_answer['answers'])):
            if 'snippets' in text_answer['answers'][i]:
                text_answer['answers'][i]['evidence'] += '\n' + text_answer['answers'][i]['snippets']

    def beautify_mm_evidence(self, mm_answer):
        for i in range(len(mm_answer['answers'])):
            concepts_found = re.findall(r'(\w+): \d\.\d+', mm_answer['answers'][i]['evidence'])
            if concepts_found is not None and len(concepts_found) > 0:
                num_concepts = len(concepts_found)
                concept_list_str = ''
                if num_concepts == 1:
                    concept_list_str = concepts_found[0]
                else:
                    concept_list_str = ', a '.join(concepts_found[:-1])+ ' and a ' + concepts_found[-1]

                snippet = ''
                if 'snippets' in mm_answer['answers'][i]:
                    snippet = mm_answer['answers'][i]['snippets']

                mm_answer['answers'][i]['evidence'] = 'Found a {} in this video.\n{}'.format(concept_list_str, snippet)

    def norm_answer_fields(self, answer):
        if type(answer) is not dict:
            return

        if 'question_type' not in answer:
            answer['question_type'] = 'UNK'
        if 'user_question' not in answer:
            answer['user_question'] = 'UNK'
        if 'answers' not in answer:
            answer['answers'] = []
        if 'highlighted_keyword' not in answer:
            answer['highlighted_keyword'] = []

    def merge(self, text_answer, mm_answer):
        '''
        Merges the answers from both pipelines
        :param text_answer: answer from text qa pipeline 
        :param mm_answer: answer from multimedia qa pipeline
        :return: merged answer
        '''

        # Ensure data is correct for processing
        self.norm_answer_fields(text_answer)
        self.norm_answer_fields(mm_answer)
        self.beautify_text_evidence(text_answer)
        self.beautify_mm_evidence(mm_answer)

        res = {}

        res['answer_summary'] = mm_answer['answer_summary']
        res['question_type'] = text_answer['question_type']
        res['user_question'] = text_answer['user_question']

        #interleave the answers
        '''
        [u'how_many', u'what', u'when', u'when_and_where', u'where', u'show_me', u'yes/no', u'who']
        '''

        if res['question_type'] in [ u'what', u'yes/no', u'who']:
            first_answer = text_answer
            second_answer = mm_answer
        else:
            first_answer = mm_answer
            second_answer = text_answer

        merged_answers = [a for a in
                          itertools.chain(*zip_longest(first_answer['answers'], second_answer['answers']))
                          if a is not None]
        merged_highlights = [a for a in
                          itertools.chain(*zip_longest(first_answer['highlighted_keyword'], second_answer['highlighted_keyword']))
                          if a is not None]
        res['answers'] = merged_answers
        res['highlighted_keyword'] = merged_highlights

        return res

def merge_test():
    m = AnswerMerger()
    text_answer = json.load(open('./data/test/textAns.json'))
    mm_answer = json.load(open('./data/test/mmAns.json'))
    merged_answer = m.merge(text_answer, mm_answer)

    print(merged_answer)

if __name__ == '__main__':
    merge_test()
