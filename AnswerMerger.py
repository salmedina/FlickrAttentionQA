from __future__ import print_function
import json
import itertools

class AnswerMerger(object):
    def __init__(self):
        pass

    def merge(self, text_answer, mm_answer):
        '''
        Merges the answers from both pipelines
        :param text_answer: answer from text qa pipeline 
        :param mm_answer: answer from multimedia qa pipeline
        :return: merged answer
        '''

        res = {}

        res['answer_summary'] = mm_answer['answer_summary']
        res['question_type'] = text_answer['question_type']
        res['user_question'] = text_answer['user_question']

        #interleave the answers
        num_text_answers = len(text_answer['answers'])
        num_mm_answers = len(mm_answer['answers'])

        merged_answers = [a for a in
                          itertools.chain(*itertools.zip_longest(text_answer['answers'], mm_answer['answers']))
                          if a is not None]
        merged_highlights = [a for a in
                          itertools.chain(*itertools.zip_longest(text_answer['highlighted_keyword'], mm_answer['highlighted_keyword']))
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
