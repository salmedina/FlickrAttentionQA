import requests
import json

class MMQAPipeline(object):
    def __init__(self, mmqa_url):
        self. url = mmqa_url

    def answer_user_question(self, userid, question):
        qa_params = {}
        qa_params['u'] = userid
        qa_params['q'] = question
        reqres = requests.get(self.url, params=qa_params)
        return json.loads(reqres.text)