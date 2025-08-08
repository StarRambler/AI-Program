from prompt import *
from utils import *
from agent import *
from langchain.chains import LLMChain
from langchain.prompts import Prompt
class Service():
    def __init__ (self):
        self.agent =Agent()
    def get_summary_message(self,message,history):
        print('总结信息')
        llm = get_llm_model()
        prompt=Prompt.from_template(SUMMARY_PROMPT_TPL)
        llm_chain =  prompt|llm
        chat_history =''
        a=[]
        q=[]
        index=0
        items=history[-4:]
        for item in items :
            if item['role'] == 'user':
                q.append(item['content'])
            elif item['role'] == 'assistant':
                a.append(item['content'])
        index=0
        while index < len(q):
            chat_history +=f'问题:{q[index]}，答案:{a[index]}\n'
            index+=1
        return llm_chain.invoke(query=message, chat_history=chat_history)
    def answer(self,message,history):
        print('调用回复函数')
        if history:
            message =self.get_summary_message(message, history)
        print('总结完成')
        return self.agent.query(message)
if __name__=='__main__':
    service =Service()
    print(service.answer('你好',[]))