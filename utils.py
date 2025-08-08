# from langchain.embeddings import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from py2neo import Graph
from config import *
from langchain_deepseek import ChatDeepSeek
from langchain_huggingface import HuggingFaceEmbeddings
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
load_dotenv()
# 词嵌入模型获取函数
def get_embeddings_model():
    model_map ={
        'openai': HuggingFaceEmbeddings()
    }
    return model_map.get('openai')
def get_llm_model():
    model_map={
        'deepseek': ChatDeepSeek(
            model = os.getenv('DEEPSEEK_LLM_MODEL'),
            temperature = os.getenv('TEMPERATURE'),
            max_tokens = os.getenv('MAX_TOKENS'),
        ),
        'qwen': ChatOpenAI(
            model=os.getenv('OPENAI_LLM_MODEL'),
            temperature=os.getenv('TEMPERATURE'),
            max_tokens=os.getenv('MAX_TOKENS'),
        )
    }
    return model_map.get(os.getenv('LLM_MODEL'))
def structured_output_parser(response_schemas):
    text = '''
    请从以下文本中，抽取出实体信息，并按json格式输出，json包含首尾的json”和
    以下是字段含义和类型，要求输出json中，必须包含下列所有字段:\n
    '''
    for schema in response_schemas:text+=schema.name+'字段，表示:'+schema.description +'，类型为:'+schema.type +'\n'
    return text
def replace_token_in_string(string, slots):
    for key,value in slots:
        string = string.replace('%'+key+'%',value)
    return string
def get_neo4j_conn():
    return Graph(
        os.getenv('NE04J_URI'),
        auth =(os.getenv('NE04J_USERNAME'),os.getenv('NE04J_PASSWORD'))
    )
if __name__ == '__main__':
    # llm_model = get_llm_model()
    # print(llm_model.invoke('什么是爱情？'))
    model = get_embeddings_model()
    print(model.model_name)  # 查看默认模型名称
    # print(model.model)  # 查看实际加载的模型对象