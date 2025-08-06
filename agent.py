from utils import *
from config import *
from prompt import *
from langchain_tavily import TavilySearch
from langchain_core.messages import HumanMessage,SystemMessage
from langchain.schema import AIMessage
from langchain_core.prompts import ChatPromptTemplate
from typing import Literal
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent,ToolNode
from langgraph.graph import END,StateGraph,MessagesState
from langchain.prompts import PromptTemplate
from langchain_chroma import Chroma
from langchain.schema import Document
from langchain.agents import Tool
from langchain.output_parsers import ResponseSchema, StructuredOutputParser
import os
from datetime import datetime
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
class Agent():
    def __init__(self):
        print('begin')
        self.vdb = Chroma(
        persist_directory = os.path.join(os.path.dirname(__file__), './data/db'),
        embedding_function = get_embeddings_model()
        )
        print('init finished')

    def retrival_func(self,  query):
        print("调用了文档查询")
        # 召回并过滤文档
        print(query)
        documents = self.vdb.similarity_search_with_relevance_scores(query, k=10)
        # print(documents)
        # exit()
        query_result = [doc[0].page_content for doc in documents if doc[1] > 0.5]
        print(documents)
        # 填充提示词并总结答案
        prompt =PromptTemplate.from_template(RETRIVAL_PROMPT_TPL)
        llm=get_llm_model()
        retrival_chain =prompt|llm
        inputs = {
            'query': query,
            'query_result': '\n\n'.join(query_result) if len(query_result) else '没有查到'
        }
        result= retrival_chain.invoke(inputs)
        # print(result)
        return result

    def graph_func(self, query):
        # 命名实体识别
        print("调用了图数据库查询")
        response_schemas =[
            ResponseSchema(type='list',name = 'Disease',description = '疾病名称实体'),
            ResponseSchema(type='list', name = 'Symptom', description = '疾病症状实本'),
            ResponseSchema(type='list',name = 'Drug',description = '药品名称实体')
        ]
        output_parser = StructuredOutputParser(response_schemas = response_schemas)
        format_instructions = structured_output_parser(response_schemas)
        ner_prompt = PromptTemplate(
            template=NER_PROMPT_TPL,
            partial_variables = {'format_instructions': format_instructions},
            input_variables = ['query']
        )
        llm = get_llm_model()
        ner_chain  = ner_prompt|llm
        result = ner_chain.invoke({
            'query': query
        })
        # print(result.content)
        # exit()
        ner_result = output_parser.parse(result.content)
        # print(ner_result)
        graph_templates = []
        for key, template in GRAPH_TEMPLATE.items():
            slot = template['slots'][0]
            slot_values = ner_result[slot]
            for value in slot_values:
                graph_templates.append({
                    'question':replace_token_in_string(template['question'],[[slot,value]]),
                    'cypher':replace_token_in_string(template['cypher'],[[slot, value]]),
                    'answer': replace_token_in_string(template['answer'], [[slot, value]]),
                })
        if not graph_templates:
            return
        graph_documents = [
            Document(page_content=template['question'], metadata=template)
            for template in graph_templates]
        # print(graph_templates)
        db = Chroma.from_documents(
            documents=graph_documents,
            embedding=get_embeddings_model(),
            persist_directory=os.path.join(os.path.dirname(__file__), './data/graph_db'),
            collection_metadata = {"allow_metadata": True}
        )
        # print(query)
        graph_documents_filter = db.similarity_search_with_relevance_scores(query, k=1)
        query_result = []
        neo4j_conn = get_neo4j_conn()
        for document in graph_documents_filter:
            question = document[0].page_content
            cypher = document[0].metadata['cypher']
            answer= document[0].metadata['answer']
            # print(cypher)
            try:
                result = neo4j_conn.run(cypher).data()
                if result and any(value for value in result[0].values()):
                    answer_str = replace_token_in_string(answer, list(result[0].items()))
                    # print(answer_str)
                    query_result.append(f'问题:{question}\n答案:{answer_str}')
            except Exception as e:
                print(f"执行cypher查询出错: {e}")
                continue
        # 总结答案
        prompt = PromptTemplate.from_template(GRAPH_PROMPT_TPL)
        llm = get_llm_model()
        graph_chain =  prompt|llm
        inputs = {
            'query': query,
            'query_result':'\n\n'.join(query_result) if len(query_result) else '没有查到'
        }
        # print(inputs)
        # print(graph_chain.invoke(inputs))
        return graph_chain.invoke(inputs)
    def search_func(self,  query):
        print("调用了网络搜索查询")
        tools = [TavilySearch(max_result=5)]
        current_time = datetime.now().strftime("%Y年%m月%d日 %H:%M:%S")
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """你是一个有用的智能助手，必须通过网络搜索查找用户提问的相关内容，并依据搜索到的内容回复用户问题，否则回复的消息将不具备实时性。
                    关于搜索相关规则如下，你必须严格遵守：
                    1.以发起查询的时刻为准，回复用户提问时，要标注查询时间；
                    2.当用户查询时间信息时，直接返回查询时间；
                    3.搜索内容具有时效性时，要以发起查询的时间为依据进行搜索
                    4.发起查询的时间为{current_time}
                    5.未搜索到的信息如实回答能查询到，不要自己回答""",
                ),
                ("placeholder", "{messages}"),
            ])

        llm = get_llm_model()
        agent_executor = prompt|create_react_agent(llm, tools)
        return agent_executor.invoke({"messages": [HumanMessage(content=query)],"current_time":current_time})
    def bulid_workflow(self):
        tools=[
            Tool.from_function(
                name='retrival_func',
                func=lambda x:self.retrival_func(x),
                description='回复医疗政策、医院科室排名、肺结核患者居家治疗、非吸烟人气肺癌流行病研究现状等问题'
            ),
            Tool.from_function(
                name='graph_func',
                func=lambda x: self.graph_func(x),
                description='回复疾病信息，治疗方法，症状等问题'
            ),
            Tool.from_function(
                name='search_func',
                func=lambda x:self.search_func(x),
                description='其他工具没有答案时，或需要查询当前时间、日期等实时信息时，通过搜索引擎回答通用类问题'
            )
        ]
        tool_node=ToolNode(tools)
        prompt=ChatPromptTemplate.from_messages(
    [
        SystemMessage(content=
            """你是一位私人健康助理，名字叫做大白，能够回答医疗相关内容。你的回答要有依据，源自于医学知识，不要自己杜撰，否则可能造成严重安全事故。
            你需要严格遵守以下规则进行回复，否则属于严重违规：
                1.你必须拒绝讨论任何关于政治，色情，暴力相关的事件或者人物。
                例如问题：普京是谁，列宁的过错，如何杀人放火，打架群殴，如何跳楼，如何制造毒药
                2.请用中文回答用户问题。
                """),

        ("placeholder","{messages}"),
    ]
)
        llm=get_llm_model()

        agent=prompt|llm.bind_tools(tools)
        def should_continue(state: MessagesState) -> Literal["tools", END]:
            message = state["messages"][-1]
            if message.tool_calls:
                # print(message.tool_calls)
                return "tools"
            return END

        def call_model(state: MessagesState):
            message = state['messages']
            # print(f"传入 agent 的消息为: {message}")
            response = agent.invoke({'messages':message})
            # print(f" agent 回复的消息为: {response}")
            return {"messages": response}

        workflow = StateGraph(MessagesState)
        workflow.add_node("agent", call_model)
        workflow.add_node('tools', tool_node)

        workflow.set_entry_point("agent")
        workflow.add_conditional_edges(
            "agent",
            should_continue
        )
        workflow.add_edge("tools", "agent")

        checkpointer = MemorySaver()
        app = workflow.compile(checkpointer=checkpointer)
        return app

    def query(self, query_str):
        return self.query_with_messages([HumanMessage(content=query_str)])
    def query_with_messages(self, messages):
        app=self.bulid_workflow()
        return app.invoke(
            {"messages": messages},
            config={"configurable": {"thread_id": 42}}  # 支持多线程上下文
        )
    def query_from_history(self, message, history):
        messages = []
        print(history)
        for item in history:
            role = item.get("role")
            content = item.get("content")
            if role == "user":
                messages.append(HumanMessage(content=content))
            elif role == "assistant":
                messages.append(AIMessage(content=content))
        messages.append(HumanMessage(content=message))
        result = self.query_with_messages(messages)
        return result["messages"][-1].content
if __name__ == '__main__':
    agent=Agent()
    # print(agent.query('你叫什么名字'))
    # print(agent.retrival_func('口腔癌补助对象都包含哪些人群？'))
    # print(agent.graph_func([],'百日咳该如何预防?'))
    # print(agent.search_func('罗小黑战记出到了第几部？'))
    print(agent.query('百日咳是什么疾病'))