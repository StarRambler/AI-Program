# from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_tavily import TavilySearch
from langchain import hub
from langchain_openai import ChatOpenAI
import asyncio
from langgraph.prebuilt import create_react_agent
from dotenv import load_dotenv
load_dotenv()
tools=[TavilySearch(max_result=1)]
prompt=hub.pull("wfh/react-agent-executor")
prompt.pretty_print()

llm=ChatOpenAI(model="deepseek-chat",base_url="https://api.deepseek.com",response_format=None)
agent_executor=create_react_agent(llm,tools,messages_modifier=prompt)
import operator
from typing import Annotated,List,Tuple,TypedDict

class PlanExecute(TypedDict):
    input: str
    plan: List[str]
    past_step:Annotated[List[Tuple],operator.add]
    response:str
from pydantic import BaseModel,Field
class Plan(BaseModel):
    '''未来要执行的计划'''
    steps:List[str]=Field(
    descriptiion="需要执行不同的步骤，应该按顺序排列"
    )
from langchain_core.prompts import ChatPromptTemplate

planner_prompt=ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "对于给定目标，提出一个简单的逐步计划。这个计划应该包含独立的任务，如果正确执行将得到正确答案。不要添加多余的步骤。最后一步的结果应该是最终答案。确保每一步都有所有必要的信息。-不要跳过步骤",
        ),
        ("placeholder","{messages}"),
    ]
)
planner=planner_prompt|ChatOpenAI(model="deepseek-chat",base_url="https://api.deepseek.com",response_format=None).with_structured_output(Plan)
from typing import Union


class Response(BaseModel):
    '''用户相应'''
    response: str


class Act(BaseModel):
    '''要执行的行为'''
    action: Union[Response, Plan] = Field(
        description="要执行的行为。如果要回应用户，使用Response。如果需要进一步使用工具获取答案，使用Plan。"
    )


# 创建一个新的提示词模板
replanner_prompt = ChatPromptTemplate.from_template(
    """对于给定的目标，提出一个简单的逐步计划。这个计划应该包含独立的任务，如果正确执行将得出正确的答案。不要添加任何多余的步骤。最后一步的结果应该是最终答案。确保每一步都有所有必要的信息 - 不要跳过步骤。

    你的目标是：
    {input}

    你的原计划是：
    {plan}

    你目前已完成的步骤是：
    {past_steps}

    相应地更新你的计划。如果不需要更多步骤并且可以返回给用户，那么就这样回应。如果需要，填写计划。只添加仍然需要完成的步骤。不要返回已完成的步骤作为计划的一部分。"""
)

# 使用指定的提示模板创建一个重新计划生成器，使用OpenAI的ChatGPT-4o模型
replanner = replanner_prompt | ChatOpenAI(
    model="deepseek-chat",
    base_url="https://api.deepseek.com",response_format=None).with_structured_output(Act)
from typing import Literal


# 定义一个异步主函数
async def main():
    # 定义一个异步函数，用于执行步骤
    async def execute_step(state: PlanExecute):
        plan = state["plan"]
        plan_str = "\n".join(f"{i + 1}. {step}" for i, step in enumerate(plan))
        task = plan[0]
        task_formatted = f"""对于以下计划：
{plan_str}\n\n你的任务是执行第{1}步，{task}。"""
        agent_response = await agent_executor.ainvoke(
            {"messages": [("user", task_formatted)]}
        )
        return {
            "past_steps": state["past_steps"] + [(task, agent_response["messages"][-1].content)],
        }

    # 定义一个异步函数，用于生成计划步骤
    async def plan_step(state: PlanExecute):
        plan = await planner.ainvoke({"messages": [("user", state["input"])]})
        return {"plan": plan.steps}

    # 定义一个异步函数，用于重新计划步骤
    async def replan_step(state: PlanExecute):
        output = await replanner.ainvoke(state)
        if isinstance(output.action, Response):
            return {"response": output.action.response}
        else:
            return {"plan": output.action.steps}

    # 定义一个函数，用于判断是否结束
    def should_end(state: PlanExecute) -> Literal["agent", "__end__"]:
        if "response" in state and state["response"]:
            return "__end__"
        else:
            return "agent"

    from langgraph.graph import StateGraph, START

    # 创建一个状态图，初始化PlanExecute
    workflow = StateGraph(PlanExecute)

    # 添加计划节点
    workflow.add_node("planner", plan_step)

    # 添加执行步骤节点
    workflow.add_node("agent", execute_step)

    # 添加重新计划节点
    workflow.add_node("replan", replan_step)

    # 设置从开始到计划节点的边
    workflow.add_edge(START, "planner")

    # 设置从计划到代理节点的边
    workflow.add_edge("planner", "agent")

    # 设置从代理到重新计划节点的边
    workflow.add_edge("agent", "replan")

    # 添加条件边，用于判断下一步操作
    workflow.add_conditional_edges(
        "replan",
        # 传入判断函数，确定下一个节点
        should_end,
    )

    # 编译状态图，生成LangChain可运行对象
    app = workflow.compile()
    # 将生成的图片保存到文件
    graph_png = app.get_graph().draw_mermaid_png()
    with open("plan_execute.png", "wb") as f:
        f.write(graph_png)
    # 设置配置，递归限制为50
    config = {"recursion_limit": 50}
    # 输入数据
    inputs = {"input": "2024年巴黎奥运会100米自由泳决赛冠军的家乡是哪里?请用中文答复"}
    # 异步执行状态图，输出结果
    async for event in app.astream(inputs, config=config):
        for k, v in event.items():
            if k != "__end__":
                print(v)

asyncio.run(main())