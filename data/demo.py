# class PlanExecute(TypedDict):
        #     input: str
        #     plan: List[str]
        #     past_steps: List[Tuple[str, str]]
        #     response: str
        #
        # class Plan(BaseModel):
        #     '''未来要执行的计划'''
        #     steps: List[str] = Field(
        #         description="需要执行不同的步骤，应该按顺序排列"
        #     )
        #
        # planner_prompt = ChatPromptTemplate.from_messages(
        #     [
        #         (
        #             "system",
        #             """对于给定目标，提出一个简单的逐步计划。这个计划应该包含独立的任务，如果正确执行将得到正确答案。不要添加多余的步骤。最后一步的结果应该是最终答案。确保每一步都有所有必要的信息。-不要跳过步骤。对于请求消息，请以json格式输出，输出格式如下：
        #
        #             {{
        #                 "steps": ["步骤1", "步骤2"]
        #                 "description"："需要执行不同的步骤，应该按顺序排列"
        #             }}""",
        #         ),
        #         ("placeholder", "{messages}"),
        #     ]
        # )
        # planner = planner_prompt | ChatOpenAI(model="qwen-plus-latest").with_structured_output(Plan)
        #
        # class Response(BaseModel):
        #     '''用户响应'''
        #     response: str
        #
        # class Act(BaseModel):
        #     '''要执行的行为'''
        #     action: Union[Response, Plan] = Field(
        #         description="要执行的行为。如果要回应用户，使用Response。如果需要进一步使用工具获取答案，使用Plan。"
        #     )
        #
        # # 创建一个新的提示词模板
        # replanner_prompt = ChatPromptTemplate.from_template(
        #     """对于给定的目标，提出一个简单的逐步计划。这个计划应该包含独立的任务，如果正确执行将得出正确的答案。不要添加任何多余的步骤。最后一步的结果应该是最终答案。确保每一步都有所有必要的信息 - 不要跳过步骤。
        #
        #     你的目标是：
        #     {input}
        #
        #     你的原计划是：
        #     {plan}
        #
        #     你目前已完成的步骤是：
        #     {past_steps}
        #
        #     相应地更新你的计划。如果不需要更多步骤并且可以返回给用户，那么就这样回应。如果需要，填写计划。只添加仍然需要完成的步骤。不要返回已完成的步骤作为计划的一部分。相应的输出要以json格式,定义如下：
        #     {{
        #         "action": {{
        #         "type": "Response" | "Plan",
        #         "response": "..."  // 如果是Response
        #         OR
        #         "steps": ["..."]    // 如果是Plan}}
        #     }}
        #     """
        # )
        #
        # # 使用指定的提示模板创建一个重新计划生成器，使用OpenAI的ChatGPT-4o模型
        # replanner = replanner_prompt | ChatOpenAI(model="qwen-plus").with_structured_output(Act)
        #
        # async def main():
        #     # 定义一个异步函数，用于执行步骤
        #     async def execute_step(state: PlanExecute):
        #         state["past_steps"] = []
        #         plan = state["plan"]
        #         plan_str = "\n".join(f"{i + 1}. {step}" for i, step in enumerate(plan))
        #         task = plan[0]
        #         task_formatted = f"""对于以下计划：
        #         {plan_str}\n\n你的任务是执行第{1}步，{task}。"""
        #         agent_response = await agent_executor.ainvoke(
        #             {"messages": [("user", task_formatted)]}
        #         )
        #
        #         # print(state)
        #         return {
        #             "past_steps": state["past_steps"] + [(task, agent_response["messages"][-1].content)],
        #         }
        #
        #     # 定义一个异步函数，用于生成计划步骤
        #     async def plan_step(state: PlanExecute):
        #         # print(state)
        #         plan = await planner.ainvoke({"messages": [("user", state["input"])]})
        #         return {"plan": plan.steps}
        #
        #     # 定义一个异步函数，用于重新计划步骤
        #     async def replan_step(state: PlanExecute):
        #         output = await replanner.ainvoke(state)
        #         if isinstance(output.action, Response):
        #             return {"response": output.action.response}
        #         else:
        #             return {"plan": output.action.steps}
        #
        #     # 定义一个函数，用于判断是否结束
        #     def should_end(state: PlanExecute) -> Literal["agent", "__end__"]:
        #         if "response" in state and state["response"]:
        #             return "__end__"
        #         else:
        #             return "agent"
        #
        #     from langgraph.graph import StateGraph, START
        #
        #     # 创建一个状态图，初始化PlanExecute
        #     workflow = StateGraph(PlanExecute)
        #
        #     # 添加计划节点
        #     workflow.add_node("planner", plan_step)
        #
        #     # 添加执行步骤节点
        #     workflow.add_node("agent", execute_step)
        #
        #     # 添加重新计划节点
        #     workflow.add_node("replan", replan_step)
        #
        #     # 设置从开始到计划节点的边
        #     workflow.add_edge(START, "planner")
        #
        #     # 设置从计划到代理节点的边
        #     workflow.add_edge("planner", "agent")
        #
        #     # 设置从代理到重新计划节点的边
        #     workflow.add_edge("agent", "replan")
        #
        #     # 添加条件边，用于判断下一步操作
        #     workflow.add_conditional_edges(
        #         "replan",
        #         # 传入判断函数，确定下一个节点
        #         should_end,
        #     )
        #
        #     # 编译状态图，生成LangChain可运行对象
        #     app = workflow.compile()
        #     # 将生成的图片保存到文件
        #     graph_png = app.get_graph().draw_mermaid_png()
        #     with open("plan_execute.png", "wb") as f:
        #         f.write(graph_png)
        #     # 设置配置，递归限制为50
        #     config = {"recursion_limit": 50}
        #     # 输入数据
        #     inputs = {"input": query}
        #     # 异步执行状态图，输出结果
        #     return await app.ainvoke(inputs, config=config)
        # return asyncio.run(main())


        ## 工作流
# prefix ="""请用中文，尽你所能回答以下问题。您可以选择使用以下一种工具: """
        # suffix ="""Begin!
        # History: {chat_history}
        # Question: {input}
        # Thought: {agent_scratchpad}
        # """
        # agent_prompt=ZeroShotAgent.create_prompt(
        #     tools=tools,
        #     prefix=prefix,
        #     suffix=suffix,
        #     input_variables=['input','agent_scratchpad','chat_history']
        #     )
        # llm = get_llm_model()
        # llm_chain = agent_prompt |llm
        # agent =ZeroShotAgent(llm_chain=llm_chain)
        # memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        # agent_chain = initialize_agent(
        #     agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
        #     tools=tools,
        #     llm=get_llm_model(),
        #     verbose=os.getenv('VERBOSE'),
        #     max_iterations=3,
        #     memory=memory
        # )
        # llm = get_llm_model()
        # llm_chain = agent_prompt |llm
        # agent =ZeroShotAgent(llm_chain=llm_chain)
        # prompt = ChatPromptTemplate.from_messages([
        #     ("system", """
        # 你是一个专业的医疗助手，能够通过工具获取信息并完成复杂医疗问题的回复。遵循以下规则：
        #
        # # 身份
        # - 名称: AI助手
        # - 风格: 友好且专业
        # - 语言: 中文
        #
        # # 工具使用
        # 当需要时，可以使用以下工具：
        # {tools}
        #
        # # 对话要求
        # 1. 保持自然对话流
        # 2. 对模糊问题主动澄清
        # 3. 长期记忆关键对话信息
        # 4. 分步思考但用自然语言回复
        #
        # # 响应格式
        # 思考过程（内部） → 工具调用（如需） → 自然语言回复
        #
        # # 当前会话
        # 工具状态：{tools}
        # 对话历史：{chat_history}
        # 用户输入：{input}
        #     """),
        #     MessagesPlaceholder(variable_name="agent_scratchpad")  # 保留给Agent的思考过程
        # ])
        # memory = ConversationBufferWindowMemory(memory_key="chat_history", return_messages=True)
        # llm = get_llm_model()
        # agent = create_conversational_react_agent(
        #     llm=llm,
        #     tools=tools,
        #     messages_modifier=prompt,
        #     memory=memory
        # )
        # agent_chain=AgentExecutor.from_agent_and_tools(
        #     agent = agent,
        #     tools = tools,
        #     memory = memory,
        #     verbose =os.getenv('VERBOSE')
        # )
        # return agent_chain.invoke({'input':query})