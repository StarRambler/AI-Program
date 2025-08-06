import gradio as gr
from agent import Agent

css ='''
.gradio-container { max-width:1050px !important; margin:20px auto !important;}
.message { padding:10px!important;font-size:14px!important;}
'''

agent=Agent()

demo = gr.ChatInterface(
    css =css,
    fn=lambda message, history: agent.query_from_history(message, history),
    title ='医疗问诊机器人',
    chatbot = gr.Chatbot(height=600, type="messages"),
    theme = gr.themes.Default(spacing_size='sm', radius_size='sm'),
    textbox=gr.Textbox(placeholder="在此输入您的问题",
                       container=False,
                       scale=7),
    examples =['你好，你叫什么名字?','口腔癌补助对象都包含哪些人群？','肺炎是一种什么病?','最新发表的关于脑神经的论文有哪些？'],
    submit_btn=gr.Button('提交',variant='primary'),
    type="messages"
)
if __name__ == '__main__':
    demo.launch(debug=False)