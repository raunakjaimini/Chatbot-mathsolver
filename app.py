import os
import streamlit as st
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.chains import LLMMathChain, LLMChain
from langchain.prompts import PromptTemplate
from langchain_community.utilities import WikipediaAPIWrapper
from langchain.agents.agent_types import AgentType
from langchain.agents import Tool, initialize_agent
from langchain.callbacks import StreamlitCallbackHandler

# Load environment variables
load_dotenv()

# Set up the Streamlit app
st.set_page_config(
    page_title="Math Problem Solver ",
    page_icon="🏆",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Apply custom CSS styling for a sleek, modern look
st.markdown("""
    <style>
    body {
        background-color: #2c2c2c;
        color: #f1f1f1;
        font-family: 'Arial', sans-serif;
    }
    h1, h2, h3 {
        color: #e0e0e0;
        text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.5);
    }
    .stButton>button {
        background-color: #4f4f4f;
        color: #f1f1f1;
        font-size: 18px;
        font-weight: bold;
        border-radius: 8px;
        padding: 10px 20px;
        border: 1px solid #606060;
        box-shadow: 3px 3px 6px rgba(0, 0, 0, 0.3);
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #696969;
    }
    .stTextArea>div>textarea {
        background-color: #3c3c3c;
        color: #f1f1f1;
        font-size: 16px;
        border-radius: 8px;
        padding: 12px;
        border: 1px solid #505050;
        box-shadow: inset 2px 2px 5px rgba(0, 0, 0, 0.3);
    }
    .stAlert, .stSpinner, .stInfo, .stWarning, .stSuccess {
        background-color: #3c3c3c;
        color: #f1f1f1;
        border-left: 4px solid #a0a0a0;
        padding: 10px;
        border-radius: 8px;
    }
    .css-1aumxhk {
        background-color: #3c3c3c;
        color: #f1f1f1;
        border-radius: 8px;
        border: 1px solid #505050;
    }
    .stTextInput>div>input {
        background-color: #3c3c3c;
        color: #f1f1f1;
        border-radius: 8px;
        padding: 10px;
        border: 1px solid #505050;
    }
    </style>
""", unsafe_allow_html=True)

st.title("Chat-Mate...Math Problem Solver🏆")

# Retrieve the Groq API key from the environment variable
groq_api_key = os.getenv("GROQ_API_KEY")

if not groq_api_key:
    st.error("API key not found. Please check your .env file.")
    st.stop()

# Initialize the LLM with Groq API
llm = ChatGroq(model="Gemma2-9b-It", groq_api_key=groq_api_key)

# Initializing the Wikipedia tool
wikipedia_wrapper = WikipediaAPIWrapper()
wikipedia_tool = Tool(
    name="Wikipedia",
    func=wikipedia_wrapper.run,
    description="A tool for searching information on various topics using Wikipedia."
)

# Initialize the Math tool
math_chain = LLMMathChain.from_llm(llm=llm)
calculator = Tool(
    name="Calculator",
    func=math_chain.run,
    description="A tool for solving math-related questions with detailed explanations."
)

# Prompt template for logical reasoning
prompt = """
You are an agent tasked with solving users' mathematical questions. Logically arrive at the solution and provide a detailed explanation, 
displaying it point by point for the question below:
Question: {question}
Answer:
"""

prompt_template = PromptTemplate(
    input_variables=["question"],
    template=prompt
)

# Combine all the tools into a chain
chain = LLMChain(llm=llm, prompt=prompt_template)

reasoning_tool = Tool(
    name="Reasoning Tool",
    func=chain.run,
    description="A tool for answering logic-based and reasoning questions."
)

# Initialize the assistant agent
assistant_agent = initialize_agent(
    tools=[wikipedia_tool, calculator, reasoning_tool],
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=False,
    handle_parsing_errors=True
)

if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "assistant", "content": "Hi, I'm a Math chatbot who can answer all your math questions!"}
    ]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg['content'])

# Interaction with the chatbot
question = st.text_area(
    "Enter your question:", 
    "What is the square of 3?"
)

if st.button("Find My Answer"):
    if question:
        with st.spinner("Generating response..."):
            st.session_state.messages.append({"role": "user", "content": question})
            st.chat_message("user").write(question)

            st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)
            try:
                response = assistant_agent.run(question, callbacks=[st_cb])
                st.session_state.messages.append({'role': 'assistant', "content": response})
                st.write('### Response:')
                st.success(response)
            except Exception as e:
                st.error(f"An error occurred: {e}")
    else:
        st.warning("Please enter a question.")
