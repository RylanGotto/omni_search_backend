from langchain.chains.llm import LLMChain
from langchain.memory import ConversationBufferMemory
from langchain.agents import (
    Tool,
    AgentExecutor,
    LLMSingleActionAgent,
    AgentOutputParser,
)
from langchain.prompts import StringPromptTemplate
from langchain import LLMChain
from typing import List, Union
from langchain.schema import AgentAction, AgentFinish
import re
from dotenv import load_dotenv
from langchain.callbacks.base import AsyncCallbackHandler
from langchain.utilities import GoogleSerperAPIWrapper
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferWindowMemory
from dotenv import load_dotenv
from generate_document import GenerateDocument
from generate_structured_data import GenerateStructuredData as GSD
from callback import StreamingLLMCallbackHandler

from dotenv import load_dotenv
from langchain.embeddings import OpenAIEmbeddings
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from langchain.vectorstores import Qdrant


from langchain.embeddings.openai import OpenAIEmbeddings

# from langchain.llms import OpenAI
from langchain.text_splitter import CharacterTextSplitter
from qdrant_client import QdrantClient, models
from callback import MyCustomHandler

import asyncio

memory = ConversationBufferWindowMemory(k=10)
load_dotenv()

from typing import Any


from langchain.chains import RetrievalQAWithSourcesChain
from langchain.chains.qa_with_sources import load_qa_with_sources_chain


async def retrieveQA(q):
    llm = ChatOpenAI(
        temperature=0,
        streaming=True,
        model="gpt-3.5-turbo-16k",
    )
    retriever = create_vectorstore()
    qa_chain = load_qa_with_sources_chain(
        llm,
        chain_type="stuff",
    )

    chain = RetrievalQAWithSourcesChain(
        combine_documents_chain=qa_chain,
        retriever=retriever,
        reduce_k_below_max_tokens=True,
        max_tokens_limit=14500,
        return_source_documents=True,
    )
    chain(q)


def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    )

    text_chunks = text_splitter.split_text(text)
    return text_chunks


def create_vectorstore():
    client = QdrantClient(host="localhost", port=6333)

    # client.recreate_collection(
    #     collection_name="socketio",
    #     vectors_config=VectorParams(size=1536, distance=Distance.COSINE),
    # )
    embeddings = OpenAIEmbeddings()
    vectorstore = Qdrant(
        client=client, collection_name="socketio", embeddings=embeddings
    )
    return vectorstore


class CustomPromptTemplate(StringPromptTemplate):
    # The template to use
    template: str
    # The list of tools available
    tools: List[Tool]

    def format(self, **kwargs) -> str:
        # Get the intermediate steps (AgentAction, Observation tuples)
        # Format them in a particular way
        intermediate_steps = kwargs.pop("intermediate_steps")
        thoughts = ""
        for action, observation in intermediate_steps:
            thoughts += action.log
            thoughts += f"\nObservation: {observation}\nThought: "
        # Set the agent_scratchpad variable to that value
        kwargs["agent_scratchpad"] = thoughts
        # Create a tools variable from the list of tools provided
        kwargs["tools"] = "\n".join(
            [f"{tool.name}: {tool.description}" for tool in self.tools]
        )
        # Create a list of tool names for the tools provided
        kwargs["tool_names"] = ", ".join([tool.name for tool in self.tools])

        return self.template.format(**kwargs)


class CustomOutputParser(AgentOutputParser):
    def parse(self, llm_output: str) -> Union[AgentAction, AgentFinish]:
        # Check if agent should finish
        if "Final Answer:" in llm_output:
            return AgentFinish(
                # Return values is generally always a dictionary with a single `output` key
                # It is not recommended to try anything else at the moment :)
                return_values={"output": llm_output},
                log=llm_output,
            )
        # Parse out the action and action input

        regex = r"Action\s*\d*\s*:(.*?)\nAction\s*\d*\s*Input\s*\d*\s*:[\s]*(.*)"
        match = re.search(regex, llm_output, re.DOTALL)
        action = ""
        action_input = ""
        if match:
            action = match.group(1).strip()
            action_input = match.group(2)
        elif "Action:" in llm_output:
            action = "Chat history"
            action_input = ""

        # Return the action and action input
        return AgentAction(
            tool=action, tool_input=action_input.strip(" ").strip('"'), log=llm_output
        )


memory = ConversationBufferMemory(return_messages=True, k=3)


from dotenv import load_dotenv
from langchain.utilities import GoogleSerperAPIWrapper
import asyncio
from newspaper import Article
import asyncio

load_dotenv()

from multiprocessing.dummy import Pool as ThreadPool


structured_data_generator = GSD()
vectorstore = create_vectorstore()


def get_article(url):
    # article.download()
    # article.parse()
    # article.nlp()
    # result = {
    #     "title": article.title,
    #     "text": article.text,
    #     "summary": article.summary,
    #     "keywords": article.keywords,
    #     "source": article.url,
    # }

    # structured_data = structured_data_generator.generate_input(result)
    try:
        doc = GenerateDocument.generate(url)
        metadata = {}
        for i, k in doc.metadata.items():
            if k:
                metadata.update({i: k})

            text_chunks = get_text_chunks(doc.page_content)
            vectorstore.add_texts(text_chunks, metadatas=[metadata])
    except:
        pass


import pprint


def dump_articles(q):
    pass


import pprint


async def articles(results):
    # pprint.pprint(results)
    urls = []
    for i, k in results.items():
        if k is isinstance(k, dict) and "descriptionLink" in k:
            urls.append(k["descriptionLink"])
        # else:
        #     for j in k:
        #         if "link" in j:
        #             urls.append(j["link"])
    # print(urls)
    pool = ThreadPool(10)

    # open the urls in their own threads
    # and return the results
    pool.map(get_article, urls)

    # close the pool and wait for the work to finish
    pool.close()
    pool.join()


async def asearch(q):
    google = GoogleSerperAPIWrapper()
    results = google.results(q)
    urls = []
    for i in results.get("organic"):
        urls.append(i.get("link"))
    print(urls)
    # return {"output": output, "results": results}


async def dump_memory(q):
    return memory.chat_memory


from langchain.callbacks.manager import AsyncCallbackManagerForToolRun

from langchain.callbacks import manager
from langchain.callbacks.base import BaseCallbackHandler
from typing import Optional


from langchain.callbacks.manager import CallbackManager


def get_tool(stream_handler):
    manager = CallbackManager([MyCustomHandler(stream_handler)])
    tools = [
        Tool(
            name="Search",
            func=asearch,
            description="useful for when you need to answer questions about current events",
            coroutine=asearch,
            callbacks=manager,
        ),
        Tool(
            name="Chat history",
            func=dump_memory,
            description="useful for when you need to answer questions previous conversation",
            coroutine=dump_memory,
        ),
        # Tool(
        #     name="Articles",
        #     func=retrieveQA,
        #     description="useful check for content of articles which have been downloaded via the Search tool",
        #     coroutine=retrieveQA,
        # ),
    ]
    return tools


template_with_history = """Answer the following questions as best you can, but speaking as a professional journalist. You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat 2 times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Answer the question with the most detail possible. If you think the output should be wrapped in HTML tags for better display in the browser please add the HTML tags. For example if a question asks for a list of items, you can wrap the output in <ul> tags.
or the question asks for a table, you can wrap the output in <table> tags. or if the question asks for code wrap the output in <code> tags. DO NOT PREFIX YOUR FINAL ANSWER with "Answer: "

Previous conversation history:
{history}

New question: {input} 
{agent_scratchpad}
{format_instructions}"""

import langchain


def get_agent(websocket) -> AgentExecutor:
    """Create a AgentExecutor for question/answering."""
    stream_handler = StreamingLLMCallbackHandler(websocket)

    tools = get_tool(websocket)

    prompt_with_history = CustomPromptTemplate(
        template=template_with_history,
        tools=tools,
        # This omits the `agent_scratchpad`, `tools`, and `tool_names` variables because those are generated dynamically
        # This includes the `intermediate_steps` variable because that is needed
        input_variables=[
            "input",
            "intermediate_steps",
            "history",
            "format_instructions",
        ],
    )

    llm = ChatOpenAI(
        temperature=0,
        streaming=True,
        model="gpt-3.5-turbo-16k",
        callbacks=[stream_handler],
    )

    llm_chain = LLMChain(llm=llm, prompt=prompt_with_history)
    tool_names = [tool.name for tool in tools]

    agent = LLMSingleActionAgent(
        llm_chain=llm_chain,
        output_parser=CustomOutputParser(),
        stop=["\nObservation:"],
        allowed_tools=tool_names,
        max_tokens_limit=16385,
    )

    agent_executor = AgentExecutor.from_agent_and_tools(
        agent=agent,
        tools=tools,
        verbose=True,
        memory=memory,
    )

    return agent_executor
