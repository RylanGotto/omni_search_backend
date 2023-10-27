# generate_user_input.py

from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate
from langchain.chat_models import ChatOpenAI
from typing import Final
import json


class GenerateStructuredData:
    def __init__(self):
        self.SANITIZE_STR: Final = "```json"

        self.chat_model = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-16k")

        self.response_schemas = [
            ResponseSchema(
                name="url",
                description="The URL of the webpage information or article where data has been retrieved from.",
            ),
            ResponseSchema(
                name="title",
                description="The title of the webpage information or article where data has been retrieved from.",
            ),
            ResponseSchema(
                name="content",
                description="The content of the webpage information or article.",
            ),
            ResponseSchema(
                name="authors",
                description="The authors of the webpage information or article.",
            ),
            ResponseSchema(
                name="publish_date",
                description="The publish date of the webpage information or article.",
            ),
            ResponseSchema(
                name="images",
                description="The images of the webpage information or article if it exists.",
            ),
            ResponseSchema(
                name="sources",
                description="Sources of where the article came from",
            ),
        ]

        self.output_parser = StructuredOutputParser.from_response_schemas(
            self.response_schemas
        )

        self.format_instructions = self.output_parser.get_format_instructions()

        self.template = """
    You will be given a series of topics and articles which may contain information on the following topics. history, politics, sports, current events, business and other potential information topics.
    Find the best corresponding match on the list of standardized names.
    The closest match will be the one with the closest semantic meaning. Not just string similarity.
    

    {format_instructions}

    Do not return a string only return your final output in valid json format put commas between objects and square brackets around all objects to make a list.
    DO NOT MAKE UP FAKE DATA.
    url INPUT:
    {url}

    RESPONSE:
"""

        self.prompt = ChatPromptTemplate(
            messages=[HumanMessagePromptTemplate.from_template(template=self.template)],
            input_variables=["url"],
            partial_variables={"format_instructions": self.format_instructions},
        )

    def generate_input(self, input_data):
        _input = self.prompt.format_prompt(url=input_data, summary=input_data)
        output = self.chat_model(_input.to_messages())
        return self.to_json_sanitize_input(output.content)

    def to_json_sanitize_input(self, input_data):
        return json.loads(input_data.strip(self.SANITIZE_STR))
