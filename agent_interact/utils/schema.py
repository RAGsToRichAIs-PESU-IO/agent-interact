from typing import List
from pydantic import BaseModel, Field
from llama_index.core import (
    SimpleDirectoryReader,
    StorageContext,
    VectorStoreIndex,
    load_index_from_storage,
    Settings,
)
from llama_index.llms.groq import Groq
from llama_index.embeddings.jinaai import JinaEmbedding
from llama_index.core.agent import ReActAgent
from llama_index.core.tools import QueryEngineTool, ToolMetadata, FunctionTool
from llama_index.core.memory import ChatMemoryBuffer
import os
from dotenv import load_dotenv
import datetime

load_dotenv()

class Document(BaseModel):
    content: str = Field(..., description="The content of the document")
    metadata: dict = Field(
        default_factory=dict, description="Metadata associated with the document"
    )

class QueryResult(BaseModel):
    answer: str = Field(..., description="The answer to the query")
    source_nodes: List[str] = Field(
        ..., description="The source nodes used to generate the answer"
    )

class NCERTTextbookAssistant:
    def __init__(
        self,
        data_path: str,
        index_path: str = "index",
    ):
        self.data_path = data_path
        self.index_path = index_path
        self.system_prompt = """
        You are an NCERT (National Council of Educational Research and Training) Textbook Assistant AI designed to help students and teachers with information from NCERT textbooks. Your knowledge comes from various NCERT textbooks across different subjects and grade levels.

        Your primary goals are to:
        1. Provide accurate information from NCERT textbooks.
        2. Explain concepts in a clear and simple manner suitable for the student's grade level.
        3. Offer examples and additional context to help students understand difficult topics.
        4. Guide students to relevant chapters or sections in their textbooks for further reading.
        5. Suggest practice questions or exercises related to the topic.

        When assisting users:
        1. Always clarify the subject and grade level if not provided.
        2. Use simple language appropriate for the student's age group.
        3. Provide step-by-step explanations for problem-solving questions.
        4. Encourage critical thinking and application of concepts.
        5. If a question is outside the scope of NCERT textbooks, politely inform the user and suggest where they might find relevant information.

        Remember, your goal is to support learning and understanding of NCERT curriculum materials. Encourage students to think independently and apply their knowledge.
        """

        self.configure_settings()
        self.index = None
        self.agent = None

        # Load or create index
        self.load_or_create_index()

    def configure_settings(self):
        Settings.llm = Groq(model="llama-3.1-70b-versatile", api_key=os.getenv("GROQ_API_KEY"))
        Settings.embed_model = JinaEmbedding(
            api_key=os.getenv("JINA_API_KEY"),
            model="jina-embeddings-v2-base-en",
        )

    def load_or_create_index(self):
        if self.check_existing_index():
            self.load_index()
        else:
            self.create_index()
        self._create_agent()

    def check_existing_index(self) -> bool:
        return os.path.exists(self.index_path)

    def load_index(self):
        print("Loading existing index...")
        storage_context = StorageContext.from_defaults(persist_dir=self.index_path)
        self.index = load_index_from_storage(storage_context)
        print("Index loaded successfully.")

    def create_index(self):
        print("Creating new index...")
        documents = SimpleDirectoryReader(
            self.data_path,
            recursive=True,
        ).load_data()
        if not documents:
            raise ValueError("No documents loaded. Check the data path.")
        self.index = VectorStoreIndex.from_documents(documents)
        self.save_index()
        print("New index created and saved successfully.")

    def _create_agent(self):
        query_engine = self.index.as_query_engine(similarity_top_k=5)
        
        search_tool = QueryEngineTool(
            query_engine=query_engine,
            metadata=ToolMetadata(
                name="ncert_textbook_search",
                description="Search the NCERT textbook database for specific information. Use this when you need to reference exact content or facts from the textbooks.",
            ),
        )
        def current_affairs(query: str) -> str:
            current_date = datetime.datetime.now().strftime("%Y-%m-%d")
            return f"As an AI language model, I don't have real-time information about current affairs. My knowledge cutoff is before the current date ({current_date}). For the most up-to-date information on current affairs, I recommend checking reliable news sources or official websites."
        
        current_affairs_tool = FunctionTool.from_defaults(
            fn=current_affairs,
            name="current_affairs",
            description="Explanation about the limitations of AI models in providing real-time information on current affairs"
        )


        def get_subject_overview(subject: str, grade: int) -> str:
            # This function would provide a general overview of a subject for a specific grade
            # You would implement the logic to retrieve this information
            return f"Overview of {subject} for grade {grade}"

        subject_overview_tool = FunctionTool.from_defaults(
            fn=get_subject_overview,
            name="get_subject_overview",
            description="Get an overview of a specific subject for a given grade level"
        )

        def suggest_practice_questions(topic: str, grade: int) -> List[str]:
            # This function would suggest practice questions for a given topic and grade
            # You would implement the logic to generate or retrieve these questions
            return [f"Practice question {i} for {topic} (Grade {grade})" for i in range(1, 4)]

        practice_questions_tool = FunctionTool.from_defaults(
            fn=suggest_practice_questions,
            name="suggest_practice_questions",
            description="Suggest practice questions for a specific topic and grade level"
        )

        self.agent = ReActAgent.from_tools(
            [search_tool, subject_overview_tool, practice_questions_tool, current_affairs_tool],
            verbose=True,
            system_prompt=self.system_prompt,
            memory=ChatMemoryBuffer.from_defaults(token_limit=4096),
        )

    def query(self, query: str) -> QueryResult:
        if not self.agent:
            raise ValueError(
                "Agent not created. There might be an issue with index loading or creation."
            )
        response = self.agent.chat(query)
        return QueryResult(
            answer=response.response,
            source_nodes=[],  # Note: ReActAgent doesn't provide source nodes directly
        )

    def save_index(self):
        os.makedirs(self.index_path, exist_ok=True)
        self.index.storage_context.persist(persist_dir=self.index_path)

# Example usage
# if __name__ == "__main__":
#     ncert_assistant = NCERTTextbookAssistant(
#         data_path="./ncert_data",
#         index_path="ncert_index"
#     )

#     result = ncert_assistant.query("Can you explain the concept of photosynthesis from the 10th grade Biology textbook?")
#     print(result.answer)

#     # For subsequent runs, just initialize the assistant again:
#     ncert_assistant = NCERTTextbookAssistant(
#         data_path="./ncert_data",
#         index_path="ncert_index"
#     )
#     result = ncert_assistant.query("What are the main themes in the poem 'Fire and Ice' from the 10th grade English literature textbook?")
#     print(result.answer)