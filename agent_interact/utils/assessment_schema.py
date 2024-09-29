from fastapi import HTTPException
import os
import json
from llama_index.core import (
    SimpleDirectoryReader,
    VectorStoreIndex,
    StorageContext,
    load_index_from_storage,
    Settings,
)
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.llms.groq import Groq
from llama_index.embeddings.jinaai import JinaEmbedding
from llama_index.core.agent import ReActAgent
from dotenv import load_dotenv

load_dotenv()

Settings.llm = Groq(model="llama-3.1-70b-versatile", api_key=os.getenv("GROQ_API_KEY"))
Settings.embed_model = JinaEmbedding(
    api_key=os.getenv("JINA_API_KEY"),
    model="jina-embeddings-v2-base-en",
)

PDF_PATH = "data/course_data.pdf"
INDEX_PATH = "saved_index"


# Load or create index
def load_or_create_index():
    if os.path.exists(INDEX_PATH):
        storage_context = StorageContext.from_defaults(persist_dir=INDEX_PATH)
        index = load_index_from_storage(storage_context)
    else:
        documents = SimpleDirectoryReader(input_files=[PDF_PATH]).load_data()
        index = VectorStoreIndex.from_documents(documents)
        index.storage_context.persist(persist_dir=INDEX_PATH)
    return index


# Initialize Generator Agent
def initialize_generator_agent(index):
    query_engine = index.as_query_engine(similarity_top_k=10)

    tools = [
        QueryEngineTool(
            query_engine=query_engine,
            metadata=ToolMetadata(
                name="study_material_query",
                description="Provides information from the study material PDF.",
            ),
        ),
    ]

    memory = ChatMemoryBuffer.from_defaults(token_limit=2048)

    custom_prompt = """
        You are a question generator designed to create questions based on study materials. Your task is to generate {question_type} questions about {topic} from the given context. Always use the study_material_query tool to gather relevant information before generating questions.

        Generate {num_questions} questions along with their correct answers.

        For Multiple Choice Questions (MCQs):
        - Generate questions with 4 options each (A, B, C, D).
        - Provide the correct answer for each question.

        For Subjective Questions:
        - Generate questions that require detailed answers.
        - Provide a model answer for each question.

        Your response must be a valid JSON object with the following structure:
        {{
            "questions": [
                {{
                    "type": "MCQ",
                    "question": "Question text here",
                    "options": [
                        "Option A text",
                        "Option B text",
                        "Option C text",
                        "Option D text"
                    ],
                    "correct_answer": "Correct option letter"
                }},
                {{
                    "type": "Subjective",
                    "question": "Question text here",
                    "model_answer": "Model answer text here"
                }}
            ]
        }}

        Ensure that the questions are diverse and cover different aspects of the topic. Use the context provided by the study_material_query tool to create accurate and relevant questions. Double-check that your response is a valid JSON object before submitting.
        """
    agent = ReActAgent.from_tools(tools, memory=memory, system_prompt=custom_prompt)
    return agent


# Generate questions
def generate_questions(agent, topic, question_type, num_questions):
    prompt = f"""Generate {num_questions} {question_type} questions about {topic}. 
    Your response must be a valid JSON object with a 'questions' key containing an array of question objects.
    Each question object should have 'type', 'question', and either 'options' and 'correct_answer' for MCQs, or 'model_answer' for subjective questions.
    Ensure that your response is a properly formatted JSON object. Double-check the JSON structure before submitting.
    """
    response = agent.chat(prompt)

    try:
        questions_data = json.loads(response.response)
        if not isinstance(questions_data, dict) or "questions" not in questions_data:
            raise ValueError("Response is not in the expected format")
        return questions_data
    except json.JSONDecodeError:
        raise HTTPException(
            status_code=500,
            detail="Failed to parse the generated questions. The output was not in valid JSON format.",
        )
    except ValueError as e:
        raise HTTPException(
            status_code=500, detail=f"Error in question generation: {str(e)}"
        )


# Evaluate user answer
def evaluate_answer(question, user_answer, correct_answer):
    prompt = f"""Evaluate the following user answer:
Question: {question}
User Answer: {user_answer}
Correct Answer: {correct_answer}

Provide your evaluation as a valid JSON object with the following structure:
{{
    "grade": "Grade here",
    "feedback": "Feedback text here"
}}

Ensure that your response is always a valid JSON object before submitting.
"""
    response = Settings.llm.complete(prompt)

    try:
        evaluation_data = json.loads(response.text)
        if (
            not isinstance(evaluation_data, dict)
            or "grade" not in evaluation_data
            or "feedback" not in evaluation_data
        ):
            raise ValueError("Response is not in the expected format")
        return evaluation_data
    except json.JSONDecodeError:
        raise HTTPException(
            status_code=500,
            detail="Failed to parse the evaluation. The output was not in valid JSON format.",
        )
    except ValueError as e:
        raise HTTPException(
            status_code=500, detail=f"Error in answer evaluation: {str(e)}"
        )
