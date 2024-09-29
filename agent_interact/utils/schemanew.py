import streamlit as st
import os
import json
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, StorageContext, load_index_from_storage, Settings
# from llama_index.llms.openai import OpenAI
from llama_index.core.tools import QueryEngineTool, ToolMetadata, FunctionTool
# from llama_index.agent.openai import OpenAIAgent
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.llms.groq import Groq
from llama_index.embeddings.jinaai import JinaEmbedding
from llama_index.core.agent import ReActAgent
from dotenv import load_dotenv

load_dotenv()

Settings.llm = Groq(model="llama-3.2-90b-text-preview", api_key=os.getenv("GROQ_API_KEY"), response_format)
Settings.embed_model = JinaEmbedding(api_key=os.getenv("JINA_API_KEY"),
            model="jina-embeddings-v2-base-en",
        )
# Set up OpenAI API key

# Define the path to your PDF file
PDF_PATH = "/Users/samarth/Documents/Samarth/io/agent-interact/agent_interact/data/RAGs_to_Rich_AIs_CoursePlan.pdf"  # Replace with the actual path to your PDF

# Define the path to save the index
INDEX_PATH = "saved_index"

# Load PDF document and create index (if not already saved)
def load_or_create_index():
    if os.path.exists(INDEX_PATH):
        # Load the existing index
        storage_context = StorageContext.from_defaults(persist_dir=INDEX_PATH)
        index = load_index_from_storage(storage_context)
    else:
        # Create a new index
        documents = SimpleDirectoryReader(input_files=[PDF_PATH]).load_data()
        # service_context = ServiceContext.from_defaults(llm=OpenAI(temperature=0, model="gpt-4"))
        index = VectorStoreIndex.from_documents(documents)
        # Save the index
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
                    "options": {{
                        "A": "Option A text",
                        "B": "Option B text",
                        "C": "Option C text",
                        "D": "Option D text"
                    }},
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
    # llm = OpenAI(temperature=0.7, model="gpt-4")
    agent = ReActAgent.from_tools(tools, memory=memory, system_prompt=custom_prompt)
    return agent

# Initialize Evaluator Agent
def initialize_evaluator_agent():
    custom_prompt = """
        You are an evaluator agent designed to assess user responses to questions. Your task is to compare the user's answer with the correct answer and provide a grade along with feedback.

        For MCQs:
        - Check if the user's selected option matches the correct answer.
        - Provide a grade (Correct/Incorrect) and brief feedback.

        For Subjective Questions:
        - Compare the user's answer with the model answer.
        - Provide a grade (Excellent/Good/Fair/Poor) based on the accuracy and completeness of the response.
        - Offer constructive feedback, highlighting strengths and areas for improvement.

        Your response must be a valid JSON object with the following structure:
        {{
            "grade": "Grade here",
            "feedback": "Feedback text here"
        }}

        Ensure that your response is always a valid JSON object before submitting.
        """

    # llm = OpenAI(temperature=0.3, model="gpt-4")
    agent = ReActAgent.from_tools([], memory=None, system_prompt=custom_prompt)
    return agent

# Initialize Main Agent
def initialize_main_agent(generator_agent, evaluator_agent):
    custom_prompt = """
    You are the main coordinator agent for a study material Q&A system. Your role is to manage the interaction between the user, the generator agent, and the evaluator agent.

    Your tasks include:
    1. Accepting user input for the topic and question type.
    2. Requesting questions from the generator agent.
    3. Presenting questions to the user and collecting their answers.
    4. Sending user answers to the evaluator agent for assessment.
    5. Presenting the evaluation results to the user.

    Ensure a smooth flow of interaction and provide clear instructions to the user at each step.
    """

    tools = [
        FunctionTool.from_defaults(
            name="generator_agent",
            description="Generates questions based on the study material.",
            fn=generator_agent.chat,
        ),
        FunctionTool.from_defaults(
            name="evaluator_agent",
            description="Evaluates user responses to questions.",
            fn=evaluator_agent.chat,
        ),
    ]

    # llm = OpenAI(temperature=0.5, model="gpt-4")
    agent = ReActAgent.from_tools(tools, memory=None, system_prompt=custom_prompt)
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
        if not isinstance(questions_data, dict) or 'questions' not in questions_data:
            raise ValueError("Response is not in the expected format")
        return questions_data
    except json.JSONDecodeError:
        st.error("Failed to parse the generated questions. The output was not in valid JSON format.")
        st.text("Raw output:")
        st.text(response.response)
        return None
    except ValueError as e:
        st.error(f"Error in question generation: {str(e)}")
        st.text("Raw output:")
        st.text(response.response)
        return None
    
# Evaluate user answer
def evaluate_answer(agent, question, user_answer, correct_answer):
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
    response = agent.chat(prompt)
    
    try:
        evaluation_data = json.loads(response.response)
        if not isinstance(evaluation_data, dict) or 'grade' not in evaluation_data or 'feedback' not in evaluation_data:
            raise ValueError("Response is not in the expected format")
        return evaluation_data
    except json.JSONDecodeError:
        st.error("Failed to parse the evaluation. The output was not in valid JSON format.")
        st.text("Raw output:")
        st.text(response.response)
        return None
    except ValueError as e:
        st.error(f"Error in answer evaluation: {str(e)}")
        st.text("Raw output:")
        st.text(response.response)
        return None

# Streamlit UI
st.title("Study Material Q&A Generator and Evaluator")

# Load or create index
index = load_or_create_index()
generator_agent = initialize_generator_agent(index)
evaluator_agent = initialize_evaluator_agent()
main_agent = initialize_main_agent(generator_agent, evaluator_agent)

st.success("Agents initialized successfully!")

# User inputs
topic = st.text_input("Enter the topic for questions (leave blank for the entire syllabus)")
question_type = st.selectbox("Select question type", ["MCQ", "Subjective"])
num_questions = st.number_input("Number of questions to generate", min_value=1, max_value=10, value=5)

if st.button("Generate Questions"):
    with st.spinner("Generating questions..."):
        questions = generate_questions(generator_agent, topic, question_type, num_questions)
        if questions:
            st.session_state.questions = questions
            st.session_state.current_question = 0
            st.session_state.evaluations = []
            st.success("Questions generated successfully!")
        else:
            st.error("Failed to generate questions. Please try again.")

if 'questions' in st.session_state:
    if st.session_state.current_question < len(st.session_state.questions['questions']):
        question = st.session_state.questions['questions'][st.session_state.current_question]
        st.subheader(f"Question {st.session_state.current_question + 1}")
        st.write(question['question'])
        
        if question['type'] == 'MCQ':
            print(question)
            user_answer = st.radio("Select your answer:", list(question['options'].keys()), format_func=lambda x: question['options'][x])
        else:
            user_answer = st.text_area("Your answer:")
        
        if st.button("Submit Answer"):
            with st.spinner("Evaluating your answer..."):
                correct_answer = question['correct_answer'] if question['type'] == 'MCQ' else question['model_answer']
                evaluation = evaluate_answer(evaluator_agent, question['question'], user_answer, correct_answer)
                if evaluation:
                    st.session_state.evaluations.append({
                        'question': question['question'],
                        'user_answer': user_answer,
                        'correct_answer': correct_answer,
                        'evaluation': evaluation
                    })
                else:
                    st.error("Failed to evaluate the answer. Please try again.")
            
            st.session_state.current_question += 1
            if st.session_state.current_question < len(st.session_state.questions['questions']):
                st.experimental_rerun()
    
    if st.session_state.current_question >= len(st.session_state.questions['questions']):
        st.success("You have completed all the questions!")
        st.subheader("Summary of All Questions and Evaluations")
        for i, eval_data in enumerate(st.session_state.evaluations):
            st.write(f"Question {i+1}: {eval_data['question']}")
            st.write(f"Your Answer: {eval_data['user_answer']}")
            st.write(f"Correct Answer: {eval_data['correct_answer']}")
            st.write(f"Grade: {eval_data['evaluation']['grade']}")
            st.write(f"Feedback: {eval_data['evaluation']['feedback']}")
            st.write("---")

# Reset button
if st.button("Start Over"):
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    st.experimental_rerun()