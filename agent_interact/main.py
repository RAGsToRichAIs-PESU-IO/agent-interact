from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
from groq import Groq
from agent_interact.utils.assessment_schema import generate_questions, evaluate_answer, load_or_create_index, initialize_generator_agent
from agent_interact.utils.faq_schema import load_or_create_index
from llama_index.core import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)

groq_client = None
groq_api_key = os.getenv("GROQ_API_KEY")
if not groq_api_key:
    raise ValueError("GROQ_API_KEY environment variable is not set")
groq_client = Groq(api_key=groq_api_key)

# Initialize index and agents
index = load_or_create_index()
generator_agent = initialize_generator_agent(index)

faq_index = load_or_create_index()
query_engine = faq_index.as_query_engine(similarity_top_k=10)
prompt = """You are a chatbot to market an educational course called RAGs to Rich AIs with LLM Agents. It is aimed at college students.
Using the context provided to you with the user's query, you should try to convince the student to join the course if they seem interested. 
If they do seem interested, you can tell them to register for the course at the website pesu.io/courses.
Some general information about the course is it is part of the PESU I/O program, it costs 1000 rupees to register and will run from October 7th to November 7th.
Do not force the course on anyone and always be polite with the user.
If the user asks something which is not related to the course, respond with 'I am sorry, I can only help you with stuff regarding the course RAGs to Rich AIs'."""
custom_prompt = PromptTemplate(prompt)
query_engine.update_prompts(
    {"response_synthesizer:summary_template": custom_prompt}
)

# Pydantic models
class QuestionRequest(BaseModel):
    topic: str
    question_type: str
    num_questions: int

class AnswerSubmission(BaseModel):
    question: str
    user_answer: str
    correct_answer: str

class EvaluationResponse(BaseModel):
    grade: str
    feedback: str

class AudioTranslation(BaseModel):
    translated: str

class Query(BaseModel):
    question: str

# API endpoints
@app.post("/generate_questions")
async def api_generate_questions(request: QuestionRequest):
    questions = generate_questions(generator_agent, request.topic, request.question_type, request.num_questions)
    return questions

@app.post("/evaluate_answer", response_model=EvaluationResponse)
async def api_evaluate_answer(submission: AnswerSubmission):
    evaluation = evaluate_answer(submission.question, submission.user_answer, submission.correct_answer)
    return evaluation

@app.post("/faq_answer")
async def answer_query(query: Query):
    try:
        response = query_engine.query(query.question)
        return {"question": query.question, "answer": response.response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")

@app.post("/audio", response_model=AudioTranslation)
async def audio_query(audio: UploadFile = File(...)):
    if not groq_client:
        raise HTTPException(status_code=500, detail="Groq client not initialized")
    try:
        audio_content = await audio.read()
        translation = groq_client.audio.translations.create(
            file=("recording.wav", audio_content),
            model="whisper-large-v3",
            prompt="Transcribe NCERT textbook related questions",
            response_format="json",
            temperature=0.0,
        )
        print(translation.text)
        return AudioTranslation(
            translated=translation
        )
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error processing audio query: {str(e)}"
        )

@app.get("/")
async def root():
    return {"message": "Welcome to the RAGs to Rich AIs server"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)


