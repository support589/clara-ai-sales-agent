"""
Clara AI Sales Agent - FastAPI Backend
This is the core backend that handles Twilio webhooks and manages the AI conversation flow.
"""

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from twilio.rest import Client
from twilio.twiml.voice_response import VoiceResponse
from openai import OpenAI
import os
from dotenv import load_dotenv
import json
from datetime import datetime

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI(title="Clara AI Sales Agent")

# Initialize Twilio client
TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN")
TWILIO_PHONE_NUMBER = os.getenv("TWILIO_PHONE_NUMBER")

twilio_client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)

# Initialize OpenAI client
# openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY")) # Moved inside endpoint to avoid proxy error during startup

# Store conversation state (in production, use a database)
conversations = {}

# The 5 qualification questions
QUALIFICATION_QUESTIONS = [
    "Am I speaking with the decision-maker for the property?",
    "What is the main reason you are considering selling the property right now?",
    "If we were to make a fast, cash offer, what is the lowest price you would be willing to accept?",
    "Can you describe the current condition of the property? Does it need any major repairs like roof, foundation, or HVAC?",
    "How quickly are you looking to close the sale?"
]

SYSTEM_PROMPT = """You are Clara, an AI Sales Agent for a real estate wholesaling company, acting as a 24/7 pre-qualification filter. Your persona is professional, empathetic, and highly focused on gathering the five key pillars of information to determine if the seller is a 'motivated seller' ready for a cash offer. Your ultimate goal is to qualify the lead and transfer only the most potential and for-sure sales to a human agent.

**The Five Pillars of Qualification (Must be answered):**
1.  **Decision Maker:** Am I speaking with the decision-maker for the property?
2.  **Motivation (The 'Why'):** What is the main reason you are considering selling the property right now?
3.  **Price Expectation:** If we were to make a fast, cash offer, what is the lowest price you would be willing to accept?
4.  **Property Condition:** Can you describe the current condition of the property? Does it need any major repairs like roof, foundation, or HVAC?
5.  **Timeline/Urgency:** How quickly are you looking to close the sale?

**Call Transfer Condition (Must be met before transfer):**
You must transfer the call to a human agent ONLY when the seller has provided answers to all five pillars AND has expressed a clear willingness to consider a fast, cash offer (e.g., a low price expectation and a fast timeline). If the seller is not motivated or is unwilling to discuss price/condition, politely end the call.

**Conversation Flow:**
- You must ask the questions one by one, in a natural, conversational way.
- After each answer, acknowledge what they said and transition smoothly to the next question.
- If they don't answer clearly, politely ask them to clarify.
- Once all 5 questions are answered, thank them and let them know a specialist will contact them soon.

**Initial Greeting:**
"Hello! This is Clara, an AI assistant from our real estate team. I'm here to gather some quick information about your property to see if it qualifies for a fast, cash offer. Do you have a few minutes to answer some questions?"

Remember: You are Clara, a gentle, professional, and highly effective pre-qualification AI agent."""


@app.get("/")
async def root():
    """Health check endpoint"""
    return {"status": "Clara AI Sales Agent is running", "version": "1.0.0"}


@app.post("/incoming-call")
async def incoming_call(request: Request):
    """
    Webhook endpoint for incoming Twilio calls.
    This endpoint is called when someone calls the Twilio number.
    """
    # Get the caller's phone number
    form_data = await request.form()
    caller_phone = form_data.get("From")
    call_sid = form_data.get("CallSid")
    
    # Initialize conversation state for this call
    if call_sid not in conversations:
        conversations[call_sid] = {
            "phone": caller_phone,
            "question_index": 0,
            "answers": [],
            "started_at": datetime.now().isoformat()
        }
    
    # Create a Twilio response
    response = VoiceResponse()
    
    # Greet the caller and ask the first question
    greeting = "Hello! This is Clara, an AI assistant from our real estate team. I'm calling to help qualify your property for sale. Let's get started with a few quick questions. First, are you the sole owner of the property?"
    
    response.say(greeting, voice="alice")
    
    # Gather the response (record up to 10 seconds of speech)
    response.gather(
        num_digits=1,
        action=f"/process-response/{call_sid}",
        method="POST",
        timeout=10,
        speech_timeout="auto",
        max_speech_time=30
    )
    
    return JSONResponse(content=str(response), media_type="application/xml")


@app.post("/process-response/{call_sid}")
async def process_response(call_sid: str, request: Request):
    """
    Webhook endpoint to process the lead's response to each question.
    """
    form_data = await request.form()
    speech_result = form_data.get("SpeechResult", "")
    
    if call_sid not in conversations:
        return JSONResponse(content={"error": "Call not found"}, status_code=404)
    
    conversation = conversations[call_sid]
    
    # Store the answer
    conversation["answers"].append({
        "question": QUALIFICATION_QUESTIONS[conversation["question_index"]],
        "answer": speech_result
    })
    
    # Use GPT-4o to generate the next response
    try:
        # Build the conversation history for GPT
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
        ]
        
        # Add previous Q&A to context
        for qa in conversation["answers"]:
            messages.append({"role": "user", "content": qa["answer"]})
            messages.append({"role": "assistant", "content": f"Thank you for that answer. {qa['question']}"})
        
        # Initialize OpenAI client here to avoid global proxy issues
        openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        
        # Get GPT's response
        gpt_response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            temperature=0.7,
            max_tokens=150
        )
        
        clara_response = gpt_response.choices[0].message.content
        
    except Exception as e:
        print(f"Error calling GPT: {e}")
        clara_response = "I apologize, there was a technical issue. Let me continue with the next question."
    
    # Move to next question
    conversation["question_index"] += 1
    
    # Create Twilio response
    response = VoiceResponse()
    
    # If we have more questions, ask the next one
    if conversation["question_index"] < len(QUALIFICATION_QUESTIONS):
        next_question = QUALIFICATION_QUESTIONS[conversation["question_index"]]
        response.say(clara_response, voice="alice")
        response.say(next_question, voice="alice")
        
        response.gather(
            num_digits=1,
            action=f"/process-response/{call_sid}",
            method="POST",
            timeout=10,
            speech_timeout="auto",
            max_speech_time=30
        )
    else:
        # All questions answered - end call
        closing_message = "Thank you so much for answering all my questions! A specialist from our team will contact you within the next 24 hours to discuss your property. Have a great day!"
        response.say(closing_message, voice="alice")
        response.hangup()
        
        # Log the qualified lead
        print(f"\n✓ QUALIFIED LEAD: {conversation['phone']}")
        print(f"Answers: {json.dumps(conversation['answers'], indent=2)}")
        
        # TODO: In production, save this to a database and send notification to your husband
    
    return JSONResponse(content=str(response), media_type="application/xml")


@app.get("/leads")
async def get_leads():
    """
    Endpoint to retrieve all qualified leads (for the dashboard).
    """
    return {
        "leads": conversations,
        "total": len(conversations)
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
