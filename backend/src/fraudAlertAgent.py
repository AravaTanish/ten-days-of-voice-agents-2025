import logging
import mysql.connector
from mysql.connector import Error
import os

from dotenv import load_dotenv
from livekit.agents import (
    Agent,
    AgentSession,
    JobContext,
    JobProcess,
    MetricsCollectedEvent,
    RoomInputOptions,
    WorkerOptions,
    cli,
    metrics,
    tokenize,
)
from livekit.plugins import murf, silero, google, deepgram, noise_cancellation
from livekit.plugins.turn_detector.multilingual import MultilingualModel
from livekit.agents import function_tool, RunContext

logger = logging.getLogger("agent")

load_dotenv(".env.local")

DB_CONFIG = {
    'host': os.getenv('DB_HOST'),
    'user': os.getenv('DB_USER'),
    'password': os.getenv('DB_PASS'),
    'database': os.getenv('DB_NAME')
}

class Assistant(Agent):
    def __init__(self):
        self.fraud_case = {
            "id": None,
            "userName": None,
            "securityIdentifier": None,
            "cardEnding": None,
            "case": None,
            "transactionName": None,
            "transactionTime": None,
            "transactionCategory": None,
            "transactionSource": None,
            "verificationStatus": None,
            "outcome": None
        }
        super().__init__(
            instructions="""
        You are a professional fraud detection representative for HDFC Bank.
        Your job is to contact customers about suspicious transactions on their account.

        FRAUD CASE FLOW:
        1. Greet the customer warmly and introduce yourself as calling from HDFC Bank's Fraud Department and your name is Alex.
        
        2. Ask for their name to load their fraud case.
        
        3. Once you have the name, call load_fraud_case tool to retrieve their case details.
        
        4. Perform simple verification by asking for their Security Identifier (a 5-digit number).
        
        5. Once they provide the Security Identifier, call verify_security_identifier tool to check if it matches.
        
        6. If verification FAILS:
           - Inform them politely that the security identifier doesn't match
           - Ask if they'd like to try again (give them ONE more attempt)
           - If they fail again, end the call and ask them to contact customer service
        
        7. If verification PASSES:
           - Read out the suspicious transaction details (merchant, amount, card ending, time, category, source).
           - Ask if they made this transaction (yes/no).
        
        8. Based on their answer, call update_fraud_case tool with:
           - "safe" if they confirm the transaction
           - "fraudulent" if they deny the transaction
        
        9. Thank them and end the call professionally.

        IMPORTANT RULES:
        - Never ask for full card numbers, PINs, or passwords.
        - Use calm, professional, and reassuring language.
        - Keep responses clear and concise.
        - ALWAYS verify the Security Identifier using the verify_security_identifier tool before proceeding.
        - Do NOT proceed with transaction verification if security check fails.
        """,
        )

    @function_tool
    async def load_fraud_case(
        self, 
        user_name: str
    ):
        """Load the fraud case for the given user name from the database.
        
        Args:
            user_name: The customer's name to look up their fraud case
        """
        try:
            conn = mysql.connector.connect(**DB_CONFIG)
            cursor = conn.cursor()
            
            query = '''
                SELECT id, userName, securityIdentifier, cardEnding, case_status, 
                       transactionName, transactionTime, transactionCategory, 
                       transactionSource, verificationStatus, outcome
                FROM fraud_cases
                WHERE userName = %s AND case_status = 'pending_review'
                LIMIT 1
            '''
            
            cursor.execute(query, (user_name,))
            result = cursor.fetchone()
            
            cursor.close()
            conn.close()
            
            if result:
                self.fraud_case = {
                    "id": result[0],
                    "userName": result[1],
                    "securityIdentifier": result[2],
                    "cardEnding": result[3],
                    "case": result[4],
                    "transactionName": result[5],
                    "transactionTime": result[6],
                    "transactionCategory": result[7],
                    "transactionSource": result[8],
                    "verificationStatus": result[9],
                    "outcome": result[10]
                }
                
                logger.info(f"Loaded fraud case for {user_name}: {self.fraud_case}")
                return f"Thank you, {user_name}. I've pulled up your account. Before we proceed, I need to verify your identity. Can you please provide your 5-digit Security Identifier?"
            else:
                logger.warning(f"No fraud case found for {user_name}")
                return f"I'm sorry, I don't see any pending fraud alerts for {user_name}. Please double-check the name or contact our main customer service line."
        
        except Error as e:
            logger.error(f"Error loading fraud case: {e}")
            return "I apologize, there was an error accessing your case. Please try again or contact our fraud department directly."

    @function_tool
    async def verify_security_identifier(
        self, 
        provided_identifier: str
    ):
        """Verify if the security identifier provided by the customer matches the one on file.
        
        Args:
            provided_identifier: The 5-digit security identifier provided by the customer
        """
        if not self.fraud_case["id"]:
            logger.error("Attempted to verify security identifier before loading fraud case")
            return "I need to load your account information first. Can you please provide your name?"
        
        # Convert to string and remove any spaces or dashes
        provided = str(provided_identifier).replace(" ", "").replace("-", "")
        expected = str(self.fraud_case["securityIdentifier"]).replace(" ", "").replace("-", "")
        
        logger.info(f"Verifying security identifier - Provided: {provided}, Expected: {expected}")
        
        if provided == expected:
            logger.info(f"Security verification SUCCESS for user {self.fraud_case['userName']}")
            return f"Thank you, your identity has been verified. Now, regarding the suspicious transaction: We detected a charge from {self.fraud_case['transactionName']} on your card ending in {self.fraud_case['cardEnding']} at {self.fraud_case['transactionTime']}, categorized as {self.fraud_case['transactionCategory']} via {self.fraud_case['transactionSource']}. Did you authorize this transaction?"
        else:
            logger.warning(f"Security verification FAILED for user {self.fraud_case['userName']} - Provided: {provided}, Expected: {expected}")
            return "I'm sorry, but the Security Identifier you provided doesn't match our records. For your security, would you like to try again, or would you prefer to call our fraud department directly?"

    @function_tool
    async def update_fraud_case(
        self, 
        case_status: str,
        customer_response: str
    ):
        try:
            conn = mysql.connector.connect(**DB_CONFIG)
            cursor = conn.cursor()
            
            update_query = '''
                UPDATE fraud_cases
                SET case_status = %s,
                    verificationStatus = 'verified',
                    outcome = %s
                WHERE id = %s
            '''
            
            cursor.execute(update_query, (case_status, customer_response, self.fraud_case["id"]))
            conn.commit()
            
            cursor.close()
            conn.close()
            
            logger.info(f"Fraud case updated: ID={self.fraud_case['id']}, Status={case_status}")
            
            if case_status == "safe":
                return f"Perfect, {self.fraud_case['userName']}. I've marked this transaction as legitimate. No further action is needed. Your card ending in {self.fraud_case['cardEnding']} remains active. Thank you for confirming, and have a great day!"
            else:
                return f"I understand, {self.fraud_case['userName']}. I've marked this as fraudulent. Your card ending in {self.fraud_case['cardEnding']} has been blocked for your protection, and we'll issue you a new card within 5-7 business days. We'll also open a dispute for this transaction. Is there anything else I can help you with today?"
        
        except Error as e:
            logger.error(f"Error updating fraud case: {e}")
            return "I apologize, there was an issue updating your case. Please contact our fraud department directly at 1-800-SECURE."


def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()


async def entrypoint(ctx: JobContext):
    # Logging setup
    # Add any other context you want in all log entries here
    ctx.log_context_fields = {
        "room": ctx.room.name,
    }

    # Set up a voice AI pipeline using OpenAI, Cartesia, AssemblyAI, and the LiveKit turn detector
    session = AgentSession(
        # Speech-to-text (STT) is your agent's ears, turning the user's speech into text that the LLM can understand
        # See all available models at https://docs.livekit.io/agents/models/stt/
        stt=deepgram.STT(model="nova-3"),
        # A Large Language Model (LLM) is your agent's brain, processing user input and generating a response
        # See all available models at https://docs.livekit.io/agents/models/llm/
        llm=google.LLM(
                model="gemini-2.5-flash",
            ),
        # Text-to-speech (TTS) is your agent's voice, turning the LLM's text into speech that the user can hear
        # See all available models as well as voice selections at https://docs.livekit.io/agents/models/tts/
        tts=murf.TTS(
                voice="en-US-matthew", 
                style="Conversation",
                tokenizer=tokenize.basic.SentenceTokenizer(min_sentence_len=2),
                text_pacing=True
            ),
        # VAD and turn detection are used to determine when the user is speaking and when the agent should respond
        # See more at https://docs.livekit.io/agents/build/turns
        turn_detection=MultilingualModel(),
        vad=ctx.proc.userdata["vad"],
        # allow the LLM to generate a response while waiting for the end of turn
        # See more at https://docs.livekit.io/agents/build/audio/#preemptive-generation
        preemptive_generation=True,
    )

    # To use a realtime model instead of a voice pipeline, use the following session setup instead.
    # (Note: This is for the OpenAI Realtime API. For other providers, see https://docs.livekit.io/agents/models/realtime/))
    # 1. Install livekit-agents[openai]
    # 2. Set OPENAI_API_KEY in .env.local
    # 3. Add `from livekit.plugins import openai` to the top of this file
    # 4. Use the following session setup instead of the version above
    # session = AgentSession(
    #     llm=openai.realtime.RealtimeModel(voice="marin")
    # )

    # Metrics collection, to measure pipeline performance
    # For more information, see https://docs.livekit.io/agents/build/metrics/
    usage_collector = metrics.UsageCollector()

    @session.on("metrics_collected")
    def _on_metrics_collected(ev: MetricsCollectedEvent):
        metrics.log_metrics(ev.metrics)
        usage_collector.collect(ev.metrics)

    async def log_usage():
        summary = usage_collector.get_summary()
        logger.info(f"Usage: {summary}")

    ctx.add_shutdown_callback(log_usage)

    # # Add a virtual avatar to the session, if desired
    # # For other providers, see https://docs.livekit.io/agents/models/avatar/
    # avatar = hedra.AvatarSession(
    #   avatar_id="...",  # See https://docs.livekit.io/agents/models/avatar/plugins/hedra
    # )
    # # Start the avatar and wait for it to join
    # await avatar.start(session, room=ctx.room)

    # Start the session, which initializes the voice pipeline and warms up the models
    await session.start(
        agent=Assistant(),
        room=ctx.room,
        room_input_options=RoomInputOptions(
            # For telephony applications, use `BVCTelephony` for best results
            noise_cancellation=noise_cancellation.BVC(),
        ),
    )

    # Join the room and connect to the user
    await ctx.connect()


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))