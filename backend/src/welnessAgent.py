import logging

from dotenv import load_dotenv
from pathlib import Path
from datetime import datetime
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
    function_tool,
    RunContext
)
from livekit.plugins import murf, silero, google, deepgram, noise_cancellation
from livekit.plugins.turn_detector.multilingual import MultilingualModel
import json

logger = logging.getLogger("agent")

load_dotenv(".env.local")

# path
WELLNESS_LOG_PATH = Path("wellness_log.json")
class WellnessAssistant(Agent):
    def __init__(self):
        # Load previous sessions to provide context
        self.previous_sessions = self._load_wellness_log()
        self.current_session = {
            "date": datetime.now().strftime("%Y-%m-%d"),
            "time": datetime.now().strftime("%H:%M:%S"),
            "mood": None,
            "energy": None,
            "goals": [],
            "notes": None
        }
        
        # Build context from previous sessions
        history_context = self._build_history_context()
        
        super().__init__(
            instructions=f"""
                You are a supportive daily wellness companion. Your role is to conduct a brief, friendly check-in with the user about their wellbeing and daily intentions.

                {history_context}

                CONVERSATION FLOW:
                1. GREETING: Start with a warm greeting. If there's previous session data, reference it briefly.
                Example: "Good morning! Last time we talked, you mentioned feeling low energy. How are you feeling today?"

                2. MOOD & ENERGY CHECK (ask ONE question at a time):
                - "How are you feeling today?" (emotional state)
                - "What's your energy level like?" (physical/mental energy)
                - "Is anything particular on your mind or stressing you out?"
                
                3. DAILY INTENTIONS:
                - "What are 1-3 things you'd like to accomplish today?"
                - "Is there anything you want to do for yourself today?" (self-care, hobbies, rest)

                4. SUPPORTIVE FEEDBACK:
                - Offer small, actionable suggestions (NOT medical advice)
                - Examples: "Breaking that into smaller steps might help", "A 5-minute walk could boost your energy", "Remember to take breaks"
                - Keep it realistic and grounded

                5. SUMMARY & CONFIRMATION:
                - Recap: mood summary and main 1-3 goals
                - Ask: "Does this sound right?"
                - Wait for confirmation

                6. SAVE SESSION:
                - Once user confirms, call the save_wellness_checkin tool with all gathered information
                - Keep the summary brief (1-2 sentences)

                IMPORTANT GUIDELINES:
                - Ask ONE question at a time, wait for response
                - Keep responses short and conversational (2-3 sentences max)
                - Never diagnose or provide medical advice
                - Be warm but not overly cheerful
                - Use natural, friendly language
                - Don't repeat the same phrasing every time
                - Focus on practical, small steps

                WHEN TO SAVE:
                - Only call save_wellness_checkin AFTER user confirms the summary is correct
                - Include ALL collected information: mood, energy, goals, and a brief note
                """,
        )

    def _load_wellness_log(self) -> list:
        try:
            if WELLNESS_LOG_PATH.exists():
                with open(WELLNESS_LOG_PATH, "r") as f:
                    data = json.load(f)
                    logger.info(f"Loaded {len(data.get('sessions', []))} previous sessions")
                    return data.get("sessions", [])
        except Exception as e:
            logger.error(f"Error loading wellness log: {e}")
        
        return []

    def _build_history_context(self) -> str:
        if not self.previous_sessions:
            return "PREVIOUS SESSIONS: None. This is the user's first check-in."
        
        # Get last 3 sessions (most recent)
        recent = self.previous_sessions[-3:]
        context = "PREVIOUS SESSIONS (for reference only - mention naturally, don't list):\n"
        
        for session in recent:
            context += f"- {session['date']}: Mood: {session['mood']}, Energy: {session['energy']}, Goals: {', '.join(session['goals'])}\n"
        
        return context

    @function_tool
    async def save_wellness_checkin(
        self, 
        mood: str,
        energy: str,
        goals: list[str],
        notes: str
    ):
        # Update current session with data from conversation
        self.current_session.update({
            "mood": mood,
            "energy": energy,
            "goals": goals,
            "notes": notes
        })
        
        try:
            # Load existing log or create new structure
            if WELLNESS_LOG_PATH.exists():
                with open(WELLNESS_LOG_PATH, "r") as f:
                    wellness_data = json.load(f)
            else:
                wellness_data = {"sessions": []}
            
            # Add this session to the log
            wellness_data["sessions"].append(self.current_session)
            
            # Save to file with nice formatting
            with open(WELLNESS_LOG_PATH, "w") as f:
                json.dump(wellness_data, f, indent=2)
            
            logger.info(f"Wellness check-in saved: {self.current_session}")
            
            # Return a warm closing message
            goals_text = ", ".join(goals) if isinstance(goals, list) else goals
            return f"Perfect! I've logged today's check-in. Wishing you the best with {goals_text}. I'll check in with you again tomorrow!"
            
        except Exception as e:
            logger.error(f"Error saving wellness check-in: {e}")
            return "I had a small issue saving our conversation, but I've noted everything you shared. Talk to you tomorrow!"


    # To add tools, use the @function_tool decorator.
    # Here's an example that adds a simple weather tool.
    # You also have to add `from livekit.agents import function_tool, RunContext` to the top of this file
    # @function_tool
    # async def lookup_weather(self, context: RunContext, location: str):
    #     """Use this tool to look up current weather information in the given location.
    #
    #     If the location is not supported by the weather service, the tool will indicate this. You must tell the user the location's weather is unavailable.
    #
    #     Args:
    #         location: The location to look up weather information for (e.g. city name)
    #     """
    #
    #     logger.info(f"Looking up weather for {location}")
    #
    #     return "sunny with a temperature of 70 degrees."


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
        agent=WellnessAssistant(),
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
