import logging

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
    # function_tool,
    # RunContext
)
from livekit.plugins import murf, silero, google, deepgram, noise_cancellation
from livekit.plugins.turn_detector.multilingual import MultilingualModel
from livekit.agents import function_tool, RunContext
import json

logger = logging.getLogger("agent")

load_dotenv(".env.local")


class Assistant(Agent):
    def __init__(self):
        self.order = {
            "drinkType": None,
            "size": None,
            "milk": None,
            "extras": [],
            "name": None
        }
        super().__init__(
            instructions="""
        You are a friendly coffee shop barista for Starbucks (or any brand you want).
        Your job is to take the customer's order through voice.

        RULES:
        1. Maintain this order structure:
        {
        "drinkType": "string",
        "size": "string",
        "milk": "string",
        "extras": ["string"],
        "name": "string"
        }

        2. Ask ONE clarifying question at a time until all fields are filled:
        - What drink would you like?
        - What size?
        - Any milk preference?
        - Any extras?
        - May I have your name?

        3. Once the order is complete, call the tool save_order with all the order details.

        4. Speak like a friendly barista.
        5. Keep responses short and simple.
        """,
        )

    @function_tool
    async def save_order(
        self, 
        context: RunContext,
        drink_type: str,
        size: str,
        milk: str,
        extras: list[str],
        name: str
    ):
        """Save the complete customer order to a JSON file.
        
        Args:
            drink_type: The type of drink (e.g., latte, cappuccino, americano)
            size: The size of the drink (e.g., tall, grande, venti)
            milk: The type of milk (e.g., whole, oat, almond, soy)
            extras: List of extra items (e.g., extra shot, whipped cream, vanilla syrup)
            name: Customer's name for the order
        """
        self.order = {
            "drinkType": drink_type,
            "size": size,
            "milk": milk,
            "extras": extras,
            "name": name
        }
        
        try:
            with open("order_summary.json", "w") as f:
                json.dump(self.order, f, indent=2)
            logger.info(f"Order saved successfully: {self.order}")
            return f"Perfect! I've saved your order for {name}. Your {size} {drink_type} with {milk} milk will be ready soon!"
        except Exception as e:
            logger.error(f"Error saving order: {e}")
            return "I apologize, there was an issue saving your order. Please let me know if you'd like to try again."
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
