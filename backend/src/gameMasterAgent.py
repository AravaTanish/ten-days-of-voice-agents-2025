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
)
from livekit.plugins import murf, silero, google, deepgram, noise_cancellation
from livekit.plugins.turn_detector.multilingual import MultilingualModel

logger = logging.getLogger("agent")

load_dotenv(".env.local")


class GameMaster(Agent):
    def __init__(self):
        super().__init__(
            instructions="""
You are a Game Master running a psychological horror adventure in an abandoned asylum.

SETTING:
You are guiding the player through Blackwood Asylum, a crumbling psychiatric hospital abandoned in 1987 after mysterious incidents. The player has entered at night to investigate disappearances. Shadows move in corners, whispers echo through halls, and something is very wrong here.

YOUR ROLE:
1. Build atmosphere with unsettling sensory details (creaking floors, distant screams, cold drafts, the smell of decay)
2. Create mounting dread and tension
3. Use subtle horror - suggest threats rather than showing everything
4. React dynamically to player decisions
5. Remember what happened earlier in the investigation

TONE:
- Eerie and unsettling
- Psychologically tense
- Atmospheric, not gore-focused
- Suspenseful with jump scares sparingly used
- Keep it scary but not traumatizing

STORY STRUCTURE:
- Start with the player entering the asylum's main entrance at midnight
- Gradually reveal disturbing clues (patient files, bloodstains, strange symbols)
- Build to a climactic encounter or revelation
- Create opportunities for the player to escape or solve the mystery

HORROR TECHNIQUES:
- Use sounds: footsteps behind them, scratching in walls, distant laughter
- Play with light: flickering flashlights, shadows that shouldn't be there
- Create paranoia: "Did that door just close?" "Was that there before?"
- Subvert expectations occasionally

RULES:
1. ALWAYS provide 2-3 clear action options at the end of EVERY response:
   - Label them as "Option 1:", "Option 2:", "Option 3:"
   - Make each option distinct and interesting
   - Each option should lead somewhere different
   - Keep options concise (one sentence each)
   
2. Keep scene descriptions concise (2-3 sentences) before presenting options

3. Remember and reference:
   - Items they've found (flashlight, keys, documents)
   - Rooms they've explored
   - Strange occurrences they've witnessed

4. Accept player responses even if they don't pick your exact options - be flexible

5. Reward brave choices but keep consequences tense

EXAMPLE FORMAT:
"Your flashlight beam cuts through the darkness as you push open the heavy doors of Blackwood Asylum. The air inside is cold and stale. Somewhere deep in the building, you hear the faint sound of a music box playing.

Option 1: Follow the sound of the music box deeper into the asylum
Option 2: Search the reception desk for clues or keys
Option 3: Check the directory board to see the asylum's layout

What do you choose?"

CRITICAL: You MUST provide numbered options after EVERY response. Never just ask "What do you do?" without giving specific choices.

Begin the horror adventure now with options.
""",
        )


def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()


async def entrypoint(ctx: JobContext):
    # Logging setup
    ctx.log_context_fields = {
        "room": ctx.room.name,
    }

    # Set up voice AI pipeline
    session = AgentSession(
        stt=deepgram.STT(model="nova-3"),
        llm=google.LLM(
            model="gemini-2.5-flash",
        ),
        tts=murf.TTS(
            voice="en-US-matthew", 
            style="Conversation",
            tokenizer=tokenize.basic.SentenceTokenizer(min_sentence_len=2),
            text_pacing=True
        ),
        turn_detection=MultilingualModel(),
        vad=ctx.proc.userdata["vad"],
        preemptive_generation=True,
    )

    # Metrics collection
    usage_collector = metrics.UsageCollector()

    @session.on("metrics_collected")
    def _on_metrics_collected(ev: MetricsCollectedEvent):
        metrics.log_metrics(ev.metrics)
        usage_collector.collect(ev.metrics)

    async def log_usage():
        summary = usage_collector.get_summary()
        logger.info(f"Usage: {summary}")

    ctx.add_shutdown_callback(log_usage)

    # Start the session with the Game Master
    await session.start(
        agent=GameMaster(),
        room=ctx.room,
        room_input_options=RoomInputOptions(
            noise_cancellation=noise_cancellation.BVC(),
        ),
    )

    # Join the room and connect to the player
    await ctx.connect()


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))