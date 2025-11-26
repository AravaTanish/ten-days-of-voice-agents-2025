import logging
import json

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


class Assistant(Agent):
    def __init__(self):
        # Store lead information instead of order
        self.lead = {
            "name": None,
            "company": None,
            "email": None,
            "role": None,
            "use_case": None,
            "team_size": None,
            "timeline": None
        }
        
        # Load FAQ data (preload company information)
        self.faq_data = self.load_faq()
        
        super().__init__(
            instructions=f"""
        You are a friendly Sales Development Representative (SDR) for Razorpay, India's leading payment gateway and financial services company.
        
        COMPANY INFORMATION:
        {self.faq_data}
        
        YOUR ROLE:
        1. Greet the visitor warmly and ask what brought them here.
        2. Understand their business needs and what they're working on.
        3. Answer their questions about Razorpay using the FAQ information provided above.
        4. Naturally collect lead information during the conversation.
        5. When they indicate they're done (e.g., "that's all", "I'm done", "thanks, bye"), call save_lead to store their information.
        
        LEAD FIELDS TO COLLECT (ask naturally during conversation):
        - Name
        - Company name
        - Email address
        - Role/Job title
        - Use case (what they want to use Razorpay for)
        - Team size
        - Timeline (now / soon / later)
        
        CONVERSATION STYLE:
        - Be warm, professional, and consultative
        - Ask open-ended questions to understand their needs
        - Don't interrogate - weave questions naturally into the conversation
        - Use the FAQ to answer product/pricing/company questions accurately
        - Don't make up information not in the FAQ
        - Keep responses conversational and concise
        
        When answering questions:
        - Base answers strictly on the FAQ information provided
        - If asked about something not in the FAQ, politely say you'll have someone from the team follow up
        - Focus on understanding their specific needs and use case
        """,
        )
    
    def load_faq(self):
        faq = """
        RAZORPAY - COMPANY FAQ
        
        WHAT WE DO:
        Razorpay is India's leading payment gateway that enables businesses to accept, process, and disburse payments online. We provide a full-stack financial solution including payment gateway, banking, lending, and business banking.
        
        PRODUCTS:
        - Payment Gateway: Accept payments via UPI, cards, netbanking, wallets (100+ payment methods)
        - Payment Links: Create and share payment links without coding
        - Payment Pages: Collect payments via customizable web pages
        - Subscriptions: Automate recurring billing
        - Smart Collect: Virtual accounts for payment collection
        - Route: Split payments and manage marketplace transactions
        - RazorpayX: Business banking with current accounts and payouts
        
        WHO IS THIS FOR:
        - Startups and small businesses
        - E-commerce platforms
        - SaaS companies
        - Marketplaces
        - Educational institutions
        - Freelancers and creators
        
        PRICING:
        - Standard pricing: 2% per transaction (no setup fee, no annual fee)
        - International cards: 3%
        - Custom pricing available for high-volume businesses
        - Free to start, pay only for successful transactions
        
        FREE TIER:
        - No setup fees or annual maintenance charges
        - Pay only for successful transactions
        - Free developer tools and APIs
        
        KEY FEATURES:
        - Instant activation and onboarding
        - Dashboard for tracking payments and analytics
        - Developer-friendly APIs and plugins
        - Auto-reconciliation
        - 24/7 customer support
        - PCI DSS compliant and secure
        
        INTEGRATION:
        - Easy integration with major platforms (Shopify, WooCommerce, etc.)
        - REST APIs for custom integration
        - Mobile SDKs for Android and iOS
        - No-code solutions available
        
        SUPPORT:
        - 24/7 customer support via chat, email, phone
        - Dedicated account manager for enterprise clients
        - Comprehensive documentation and resources
        """
        return faq
    
    @function_tool
    async def save_lead(
        self, 
        context: RunContext,
        name: str,
        company: str,
        email: str,
        role: str,
        use_case: str,
        team_size: str,
        timeline: str
    ):
        self.lead = {
            "name": name,
            "company": company,
            "email": email,
            "role": role,
            "use_case": use_case,
            "team_size": team_size,
            "timeline": timeline
        }
        
        try:
            with open("lead_summary.json", "w") as f:
                json.dump(self.lead, f, indent=2)
            logger.info(f"Lead saved successfully: {self.lead}")
            
            # Create verbal summary
            summary = f"Perfect! Let me summarize what we discussed. You're {name} from {company}, working as a {role}. "
            summary += f"You're looking to use Razorpay for {use_case} with a team of {team_size} people, "
            summary += f"and your timeline is {timeline}. "
            summary += "I've saved all your details. Someone from our team will reach out to you shortly. Thank you for your interest in Razorpay!"
            
            return summary
        except Exception as e:
            logger.error(f"Error saving lead: {e}")
            return "I apologize, there was an issue saving your information. Please let me know if you'd like to try again."
    
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
    # Hint - You can also preload and preprocess FAQ data here for better performance
    # For example: proc.userdata["faq"] = load_and_process_faq()


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
            tokenizer=tokenize.basic.SentenceTokenizer(
                min_sentence_len=15,  # Longer sentences
            )
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