import logging
import json
from datetime import datetime

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
        # Load catalog
        with open("shared-data/catalog.json", "r") as f:
            self.catalog = json.load(f)
        
        # Initialize cart
        self.cart = []
        
        super().__init__(
            instructions="""
        You are a friendly grocery shopping assistant for Blinkit, India's quick commerce platform.
        Your job is to help customers order groceries and food items through voice.

        AVAILABLE ITEMS:
        - Vegetables: Tomatoes, Onions, Potatoes, Capsicum
        - Dairy: Milk (Amul), Butter (Amul), Cheese Slices (Britannia), Yogurt (Mother Dairy)
        - Pantry: Pasta (Del Monte), Pasta Sauce (Kissan), Bread (Britannia), Peanut Butter (Sundrop), Rice (India Gate)
        - Snacks: Potato Chips (Lays), Chocolate Bar (Dairy Milk), Cookies (Parle-G)
        - Beverages: Coffee Powder (Nescafe), Tea (Tata Tea), Soft Drink (Coca Cola)

        CAPABILITIES:
        1. Add items to cart with quantities (e.g., "2 kg tomatoes")
        2. Handle ingredient requests (e.g., "I need ingredients for pasta" → adds pasta, sauce, tomatoes, capsicum)
        3. Show current cart when asked
        4. Remove items from cart
        5. Place final order

        RULES:
        1. Greet warmly: "Hello! Welcome to Blinkit. I can help you order groceries. What would you like today?"
        2. For each item request:
           - Confirm the item and quantity
           - Use add_to_cart tool
        3. For ingredient requests ("ingredients for pasta", "what I need for tea"):
           - Use add_ingredients tool to add multiple items at once
        4. When user asks "what's in my cart?" or "show my cart":
           - Use show_cart tool
        5. If user wants to remove something:
           - Use remove_from_cart tool
        6. When user says "I'm done", "place order", or "checkout":
           - Use place_order tool to save the final order
        7. Keep responses natural and conversational
        8. Always confirm actions taken
        """,
        )

    def _find_item(self, item_name: str):
        """Find item in catalog by name (case-insensitive)"""
        item_name_lower = item_name.lower()
        for category, items in self.catalog["categories"].items():
            for item in items:
                if item_name_lower in item["name"].lower():
                    return item
        return None

    @function_tool
    async def add_to_cart(
        self, 
        item_name: str,
        quantity: float
    ):
        """Add a single item to the cart.
        
        Args:
            item_name: The name of the item (e.g., "tomatoes", "milk", "bread")
            quantity: The quantity to add (e.g., 2 for 2 kg or 2 units)
        """
        item = self._find_item(item_name)
        if not item:
            return f"Sorry, I couldn't find '{item_name}' in our catalog. Could you try another item?"
        
        # Check if item already in cart
        for cart_item in self.cart:
            if cart_item["id"] == item["id"]:
                cart_item["quantity"] += quantity
                return f"Great! I've added {quantity} {item['unit']} of {item['name']} to your cart. You now have {cart_item['quantity']} {item['unit']} total."
        
        # Add new item
        self.cart.append({
            "id": item["id"],
            "name": item["name"],
            "price": item["price"],
            "unit": item["unit"],
            "quantity": quantity
        })
        
        return f"Added {quantity} {item['unit']} of {item['name']} (₹{item['price']}/{item['unit']}) to your cart!"

    @function_tool
    async def add_ingredients(
        self, 
        context: RunContext,
        dish_name: str
    ):
        """Add all ingredients needed for a specific dish or meal.
        
        Args:
            dish_name: The name of the dish (e.g., "pasta", "peanut butter sandwich", "tea", "breakfast")
        """
        dish_lower = dish_name.lower()
        
        # Find matching recipe
        recipe_items = None
        for recipe_name, item_ids in self.catalog["recipes"].items():
            if recipe_name in dish_lower or dish_lower in recipe_name:
                recipe_items = item_ids
                break
        
        if not recipe_items:
            return f"I don't have a recipe for '{dish_name}' in my system. Would you like to add specific items instead?"
        
        added_items = []
        for item_id in recipe_items:
            # Find item in catalog
            item = None
            for category, items in self.catalog["categories"].items():
                for cat_item in items:
                    if cat_item["id"] == item_id:
                        item = cat_item
                        break
                if item:
                    break
            
            if item:
                # Add to cart (quantity 1 for each ingredient)
                cart_item = {
                    "id": item["id"],
                    "name": item["name"],
                    "price": item["price"],
                    "unit": item["unit"],
                    "quantity": 1
                }
                
                # Check if already in cart
                found = False
                for existing in self.cart:
                    if existing["id"] == item["id"]:
                        existing["quantity"] += 1
                        found = True
                        break
                
                if not found:
                    self.cart.append(cart_item)
                
                added_items.append(item["name"])
        
        items_list = ", ".join(added_items)
        return f"Perfect! I've added ingredients for {dish_name} to your cart: {items_list}. Anything else?"

    @function_tool
    async def show_cart(self, context: RunContext):
        """Show all items currently in the cart with quantities and prices."""
        if not self.cart:
            return "Your cart is empty. What would you like to order?"
        
        cart_summary = "Here's what's in your cart:\n"
        total = 0
        
        for item in self.cart:
            item_total = item["price"] * item["quantity"]
            total += item_total
            cart_summary += f"- {item['name']}: {item['quantity']} {item['unit']} (₹{item_total})\n"
        
        cart_summary += f"\nTotal: ₹{total}"
        return cart_summary

    @function_tool
    async def remove_from_cart(
        self, 
        context: RunContext,
        item_name: str
    ):
        """Remove an item from the cart.
        
        Args:
            item_name: The name of the item to remove
        """
        item_name_lower = item_name.lower()
        
        for i, cart_item in enumerate(self.cart):
            if item_name_lower in cart_item["name"].lower():
                removed_item = self.cart.pop(i)
                return f"I've removed {removed_item['name']} from your cart."
        
        return f"I couldn't find '{item_name}' in your cart."

    @function_tool
    async def place_order(self, context: RunContext):
        """Place the final order and save it to a JSON file."""
        if not self.cart:
            return "Your cart is empty. Please add some items before placing an order."
        
        # Calculate total
        total = sum(item["price"] * item["quantity"] for item in self.cart)
        
        # Create order object
        order = {
            "order_id": f"BLK{datetime.now().strftime('%Y%m%d%H%M%S')}",
            "timestamp": datetime.now().isoformat(),
            "items": self.cart,
            "total": total,
            "status": "placed"
        }
        
        try:
            # Save to JSON file
            with open("current_order.json", "w") as f:
                json.dump(order, f, indent=2)
            
            logger.info(f"Order placed successfully: {order['order_id']}")
            
            # Clear cart
            self.cart = []
            
            return f"Awesome! Your order {order['order_id']} has been placed. Total amount: ₹{total}. Your groceries will be delivered in 10 minutes! Thank you for shopping with Blinkit!"
        
        except Exception as e:
            logger.error(f"Error placing order: {e}")
            return "I apologize, there was an issue placing your order. Please try again."


def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()


async def entrypoint(ctx: JobContext):
    ctx.log_context_fields = {
        "room": ctx.room.name,
    }

    session = AgentSession(
        stt=deepgram.STT(model="nova-3"),
        llm=google.LLM(model="gemini-2.5-flash"),
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

    usage_collector = metrics.UsageCollector()

    @session.on("metrics_collected")
    def _on_metrics_collected(ev: MetricsCollectedEvent):
        metrics.log_metrics(ev.metrics)
        usage_collector.collect(ev.metrics)

    async def log_usage():
        summary = usage_collector.get_summary()
        logger.info(f"Usage: {summary}")

    ctx.add_shutdown_callback(log_usage)

    await session.start(
        agent=Assistant(),
        room=ctx.room,
        room_input_options=RoomInputOptions(
            noise_cancellation=noise_cancellation.BVC(),
        ),
    )

    await ctx.connect()


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))