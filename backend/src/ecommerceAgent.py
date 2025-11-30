import logging
import os
import sys
from typing import Optional

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
    function_tool,
    RunContext
)
from livekit.plugins import murf, silero, google, deepgram, noise_cancellation
from livekit.plugins.turn_detector.multilingual import MultilingualModel

# Add day9_data directory to path to import catalog
day9_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "day9_data")
if day9_path not in sys.path:
    sys.path.insert(0, day9_path)

print(f"Looking for catalog.py in: {day9_path}")
print(f"Files in day9_data: {os.listdir(day9_path) if os.path.exists(day9_path) else 'Directory not found'}")

try:
    from catalog import list_products, create_order, get_last_order, get_product_by_name
    print("✓ Successfully imported catalog functions")
except ImportError as e:
    print(f"✗ Failed to import catalog: {e}")
    raise

logger = logging.getLogger("agent")

load_dotenv(".env.local")


class ShoppingAssistant(Agent):
    def __init__(self):
        self.conversation_context = {
            "last_products_shown": [],
            "current_category": None
        }
        self.cart = []  # Shopping cart: list of {product_id, product_name, quantity, size, price}
        super().__init__(
            instructions="""
You are a friendly voice shopping assistant for Amazon online store.
Your job is to help customers browse products, manage their cart, and place orders.

CRITICAL RULES FOR BROWSING:
1. When a user asks "what hoodies do you have" or "show me mugs":
   - IMMEDIATELY call browse_catalog with the correct category
   - READ OUT the complete product list returned by the tool
   - Describe each product with: name, price, description, color, and sizes (if applicable)
   - Keep your voice natural but make sure to mention all the key details

2. Product categories (use these EXACT values):
   - For mugs: use category="mug", max_price=0, color="", keyword=""
   - For t-shirts: use category="tshirt", max_price=0, color="", keyword=""
   - For hoodies: use category="hoodie", max_price=0, color="", keyword=""
   - For bottles: use category="bottle", max_price=0, color="", keyword=""
   - For caps: use category="cap", max_price=0, color="", keyword=""
   - IMPORTANT: You MUST provide all 4 parameters. Use empty string "" for unused text filters and 0 for unused price filter.

3. When describing products:
   - Say the number (first, second, third)
   - Say the full product name
   - Mention the price clearly
   - Highlight key features from the description
   - If it's clothing, mention available sizes

4. Examples of good responses:
   User: "What hoodies do you have?"
   You should call: browse_catalog(category="hoodie", max_price=0, color="", keyword="")
   Then say: "I found 2 hoodies. First, the Black Pullover Hoodie for 1800 rupees. It's a warm black hoodie with a front pocket, available in sizes S, M, L, and XL. Second, the Grey Zip Hoodie for 2100 rupees. It's a grey hoodie with full zip closure, also available in sizes S, M, L, and XL."
   
   User: "Show me black hoodies under 2000"
   You should call: browse_catalog(category="hoodie", max_price=2000, color="black", keyword="")

5. SHOPPING CART WORKFLOW:
   - When user says "add to cart", "I'll take that", "add the first one": Use add_to_cart tool
   - When user says "remove from cart", "delete that item": Use remove_from_cart tool
   - When user says "show my cart", "what's in my cart": Use show_cart tool
   - When user says "place order", "confirm order", "checkout": Use place_order tool
   - IMPORTANT: Only place_order should actually create the order. add_to_cart just adds to cart.

6. For adding to cart:
   - Ask for size if it's clothing and they haven't specified
   - Use add_to_cart with product_name, quantity, and size
   - Confirm what was added to cart

7. For removing from cart:
   - Ask for size if it's clothing and they haven't specified, if that sized item is not in cart then say that item with that size id not in cart
   - Use remove_from_cart with product_name, and size
   - Confirm what was removed from cart
   
8. For placing orders:
   - User must explicitly say "place order", "confirm order", or "checkout"
   - Before placing, ask "Would you like to place this order?" if they haven't confirmed
   - Only call place_order when user confirms
   
9. Be conversational but thorough:
   - Don't just say "I found products" - describe them!
   - Keep responses natural but informative
   - Help users make decisions by highlighting differences
   - Always confirm cart actions

Available categories: mug, tshirt, hoodie, bottle, cap
Prices are in Indian Rupees (INR).
""",
        )

    @function_tool
    async def browse_catalog(
        self, 
        category: str,
        max_price: int,
        color: str,
        keyword: str
    ):
        """Browse the product catalog with optional filters. Call this whenever the user asks about products.
        
        Args:
            category: Product category - use exactly one of: 'mug', 'tshirt', 'hoodie', 'bottle', 'cap'. Use empty string "" to see all products.
            max_price: Maximum price in INR (Indian Rupees). Use 0 for no price limit.
            color: Product color like 'black', 'white', 'blue', 'grey', etc. Use empty string "" for any color.
            keyword: Search keyword in product name or description. Use empty string "" for no keyword search.
        """
        try:
            # Convert empty strings and 0 to None for filtering
            cat = category if category and category.strip() else None
            price = max_price if max_price > 0 else None
            col = color if color and color.strip() else None
            kw = keyword if keyword and keyword.strip() else None
            
            logger.info(f"Browsing catalog with filters - category: {cat}, max_price: {price}, color: {col}, keyword: {kw}")
            
            products = list_products(
                category=cat,
                max_price=price,
                color=col,
                keyword=kw
            )
            
            # Store products in context for reference
            self.conversation_context["last_products_shown"] = products
            if category:
                self.conversation_context["current_category"] = category
            
            logger.info(f"Found {len(products)} products")
            
            if not products:
                return "I couldn't find any products matching those criteria. Would you like to try different filters or browse another category?"
            
            # Format product list for voice response
            if len(products) == 0:
                return "I couldn't find any products matching those criteria. Would you like to try different filters or browse another category?"
            
            response = f"I found {len(products)} product{'s' if len(products) != 1 else ''}:\n\n"
            
            # Show first 3-5 products with clear numbering and details
            num_to_show = min(len(products), 5)
            for i, product in enumerate(products[:num_to_show], 1):
                response += f"{i}. {product['name']}\n"
                response += f"   Price: ₹{product['price']}\n"
                response += f"   {product['description']}\n"
                if product.get('color'):
                    response += f"   Color: {product['color']}\n"
                if product['category'] in ['tshirt', 'hoodie'] and 'attributes' in product:
                    if 'sizes' in product['attributes']:
                        response += f"   Sizes: {', '.join(product['attributes']['sizes'])}\n"
                response += "\n"
            
            if len(products) > num_to_show:
                response += f"\nI have {len(products) - num_to_show} more options. Would you like to hear about them?"
            
            logger.info(f"Formatted response with {num_to_show} products")
            return response
            
        except Exception as e:
            logger.error(f"Error browsing catalog: {e}", exc_info=True)
            return "Sorry, I had trouble accessing the catalog. Please try again."

    @function_tool
    async def add_to_cart(
        self,
        product_name: str,
        quantity: int, 
        size: str
    ):
        try:
            logger.info(f"Adding to cart - product_name: {product_name}, quantity: {quantity}, size: {size}")
            product = None
            product_id = None
            
            # Case 1: Direct product ID
            
            product = get_product_by_name(product_name)
            product_id = product['id']
            
            if not product:
                logger.warning(f"Could not resolve product name: {product_name}")
                return "I'm not sure which product you mean. Could you specify which one by saying 'the first one', 'the second one', or the product name?"
        
            logger.info(f"Resolved to product: {product['name']} (ID: {product_id})")
            
            # Create cart item
            cart_item = {
                "product_id": product_id,
                "product_name": product['name'],
                "quantity": quantity if quantity > 0 else 1,
                "price": product['price']
            }
            if size and size.strip():
                cart_item["size"] = size
            
            # Add to cart
            self.cart.append(cart_item)
            
            # Format confirmation
            response = f"Great! I've added {cart_item['quantity']} x {product['name']} "
            if size and size.strip():
                response += f"(size {size}) "
            response += f"to your cart for ₹{product['price'] * cart_item['quantity']}. "
            response += f"Your cart now has {len(self.cart)} item{'s' if len(self.cart) != 1 else ''}. "
            response += "Would you like to continue shopping or view your cart?"
            
            logger.info(f"Added to cart. Cart now has {len(self.cart)} items")
            return response
            
        except Exception as e:
            logger.error(f"Error adding to cart: {e}", exc_info=True)
            return "I'm sorry, there was an issue adding that to your cart. Could you try again?"

    @function_tool
    async def remove_from_cart(
        self,
        product_name: str,
        size: str 
    ):
        try:
            logger.info(f"Removing from cart - product_name: {product_name}, size: {size}")
            
            if not self.cart:
                return "Your cart is empty. There's nothing to remove."
            
            removed_item = None
            for idx, item in enumerate(self.cart):
                if item["product_name"] == product_name:
                    if size and item["size"] and item["size"] == size:
                        removed_item = self.cart.pop(idx)
                        break
            
            response = f"I've removed {removed_item['product_name']} from your cart. "
            if self.cart:
                response += f"You now have {len(self.cart)} item{'s' if len(self.cart) != 1 else ''} remaining in your cart."
            else:
                response += "Your cart is now empty."
            
            logger.info(f"Removed item from cart. Cart now has {len(self.cart)} items")
            return response
            
        except Exception as e:
            logger.error(f"Error removing from cart: {e}", exc_info=True)
            return "I'm sorry, there was an issue removing that item. Could you try again?"


    @function_tool
    async def show_cart(self):
        """Show the current contents of the shopping cart."""
        try:
            logger.info(f"Showing cart with {len(self.cart)} items")
            
            if not self.cart:
                return "Your cart is empty. Browse our products and add items to get started!"
            
            response = f"Your cart has {len(self.cart)} item{'s' if len(self.cart) != 1 else ''}:\n\n"
            
            total = 0
            for i, item in enumerate(self.cart, 1):
                item_total = item['price'] * item['quantity']
                total += item_total
                
                response += f"{i}. {item['quantity']} x {item['product_name']}"
                if 'size' in item:
                    response += f" (size {item['size']})"
                response += f" - ₹{item_total}\n"
            
            response += f"\nCart Total: ₹{total}\n\n"
            response += "Would you like to place your order or continue shopping?"
            
            return response
            
        except Exception as e:
            logger.error(f"Error showing cart: {e}", exc_info=True)
            return "I'm sorry, I couldn't retrieve your cart right now."


    @function_tool
    async def place_order(self):
        """Place an order with all items currently in the cart. Only call this when user explicitly confirms they want to place/checkout the order."""
        try:
            logger.info(f"Placing order with {len(self.cart)} items")
            
            if not self.cart:
                return "Your cart is empty. Please add some items before placing an order."
            
            # Build line items from cart
            line_items = []
            for cart_item in self.cart:
                line_item = {
                    "product_name": cart_item['product_name'],
                    "quantity": cart_item['quantity']
                }
                if 'size' in cart_item:
                    line_item['size'] = cart_item['size']
                line_items.append(line_item)
            
            # Create order
            order = create_order(line_items)
            
            # Format confirmation
            response = f"Excellent! Your order has been placed successfully. Order ID: {order['id']}.\n\n"
            response += "Order Summary:\n"
            for item in order['items']:
                response += f"- {item['quantity']} x {item['product_name']}"
                if 'size' in item:
                    response += f" (size {item['size']})"
                response += f" - ₹{item['item_total']}\n"
            
            response += f"\nTotal Amount: ₹{order['total']}\n"
            response += f"Status: {order['status'].title()}\n\n"
            response += "Thank you for your order! Is there anything else I can help you with?"
            
            # Clear cart after successful order
            self.cart = []
            logger.info(f"Order placed successfully: {order['id']}. Cart cleared.")
            
            return response
            
        except Exception as e:
            logger.error(f"Error placing order: {e}", exc_info=True)
            return "I'm sorry, there was an issue placing your order. Your cart is still saved. Please try again."

    @function_tool
    async def view_last_order(self):
        """View the most recent order placed. Use this when user asks about their order."""
        try:
            order = get_last_order()
            
            if not order:
                return "You haven't placed any orders yet. Would you like to browse our catalog?"
            
            # Format order summary
            response = f"Your last order, Order ID {order['id']}, was placed on {order['created_at'][:10]}. "
            response += "You ordered: "
            
            for i, item in enumerate(order['items']):
                if i > 0:
                    response += "and "
                response += f"{item['quantity']} {item['product_name']} "
                if 'size' in item:
                    response += f"in size {item['size']} "
                response += f"for {item['item_total']} rupees, "
            
            response += f"Total amount: {order['total']} rupees. Status: {order['status']}."
            
            logger.info(f"Viewed order: {order['id']}")
            return response
            
        except Exception as e:
            logger.error(f"Error viewing order: {e}", exc_info=True)
            return "Sorry, I couldn't retrieve your order information right now."


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

    # Start the session
    await session.start(
        agent=ShoppingAssistant(),
        room=ctx.room,
        room_input_options=RoomInputOptions(
            noise_cancellation=noise_cancellation.BVC(),
        ),
    )

    # Join the room and connect to the user
    await ctx.connect()


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))