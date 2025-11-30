import json
import os
from datetime import datetime
from typing import Optional

# Get the directory where this script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PRODUCTS_FILE = os.path.join(SCRIPT_DIR, "products.json")
ORDERS_FILE = os.path.join(SCRIPT_DIR, "orders.json")


def load_products() -> list[dict]:
    """Load products from JSON file."""
    try:
        with open(PRODUCTS_FILE, "r") as f:
            products = json.load(f)
            print(f"✓ Loaded {len(products)} products from {PRODUCTS_FILE}")
            return products
    except FileNotFoundError:
        print(f"✗ Products file not found: {PRODUCTS_FILE}")
        return []
    except Exception as e:
        print(f"✗ Error loading products: {e}")
        return []


def load_orders() -> list[dict]:
    """Load orders from JSON file."""
    try:
        with open(ORDERS_FILE, "r") as f:
            data = json.load(f)
            # Ensure it's a list, not a dict
            if isinstance(data, list):
                print(f"Loaded {len(data)} orders from {ORDERS_FILE}")
                return data
            else:
                print(f"Orders file contains {type(data)}, resetting to empty list")
                return []
    except FileNotFoundError:
        print(f"Orders file not found, creating new one")
        return []
    except Exception as e:
        print(f"Error loading orders: {e}")
        return []


def save_orders(orders: list[dict]) -> None:
    """Save orders to JSON file."""
    with open(ORDERS_FILE, "w") as f:
        json.dump(orders, f, indent=2)


def list_products(
    category: Optional[str] = None,
    max_price: Optional[int] = None,
    color: Optional[str] = None,
    keyword: Optional[str] = None
) -> list[dict]:
    """
    List products with optional filters.
    
    Args:
        category: Filter by category (e.g., 'mug', 'tshirt', 'hoodie')
        max_price: Filter by maximum price
        color: Filter by color
        keyword: Search in name or description
    
    Returns:
        List of matching products
    """
    products = load_products()
    print(f"Starting with {len(products)} total products")
    filtered = products
    
    if category:
        print(f"Filtering by category: {category}")
        filtered = [p for p in filtered if p.get("category", "").lower() == category.lower()]
        print(f"After category filter: {len(filtered)} products")
    
    if max_price:
        print(f"Filtering by max_price: {max_price}")
        filtered = [p for p in filtered if p.get("price", 0) <= max_price]
        print(f"After price filter: {len(filtered)} products")
    
    if color:
        print(f"Filtering by color: {color}")
        filtered = [p for p in filtered if p.get("color", "").lower() == color.lower()]
        print(f"After color filter: {len(filtered)} products")
    
    if keyword:
        print(f"Filtering by keyword: {keyword}")
        keyword_lower = keyword.lower()
        filtered = [
            p for p in filtered
            if keyword_lower in p.get("name", "").lower()
            or keyword_lower in p.get("description", "").lower()
        ]
        print(f"After keyword filter: {len(filtered)} products")
    
    print(f"Returning {len(filtered)} products")
    return filtered


def get_product_by_name(product_name: str) -> Optional[dict]:
    """Get a specific product by name."""
    products = load_products()
    for product in products:
        if product["name"] == product_name:
            return product
    return None


def create_order(line_items: list[dict]) -> dict:
    """
    Create a new order.
    
    Args:
        line_items: List of items like [{"product_id": "mug-001", "quantity": 2, "size": "M"}]
    
    Returns:
        Created order object
    """
    orders = load_orders()
    
    # Generate order ID
    order_id = f"order-{len(orders) + 1:04d}"
    
    # Calculate total and build order items
    total = 0
    order_items = []
    
    for item in line_items:
        product = get_product_by_name(item["product_name"])
        if not product:
            continue
        
        quantity = item.get("quantity", 1)
        item_total = product["price"] * quantity
        total += item_total
        
        order_item = {
            "product_id": product["id"],
            "product_name": product["name"],
            "quantity": quantity,
            "price": product["price"],
            "item_total": item_total
        }
        
        # Add optional attributes like size
        if "size" in item:
            order_item["size"] = item["size"]
        
        order_items.append(order_item)
    
    # Create order object
    order = {
        "id": order_id,
        "items": order_items,
        "total": total,
        "currency": "INR",
        "created_at": datetime.now().isoformat(),
        "status": "confirmed"
    }
    
    # Save order
    orders.append(order)
    save_orders(orders)
    
    return order


def get_last_order() -> Optional[dict]:
    """Get the most recent order."""
    orders = load_orders()
    if orders and len(orders) > 0:
        return orders[-1]
    return None


def get_order_by_id(order_id: str) -> Optional[dict]:
    """Get a specific order by ID."""
    orders = load_orders()
    for order in orders:
        if order["id"] == order_id:
            return order
    return None