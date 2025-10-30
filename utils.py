import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
import re

def format_currency(amount: float, currency: str = "INR") -> str:
    """Format amount as currency string"""
    if pd.isna(amount) or amount is None:
        return "₹0.00"
    
    currency_symbols = {
        "INR": "₹"
    }
    
    symbol = currency_symbols.get(currency, "₹")
    return f"{symbol}{amount:,.2f}"

def format_dual_currency(amount_inr: float) -> str:
    """Format amount in INR only"""
    if pd.isna(amount_inr) or amount_inr is None:
        return "₹0.00"
    
    return f"₹{amount_inr:,.2f}"

def calculate_days_until_expiry(expiry_date) -> int:
    """Calculate days until expiry date"""
    if pd.isna(expiry_date) or expiry_date is None:
        return 999  # Return large number for items without expiry date
    
    if isinstance(expiry_date, str):
        try:
            expiry_date = datetime.strptime(expiry_date, "%Y-%m-%d").date()
        except ValueError:
            return 999
    
    if hasattr(expiry_date, 'date'):
        expiry_date = expiry_date.date()
    
    today = datetime.now().date()
    delta = expiry_date - today
    return delta.days

def calculate_stock_status(current_stock: int, minimum_stock: int) -> str:
    """Determine stock status based on current and minimum levels"""
    if current_stock == 0:
        return "Out of Stock"
    elif current_stock <= minimum_stock:
        return "Low Stock"
    elif current_stock <= minimum_stock * 1.5:
        return "Warning"
    else:
        return "Normal"

def get_stock_status_color(status: str) -> str:
    """Get color code for stock status"""
    color_map = {
        "Out of Stock": "#FF4136",    # Red
        "Low Stock": "#FF851B",       # Orange
        "Warning": "#FFDC00",         # Yellow
        "Normal": "#2ECC40"           # Green
    }
    return color_map.get(status, "#808080")  # Gray as default

def generate_alerts(db) -> List[Dict[str, Any]]:
    """Generate system alerts based on current data"""
    alerts = []
    
    try:
        # Low stock alerts
        low_stock_items = get_low_stock_items(db)
        for item in low_stock_items:
            alerts.append({
                'type': 'warning' if item['current_stock'] > 0 else 'critical',
                'category': 'stock',
                'message': f"Low stock alert: {item['drug_name']} has only {item['current_stock']} units remaining (minimum: {item['minimum_stock']})",
                'drug_id': item['id'],
                'priority': 'high' if item['current_stock'] == 0 else 'medium'
            })
        
        # Expiry alerts
        expiring_items = get_expiring_items(db)
        for item in expiring_items:
            days_until_expiry = item['days_until_expiry']
            if days_until_expiry <= 7:
                alert_type = 'critical'
                priority = 'high'
            elif days_until_expiry <= 30:
                alert_type = 'warning'
                priority = 'medium'
            else:
                alert_type = 'info'
                priority = 'low'
            
            alerts.append({
                'type': alert_type,
                'category': 'expiry',
                'message': f"Expiry alert: {item['drug_name']} (Batch: {item['batch_number']}) expires in {days_until_expiry} days",
                'drug_id': item.get('id'),
                'priority': priority
            })
        
        # High value at risk alerts
        high_value_expiring = get_high_value_expiring_items(db)
        for item in high_value_expiring:
            alerts.append({
                'type': 'warning',
                'category': 'financial',
                'message': f"High value at risk: {item['drug_name']} worth {format_currency(item['value_at_risk'])} expiring soon",
                'drug_id': item.get('id'),
                'priority': 'medium'
            })
        
        # Unusual consumption pattern alerts
        consumption_anomalies = detect_consumption_anomalies(db)
        for anomaly in consumption_anomalies:
            alerts.append({
                'type': 'info',
                'category': 'consumption',
                'message': f"Consumption anomaly detected: {anomaly['drug_name']} usage {anomaly['change_description']}",
                'drug_id': anomaly.get('drug_id'),
                'priority': 'low'
            })
        
        # Reorder suggestions
        reorder_alerts = get_reorder_alerts(db)
        for alert in reorder_alerts:
            alerts.append({
                'type': 'info',
                'category': 'reorder',
                'message': f"Reorder suggestion: {alert['drug_name']} - suggested quantity: {alert['suggested_quantity']}",
                'drug_id': alert.get('drug_id'),
                'priority': alert.get('priority', 'medium')
            })
    
    except Exception as e:
        alerts.append({
            'type': 'critical',
            'category': 'system',
            'message': f"Error generating alerts: {str(e)}",
            'priority': 'high'
        })
    
    # Sort alerts by priority (high -> medium -> low)
    priority_order = {'high': 3, 'medium': 2, 'low': 1}
    alerts.sort(key=lambda x: priority_order.get(x['priority'], 0), reverse=True)
    
    return alerts[:10]  # Return top 10 alerts

def get_low_stock_items(db) -> List[Dict]:
    """Get items with low stock levels"""
    try:
        conn = db.get_connection()
        query = '''
            SELECT id, drug_name, current_stock, minimum_stock, unit_price
            FROM inventory 
            WHERE current_stock <= minimum_stock
            ORDER BY (CAST(current_stock AS FLOAT) / minimum_stock) ASC
            LIMIT 10
        '''
        cursor = conn.cursor()
        cursor.execute(query)
        items = []
        for row in cursor.fetchall():
            items.append({
                'id': row[0],
                'drug_name': row[1],
                'current_stock': row[2],
                'minimum_stock': row[3],
                'unit_price': row[4]
            })
        conn.close()
        return items
    except Exception:
        return []

def get_expiring_items(db) -> List[Dict]:
    """Get items expiring within 90 days"""
    try:
        conn = db.get_connection()
        query = '''
            SELECT id, drug_name, batch_number, current_stock, expiry_date, unit_price,
                   CASE 
                       WHEN julianday(expiry_date) - julianday('now') <= 0 THEN 0
                       ELSE CAST(julianday(expiry_date) - julianday('now') AS INTEGER)
                   END as days_until_expiry
            FROM inventory
            WHERE expiry_date <= date('now', '+90 days')
            ORDER BY days_until_expiry ASC
            LIMIT 20
        '''
        cursor = conn.cursor()
        cursor.execute(query)
        items = []
        for row in cursor.fetchall():
            items.append({
                'id': row[0],
                'drug_name': row[1],
                'batch_number': row[2],
                'current_stock': row[3],
                'expiry_date': row[4],
                'unit_price': row[5],
                'days_until_expiry': row[6]
            })
        conn.close()
        return items
    except Exception:
        return []

def get_high_value_expiring_items(db, min_value: float = 1000.0) -> List[Dict]:
    """Get high-value items that are expiring"""
    try:
        conn = db.get_connection()
        query = '''
            SELECT id, drug_name, batch_number, current_stock, expiry_date, unit_price,
                   (current_stock * unit_price) as value_at_risk,
                   CASE 
                       WHEN julianday(expiry_date) - julianday('now') <= 0 THEN 0
                       ELSE CAST(julianday(expiry_date) - julianday('now') AS INTEGER)
                   END as days_until_expiry
            FROM inventory
            WHERE expiry_date <= date('now', '+60 days')
            AND (current_stock * unit_price) >= ?
            ORDER BY value_at_risk DESC
            LIMIT 10
        '''
        cursor = conn.cursor()
        cursor.execute(query, (min_value,))
        items = []
        for row in cursor.fetchall():
            items.append({
                'id': row[0],
                'drug_name': row[1],
                'batch_number': row[2],
                'current_stock': row[3],
                'expiry_date': row[4],
                'unit_price': row[5],
                'value_at_risk': row[6],
                'days_until_expiry': row[7]
            })
        conn.close()
        return items
    except Exception:
        return []

def detect_consumption_anomalies(db) -> List[Dict]:
    """Detect unusual consumption patterns"""
    try:
        conn = db.get_connection()
        
        # Get consumption data for the last 30 days vs previous 30 days
        query = '''
            SELECT i.id, i.drug_name,
                   AVG(CASE WHEN cp.date >= date('now', '-30 days') THEN cp.quantity_consumed ELSE 0 END) as recent_avg,
                   AVG(CASE WHEN cp.date BETWEEN date('now', '-60 days') AND date('now', '-30 days') THEN cp.quantity_consumed ELSE 0 END) as previous_avg
            FROM inventory i
            LEFT JOIN consumption_patterns cp ON i.id = cp.drug_id
            WHERE cp.date >= date('now', '-60 days')
            GROUP BY i.id, i.drug_name
            HAVING recent_avg > 0 AND previous_avg > 0
        '''
        
        cursor = conn.cursor()
        cursor.execute(query)
        anomalies = []
        
        for row in cursor.fetchall():
            drug_id, drug_name, recent_avg, previous_avg = row
            
            if recent_avg > 0 and previous_avg > 0:
                change_ratio = recent_avg / previous_avg
                
                # Detect significant changes (>50% increase or decrease)
                if change_ratio > 1.5:
                    anomalies.append({
                        'drug_id': drug_id,
                        'drug_name': drug_name,
                        'change_description': f'increased by {((change_ratio - 1) * 100):.1f}%',
                        'recent_avg': recent_avg,
                        'previous_avg': previous_avg
                    })
                elif change_ratio < 0.5:
                    anomalies.append({
                        'drug_id': drug_id,
                        'drug_name': drug_name,
                        'change_description': f'decreased by {((1 - change_ratio) * 100):.1f}%',
                        'recent_avg': recent_avg,
                        'previous_avg': previous_avg
                    })
        
        conn.close()
        return anomalies[:5]  # Return top 5 anomalies
    
    except Exception:
        return []

def get_reorder_alerts(db) -> List[Dict]:
    """Get reorder alerts based on current stock and consumption patterns"""
    try:
        conn = db.get_connection()
        query = '''
            SELECT i.id, i.drug_name, i.current_stock, i.minimum_stock,
                   AVG(cp.quantity_consumed) as avg_daily_consumption
            FROM inventory i
            LEFT JOIN consumption_patterns cp ON i.id = cp.drug_id 
            WHERE cp.date >= date('now', '-30 days')
            GROUP BY i.id, i.drug_name, i.current_stock, i.minimum_stock
            HAVING i.current_stock <= i.minimum_stock * 1.5
        '''
        
        cursor = conn.cursor()
        cursor.execute(query)
        alerts = []
        
        for row in cursor.fetchall():
            drug_id, drug_name, current_stock, minimum_stock, avg_daily_consumption = row
            
            # Calculate suggested quantity
            if avg_daily_consumption and avg_daily_consumption > 0:
                # Order enough for 30 days plus safety stock
                suggested_quantity = int(avg_daily_consumption * 30 + minimum_stock - current_stock)
                suggested_quantity = max(suggested_quantity, minimum_stock)
                
                priority = 'high' if current_stock <= minimum_stock else 'medium'
                
                alerts.append({
                    'drug_id': drug_id,
                    'drug_name': drug_name,
                    'current_stock': current_stock,
                    'minimum_stock': minimum_stock,
                    'suggested_quantity': suggested_quantity,
                    'avg_daily_consumption': avg_daily_consumption,
                    'priority': priority
                })
        
        conn.close()
        return alerts[:5]  # Return top 5 reorder alerts
    
    except Exception:
        return []

def calculate_inventory_turnover(db, drug_name: str = None) -> Dict[str, float]:
    """Calculate inventory turnover metrics"""
    try:
        conn = db.get_connection()
        
        if drug_name:
            # Calculate for specific drug
            query = '''
                SELECT 
                    SUM(cp.quantity_consumed * i.unit_price) as total_consumption_value,
                    AVG(i.current_stock * i.unit_price) as avg_inventory_value
                FROM consumption_patterns cp
                JOIN inventory i ON cp.drug_id = i.id
                WHERE i.drug_name = ? AND cp.date >= date('now', '-365 days')
                GROUP BY i.drug_name
            '''
            cursor = conn.cursor()
            cursor.execute(query, (drug_name,))
        else:
            # Calculate for entire inventory
            query = '''
                SELECT 
                    SUM(cp.quantity_consumed * i.unit_price) as total_consumption_value,
                    AVG(i.current_stock * i.unit_price) as avg_inventory_value
                FROM consumption_patterns cp
                JOIN inventory i ON cp.drug_id = i.id
                WHERE cp.date >= date('now', '-365 days')
            '''
            cursor = conn.cursor()
            cursor.execute(query)
        
        result = cursor.fetchone()
        conn.close()
        
        if result and result[0] and result[1]:
            total_consumption_value, avg_inventory_value = result
            turnover_ratio = total_consumption_value / avg_inventory_value
            days_in_inventory = 365 / turnover_ratio if turnover_ratio > 0 else 365
            
            return {
                'turnover_ratio': turnover_ratio,
                'days_in_inventory': days_in_inventory,
                'total_consumption_value': total_consumption_value,
                'avg_inventory_value': avg_inventory_value
            }
        
        return {
            'turnover_ratio': 0,
            'days_in_inventory': 365,
            'total_consumption_value': 0,
            'avg_inventory_value': 0
        }
    
    except Exception:
        return {
            'turnover_ratio': 0,
            'days_in_inventory': 365,
            'total_consumption_value': 0,
            'avg_inventory_value': 0
        }

def validate_batch_number(batch_number: str) -> bool:
    """Validate batch number format"""
    if not batch_number:
        return False
    
    # Basic validation - batch number should be alphanumeric and 3-20 characters
    pattern = r'^[A-Z0-9]{3,20}$'
    return bool(re.match(pattern, batch_number.upper()))

def sanitize_drug_name(drug_name: str) -> str:
    """Sanitize drug name for database storage"""
    if not drug_name:
        return ""
    
    # Remove extra whitespace and standardize format
    sanitized = re.sub(r'\s+', ' ', drug_name.strip())
    
    # Capitalize first letter of each word
    sanitized = ' '.join(word.capitalize() for word in sanitized.split())
    
    return sanitized

def calculate_safety_stock(avg_daily_usage: float, lead_time_days: int, 
                          service_level: float = 0.95) -> int:
    """Calculate safety stock based on usage patterns and service level"""
    if avg_daily_usage <= 0:
        return 0
    
    # Simplified safety stock calculation
    # In practice, this would use standard deviation of demand and lead time
    safety_factor = {
        0.90: 1.28,  # 90% service level
        0.95: 1.65,  # 95% service level
        0.99: 2.33   # 99% service level
    }.get(service_level, 1.65)
    
    # Assume 20% coefficient of variation for demand
    demand_std = avg_daily_usage * 0.2
    lead_time_std = lead_time_days * 0.1  # 10% lead time variation
    
    # Safety stock formula: Z * sqrt(LT * σd² + d² * σLT²)
    # Simplified version
    safety_stock = safety_factor * np.sqrt(
        lead_time_days * (demand_std ** 2) + 
        (avg_daily_usage ** 2) * (lead_time_std ** 2)
    )
    
    return max(1, int(safety_stock))

def generate_order_number(prefix: str = "PO") -> str:
    """Generate unique order number"""
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    return f"{prefix}{timestamp}"

def parse_expiry_date(date_string: str) -> Optional[datetime]:
    """Parse expiry date from various string formats"""
    if not date_string:
        return None
    
    # Common date formats
    formats = [
        "%Y-%m-%d",
        "%d/%m/%Y",
        "%m/%d/%Y",
        "%d-%m-%Y",
        "%Y/%m/%d",
        "%b %Y",
        "%B %Y"
    ]
    
    for fmt in formats:
        try:
            return datetime.strptime(date_string.strip(), fmt)
        except ValueError:
            continue
    
    return None

def calculate_reorder_point(avg_daily_usage: float, lead_time_days: int, 
                           safety_stock: int = 0) -> int:
    """Calculate reorder point"""
    if avg_daily_usage <= 0:
        return safety_stock
    
    reorder_point = (avg_daily_usage * lead_time_days) + safety_stock
    return max(1, int(reorder_point))

def get_consumption_forecast_accuracy(predicted: List[float], 
                                    actual: List[float]) -> Dict[str, float]:
    """Calculate forecast accuracy metrics"""
    if len(predicted) != len(actual) or len(predicted) == 0:
        return {'mape': 0, 'mae': 0, 'mse': 0, 'accuracy': 0}
    
    predicted = np.array(predicted)
    actual = np.array(actual)
    
    # Mean Absolute Percentage Error
    mape = np.mean(np.abs((actual - predicted) / np.where(actual != 0, actual, 1))) * 100
    
    # Mean Absolute Error
    mae = np.mean(np.abs(actual - predicted))
    
    # Mean Squared Error
    mse = np.mean((actual - predicted) ** 2)
    
    # Accuracy (1 - MAPE/100)
    accuracy = max(0, 1 - mape/100)
    
    return {
        'mape': mape,
        'mae': mae,
        'mse': mse,
        'accuracy': accuracy
    }

def format_percentage(value: float) -> str:
    """Format value as percentage"""
    if pd.isna(value) or value is None:
        return "0.0%"
    return f"{value:.1%}"

def get_alert_priority_color(priority: str) -> str:
    """Get color for alert priority"""
    colors = {
        'high': '#FF4136',      # Red
        'medium': '#FF851B',    # Orange  
        'low': '#0074D9'        # Blue
    }
    return colors.get(priority.lower(), '#808080')  # Gray as default

def clean_numeric_input(value: Any) -> Optional[float]:
    """Clean and validate numeric input"""
    if value is None or pd.isna(value):
        return None
    
    if isinstance(value, (int, float)):
        return float(value) if value >= 0 else None
    
    if isinstance(value, str):
        # Remove currency symbols and commas
        cleaned = re.sub(r'[^\d.-]', '', value.strip())
        try:
            result = float(cleaned)
            return result if result >= 0 else None
        except ValueError:
            return None
    
    return None

def calculate_abc_classification(inventory_data: pd.DataFrame) -> pd.DataFrame:
    """Calculate ABC classification based on inventory value"""
    if inventory_data.empty:
        return inventory_data
    
    # Calculate value for each item
    inventory_data['total_value'] = inventory_data['current_stock'] * inventory_data['unit_price']
    
    # Sort by value in descending order
    sorted_data = inventory_data.sort_values('total_value', ascending=False).copy()
    
    # Calculate cumulative percentage
    sorted_data['cumulative_value'] = sorted_data['total_value'].cumsum()
    total_value = sorted_data['total_value'].sum()
    sorted_data['cumulative_percentage'] = sorted_data['cumulative_value'] / total_value * 100
    
    # Assign ABC classification
    def assign_class(percentage):
        if percentage <= 70:
            return 'A'
        elif percentage <= 90:
            return 'B'
        else:
            return 'C'
    
    sorted_data['abc_class'] = sorted_data['cumulative_percentage'].apply(assign_class)
    
    return sorted_data

def get_seasonal_adjustment_factor(date: datetime, drug_category: str) -> float:
    """Get seasonal adjustment factor for demand forecasting"""
    month = date.month
    
    # Seasonal patterns by drug category
    seasonal_patterns = {
        'respiratory': {
            # Higher demand in winter months
            1: 1.3, 2: 1.3, 3: 1.1, 4: 0.9, 5: 0.8, 6: 0.7,
            7: 0.7, 8: 0.8, 9: 0.9, 10: 1.1, 11: 1.2, 12: 1.3
        },
        'cardiovascular': {
            # Relatively stable year-round with slight winter increase
            1: 1.1, 2: 1.1, 3: 1.0, 4: 1.0, 5: 0.9, 6: 0.9,
            7: 0.9, 8: 0.9, 9: 1.0, 10: 1.0, 11: 1.1, 12: 1.1
        },
        'analgesics': {
            # Stable throughout year
            1: 1.0, 2: 1.0, 3: 1.0, 4: 1.0, 5: 1.0, 6: 1.0,
            7: 1.0, 8: 1.0, 9: 1.0, 10: 1.0, 11: 1.0, 12: 1.0
        }
    }
    
    pattern = seasonal_patterns.get(drug_category.lower(), 
                                   {i: 1.0 for i in range(1, 13)})
    
    return pattern.get(month, 1.0)
