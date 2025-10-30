"""
Enhanced Smart Recommendation Engine
AI-powered intelligent recommendations with user behavior tracking and trend analysis
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from scipy import stats

class SmartRecommendationEngine:
    """Enhanced intelligent recommendation system with ML-driven insights"""
    
    def __init__(self, db_manager):
        self.db = db_manager
        
    def get_personalized_recommendations(self, user_role='pharmacist', user_id=None):
        """
        Generate comprehensive personalized recommendations with advanced scoring
        
        Integrates multiple data sources and applies intelligent priority scoring
        """
        recommendations = []
        
        low_stock = self._analyze_low_stock_items()
        expiring_items = self._analyze_expiring_items()
        overstock = self._analyze_overstock_items()
        slow_movers = self._analyze_slow_moving_items()
        high_demand = self._analyze_high_demand_items()
        seasonal_opportunities = self._analyze_seasonal_opportunities()
        supplier_issues = self._analyze_supplier_performance()
        
        recommendations.extend(self._generate_stock_recommendations(low_stock))
        recommendations.extend(self._generate_expiry_recommendations(expiring_items))
        recommendations.extend(self._generate_overstock_recommendations(overstock))
        recommendations.extend(self._generate_slow_mover_recommendations(slow_movers))
        recommendations.extend(self._generate_demand_recommendations(high_demand))
        recommendations.extend(self._generate_seasonal_recommendations(seasonal_opportunities))
        recommendations.extend(self._generate_supplier_recommendations(supplier_issues))
        
        scored_recommendations = self._score_recommendations(recommendations)
        
        return sorted(scored_recommendations, key=lambda x: x['priority_score'], reverse=True)[:20]
    
    def _analyze_low_stock_items(self):
        """Identify items with critical stock levels"""
        conn = self.db.get_connection()
        query = """
            SELECT i.drug_name, i.category, i.current_stock, i.minimum_stock,
                   i.unit_price, (i.minimum_stock - i.current_stock) as shortage,
                   COALESCE(AVG(cp.quantity_consumed), 0) as avg_daily_consumption
            FROM inventory i
            LEFT JOIN consumption_patterns cp ON i.id = cp.drug_id
              AND cp.date >= DATE('now', '-30 days')
            WHERE i.current_stock < i.minimum_stock
            GROUP BY i.drug_name, i.category, i.current_stock, i.minimum_stock, i.unit_price
            ORDER BY shortage DESC
            LIMIT 20
        """
        df = pd.read_sql_query(query, conn)
        conn.close()
        return df
    
    def _analyze_expiring_items(self):
        """Identify items with expiry risks and potential wastage"""
        conn = self.db.get_connection()
        query = """
            SELECT i.drug_name, i.category, i.current_stock, i.expiry_date, i.unit_price,
                   CAST(JULIANDAY(i.expiry_date) - JULIANDAY('now') AS INTEGER) as days_to_expiry,
                   (i.current_stock * i.unit_price) as potential_loss,
                   COALESCE(AVG(cp.quantity_consumed), 0) as avg_daily_consumption
            FROM inventory i
            LEFT JOIN consumption_patterns cp ON i.id = cp.drug_id
              AND cp.date >= DATE('now', '-30 days')
            WHERE i.expiry_date IS NOT NULL
              AND JULIANDAY(i.expiry_date) - JULIANDAY('now') BETWEEN 0 AND 90
              AND i.current_stock > 0
            GROUP BY i.drug_name, i.category, i.current_stock, i.expiry_date, i.unit_price
            ORDER BY days_to_expiry
            LIMIT 20
        """
        df = pd.read_sql_query(query, conn)
        conn.close()
        return df
    
    def _analyze_overstock_items(self):
        """Identify overstocked items with tied capital"""
        conn = self.db.get_connection()
        query = """
            SELECT i.drug_name, i.category, i.current_stock, i.minimum_stock,
                   (i.current_stock - i.minimum_stock) as excess_stock,
                   (i.current_stock * i.unit_price) as tied_capital,
                   COALESCE(AVG(cp.quantity_consumed), 0) as avg_daily_consumption
            FROM inventory i
            LEFT JOIN consumption_patterns cp ON i.id = cp.drug_id
              AND cp.date >= DATE('now', '-30 days')
            WHERE i.current_stock > i.minimum_stock * 3
            GROUP BY i.drug_name, i.category, i.current_stock, i.minimum_stock
            HAVING avg_daily_consumption < (i.current_stock / 90)
            ORDER BY tied_capital DESC
            LIMIT 15
        """
        df = pd.read_sql_query(query, conn)
        conn.close()
        return df
    
    def _analyze_slow_moving_items(self):
        """Identify slow-moving inventory with low turnover"""
        conn = self.db.get_connection()
        query = """
            SELECT i.drug_name, i.category, i.current_stock,
                   COALESCE(SUM(cp.quantity_consumed), 0) as total_consumed_90d,
                   (i.current_stock * i.unit_price) as inventory_value
            FROM inventory i
            LEFT JOIN consumption_patterns cp ON i.id = cp.drug_id
              AND cp.date >= DATE('now', '-90 days')
            WHERE i.current_stock > 0
            GROUP BY i.drug_name, i.category, i.current_stock, i.unit_price
            HAVING total_consumed_90d < 10 OR total_consumed_90d IS NULL
            ORDER BY inventory_value DESC
            LIMIT 15
        """
        df = pd.read_sql_query(query, conn)
        conn.close()
        return df
    
    def _analyze_high_demand_items(self):
        """Identify high-demand items with growth trends"""
        conn = self.db.get_connection()
        query = """
            SELECT i.drug_name, i.category, i.current_stock, i.minimum_stock, i.unit_price,
                   SUM(CASE WHEN cp.date >= DATE('now', '-30 days') THEN cp.quantity_consumed ELSE 0 END) as last_30d,
                   SUM(CASE WHEN cp.date >= DATE('now', '-60 days') AND cp.date < DATE('now', '-30 days') 
                       THEN cp.quantity_consumed ELSE 0 END) as prev_30d
            FROM inventory i
            JOIN consumption_patterns cp ON i.id = cp.drug_id
            WHERE cp.date >= DATE('now', '-60 days')
            GROUP BY i.drug_name, i.category, i.current_stock, i.minimum_stock, i.unit_price
            HAVING last_30d > prev_30d * 1.2 AND last_30d > 10
            ORDER BY (last_30d - prev_30d) DESC
            LIMIT 15
        """
        df = pd.read_sql_query(query, conn)
        conn.close()
        return df
    
    def _analyze_seasonal_opportunities(self):
        """Identify seasonal demand patterns and opportunities"""
        conn = self.db.get_connection()
        current_month = datetime.now().month
        current_month_str = str(current_month).zfill(2)
        
        query = """
            SELECT i.drug_name, i.category,
                   AVG(CASE WHEN strftime('%m', cp.date) = ? THEN cp.quantity_consumed ELSE 0 END) as current_month_avg,
                   AVG(cp.quantity_consumed) as overall_avg
            FROM inventory i
            JOIN consumption_patterns cp ON i.id = cp.drug_id
            WHERE cp.date >= DATE('now', '-365 days')
            GROUP BY i.drug_name, i.category
            HAVING current_month_avg > overall_avg * 1.3
            ORDER BY (current_month_avg - overall_avg) DESC
            LIMIT 10
        """
        df = pd.read_sql_query(query, conn, params=(current_month_str,))
        conn.close()
        return df
    
    def _analyze_supplier_performance(self):
        """Analyze supplier performance issues"""
        conn = self.db.get_connection()
        query = """
            SELECT s.name, s.reliability_score, s.quality_score, s.cost_rating,
                   s.lead_time_days
            FROM suppliers s
            WHERE s.reliability_score < 3.5 OR s.quality_score < 3.5 OR s.lead_time_days > 10
            ORDER BY (s.reliability_score + s.quality_score) / 2
            LIMIT 10
        """
        try:
            df = pd.read_sql_query(query, conn)
        except:
            df = pd.DataFrame()
        conn.close()
        return df
    
    def _generate_stock_recommendations(self, low_stock_df):
        """Generate critical stock recommendations with urgency scoring"""
        recommendations = []
        for _, row in low_stock_df.iterrows():
            days_until_stockout = row['current_stock'] / row['avg_daily_consumption'] if row['avg_daily_consumption'] > 0 else 999
            
            if days_until_stockout < 3:
                urgency = 'Critical'
                priority = 98
            elif days_until_stockout < 7:
                urgency = 'High'
                priority = 92
            else:
                urgency = 'Medium'
                priority = 85
            
            recommendations.append({
                'type': 'RESTOCK',
                'category': 'Inventory Management',
                'title': f'⚠️ URGENT: Restock {row["drug_name"]}',
                'description': f'Critical shortage: Current stock ({row["current_stock"]} units) is {row["shortage"]} units below minimum. Daily consumption: {row["avg_daily_consumption"]:.1f} units. Stockout in {days_until_stockout:.0f} days.',
                'action': f'Place immediate purchase order for {row["shortage"] * 2} units',
                'impact': 'Critical',
                'urgency': urgency,
                'estimated_cost': row['unit_price'] * row['shortage'] * 2,
                'estimated_savings': row['unit_price'] * row['shortage'] * 3,
                'priority_score': priority,
                'days_until_impact': int(days_until_stockout)
            })
        return recommendations
    
    def _generate_expiry_recommendations(self, expiring_df):
        """Generate wastage prevention recommendations"""
        recommendations = []
        for _, row in expiring_df.iterrows():
            days_to_expiry = row['days_to_expiry']
            consumption_rate = row['avg_daily_consumption']
            can_sell = consumption_rate * days_to_expiry if consumption_rate > 0 else 0
            expected_wastage = max(0, row['current_stock'] - can_sell)
            
            if days_to_expiry <= 15:
                urgency = 'Critical'
                priority = 96
                action = f'IMMEDIATE: Discount 30-40% or transfer to high-demand location'
            elif days_to_expiry <= 30:
                urgency = 'High'
                priority = 88
                action = 'Implement promotional pricing (20-25% discount)'
            elif days_to_expiry <= 60:
                urgency = 'Medium'
                priority = 70
                action = 'Monitor closely and plan promotional activities'
            else:
                urgency = 'Low'
                priority = 55
                action = 'Continue normal operations with regular monitoring'
            
            recommendations.append({
                'type': 'EXPIRY_ALERT',
                'category': 'Wastage Prevention',
                'title': f'⏰ {row["drug_name"]} expiring in {days_to_expiry} days',
                'description': f'{row["current_stock"]} units will expire. Daily consumption: {consumption_rate:.1f} units. Expected wastage: {expected_wastage:.0f} units (₹{expected_wastage * row["unit_price"]:.2f})',
                'action': action,
                'impact': 'High' if row['potential_loss'] > 1000 else 'Medium',
                'urgency': urgency,
                'estimated_cost': 0,
                'estimated_savings': expected_wastage * row['unit_price'] * 0.7,
                'priority_score': priority,
                'days_until_impact': days_to_expiry
            })
        return recommendations
    
    def _generate_overstock_recommendations(self, overstock_df):
        """Generate cost optimization recommendations for overstock"""
        recommendations = []
        for _, row in overstock_df.iterrows():
            monthly_holding_cost = row['tied_capital'] * 0.015
            
            recommendations.append({
                'type': 'REDUCE_STOCK',
                'category': 'Cost Optimization',
                'title': f'💰 Optimize stock levels for {row["drug_name"]}',
                'description': f'Excess inventory of {row["excess_stock"]} units (₹{row["tied_capital"]:.2f} tied up). Daily consumption: {row["avg_daily_consumption"]:.1f} units. Holding cost: ₹{monthly_holding_cost:.2f}/month.',
                'action': f'Reduce stock by {int(row["excess_stock"] * 0.6)} units via supplier returns or branch transfers',
                'impact': 'Medium',
                'urgency': 'Low',
                'estimated_cost': row['tied_capital'] * 0.02,
                'estimated_savings': monthly_holding_cost * 6,
                'priority_score': 60,
                'days_until_impact': 90
            })
        return recommendations
    
    def _generate_slow_mover_recommendations(self, slow_movers_df):
        """Generate recommendations for slow-moving items"""
        recommendations = []
        for _, row in slow_movers_df.iterrows():
            recommendations.append({
                'type': 'SLOW_MOVER',
                'category': 'Inventory Optimization',
                'title': f'📉 Review slow-moving item: {row["drug_name"]}',
                'description': f'Low turnover: Only {row["total_consumed_90d"]:.0f} units in 90 days. Stock value: ₹{row["inventory_value"]:.2f}. Consider product review.',
                'action': 'Reduce minimum stock levels or consider discontinuation. Implement clearance promotion.',
                'impact': 'Medium',
                'urgency': 'Low',
                'estimated_cost': row['inventory_value'] * 0.1,
                'estimated_savings': row['inventory_value'] * 0.15,
                'priority_score': 45,
                'days_until_impact': 120
            })
        return recommendations
    
    def _generate_demand_recommendations(self, high_demand_df):
        """Generate growth opportunity recommendations"""
        recommendations = []
        for _, row in high_demand_df.iterrows():
            growth_rate = ((row['last_30d'] - row['prev_30d']) / (row['prev_30d'] + 1)) * 100
            revenue_opportunity = row['last_30d'] * row['unit_price'] * (growth_rate / 100) * 3
            
            recommendations.append({
                'type': 'INCREASE_STOCK',
                'category': 'Growth Opportunity',
                'title': f'📈 Capitalize on growing demand: {row["drug_name"]}',
                'description': f'Strong growth: +{growth_rate:.1f}% demand increase (from {row["prev_30d"]:.0f} to {row["last_30d"]:.0f} units). Revenue opportunity: ₹{revenue_opportunity:.2f} over next quarter.',
                'action': f'Increase minimum stock from {row["minimum_stock"]} to {int(row["minimum_stock"] * 1.5)} units. Secure additional supply.',
                'impact': 'High',
                'urgency': 'Medium',
                'estimated_cost': row['minimum_stock'] * row['unit_price'] * 0.5,
                'estimated_savings': revenue_opportunity * 0.25,
                'priority_score': 78,
                'days_until_impact': 30
            })
        return recommendations
    
    def _generate_seasonal_recommendations(self, seasonal_df):
        """Generate seasonal opportunity recommendations"""
        recommendations = []
        for _, row in seasonal_df.iterrows():
            seasonal_factor = row['current_month_avg'] / row['overall_avg'] if row['overall_avg'] > 0 else 1
            
            recommendations.append({
                'type': 'SEASONAL',
                'category': 'Seasonal Opportunity',
                'title': f'🌟 Seasonal peak for {row["drug_name"]}',
                'description': f'Current month shows {seasonal_factor:.1f}x higher demand. Historical average: {row["overall_avg"]:.1f} units, Current month: {row["current_month_avg"]:.1f} units.',
                'action': f'Increase inventory by {int((seasonal_factor - 1) * 100)}% to meet seasonal demand.',
                'impact': 'Medium',
                'urgency': 'Medium',
                'estimated_cost': row['current_month_avg'] * 5,
                'estimated_savings': row['current_month_avg'] * 8,
                'priority_score': 68,
                'days_until_impact': 15
            })
        return recommendations
    
    def _generate_supplier_recommendations(self, supplier_df):
        """Generate supplier performance recommendations"""
        recommendations = []
        for _, row in supplier_df.iterrows():
            issues = []
            if row['reliability_score'] < 3.5:
                issues.append(f'Low reliability ({row["reliability_score"]:.1f}/5)')
            if row['quality_score'] < 3.5:
                issues.append(f'Quality concerns ({row["quality_score"]:.1f}/5)')
            if row['lead_time_days'] > 10:
                issues.append(f'Slow delivery ({row["lead_time_days"]:.0f} days)')
            
            recommendations.append({
                'type': 'SUPPLIER_REVIEW',
                'category': 'Supplier Management',
                'title': f'🔍 Review supplier: {row["name"]}',
                'description': f'Performance issues: {", ".join(issues)}. Consider alternative suppliers.',
                'action': 'Evaluate alternative suppliers and negotiate performance improvements',
                'impact': 'Medium',
                'urgency': 'Low',
                'estimated_cost': 500,
                'estimated_savings': 2000,
                'priority_score': 50,
                'days_until_impact': 60
            })
        return recommendations
    
    def _score_recommendations(self, recommendations):
        """Apply intelligent scoring with multiple factors"""
        for rec in recommendations:
            impact_multiplier = {'Critical': 1.5, 'High': 1.3, 'Medium': 1.0, 'Low': 0.7}.get(rec['impact'], 1.0)
            urgency_multiplier = {'Critical': 1.6, 'High': 1.3, 'Medium': 1.0, 'Low': 0.6}.get(rec['urgency'], 1.0)
            
            roi = 0
            if rec.get('estimated_cost', 0) > 0:
                roi = (rec.get('estimated_savings', 0) - rec.get('estimated_cost', 0)) / rec.get('estimated_cost', 1)
            
            roi_multiplier = min(1.5, max(0.5, 1.0 + roi * 0.1))
            
            time_urgency = 1.0
            if rec.get('days_until_impact', 999) < 7:
                time_urgency = 1.4
            elif rec.get('days_until_impact', 999) < 30:
                time_urgency = 1.2
            
            rec['priority_score'] = (
                rec['priority_score'] * 
                impact_multiplier * 
                urgency_multiplier * 
                roi_multiplier * 
                time_urgency
            )
            
            rec['roi_ratio'] = roi if roi > 0 else 0
        
        return recommendations
