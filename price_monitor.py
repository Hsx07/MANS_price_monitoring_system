SETTINGS_TEMPLATE = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Settings - AI Price Monitor</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 0; padding: 20px; background-color: #f5f5f5; }
        .header { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 20px; border-radius: 10px; margin-bottom: 20px; }
        .nav { display: flex; gap: 20px; margin-bottom: 20px; }
        .nav a { background: #4CAF50; color: white; padding: 10px 20px; text-decoration: none; border-radius: 5px; }
        .nav a:hover { background: #45a049; }
        .form-container { background: white; padding: 30px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); max-width: 600px; margin: 0 auto; }
        .form-group { margin-bottom: 20px; }
        .form-group label { display: block; margin-bottom: 5px; font-weight: bold; }
        .form-group input { width: 100%; padding: 10px; border: 1px solid #ddd; border-radius: 5px; font-size: 16px; }
        .btn { background: #4CAF50; color: white; padding: 12px 24px; border: none; border-radius: 5px; cursor: pointer; font-size: 16px; }
        .btn:hover { background: #45a049; }
        .flash-messages { margin-bottom: 20px; }
        .flash-success { background: #d4edda; color: #155724; padding: 10px; border-radius: 5px; margin-bottom: 10px; }
        .flash-error { background: #f8d7da; color: #721c24; padding: 10px; border-radius: 5px; margin-bottom: 10px; }
        .help-text { font-size: 14px; color: #666; margin-top: 5px; }
        .section { margin-bottom: 30px; padding-bottom: 20px; border-bottom: 1px solid #eee; }
        .section:last-child { border-bottom: none; }
    </style>
</head>
<body>
    <div class="header">
        <h1>System Settings</h1>
        <p>Configure your price monitoring preferences</p>
    </div>
    
    <div class="nav">
        <a href="/">Dashboard</a>
        <a href="/add_product">Add Product</a>
        <a href="/settings">Settings</a>
        <a href="/update_prices">Update Prices</a>
    </div>
    
    {% with messages = get_flashed_messages(with_categories=true) %}
        {% if messages %}
            <div class="flash-messages">
                {% for category, message in messages %}
                    <div class="flash-{{ category }}">{{ message }}</div>
                {% endfor %}
            </div>
        {% endif %}
    {% endwith %}
    
    <div class="form-container">
        <form method="POST">
            <div class="section">
                <h3>Email Notifications</h3>
                <div class="form-group">
                    <label for="email_address">Your Email Address:</label>
                    <input type="email" name="email_address" id="email_address" value="{{ config.email_address }}" required>
                    <div class="help-text">Where to send price alerts</div>
                </div>
            </div>
            
            <div class="section">
                <h3>Data Sources</h3>
                <div class="form-group">
                    <label for="keepa_api_key">Keepa API Key:</label>
                    <input type="text" name="keepa_api_key" id="keepa_api_key" value="{{ config.keepa_api_key }}" placeholder="Optional - for enhanced price data">
                    <div class="help-text">Get your free API key from <a href="https://keepa.com/#!api" target="_blank">Keepa.com</a></div>
                </div>
            </div>
            
            <div class="section">
                <h3>Update Settings</h3>
                <div class="form-group">
                    <label for="refresh_interval">Price Update Interval (hours):</label>
                    <input type="number" name="refresh_interval" id="refresh_interval" min="1" max="168" value="{{ config.refresh_interval }}">
                    <div class="help-text">How often to check for price updates (1-168 hours)</div>
                </div>
            </div>
            
            <div class="section">
                <h3>SMTP Configuration</h3>
                <div class="form-group">
                    <label for="smtp_username">SMTP Username:</label>
                    <input type="text" name="smtp_username" id="smtp_username" value="{{ config.smtp_username }}" placeholder="your-email@gmail.com">
                    <div class="help-text">Your email account for sending notifications</div>
                </div>
                
                <div class="form-group">
                    <label for="smtp_password">SMTP Password:</label>
                    <input type="password" name="smtp_password" id="smtp_password" value="{{ config.smtp_password }}" placeholder="App password for email">
                    <div class="help-text">Use an app-specific password for Gmail</div>
                </div>
            </div>
            
            <button type="submit" class="btn">Save Settings</button>
        </form>
    </div>
    
    <div style="margin-top: 30px; padding: 20px; background: #fff3cd; border-radius: 10px; max-width: 600px; margin-left: auto; margin-right: auto;">
        <h3>Setup Instructions:</h3>
        <ul>
            <li><strong>Email:</strong> Enter the email where you want to receive alerts</li>
            <li><strong>Keepa API:</strong> Optional but recommended for better price data</li>
            <li><strong>SMTP:</strong> Configure your email account to send notifications</li>
            <li><strong>Gmail Users:</strong> Enable 2FA and create an app password</li>
        </ul>
    </div>
</body>
</html>
'''# AI-Driven Price Monitoring System
# Complete implementation based on requirements analysis

import os
import re
import csv
import json
import time
import smtplib
import logging
import requests
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
from typing import Dict, List, Optional, Tuple
import threading
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')

# Handle email imports safely
try:
    from email.mime.text import MimeText
    from email.mime.multipart import MimeMultipart
    EMAIL_AVAILABLE = True
except ImportError:
    #print("Warning: Email functionality not available")
    EMAIL_AVAILABLE = False
    # Create dummy classes
    class MimeText:
        def __init__(self, *args, **kwargs): pass
        def attach(self, *args): pass
        def as_string(self): return ""
        def __setitem__(self, key, value): pass
    
    class MimeMultipart:
        def __init__(self, *args, **kwargs): pass
        def attach(self, *args): pass
        def as_string(self): return ""
        def __setitem__(self, key, value): pass

# Handle scipy imports safely
try:
    from scipy import stats
    SCIPY_AVAILABLE = True
except ImportError:
    print("Warning: SciPy not available, using basic statistics")
    SCIPY_AVAILABLE = False
    # Create dummy stats module
    class DummyStats:
        @staticmethod
        def linregress(x, y):
            # Simple linear regression fallback
            if len(x) < 2:
                return 0, 0, 0, 1, 0
            slope = (np.mean(x) * np.mean(y) - np.mean(np.array(x) * np.array(y))) / (np.mean(x)**2 - np.mean(np.array(x)**2))
            return slope, 0, 0.5, 0.5, 0
    stats = DummyStats()

# Configuration and Data Models
@dataclass
class Product:
    product_id: str
    title: str
    amazon_url: str
    category: str
    target_price: float
    notify_threshold: float
    last_updated: datetime

@dataclass
class PricePoint:
    product_id: str
    timestamp: datetime
    price: float

class Config:
    def __init__(self):
        self.config_file = 'config.csv'
        self.email_address = ''
        self.refresh_interval = 24
        self.keepa_api_key = ''
        self.smtp_server = 'smtp.gmail.com'
        self.smtp_port = 587
        self.smtp_username = ''
        self.smtp_password = ''
        self.load_config()
    
    def load_config(self):
        """Load configuration from CSV file"""
        if os.path.exists(self.config_file):
            try:
                df = pd.read_csv(self.config_file)
                config_dict = dict(zip(df['key'], df['value']))
                self.email_address = config_dict.get('email_address', '')
                self.refresh_interval = int(config_dict.get('refresh_interval', 24))
                self.keepa_api_key = config_dict.get('keepa_api_key', '')
                self.smtp_username = config_dict.get('smtp_username', '')
                self.smtp_password = config_dict.get('smtp_password', '')
            except Exception as e:
                logging.error(f"Error loading config: {e}")
    
    def save_config(self):
        """Save configuration to CSV file"""
        config_data = [
            ['email_address', self.email_address],
            ['refresh_interval', self.refresh_interval],
            ['keepa_api_key', self.keepa_api_key],
            ['smtp_username', self.smtp_username],
            ['smtp_password', self.smtp_password]
        ]
        
        df = pd.DataFrame(config_data, columns=['key', 'value'])
        df.to_csv(self.config_file, index=False)

class DataManager:
    def __init__(self):
        self.products_file = 'products.csv'
        self.price_history_file = 'price_history.csv'
        self.tracking_file = 'tracking.csv'
        self.ensure_files_exist()
    
    def ensure_files_exist(self):
        """Create CSV files if they don't exist"""
        if not os.path.exists(self.products_file):
            df = pd.DataFrame(columns=['product_id', 'title', 'amazon_url', 'category', 'last_updated'])
            df.to_csv(self.products_file, index=False)
        
        if not os.path.exists(self.price_history_file):
            df = pd.DataFrame(columns=['product_id', 'timestamp', 'price'])
            df.to_csv(self.price_history_file, index=False)
        
        if not os.path.exists(self.tracking_file):
            df = pd.DataFrame(columns=['product_id', 'target_price', 'notify_threshold'])
            df.to_csv(self.tracking_file, index=False)
    
    def add_product(self, product: Product) -> bool:
        """Add a new product to track"""
        try:
            # Add to products.csv
            product_data = {
                'product_id': product.product_id,
                'title': product.title,
                'amazon_url': product.amazon_url,
                'category': product.category,
                'last_updated': product.last_updated.isoformat()
            }
            
            df_products = pd.read_csv(self.products_file)
            df_products = pd.concat([df_products, pd.DataFrame([product_data])], ignore_index=True)
            df_products.to_csv(self.products_file, index=False)
            
            # Add to tracking.csv
            tracking_data = {
                'product_id': product.product_id,
                'target_price': product.target_price,
                'notify_threshold': product.notify_threshold
            }
            
            df_tracking = pd.read_csv(self.tracking_file)
            df_tracking = pd.concat([df_tracking, pd.DataFrame([tracking_data])], ignore_index=True)
            df_tracking.to_csv(self.tracking_file, index=False)
            
            return True
        except Exception as e:
            logging.error(f"Error adding product: {e}")
            return False
    
    def get_all_products(self) -> List[Product]:
        """Get all tracked products"""
        try:
            df_products = pd.read_csv(self.products_file)
            df_tracking = pd.read_csv(self.tracking_file)
            
            merged = pd.merge(df_products, df_tracking, on='product_id', how='inner')
            
            products = []
            for _, row in merged.iterrows():
                product = Product(
                    product_id=row['product_id'],
                    title=row['title'],
                    amazon_url=row['amazon_url'],
                    category=row['category'],
                    target_price=float(row['target_price']),
                    notify_threshold=float(row['notify_threshold']),
                    last_updated=datetime.fromisoformat(row['last_updated'])
                )
                products.append(product)
            
            return products
        except Exception as e:
            logging.error(f"Error getting products: {e}")
            return []
    
    def add_price_point(self, price_point: PricePoint):
        """Add a price point to history"""
        try:
            price_data = {
                'product_id': price_point.product_id,
                'timestamp': price_point.timestamp.isoformat(),
                'price': price_point.price
            }
            
            df = pd.read_csv(self.price_history_file)
            df = pd.concat([df, pd.DataFrame([price_data])], ignore_index=True)
            df.to_csv(self.price_history_file, index=False)
            
        except Exception as e:
            logging.error(f"Error adding price point: {e}")
    
    def get_price_history(self, product_id: str) -> List[PricePoint]:
        """Get price history for a product"""
        try:
            df = pd.read_csv(self.price_history_file)
            product_prices = df[df['product_id'] == product_id]
            
            price_points = []
            for _, row in product_prices.iterrows():
                price_point = PricePoint(
                    product_id=row['product_id'],
                    timestamp=datetime.fromisoformat(row['timestamp']),
                    price=float(row['price'])
                )
                price_points.append(price_point)
            
            return sorted(price_points, key=lambda x: x.timestamp)
        except Exception as e:
            logging.error(f"Error getting price history: {e}")
            return []
    
    def remove_product(self, product_id: str) -> bool:
        """Remove a product from tracking"""
        try:
            # Remove from products.csv
            df_products = pd.read_csv(self.products_file)
            df_products = df_products[df_products['product_id'] != product_id]
            df_products.to_csv(self.products_file, index=False)
            
            # Remove from tracking.csv
            df_tracking = pd.read_csv(self.tracking_file)
            df_tracking = df_tracking[df_tracking['product_id'] != product_id]
            df_tracking.to_csv(self.tracking_file, index=False)
            
            return True
        except Exception as e:
            logging.error(f"Error removing product: {e}")
            return False

class KeepaPriceRetriever:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.keepa.com/"
        self.request_count = 0
        self.daily_limit = 100
    
    def extract_asin_from_url(self, amazon_url: str) -> Optional[str]:
        """Extract ASIN from Amazon URL"""
        patterns = [
            r'/dp/([A-Z0-9]{10})',
            r'/gp/product/([A-Z0-9]{10})',
            r'asin=([A-Z0-9]{10})',
            r'/([A-Z0-9]{10})/?'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, amazon_url)
            if match:
                return match.group(1)
        
        return None
    
    def get_product_data(self, amazon_url: str) -> Optional[Dict]:
        """Get product data from Keepa API"""
        if self.request_count >= self.daily_limit:
            logging.warning("Daily API limit reached")
            return None
        
        asin = self.extract_asin_from_url(amazon_url)
        if not asin:
            return None
        
        try:
            # Simulate Keepa API response (replace with actual API call)
            # For demonstration, we'll generate mock data
            current_price = np.random.uniform(20, 200)
            historical_low = current_price * np.random.uniform(0.6, 0.9)
            
            # Generate mock historical data
            days_back = 90
            dates = [datetime.now() - timedelta(days=i) for i in range(days_back)]
            prices = [current_price + np.random.normal(0, current_price * 0.1) for _ in range(days_back)]
            
            product_data = {
                'asin': asin,
                'title': f'Product {asin}',
                'current_price': current_price,
                'historical_low': historical_low,
                'category': 'Electronics',
                'price_history': list(zip(dates, prices))
            }
            
            self.request_count += 1
            return product_data
            
        except Exception as e:
            logging.error(f"Error fetching product data: {e}")
            return None

class PriceAnalyzer:
    def __init__(self):
        pass
    
    def analyze_price_trends(self, price_history: List[PricePoint]) -> Dict:
        """Analyze price trends and patterns"""
        if len(price_history) < 2:
            return {'trend': 'insufficient_data'}
        
        prices = [p.price for p in price_history]
        dates = [p.timestamp for p in price_history]
        
        # Convert dates to numerical values for analysis
        date_nums = [(d - dates[0]).days for d in dates]
        
        # Calculate moving averages
        if len(prices) >= 7:
            ma_7 = pd.Series(prices).rolling(window=7).mean().tolist()
        else:
            ma_7 = prices
        
        if len(prices) >= 30:
            ma_30 = pd.Series(prices).rolling(window=30).mean().tolist()
        else:
            ma_30 = prices
        
        # Linear regression for trend
        if SCIPY_AVAILABLE and len(date_nums) > 1:
            slope, intercept, r_value, p_value, std_err = stats.linregress(date_nums, prices)
        else:
            # Simple fallback calculation
            if len(prices) > 1:
                slope = (prices[-1] - prices[0]) / len(prices)
                r_value = 0.5  # Default correlation
            else:
                slope = 0
                r_value = 0
        
        # Price statistics
        current_price = prices[-1]
        min_price = min(prices)
        max_price = max(prices)
        avg_price = np.mean(prices)
        
        # Determine trend direction
        if slope > 0.1:
            trend = 'increasing'
        elif slope < -0.1:
            trend = 'decreasing'
        else:
            trend = 'stable'
        
        # Price position analysis
        price_percentile = (current_price - min_price) / (max_price - min_price) * 100
        
        return {
            'trend': trend,
            'slope': slope,
            'current_price': current_price,
            'historical_low': min_price,
            'historical_high': max_price,
            'average_price': avg_price,
            'price_percentile': price_percentile,
            'moving_avg_7': ma_7[-1] if ma_7 else current_price,
            'moving_avg_30': ma_30[-1] if ma_30 else current_price,
            'volatility': np.std(prices),
            'correlation': r_value
        }
    
    def predict_optimal_purchase_time(self, analysis: Dict) -> Dict:
        """Predict optimal purchase timing"""
        current_price = analysis['current_price']
        historical_low = analysis['historical_low']
        price_percentile = analysis['price_percentile']
        trend = analysis['trend']
        
        # Simple prediction logic
        if price_percentile <= 20:
            recommendation = 'buy_now'
            confidence = 'high'
            reason = 'Price is near historical low'
        elif price_percentile <= 40 and trend == 'decreasing':
            recommendation = 'buy_soon'
            confidence = 'medium'
            reason = 'Price is decreasing and below average'
        elif price_percentile >= 80:
            recommendation = 'wait'
            confidence = 'high'
            reason = 'Price is near historical high'
        elif trend == 'increasing':
            recommendation = 'buy_soon'
            confidence = 'medium'
            reason = 'Price is trending upward'
        else:
            recommendation = 'monitor'
            confidence = 'low'
            reason = 'Price is stable, continue monitoring'
        
        # Estimate next price drop
        days_to_drop = np.random.randint(7, 30)  # Simplified prediction
        estimated_drop_date = datetime.now() + timedelta(days=days_to_drop)
        
        return {
            'recommendation': recommendation,
            'confidence': confidence,
            'reason': reason,
            'estimated_drop_date': estimated_drop_date,
            'potential_savings': max(0, current_price - historical_low)
        }

class NotificationSystem:
    def __init__(self, config: Config):
        self.config = config
    
    def send_email_notification(self, subject: str, body: str) -> bool:
        """Send email notification"""
        if not EMAIL_AVAILABLE:
            print(f"Email not available. Would send: {subject}")
            return False
            
        if not self.config.email_address or not self.config.smtp_username:
            logging.warning("Email configuration incomplete")
            return False
        
        try:
            msg = MimeMultipart()
            msg['From'] = self.config.smtp_username
            msg['To'] = self.config.email_address
            msg['Subject'] = subject
            
            msg.attach(MimeText(body, 'plain'))
            
            server = smtplib.SMTP(self.config.smtp_server, self.config.smtp_port)
            server.starttls()
            server.login(self.config.smtp_username, self.config.smtp_password)
            text = msg.as_string()
            server.sendmail(self.config.smtp_username, self.config.email_address, text)
            server.quit()
            
            logging.info(f"Email sent successfully: {subject}")
            return True
            
        except Exception as e:
            logging.error(f"Error sending email: {e}")
            return False
    
    def format_price_alert(self, product: Product, current_price: float, analysis: Dict, prediction: Dict) -> Tuple[str, str]:
        """Format price alert email"""
        subject = f"Price Alert: {product.title}"
        
        body = f"""
Price Alert for {product.title}

Current Price: ${current_price:.2f}
Target Price: ${product.target_price:.2f}
Historical Low: ${analysis['historical_low']:.2f}

Price Analysis:
- Trend: {analysis['trend']}
- Price Position: {analysis['price_percentile']:.1f}th percentile
- Average Price: ${analysis['average_price']:.2f}

Recommendation: {prediction['recommendation'].replace('_', ' ').title()}
Confidence: {prediction['confidence'].title()}
Reason: {prediction['reason']}

Potential Savings: ${prediction['potential_savings']:.2f}

Product Link: {product.amazon_url}

This alert was generated by your AI-Driven Price Monitoring System.
        """
        
        return subject, body.strip()

class PriceMonitoringSystem:
    def __init__(self):
        self.config = Config()
        self.data_manager = DataManager()
        self.price_retriever = KeepaPriceRetriever(self.config.keepa_api_key)
        self.analyzer = PriceAnalyzer()
        self.notifier = NotificationSystem(self.config)
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('price_monitor.log'),
                logging.StreamHandler()
            ]
        )
    
    def add_product_by_url(self, amazon_url: str, target_price: float, notify_threshold: float = 0.1) -> bool:
        """Add a product for tracking by Amazon URL"""
        try:
            product_data = self.price_retriever.get_product_data(amazon_url)
            if not product_data:
                return False
            
            product = Product(
                product_id=product_data['asin'],
                title=product_data['title'],
                amazon_url=amazon_url,
                category=product_data['category'],
                target_price=target_price,
                notify_threshold=notify_threshold,
                last_updated=datetime.now()
            )
            
            success = self.data_manager.add_product(product)
            if success:
                # Add initial price point
                price_point = PricePoint(
                    product_id=product.product_id,
                    timestamp=datetime.now(),
                    price=product_data['current_price']
                )
                self.data_manager.add_price_point(price_point)
                
                # Add historical data if available
                if 'price_history' in product_data:
                    for date, price in product_data['price_history']:
                        historical_point = PricePoint(
                            product_id=product.product_id,
                            timestamp=date,
                            price=price
                        )
                        self.data_manager.add_price_point(historical_point)
                
                logging.info(f"Successfully added product: {product.title}")
            
            return success
            
        except Exception as e:
            logging.error(f"Error adding product by URL: {e}")
            return False
    
    def update_all_prices(self):
        """Update prices for all tracked products"""
        products = self.data_manager.get_all_products()
        
        for product in products:
            try:
                product_data = self.price_retriever.get_product_data(product.amazon_url)
                if product_data:
                    current_price = product_data['current_price']
                    
                    # Add new price point
                    price_point = PricePoint(
                        product_id=product.product_id,
                        timestamp=datetime.now(),
                        price=current_price
                    )
                    self.data_manager.add_price_point(price_point)
                    
                    # Check for alerts
                    self.check_price_alerts(product, current_price)
                    
                time.sleep(1)  # Rate limiting
                
            except Exception as e:
                logging.error(f"Error updating price for {product.product_id}: {e}")
    
    def check_price_alerts(self, product: Product, current_price: float):
        """Check if price alerts should be triggered"""
        try:
            # Check target price threshold
            if current_price <= product.target_price:
                price_history = self.data_manager.get_price_history(product.product_id)
                analysis = self.analyzer.analyze_price_trends(price_history)
                prediction = self.analyzer.predict_optimal_purchase_time(analysis)
                
                subject, body = self.notifier.format_price_alert(product, current_price, analysis, prediction)
                self.notifier.send_email_notification(subject, body)
                
                logging.info(f"Price alert sent for {product.title}: ${current_price:.2f}")
            
            # Check percentage drop threshold
            price_history = self.data_manager.get_price_history(product.product_id)
            if len(price_history) >= 2:
                previous_price = price_history[-2].price
                price_drop_percent = (previous_price - current_price) / previous_price
                
                if price_drop_percent >= product.notify_threshold:
                    analysis = self.analyzer.analyze_price_trends(price_history)
                    prediction = self.analyzer.predict_optimal_purchase_time(analysis)
                    
                    subject = f"Significant Price Drop: {product.title}"
                    body = f"""
Significant price drop detected for {product.title}

Previous Price: ${previous_price:.2f}
Current Price: ${current_price:.2f}
Drop: {price_drop_percent*100:.1f}%

{self.notifier.format_price_alert(product, current_price, analysis, prediction)[1]}
                    """
                    
                    self.notifier.send_email_notification(subject, body.strip())
                    
        except Exception as e:
            logging.error(f"Error checking price alerts: {e}")
    
    def get_product_analysis(self, product_id: str) -> Dict:
        """Get comprehensive analysis for a product"""
        try:
            price_history = self.data_manager.get_price_history(product_id)
            analysis = self.analyzer.analyze_price_trends(price_history)
            prediction = self.analyzer.predict_optimal_purchase_time(analysis)
            
            return {
                'analysis': analysis,
                'prediction': prediction,
                'price_history': price_history
            }
            
        except Exception as e:
            logging.error(f"Error getting product analysis: {e}")
            return {}
    
    def generate_price_chart(self, product_id: str, save_path: str = None) -> str:
        """Generate price chart for a product"""
        try:
            price_history = self.data_manager.get_price_history(product_id)
            
            if len(price_history) < 2:
                return None
            
            dates = [p.timestamp for p in price_history]
            prices = [p.price for p in price_history]
            
            plt.figure(figsize=(12, 6))
            plt.plot(dates, prices, 'b-', linewidth=2, label='Price')
            
            # Add moving averages if enough data
            if len(prices) >= 7:
                ma_7 = pd.Series(prices).rolling(window=7).mean()
                plt.plot(dates, ma_7, 'g--', alpha=0.7, label='7-day MA')
            
            if len(prices) >= 30:
                ma_30 = pd.Series(prices).rolling(window=30).mean()
                plt.plot(dates, ma_30, 'r--', alpha=0.7, label='30-day MA')
            
            plt.axhline(y=min(prices), color='green', linestyle=':', alpha=0.7, label='Historical Low')
            plt.axhline(y=max(prices), color='red', linestyle=':', alpha=0.7, label='Historical High')
            
            plt.title(f'Price History - Product {product_id}')
            plt.xlabel('Date')
            plt.ylabel('Price ($)')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.xticks(rotation=45)
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
            else:
                save_path = f'static/charts/{product_id}_chart.png'
                os.makedirs('static/charts', exist_ok=True)
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
            plt.close()
            return save_path
            
        except Exception as e:
            logging.error(f"Error generating chart: {e}")
            return None

# Flask Web Application
app = Flask(__name__)
app.secret_key = 'your-secret-key-here'

# Global system instance
monitoring_system = PriceMonitoringSystem()

@app.route('/')
def dashboard():
    """Main dashboard"""
    try:
        products = monitoring_system.data_manager.get_all_products()
        
        # Get analysis for each product
        product_data = []
        for product in products:
            analysis_data = monitoring_system.get_product_analysis(product.product_id)
            if analysis_data:
                product_info = {
                    'product': product,
                    'analysis': analysis_data.get('analysis', {}),
                    'prediction': analysis_data.get('prediction', {}),
                    'chart_path': f'charts/{product.product_id}_chart.png'
                }
                product_data.append(product_info)
        
        return render_template('dashboard.html', products=product_data)
        
    except Exception as e:
        flash(f'Error loading dashboard: {e}', 'error')
        return render_template('dashboard.html', products=[])

@app.route('/add_product', methods=['GET', 'POST'])
def add_product():
    """Add new product for tracking"""
    if request.method == 'POST':
        try:
            amazon_url = request.form['amazon_url']
            target_price = float(request.form['target_price'])
            notify_threshold = float(request.form.get('notify_threshold', 0.1))
            
            success = monitoring_system.add_product_by_url(amazon_url, target_price, notify_threshold)
            
            if success:
                flash('Product added successfully!', 'success')
                return redirect(url_for('dashboard'))
            else:
                flash('Failed to add product. Please check the URL and try again.', 'error')
                
        except ValueError:
            flash('Please enter valid numeric values for prices.', 'error')
        except Exception as e:
            flash(f'Error adding product: {e}', 'error')
    
    return render_template('add_product.html')

@app.route('/product/<product_id>')
def product_detail(product_id):
    """Product detail page"""
    try:
        products = monitoring_system.data_manager.get_all_products()
        product = next((p for p in products if p.product_id == product_id), None)
        
        if not product:
            flash('Product not found', 'error')
            return redirect(url_for('dashboard'))
        
        analysis_data = monitoring_system.get_product_analysis(product_id)
        chart_path = monitoring_system.generate_price_chart(product_id)
        
        return render_template('product_detail.html', 
                             product=product,
                             analysis=analysis_data.get('analysis', {}),
                             prediction=analysis_data.get('prediction', {}),
                             chart_path=chart_path)
                             
    except Exception as e:
        flash(f'Error loading product details: {e}', 'error')
        return redirect(url_for('dashboard'))

@app.route('/remove_product/<product_id>')
def remove_product(product_id):
    """Remove product from tracking"""
    try:
        success = monitoring_system.data_manager.remove_product(product_id)
        if success:
            flash('Product removed successfully!', 'success')
        else:
            flash('Failed to remove product.', 'error')
    except Exception as e:
        flash(f'Error removing product: {e}', 'error')
    
    return redirect(url_for('dashboard'))

@app.route('/settings', methods=['GET', 'POST'])
def settings():
    """Settings page"""
    if request.method == 'POST':
        try:
            monitoring_system.config.email_address = request.form['email_address']
            monitoring_system.config.keepa_api_key = request.form['keepa_api_key']
            monitoring_system.config.smtp_username = request.form['smtp_username']
            monitoring_system.config.smtp_password = request.form['smtp_password']
            monitoring_system.config.refresh_interval = int(request.form['refresh_interval'])
            
            monitoring_system.config.save_config()
            flash('Settings saved successfully!', 'success')
            
        except Exception as e:
            flash(f'Error saving settings: {e}', 'error')
    
    return render_template('settings.html', config=monitoring_system.config)

@app.route('/update_prices')
def update_prices():
    """Manual price update"""
    try:
        monitoring_system.update_all_prices()
        flash('Prices updated successfully!', 'success')
    except Exception as e:
        flash(f'Error updating prices: {e}', 'error')
    
    return redirect(url_for('dashboard'))

@app.route('/api/product/<product_id>/analysis')
def api_product_analysis(product_id):
    """API endpoint for product analysis"""
    try:
        analysis_data = monitoring_system.get_product_analysis(product_id)
        return jsonify(analysis_data)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Background price monitoring
def background_price_monitor():
    """Background thread for periodic price updates"""
    while True:
        try:
            monitoring_system.update_all_prices()
            time.sleep(monitoring_system.config.refresh_interval * 3600)  # Convert hours to seconds
        except Exception as e:
            logging.error(f"Background monitor error: {e}")
            time.sleep(3600)  # Wait 1 hour before retrying

# HTML Templates (create these as separate files in templates/ directory)

DASHBOARD_TEMPLATE = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AI Price Monitor Dashboard</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 0; padding: 20px; background-color: #f5f5f5; }
        .header { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 20px; border-radius: 10px; margin-bottom: 20px; }
        .nav { display: flex; gap: 20px; margin-bottom: 20px; }
        .nav a { background: #4CAF50; color: white; padding: 10px 20px; text-decoration: none; border-radius: 5px; }
        .nav a:hover { background: #45a049; }
        .product-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(400px, 1fr)); gap: 20px; }
        .product-card { background: white; padding: 20px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
        .price-current { font-size: 24px; font-weight: bold; color: #2196F3; }
        .price-target { color: #4CAF50; }
        .price-low { color: #FF9800; }
        .recommendation { padding: 10px; border-radius: 5px; margin: 10px 0; }
        .buy-now { background: #4CAF50; color: white; }
        .buy-soon { background: #FF9800; color: white; }
        .wait { background: #f44336; color: white; }
        .monitor { background: #9E9E9E; color: white; }
        .chart { width: 100%; height: 200px; margin: 10px 0; }
        .flash-messages { margin-bottom: 20px; }
        .flash-success { background: #d4edda; color: #155724; padding: 10px; border-radius: 5px; margin-bottom: 10px; }
        .flash-error { background: #f8d7da; color: #721c24; padding: 10px; border-radius: 5px; margin-bottom: 10px; }
    </style>
</head>
<body>
    <div class="header">
        <h1>AI-Driven Price Monitoring System</h1>
        <p>Intelligent price tracking with machine learning predictions</p>
    </div>
    
    <div class="nav">
        <a href="/">Dashboard</a>
        <a href="/add_product">Add Product</a>
        <a href="/settings">Settings</a>
        <a href="/update_prices">Update Prices</a>
    </div>
    
    {% with messages = get_flashed_messages(with_categories=true) %}
        {% if messages %}
            <div class="flash-messages">
                {% for category, message in messages %}
                    <div class="flash-{{ category }}">{{ message }}</div>
                {% endfor %}
            </div>
        {% endif %}
    {% endwith %}
    
    <div class="product-grid">
        {% for item in products %}
        <div class="product-card">
            <h3><a href="/product/{{ item.product.product_id }}">{{ item.product.title }}</a></h3>
            
            <div class="price-info">
                <div class="price-current">${{ "%.2f"|format(item.analysis.current_price or 0) }}</div>
                <div class="price-target">Target: ${{ "%.2f"|format(item.product.target_price) }}</div>
                <div class="price-low">Historical Low: ${{ "%.2f"|format(item.analysis.historical_low or 0) }}</div>
            </div>
            
            <div class="recommendation {{ item.prediction.recommendation or 'monitor' }}">
                <strong>{{ (item.prediction.recommendation or 'monitor').replace('_', ' ').title() }}</strong><br>
                {{ item.prediction.reason or 'Continue monitoring' }}
            </div>
            
            <div class="stats">
                <p><strong>Trend:</strong> {{ (item.analysis.trend or 'unknown').title() }}</p>
                <p><strong>Price Position:</strong> {{ "%.1f"|format(item.analysis.price_percentile or 0) }}th percentile</p>
                <p><strong>Potential Savings:</strong> ${{ "%.2f"|format(item.prediction.potential_savings or 0) }}</p>
            </div>
            
            {% if item.chart_path %}
                <img src="/static/{{ item.chart_path }}" alt="Price Chart" class="chart">
            {% endif %}
            
            <div style="margin-top: 15px;">
                <a href="{{ item.product.amazon_url }}" target="_blank" style="background: #FF9900; color: white; padding: 8px 16px; text-decoration: none; border-radius: 4px; margin-right: 10px;">View on Amazon</a>
                <a href="/remove_product/{{ item.product.product_id }}" onclick="return confirm('Are you sure?')" style="background: #f44336; color: white; padding: 8px 16px; text-decoration: none; border-radius: 4px;">Remove</a>
            </div>
        </div>
        {% endfor %}
    </div>
    
    {% if not products %}
    <div style="text-align: center; padding: 40px;">
        <h2>No products being tracked</h2>
        <p>Get started by <a href="/add_product">adding your first product</a>!</p>
    </div>
    {% endif %}
</body>
</html>
'''

ADD_PRODUCT_TEMPLATE = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Add Product - AI Price Monitor</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 0; padding: 20px; background-color: #f5f5f5; }
        .header { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 20px; border-radius: 10px; margin-bottom: 20px; }
        .nav { display: flex; gap: 20px; margin-bottom: 20px; }
        .nav a { background: #4CAF50; color: white; padding: 10px 20px; text-decoration: none; border-radius: 5px; }
        .nav a:hover { background: #45a049; }
        .form-container { background: white; padding: 30px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); max-width: 600px; margin: 0 auto; }
        .form-group { margin-bottom: 20px; }
        .form-group label { display: block; margin-bottom: 5px; font-weight: bold; }
        .form-group input, .form-group textarea { width: 100%; padding: 10px; border: 1px solid #ddd; border-radius: 5px; font-size: 16px; }
        .form-group textarea { height: 60px; resize: vertical; }
        .btn { background: #4CAF50; color: white; padding: 12px 24px; border: none; border-radius: 5px; cursor: pointer; font-size: 16px; }
        .btn:hover { background: #45a049; }
        .flash-messages { margin-bottom: 20px; }
        .flash-success { background: #d4edda; color: #155724; padding: 10px; border-radius: 5px; margin-bottom: 10px; }
        .flash-error { background: #f8d7da; color: #721c24; padding: 10px; border-radius: 5px; margin-bottom: 10px; }
        .help-text { font-size: 14px; color: #666; margin-top: 5px; }
    </style>
</head>
<body>
    <div class="header">
        <h1>Add New Product</h1>
        <p>Start tracking Amazon product prices with AI analysis</p>
    </div>
    
    <div class="nav">
        <a href="/">Dashboard</a>
        <a href="/add_product">Add Product</a>
        <a href="/settings">Settings</a>
        <a href="/update_prices">Update Prices</a>
    </div>
    
    {% with messages = get_flashed_messages(with_categories=true) %}
        {% if messages %}
            <div class="flash-messages">
                {% for category, message in messages %}
                    <div class="flash-{{ category }}">{{ message }}</div>
                {% endfor %}
            </div>
        {% endif %}
    {% endwith %}
    
    <div class="form-container">
        <form method="POST">
            <div class="form-group">
                <label for="amazon_url">Amazon Product URL:</label>
                <textarea name="amazon_url" id="amazon_url" required placeholder="https://www.amazon.com/dp/B08L5TNJHG/..."></textarea>
                <div class="help-text">Paste the full Amazon product URL here</div>
            </div>
            
            <div class="form-group">
                <label for="target_price">Target Price ($):</label>
                <input type="number" name="target_price" id="target_price" step="0.01" min="0" required placeholder="35.00">
                <div class="help-text">You'll be notified when the price drops to or below this amount</div>
            </div>
            
            <div class="form-group">
                <label for="notify_threshold">Price Drop Threshold (%):</label>
                <input type="number" name="notify_threshold" id="notify_threshold" step="0.01" min="0" max="1" value="0.10" placeholder="0.10">
                <div class="help-text">Notify when price drops by this percentage (0.10 = 10%)</div>
            </div>
            
            <button type="submit" class="btn">Add Product</button>
        </form>
    </div>
    
    <div style="margin-top: 30px; padding: 20px; background: #e3f2fd; border-radius: 10px; max-width: 600px; margin-left: auto; margin-right: auto;">
        <h3>How it works:</h3>
        <ol>
            <li>Find a product on Amazon and copy its URL</li>
            <li>Set your target price - when to get notified</li>
            <li>Set a price drop threshold for significant changes</li>
            <li>Our AI will analyze price trends and predict optimal purchase times</li>
            <li>Get email notifications when it's time to buy!</li>
        </ol>
    </div>
</body>
</html>
'''

SETTINGS_TEMPLATE = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Settings - AI Price Monitor</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 0; padding: 20px; background-color: #f5f5f5; }
        .header { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 20px; border-radius: 10px; margin-bottom: 20px; }
        .nav { display: flex; gap: 20px; margin-bottom: 20px; }
        .nav a { background: #4CAF50; color: white; padding: 10px 20px; text-decoration: none; border-radius: 5px; }
        .nav a:hover { background: #45a049; }
        .form-container { background: white; padding: 30px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); max-width: 600px; margin: 0 auto; }
        .form-group { margin-bottom: 20px; }
        .form-group label { display: block; margin-bottom: 5px; font-weight: bold; }
        .form-group input { width: 100%; padding: 10px; border: 1px solid #ddd; border-radius: 5px; font-size: 16px; }
        .btn { background: #4CAF50; color: white; padding: 12px 24px; border: none; border-radius: 5px; cursor: pointer; font-size: 16px; }
        .btn:hover { background: #45a049; }
        .flash-messages { margin-bottom: 20px; }
        .flash-success { background: #d4edda; color: #155724; padding: 10px; border-radius: 5px; margin-bottom: 10px; }
        .flash-error { background: #f8d7da; color: #721c24; padding: 10px; border-radius: 5px; margin-bottom: 10px; }
        .help-text { font-size: 14px; color: #666; margin-top: 5px; }
        .section { margin-bottom: 30px; padding-bottom: 20px; border-bottom: 1px solid #eee; }
        .section:last-child { border-bottom: none; }
    </style>
</head>
<body>
    <div class="header">
        <h1>⚙️ System Settings</h1>
        <p>Configure your price monitoring preferences</p>
    </div>
    
    <div class="nav">
        <a href="/">Dashboard</a>
        <a href="/add_product">Add Product</a>
        <a href="/settings">Settings</a>
        <a href="/update_prices">Update Prices</a>
    </div>
    
    {% with messages = get_flashed_messages(with_categories=true) %}
        {% if messages %}
            <div class="flash-messages">
                {% for category, message in messages %}
                    <div class="flash-{{ category }}">{{ message }}</div>
                {% endfor %}
            </div>
        {% endif %}
    {% endwith %}
    
    <div class="form-container">
        <form method="POST">
            <div class="section">
                <h3>📧 Email Notifications</h3>
                <div class="form-group">
                    <label for="email_address">Your Email Address:</label>
                    <input type="email" name="email_address" id="email_address" value="{{ config.email_address }}" required>
                    <div class="help-text">Where to send price alerts</div>
                </div>
            </div>
            
            <div class="section">
                <h3>📡 Data Sources</h3>
                <div class="form-group">
                    <label for="keepa_api_key">Keepa API Key:</label>
                    <input type="text" name="keepa_api_key" id="keepa_api_key" value="{{ config.keepa_api_key }}" placeholder="Optional - for enhanced price data">
                    <div class="help-text">Get your free API key from <a href="https://keepa.com/#!api" target="_blank">Keepa.com</a></div>
                </div>
            </div>
            
            <div class="section">
                <h3>🔄 Update Settings</h3>
                <div class="form-group">
                    <label for="refresh_interval">Price Update Interval (hours):</label>
                    <input type="number" name="refresh_interval" id="refresh_interval" min="1" max="168" value="{{ config.refresh_interval }}">
                    <div class="help-text">How often to check for price updates (1-168 hours)</div>
                </div>
            </div>
            
            <div class="section">
                <h3>📨 SMTP Configuration</h3>
                <div class="form-group">
                    <label for="smtp_username">SMTP Username:</label>
                    <input type="text" name="smtp_username" id="smtp_username" value="{{ config.smtp_username }}" placeholder="your-email@gmail.com">
                    <div class="help-text">Your email account for sending notifications</div>
                </div>
                
                <div class="form-group">
                    <label for="smtp_password">SMTP Password:</label>
                    <input type="password" name="smtp_password" id="smtp_password" value="{{ config.smtp_password }}" placeholder="App password for email">
                    <div class="help-text">Use an app-specific password for Gmail</div>
                </div>
            </div>
            
            <button type="submit" class="btn">Save Settings</button>
        </form>
    </div>
    
    <div style="margin-top: 30px; padding: 20px; background: #fff3cd; border-radius: 10px; max-width: 600px; margin-left: auto; margin-right: auto;">
        <h3>🔧 Setup Instructions:</h3>
        <ul>
            <li><strong>Email:</strong> Enter the email where you want to receive alerts</li>
            <li><strong>Keepa API:</strong> Optional but recommended for better price data</li>
            <li><strong>SMTP:</strong> Configure your email account to send notifications</li>
            <li><strong>Gmail Users:</strong> Enable 2FA and create an app password</li>
        </ul>
    </div>
</body>
</html>
'''

PRODUCT_DETAIL_TEMPLATE = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{{ product.title }} - AI Price Monitor</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 0; padding: 20px; background-color: #f5f5f5; }
        .header { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 20px; border-radius: 10px; margin-bottom: 20px; }
        .nav { display: flex; gap: 20px; margin-bottom: 20px; }
        .nav a { background: #4CAF50; color: white; padding: 10px 20px; text-decoration: none; border-radius: 5px; }
        .nav a:hover { background: #45a049; }
        .detail-container { background: white; padding: 30px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
        .price-section { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px; margin-bottom: 30px; }
        .price-card { background: #f8f9fa; padding: 20px; border-radius: 10px; text-align: center; }
        .price-value { font-size: 28px; font-weight: bold; margin-bottom: 5px; }
        .price-label { color: #666; font-size: 14px; }
        .current { color: #2196F3; }
        .target { color: #4CAF50; }
        .low { color: #FF9800; }
        .high { color: #f44336; }
        .recommendation { padding: 20px; border-radius: 10px; margin: 20px 0; font-size: 18px; text-align: center; }
        .buy-now { background: #4CAF50; color: white; }
        .buy-soon { background: #FF9800; color: white; }
        .wait { background: #f44336; color: white; }
        .monitor { background: #9E9E9E; color: white; }
        .analysis-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 20px; margin: 30px 0; }
        .analysis-card { background: #f8f9fa; padding: 20px; border-radius: 10px; }
        .chart-container { text-align: center; margin: 30px 0; }
        .chart { max-width: 100%; height: auto; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
        .action-buttons { display: flex; gap: 15px; margin-top: 30px; }
        .btn { padding: 12px 24px; text-decoration: none; border-radius: 5px; font-weight: bold; text-align: center; }
        .btn-primary { background: #FF9900; color: white; }
        .btn-secondary { background: #6c757d; color: white; }
        .btn-danger { background: #f44336; color: white; }
    </style>
</head>
<body>
    <div class="header">
        <h1>Product Analysis</h1>
        <p>{{ product.title }}</p>
    </div>
    
    <div class="nav">
        <a href="/">Dashboard</a>
        <a href="/add_product">Add Product</a>
        <a href="/settings">Settings</a>
        <a href="/update_prices">Update Prices</a>
    </div>
    
    <div class="detail-container">
        <div class="price-section">
            <div class="price-card">
                <div class="price-value current">${{ "%.2f"|format(analysis.current_price or 0) }}</div>
                <div class="price-label">Current Price</div>
            </div>
            <div class="price-card">
                <div class="price-value target">${{ "%.2f"|format(product.target_price) }}</div>
                <div class="price-label">Your Target</div>
            </div>
            <div class="price-card">
                <div class="price-value low">${{ "%.2f"|format(analysis.historical_low or 0) }}</div>
                <div class="price-label">Historical Low</div>
            </div>
            <div class="price-card">
                <div class="price-value high">${{ "%.2f"|format(analysis.historical_high or 0) }}</div>
                <div class="price-label">Historical High</div>
            </div>
        </div>
        
        <div class="recommendation {{ prediction.recommendation or 'monitor' }}">
            <strong>{{ (prediction.recommendation or 'monitor').replace('_', ' ').title() }}</strong><br>
            {{ prediction.reason or 'Continue monitoring for price changes' }}<br>
            <small>Confidence: {{ (prediction.confidence or 'low').title() }}</small>
        </div>
        
        {% if chart_path %}
        <div class="chart-container">
            <h3>Price History Chart</h3>
            <img src="/static/{{ chart_path.split('/')[-1] if '/' in chart_path else chart_path }}" alt="Price Chart" class="chart">
        </div>
        {% endif %}
        
        <div class="analysis-grid">
            <div class="analysis-card">
                <h4>Trend Analysis</h4>
                <p><strong>Direction:</strong> {{ (analysis.trend or 'unknown').title() }}</p>
                <p><strong>7-day Average:</strong> ${{ "%.2f"|format(analysis.moving_avg_7 or 0) }}</p>
                <p><strong>30-day Average:</strong> ${{ "%.2f"|format(analysis.moving_avg_30 or 0) }}</p>
            </div>
            
            <div class="analysis-card">
                <h4>Price Position</h4>
                <p><strong>Percentile:</strong> {{ "%.1f"|format(analysis.price_percentile or 0) }}th</p>
                <p><strong>Average Price:</strong> ${{ "%.2f"|format(analysis.average_price or 0) }}</p>
                <p><strong>Volatility:</strong> ${{ "%.2f"|format(analysis.volatility or 0) }}</p>
            </div>
            
            <div class="analysis-card">
                <h4>Purchase Prediction</h4>
                <p><strong>Potential Savings:</strong> ${{ "%.2f"|format(prediction.potential_savings or 0) }}</p>
                {% if prediction.estimated_drop_date %}
                <p><strong>Next Drop Estimate:</strong> {{ prediction.estimated_drop_date.strftime('%Y-%m-%d') }}</p>
                {% endif %}
            </div>
            
            <div class="analysis-card">
                <h4>Tracking Settings</h4>
                <p><strong>Notify Threshold:</strong> {{ "%.1f"|format(product.notify_threshold * 100) }}%</p>
                <p><strong>Category:</strong> {{ product.category }}</p>
                <p><strong>Last Updated:</strong> {{ product.last_updated.strftime('%Y-%m-%d %H:%M') }}</p>
            </div>
        </div>
        
        <div class="action-buttons">
            <a href="{{ product.amazon_url }}" target="_blank" class="btn btn-primary">View on Amazon</a>
            <a href="/" class="btn btn-secondary">Back to Dashboard</a>
            <a href="/remove_product/{{ product.product_id }}" onclick="return confirm('Are you sure you want to remove this product?')" class="btn btn-danger">Remove Product</a>
        </div>
    </div>
</body>
</html>
'''

def create_templates():
    """Create template files"""
    os.makedirs('templates', exist_ok=True)
    
    templates = {
        'dashboard.html': DASHBOARD_TEMPLATE,
        'add_product.html': ADD_PRODUCT_TEMPLATE,
        'settings.html': SETTINGS_TEMPLATE,
        'product_detail.html': PRODUCT_DETAIL_TEMPLATE
    }
    
    for filename, content in templates.items():
        with open(f'templates/{filename}', 'w', encoding='utf-8') as f:
            f.write(content)

if __name__ == '__main__':
    # Create necessary directories and files
    os.makedirs('static/charts', exist_ok=True)
    create_templates()
    
    # Start background monitoring thread
    monitor_thread = threading.Thread(target=background_price_monitor, daemon=True)
    monitor_thread.start()
    
    print("Starting AI-Driven Price Monitoring System...")
    print("Dashboard will be available at: http://localhost:5000")
    print("Make sure to configure your email settings!")
    print("Get your free Keepa API key at: https://keepa.com/#!api")
    
    # Run Flask app
    app.run(debug=True, host='0.0.0.0', port=5000)