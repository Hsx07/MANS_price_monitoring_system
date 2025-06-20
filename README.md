# AI-Driven Price Monitoring System

Track Amazon product prices and get smart email alerts when it's time to buy. Uses AI to analyze price trends and predict optimal purchase timing.

## Quick Start

1. **Install & Setup**
   ```bash
   python run.py
   python price_monitor.py
   ```
   Open http://localhost:5000

2. **Configure Email** (Required for alerts)
   - Go to Settings in the web interface
   - Enter your Gmail address
   - Get Gmail App Password (see below)
   - Enter the 16-character App Password

3. **Test It Works**
   ```bash
   python price_monitor_tester.py
   ```
   This script simulates price drops and tests email alerts. Choose option 2 to test email configuration, then option 1 to simulate a product price dropping to your target price and receive a real alert email.

## Setup Output

After running `python run.py`, you should see output like this showing successful installation:

![Setup Output](https://i.imgur.com/DqyyeKb.png)

*The setup script automatically installs dependencies and creates all necessary files (config.csv, products.csv, etc.)*

## Gmail App Password Setup

If you have 2-Factor Authentication on Gmail (you should!), you need an App Password:

1. Go to [Google Account Security](https://myaccount.google.com/security)
2. Click "App passwords" 
3. Generate password for "Mail"
4. Copy the 16-character password (like `abcd efgh ijkl mnop`)
5. Use this in Settings, NOT your regular Gmail password

## How to Use

1. **Add Products**: Click "Add Product", paste Amazon URL, set target price
2. **Get Alerts**: Receive emails when prices drop to your target

The system checks prices daily and sends beautiful email alerts with savings analysis.

## Features

- Smart price tracking
- Email notifications for price drops
- Purchase timing recommendations
- Web dashboard for easy management

## Requirements

- Python 3.8+
- Gmail account
- Internet connection

## Files

- `price_monitor.py` - Main application
- `price_monitor_tester.py` - Simulates price drops and tests email alerts
- `run.py` - Setup script
- `config.csv` - Stores your email settings and system preferences (auto-created)

## Troubleshooting

**Email not working?**
- Make sure you're using Gmail App Password, not regular password
- Check 2-Factor Authentication is enabled

**Product not found?**
- Use full Amazon URL from browser address bar
- Make sure product is available in your region



---

That's it! Start saving money with AI-powered price monitoring.
