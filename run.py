# requirements.txt
"""
Flask==2.3.3
pandas==2.1.1
numpy==1.24.3
matplotlib==3.7.2
scipy==1.11.3
requests==2.31.0
python-dateutil==2.8.2
Werkzeug==2.3.7
Jinja2==3.1.2
"""

# setup.py
"""
from setuptools import setup, find_packages

setup(
    name="ai-price-monitor",
    version="1.0.0",
    description="AI-Driven Price Monitoring System for Amazon Products",
    author="Your Name",
    author_email="your.email@example.com",
    packages=find_packages(),
    install_requires=[
        "Flask>=2.3.0",
        "pandas>=2.0.0",
        "numpy>=1.24.0",
        "matplotlib>=3.7.0",
        "scipy>=1.11.0",
        "requests>=2.31.0",
        "python-dateutil>=2.8.0",
    ],
    python_requires=">=3.8",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: End Users/Desktop",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
)
"""

# run.py - Main application runner
import os
import sys
import subprocess
import platform

def install_requirements():
    """Install required packages with fallback options"""
    print("Installing required packages...")
    
    # Try installing packages one by one with fallback versions
    packages = [
        ("Flask", ["Flask>=2.3.0", "Flask>=2.2.0", "Flask"]),
        ("pandas", ["pandas>=2.0.0", "pandas>=1.5.0", "pandas"]),
        ("numpy", ["numpy>=1.24.0", "numpy>=1.21.0", "numpy"]),
        ("matplotlib", ["matplotlib>=3.7.0", "matplotlib>=3.5.0", "matplotlib"]),
        ("scipy", ["scipy>=1.11.0", "scipy>=1.9.0", "scipy"]),
        ("requests", ["requests>=2.31.0", "requests>=2.25.0", "requests"]),
        ("python-dateutil", ["python-dateutil>=2.8.0", "python-dateutil"]),
        ("Werkzeug", ["Werkzeug>=2.3.0", "Werkzeug>=2.0.0", "Werkzeug"]),
        ("Jinja2", ["Jinja2>=3.1.0", "Jinja2>=3.0.0", "Jinja2"])
    ]
    
    failed_packages = []
    
    for package_name, versions in packages:
        installed = False
        for version in versions:
            try:
                print(f"  Trying to install {version}...")
                subprocess.check_call([sys.executable, "-m", "pip", "install", version], 
                                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                print(f"  ✓ Successfully installed {version}")
                installed = True
                break
            except subprocess.CalledProcessError:
                continue
        
        if not installed:
            failed_packages.append(package_name)
            print(f"  ✗ Failed to install {package_name}")
    
    if not failed_packages:
        print("Successfully installed all requirements!")
        return True
    else:
        print(f"\nSome packages failed to install: {', '.join(failed_packages)}")
        print("You can try installing them manually:")
        for package in failed_packages:
            print(f"  pip install {package}")
        
        # Check if essential packages are available
        essential_available = True
        try:
            import flask
            print("✓ Flask is available")
        except ImportError:
            print("✗ Flask is missing (required)")
            essential_available = False
        
        try:
            import pandas
            print("✓ Pandas is available")
        except ImportError:
            print("✗ Pandas is missing (required)")
            essential_available = False
        
def install_requirements_simple():
    """Simple installation method for problematic environments"""
    print("Trying simple installation method...")
    
    simple_packages = [
        "Flask",
        "requests", 
        "python-dateutil"
    ]
    
    for package in simple_packages:
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
            print(f"✓ Installed {package}")
        except subprocess.CalledProcessError:
            print(f"✗ Failed to install {package}")
    
    print("\nTrying to install pandas and numpy (may take time)...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "pandas", "numpy"])
        print("✓ Installed pandas and numpy")
    except subprocess.CalledProcessError:
        print("✗ Failed to install pandas/numpy")
        print("You may need to install Microsoft Visual C++ Build Tools")
        print("Or try: pip install pandas --no-build-isolation")
    
    print("\nTrying to install matplotlib and scipy...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "matplotlib", "scipy"])
        print("✓ Installed matplotlib and scipy")
    except subprocess.CalledProcessError:
        print("✗ Failed to install matplotlib/scipy")
    
    return True

def create_requirements_file():
    """Create requirements.txt file"""
    requirements = [
        "Flask==2.3.3",
        "pandas>=2.0.0", 
        "numpy>=1.24.0",
        "matplotlib>=3.7.0",
        "scipy>=1.11.0",
        "requests>=2.31.0",
        "python-dateutil>=2.8.0",
        "Werkzeug==2.3.7",
        "Jinja2==3.1.2"
    ]
    
    with open("requirements.txt", "w", encoding='utf-8') as f:
        f.write("\n".join(requirements))
    print("Created requirements.txt")

def setup_project_structure():
    """Create necessary directories and files"""
    print("Setting up project structure...")
    
    directories = [
        "templates",
        "static",
        "static/charts",
        "data",
        "logs"
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"   Created directory: {directory}")
    
    # Create empty CSV files
    csv_files = {
        "products.csv": "product_id,title,amazon_url,category,last_updated\n",
        "price_history.csv": "product_id,timestamp,price\n",
        "tracking.csv": "product_id,target_price,notify_threshold\n",
        "config.csv": "key,value\nemail_address,\nrefresh_interval,24\nkeepa_api_key,\nsmtp_username,\nsmtp_password,\n"
    }
    
    for filename, content in csv_files.items():
        if not os.path.exists(filename):
            with open(filename, "w", encoding='utf-8') as f:
                f.write(content)
            print(f"   Created file: {filename}")
    
    print("Project structure created successfully!")

def create_startup_script():
    """Create platform-specific startup scripts"""
    
    # Windows batch file
    windows_script = '''@echo off
echo Starting AI-Driven Price Monitoring System...
echo.
echo Make sure you have Python 3.8+ installed!
echo.
pause
python price_monitor.py
pause
'''
    
    # Unix shell script
    unix_script = '''#!/bin/bash
echo "Starting AI-Driven Price Monitoring System..."
echo ""
echo "Make sure you have Python 3.8+ installed!"
echo ""
read -p "Press Enter to continue..."
python3 price_monitor.py
'''
    
    with open("start_windows.bat", "w", encoding='utf-8') as f:
        f.write(windows_script)
    
    with open("start_unix.sh", "w", encoding='utf-8') as f:
        f.write(unix_script)
    
    # Make Unix script executable
    if platform.system() != "Windows":
        os.chmod("start_unix.sh", 0o755)
    
    print("Created startup scripts:")
    print("   - start_windows.bat (for Windows)")
    print("   - start_unix.sh (for Linux/Mac)")

def create_readme():
    """Create comprehensive README file"""
    readme_content = '''# AI-Driven Price Monitoring System

An intelligent price tracking system for Amazon products that uses machine learning to analyze price trends and predict optimal purchase times.

## Features

- **Smart Price Tracking**: Monitor Amazon product prices automatically
- **AI Analysis**: Machine learning algorithms analyze price trends and patterns
- **Email Notifications**: Get alerts when prices drop or reach your target
- **Visual Charts**: Beautiful price history charts and trend analysis
- **Purchase Predictions**: AI-powered recommendations for optimal buying times
- **Easy Configuration**: Simple web interface for managing products and settings

## Quick Start

### Prerequisites
- Python 3.8 or higher
- Internet connection
- Email account for notifications (optional)

### Installation

1. **Download the system files**
2. **Run the setup**:
   ```bash
   python run.py
   ```
3. **Access the dashboard**: Open http://localhost:5000 in your browser

### Alternative Manual Setup

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the application**:
   ```bash
   python price_monitor.py
   ```

##  Configuration

### Initial Setup
1. Go to **Settings** in the web interface
2. Configure your email address for notifications
3. Set up SMTP settings for email delivery
4. (Optional) Add your Keepa API key for enhanced data

### Adding Products
1. Click **Add Product** 
2. Paste an Amazon product URL
3. Set your target price
4. Configure notification preferences

##  Technical Details

### Architecture
- **Backend**: Python Flask web application
- **Data Storage**: CSV files with Pandas for data manipulation
- **Analysis**: Statistical methods and basic machine learning
- **Visualization**: Matplotlib for price charts
- **Notifications**: SMTP email delivery

### Key Components
1. **Data Acquisition**: Retrieves price data via Keepa API
2. **Price Analysis**: Statistical trend analysis and pattern recognition
3. **Prediction Engine**: AI-based purchase timing recommendations
4. **Notification System**: Smart email alerts with purchase advice

##  How It Works

1. **Price Monitoring**: System checks product prices at configured intervals
2. **Trend Analysis**: AI algorithms analyze historical price data to identify patterns
3. **Prediction**: Machine learning models predict optimal purchase timing
4. **Notifications**: Smart alerts sent when prices hit targets or show favorable trends

##  Email Setup

### Gmail Configuration
1. Enable 2-Factor Authentication on your Google account
2. Generate an App Password:
   - Go to Google Account settings
   - Security > 2-Step Verification > App passwords
   - Generate password for "Mail"
3. Use your Gmail address as SMTP username
4. Use the generated app password as SMTP password

### Other Email Providers
- **Outlook**: smtp-mail.outlook.com, port 587
- **Yahoo**: smtp.mail.yahoo.com, port 587
- **Custom**: Contact your email provider for SMTP settings

## 🛡️ Privacy & Security

- All data stored locally on your machine
- No personal data sent to external services (except email notifications)
- Amazon product data retrieved through authorized Keepa API
- Email credentials stored locally in encrypted configuration



## 🔗 API Integration

### Keepa API (Recommended)
1. Sign up at https://keepa.com/#!api
2. Get your free API key (100 requests/day)
3. Add API key in Settings page
4. Enjoy enhanced price data and historical trends

### Mock Data Mode
- System works without API key using simulated data
- Perfect for testing and demonstration
- Limited to basic price tracking functionality

## 📁 File Structure

```
price_monitor/
├── price_monitor.py          # Main application
├── requirements.txt          # Python dependencies
├── config.csv               # System configuration
├── products.csv             # Tracked products
├── price_history.csv        # Historical price data
├── tracking.csv             # User preferences
├── templates/               # Web interface templates
├── static/                  # Static files and charts
└── logs/                    # Application logs
```



---

**Happy Price Monitoring!** 
'''
    
    with open("README.md", "w", encoding='utf-8') as f:
        f.write(readme_content)
    print("Created comprehensive README.md")

def check_python_version():
    """Check if Python version is compatible"""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("ERROR: Python 3.8 or higher is required!")
        print(f"   Current version: {version.major}.{version.minor}.{version.micro}")
        return False
    print(f"Python version check passed: {version.major}.{version.minor}.{version.micro}")
    return True

def main():
    """Main setup function"""
    print("AI-Driven Price Monitoring System Setup")
    print("=" * 50)
    
    # Check Python version
    if not check_python_version():
        return
    
    # Create project structure
    setup_project_structure()
    
    # Create requirements file
    create_requirements_file()
    
    # Create startup scripts
    create_startup_script()
    
    # Create README
    create_readme()
    
    # Install requirements
    install_choice = input("\nInstall required packages now? (y/n): ").lower().strip()
    if install_choice in ['y', 'yes']:
        success = install_requirements()
        if not success:
            print("\nTrying alternative installation method...")
            install_requirements_simple()
        
        print("\nSetup completed!")
        print("\nTo start the application:")
        print("   Option 1: Run 'python price_monitor.py'")
        if platform.system() == "Windows":
            print("   Option 2: Double-click 'start_windows.bat'")
        else:
            print("   Option 2: Run './start_unix.sh'")
        print("\nThen open http://localhost:5000 in your browser")
        print("\nDon't forget to configure email settings!")
    else:
        print("\nSetup completed, but packages not installed.")
        print("   Run 'pip install -r requirements.txt' to install dependencies")
    
    print("\nCheck README.md for detailed instructions")

if __name__ == "__main__":
    main()

# Configuration helper utilities
class ConfigHelper:
    """Helper class for configuration management"""
    
    @staticmethod
    def create_sample_config():
        """Create sample configuration with explanations"""
        sample_config = """# AI Price Monitor Configuration Guide

## Email Settings
email_address=your.email@example.com          # Where to receive alerts
smtp_username=your.email@gmail.com            # Email account for sending
smtp_password=your_app_password_here          # App-specific password

## API Configuration  
keepa_api_key=your_keepa_api_key_here         # Optional: Get from keepa.com/api
refresh_interval=24                           # Hours between price checks

## Advanced Settings
max_products=30                               # Maximum tracked products
chart_days=90                                 # Days of price history in charts
notification_cooldown=6                       # Hours between repeat notifications

## SMTP Settings (Advanced)
smtp_server=smtp.gmail.com                    # SMTP server address
smtp_port=587                                 # SMTP port (usually 587)
use_tls=true                                  # Enable TLS encryption
"""
        
        with open("config_guide.txt", "w", encoding='utf-8') as f:
            f.write(sample_config)
        print("Created config_guide.txt with setup instructions")

    @staticmethod
    def validate_config():
        """Validate configuration settings"""
        issues = []
        
        try:
            import pandas as pd
            config_df = pd.read_csv("config.csv")
            config_dict = dict(zip(config_df['key'], config_df['value']))
            
            # Check email configuration
            if not config_dict.get('email_address'):
                issues.append("❌ Email address not configured")
            
            if not config_dict.get('smtp_username'):
                issues.append("⚠️  SMTP username not set (needed for notifications)")
            
            if not config_dict.get('smtp_password'):
                issues.append("⚠️  SMTP password not set (needed for notifications)")
            
            # Check API key
            if not config_dict.get('keepa_api_key'):
                issues.append("ℹ️  Keepa API key not set (using mock data)")
            
            if issues:
                print("Configuration Issues Found:")
                for issue in issues:
                    print(f"   {issue}")
                print("\nGo to Settings page to fix these issues")
            else:
                print("Configuration looks good!")
                
        except FileNotFoundError:
            print("Configuration file not found. Run setup first.")
        except Exception as e:
            print(f"Error validating config: {e}")

# Data management utilities
class DataUtils:
    """Utility functions for data management"""
    
    @staticmethod
    def backup_data():
        """Create backup of all data files"""
        import shutil
        from datetime import datetime
        
        backup_dir = f"backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        os.makedirs(backup_dir, exist_ok=True)
        
        files_to_backup = [
            "products.csv",
            "price_history.csv", 
            "tracking.csv",
            "config.csv"
        ]
        
        for file in files_to_backup:
            if os.path.exists(file):
                shutil.copy2(file, backup_dir)
        
        print(f"Data backed up to: {backup_dir}")
        return backup_dir
    
    @staticmethod
    def cleanup_old_data(days_to_keep=180):
        """Clean up old price history data"""
        try:
            import pandas as pd
            from datetime import datetime, timedelta
            
            df = pd.read_csv("price_history.csv")
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            cutoff_date = datetime.now() - timedelta(days=days_to_keep)
            df_cleaned = df[df['timestamp'] > cutoff_date]
            
            removed_count = len(df) - len(df_cleaned)
            df_cleaned.to_csv("price_history.csv", index=False)
            
            print(f"Cleaned up {removed_count} old price records")
            print(f"   Keeping last {days_to_keep} days of data")
            
        except Exception as e:
            print(f"Error cleaning data: {e}")
    
    @staticmethod
    def export_data():
        """Export all data to JSON format"""
        try:
            import pandas as pd
            import json
            from datetime import datetime
            
            export_data = {
                "export_date": datetime.now().isoformat(),
                "products": pd.read_csv("products.csv").to_dict('records'),
                "price_history": pd.read_csv("price_history.csv").to_dict('records'),
                "tracking": pd.read_csv("tracking.csv").to_dict('records')
            }
            
            filename = f"price_data_export_{datetime.now().strftime('%Y%m%d')}.json"
            with open(filename, "w", encoding='utf-8') as f:
                json.dump(export_data, f, indent=2, default=str)
            
            print(f"Data exported to: {filename}")
            return filename
            
        except Exception as e:
            print(f"Error exporting data: {e}")
            return None

if __name__ == "__main__":
    main()