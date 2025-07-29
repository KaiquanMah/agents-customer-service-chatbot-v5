#!/usr/bin/env python3
"""
Setup script for Data Analyst Chatbot - Assignment 2
Helps verify installation and configuration
"""

import os
import sys
import pandas as pd
from pathlib import Path

def check_python_version():
    """Check if Python version is compatible"""
    if sys.version_info < (3, 8):
        print("❌ Python 3.8+ required. Current version:", sys.version)
        return False
    print(f"✅ Python version: {sys.version.split()[0]}")
    return True

def check_dependencies():
    """Check if required packages are installed"""
    required_packages = [
        'streamlit',
        'pandas', 
        'langchain',
        'langchain_openai',
        'langgraph',
        'openai'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package} installed")
        except ImportError:
            missing_packages.append(package)
            print(f"❌ {package} missing")
    
    if missing_packages:
        print(f"\n🔧 Install missing packages:")
        print(f"pip install {' '.join(missing_packages)}")
        return False
    
    return True

def check_dataset():
    """Check if dataset file exists"""
    dataset_path = "Bitext_Sample_Customer_Support_Training_Dataset_27K_responses-v11.csv"
    
    if os.path.exists(dataset_path):
        try:
            df = pd.read_csv(dataset_path)
            print(f"✅ Dataset found: {len(df)} rows")
            
            # Check required columns
            required_columns = ['instruction', 'category', 'intent', 'response']
            missing_columns = [col for col in required_columns if col not in df.columns]
            
            if missing_columns:
                print(f"❌ Missing columns in dataset: {missing_columns}")
                return False
            else:
                print(f"✅ Dataset columns: {list(df.columns)}")
                return True
                
        except Exception as e:
            print(f"❌ Error reading dataset: {e}")
            return False
    else:
        print(f"❌ Dataset not found: {dataset_path}")
        print("📁 Place the dataset file in the project root directory")
        return False

def check_api_key():
    """Check if API key is configured"""
    from dotenv import load_dotenv
    load_dotenv()
    api_key = os.getenv("NEBIUS_YAIR")
    
    if api_key:
        print("✅ NEBIUS_YAIR API key configured")
        return True
    else:
        print("❌ NEBIUS_YAIR API key not found")
        print("🔑 Set your API key:")
        print("   export NEBIUS_YAIR='your-api-key'")
        print("   or create a .env file")
        return False

def create_env_template():
    """Create .env template file"""
    env_template = """# API Configuration
NEBIUS_YAIR=your-nebius-api-key-here

# Optional: Enable debug mode
DEBUG=false
"""
    
    if not os.path.exists(".env"):
        with open(".env.template", "w") as f:
            f.write(env_template)
        print("📝 Created .env.template file")
        print("   Copy to .env and add your API key")

def main():
    """Run all setup checks"""
    print("🚀 Data Analyst Chatbot - Assignment 2 Setup Check\n")
    
    checks = [
        ("Python Version", check_python_version),
        ("Dependencies", check_dependencies),
        ("Dataset", check_dataset),
        ("API Key", check_api_key)
    ]
    
    results = []
    
    for check_name, check_func in checks:
        print(f"\n📋 Checking {check_name}...")
        results.append(check_func())
    
    create_env_template()
    
    print(f"\n{'='*50}")
    if all(results):
        print("🎉 All checks passed! Ready to run the application:")
        print("   streamlit run streamlit_app.py")
    else:
        print("⚠️  Some checks failed. Please fix the issues above.")
        print("💡 After fixing issues, run this script again.")
    
    print(f"{'='*50}")

if __name__ == "__main__":
    main()
