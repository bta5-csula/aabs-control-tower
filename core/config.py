"""
AABS Control Tower Configuration
"""

class Config:
    APP_NAME = "AABS Control Tower"
    VERSION = "8.3"
    
    # Data Paths
    GBI_PATH = 'uploads/GB_AnalyticsData.xlsx'
    ERPSIM_PATH = 'uploads/ERPSIM.xlsx'
    
    # ML Model Paths
    RISK_MODEL_PATH = 'ml/order_risk_model.pkl'
    DEMAND_MODEL_PATH = 'ml/demand_forecast_model.pkl'
    
    # UI Settings
    AUTO_REFRESH_INTERVAL = 30
    CSS_PATH = 'assets/style.css'
    
    # Branding
    BRAND = "AABS"
    TAGLINE = "Enterprise Decision Intelligence Platform"
