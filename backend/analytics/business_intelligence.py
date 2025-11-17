"""
Business Intelligence Dashboard for Insurance Lead Scoring Platform

Provides real-time business intelligence, KPI monitoring, and strategic insights
with interactive dashboards and automated alerting.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from enum import Enum
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

logger = logging.getLogger(__name__)

class KPIType(Enum):
    REVENUE = "revenue"
    CONVERSION = "conversion"
    EFFICIENCY = "efficiency"
    QUALITY = "quality"
    GROWTH = "growth"
    RISK = "risk"

class AlertSeverity(Enum):
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"

@dataclass
class KPI:
    kpi_id: str
    name: str
    description: str
    kpi_type: KPIType
    current_value: float
    target_value: float
    threshold_warning: float
    threshold_critical: float
    unit: str
    calculation_method: str
    last_updated: datetime = field(default_factory=datetime.utcnow)

@dataclass
class BusinessAlert:
    alert_id: str
    kpi_id: str
    severity: AlertSeverity
    message: str
    current_value: float
    threshold_value: float
    created_at: datetime = field(default_factory=datetime.utcnow)
    acknowledged: bool = False

class BusinessIntelligenceDashboard:
    """Advanced business intelligence dashboard with real-time monitoring"""
    
    def __init__(self):
        self.kpis = {}
        self.alerts = {}
        self.dashboard_configs = {}
        self.real_time_data = {}
        self._initialize_default_kpis()
    
    def _initialize_default_kpis(self):
        """Initialize default KPIs for insurance lead scoring"""
        
        default_kpis = [
            KPI(
                kpi_id="lead_conversion_rate",
                name="Lead Conversion Rate",
                description="Percentage of leads that convert to customers",
                kpi_type=KPIType.CONVERSION,
                current_value=0.0,
                target_value=0.15,
                threshold_warning=0.12,
                threshold_critical=0.10,
                unit="%",
                calculation_method="converted_leads / total_leads"
            ),
            KPI(
                kpi_id="revenue_per_lead",
                name="Revenue Per Lead",
                description="Average revenue generated per lead",
                kpi_type=KPIType.REVENUE,
                current_value=0.0,
                target_value=2500.0,
                threshold_warning=2000.0,
                threshold_critical=1500.0,
                unit="$",
                calculation_method="total_revenue / total_leads"
            ),
            KPI(
                kpi_id="lead_quality_score",
                name="Lead Quality Score",
                description="Average quality score of incoming leads",
                kpi_type=KPIType.QUALITY,
                current_value=0.0,
                target_value=75.0,
                threshold_warning=65.0,
                threshold_critical=55.0,
                unit="points",
                calculation_method="avg(lead_scores)"
            ),
            KPI(
                kpi_id="cost_per_acquisition",
                name="Cost Per Acquisition",
                description="Average cost to acquire a new customer",
                kpi_type=KPIType.EFFICIENCY,
                current_value=0.0,
                target_value=500.0,
                threshold_warning=750.0,
                threshold_critical=1000.0,
                unit="$",
                calculation_method="total_marketing_cost / conversions"
            ),
            KPI(
                kpi_id="monthly_growth_rate",
                name="Monthly Growth Rate",
                description="Month-over-month growth in lead volume",
                kpi_type=KPIType.GROWTH,
                current_value=0.0,
                target_value=0.10,
                threshold_warning=0.05,
                threshold_critical=0.0,
                unit="%",
                calculation_method="(current_month - previous_month) / previous_month"
            )
        ]
        
        for kpi in default_kpis:
            self.kpis[kpi.kpi_id] = kpi
    
    async def update_kpi_values(self, data_source: str = "default") -> Dict[str, Any]:
        """Update all KPI values from data sources"""
        
        try:
            # Fetch latest data
            data = await self._fetch_kpi_data(data_source)
            
            updated_kpis = []
            new_alerts = []
            
            for kpi_id, kpi in self.kpis.items():
                old_value = kpi.current_value
                new_value = await self._calculate_kpi_value(kpi, data)
                
                # Update KPI
                kpi.current_value = new_value
                kpi.last_updated = datetime.now(datetime.UTC)
                updated_kpis.append(kpi_id)
                
                # Check for alerts
                alert = self._check_kpi_thresholds(kpi, old_value, new_value)
                if alert:
                    self.alerts[alert.alert_id] = alert
                    new_alerts.append(alert)
            
            return {
                'updated_kpis': updated_kpis,
                'new_alerts': len(new_alerts),
                'alerts': [alert.alert_id for alert in new_alerts],
                'timestamp': datetime.now(datetime.UTC).isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error updating KPI values: {e}")
            raise
    
    async def _calculate_kpi_value(self, kpi: KPI, data: pd.DataFrame) -> float:
        """Calculate KPI value based on data and calculation method"""
        
        try:
            if kpi.kpi_id == "lead_conversion_rate":
                if 'converted' in data.columns and len(data) > 0:
                    return data['converted'].mean() * 100
                return 0.0
            
            elif kpi.kpi_id == "revenue_per_lead":
                if 'revenue' in data.columns and len(data) > 0:
                    return data['revenue'].sum() / len(data)
                return 0.0
            
            elif kpi.kpi_id == "lead_quality_score":
                if 'quality_score' in data.columns and len(data) > 0:
                    return data['quality_score'].mean()
                return 0.0
            
            elif kpi.kpi_id == "cost_per_acquisition":
                if 'marketing_cost' in data.columns and 'converted' in data.columns:
                    total_cost = data['marketing_cost'].sum()
                    conversions = data['converted'].sum()
                    return total_cost / conversions if conversions > 0 else 0.0
                return 0.0
            
            elif kpi.kpi_id == "monthly_growth_rate":
                if 'timestamp' in data.columns and len(data) > 0:
                    # Calculate month-over-month growth
                    current_month = data[data['timestamp'].dt.month == datetime.now().month]
                    previous_month = data[data['timestamp'].dt.month == datetime.now().month - 1]
                    
                    if len(previous_month) > 0:
                        growth = (len(current_month) - len(previous_month)) / len(previous_month)
                        return growth * 100
                return 0.0
            
            return 0.0
            
        except Exception as e:
            logger.error(f"Error calculating KPI {kpi.kpi_id}: {e}")
            return 0.0
    
    def _check_kpi_thresholds(self, kpi: KPI, old_value: float, new_value: float) -> Optional[BusinessAlert]:
        """Check if KPI value crosses alert thresholds"""
        
        # Determine if this is a "higher is better" or "lower is better" KPI
        higher_is_better = kpi.kpi_type in [KPIType.REVENUE, KPIType.CONVERSION, KPIType.QUALITY, KPIType.GROWTH]
        
        if higher_is_better:
            if new_value <= kpi.threshold_critical:
                severity = AlertSeverity.CRITICAL
                message = f"{kpi.name} has dropped to {new_value:.2f} {kpi.unit}, below critical threshold of {kpi.threshold_critical:.2f}"
            elif new_value <= kpi.threshold_warning:
                severity = AlertSeverity.WARNING
                message = f"{kpi.name} has dropped to {new_value:.2f} {kpi.unit}, below warning threshold of {kpi.threshold_warning:.2f}"
            else:
                return None
        else:
            if new_value >= kpi.threshold_critical:
                severity = AlertSeverity.CRITICAL
                message = f"{kpi.name} has risen to {new_value:.2f} {kpi.unit}, above critical threshold of {kpi.threshold_critical:.2f}"
            elif new_value >= kpi.threshold_warning:
                severity = AlertSeverity.WARNING
                message = f"{kpi.name} has risen to {new_value:.2f} {kpi.unit}, above warning threshold of {kpi.threshold_warning:.2f}"
            else:
                return None
        
        return BusinessAlert(
            alert_id=f"{kpi.kpi_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            kpi_id=kpi.kpi_id,
            severity=severity,
            message=message,
            current_value=new_value,
            threshold_value=kpi.threshold_critical if severity == AlertSeverity.CRITICAL else kpi.threshold_warning
        )
    
    async def generate_executive_dashboard(self) -> Dict[str, Any]:
        """Generate executive-level dashboard data"""
        
        # Update KPIs first
        await self.update_kpi_values()
        
        # Organize KPIs by type
        kpis_by_type = {}
        for kpi in self.kpis.values():
            if kpi.kpi_type not in kpis_by_type:
                kpis_by_type[kpi.kpi_type] = []
            kpis_by_type[kpi.kpi_type].append({
                'id': kpi.kpi_id,
                'name': kpi.name,
                'current_value': kpi.current_value,
                'target_value': kpi.target_value,
                'unit': kpi.unit,
                'performance': (kpi.current_value / kpi.target_value * 100) if kpi.target_value > 0 else 0,
                'status': self._get_kpi_status(kpi)
            })
        
        # Get recent alerts
        recent_alerts = [
            {
                'id': alert.alert_id,
                'severity': alert.severity.value,
                'message': alert.message,
                'created_at': alert.created_at.isoformat(),
                'acknowledged': alert.acknowledged
            }
            for alert in sorted(self.alerts.values(), key=lambda x: x.created_at, reverse=True)[:10]
        ]
        
        # Generate trend data
        trend_data = await self._generate_trend_data()
        
        return {
            'kpis_by_type': {kpi_type.value: kpis for kpi_type, kpis in kpis_by_type.items()},
            'recent_alerts': recent_alerts,
            'alert_summary': {
                'total': len(self.alerts),
                'critical': len([a for a in self.alerts.values() if a.severity == AlertSeverity.CRITICAL]),
                'warning': len([a for a in self.alerts.values() if a.severity == AlertSeverity.WARNING]),
                'unacknowledged': len([a for a in self.alerts.values() if not a.acknowledged])
            },
            'trend_data': trend_data,
            'last_updated': datetime.now(datetime.UTC).isoformat()
        }
    
    def _get_kpi_status(self, kpi: KPI) -> str:
        """Get status indicator for KPI"""
        
        higher_is_better = kpi.kpi_type in [KPIType.REVENUE, KPIType.CONVERSION, KPIType.QUALITY, KPIType.GROWTH]
        
        if higher_is_better:
            if kpi.current_value >= kpi.target_value:
                return "excellent"
            elif kpi.current_value >= kpi.threshold_warning:
                return "good"
            elif kpi.current_value >= kpi.threshold_critical:
                return "warning"
            else:
                return "critical"
        else:
            if kpi.current_value <= kpi.target_value:
                return "excellent"
            elif kpi.current_value <= kpi.threshold_warning:
                return "good"
            elif kpi.current_value <= kpi.threshold_critical:
                return "warning"
            else:
                return "critical"
    
    async def _generate_trend_data(self) -> Dict[str, Any]:
        """Generate trend data for dashboard charts"""
        
        # This would typically fetch historical data
        # For now, generating sample trend data
        
        dates = pd.date_range(start=datetime.now() - timedelta(days=30), end=datetime.now(), freq='D')
        
        trends = {}
        for kpi_id, kpi in self.kpis.items():
            # Generate sample trend data
            base_value = kpi.current_value
            trend_values = []
            
            for i, date in enumerate(dates):
                # Add some realistic variation
                variation = np.random.normal(0, base_value * 0.1)
                trend_value = max(0, base_value + variation)
                trend_values.append(trend_value)
            
            trends[kpi_id] = {
                'dates': [date.isoformat() for date in dates],
                'values': trend_values,
                'name': kpi.name,
                'unit': kpi.unit
            }
        
        return trends

# Global business intelligence dashboard instance
bi_dashboard = BusinessIntelligenceDashboard()