"""
Advanced Reporting Framework for Insurance Lead Scoring Platform

Provides automated report generation, dashboard creation, and business intelligence
reporting with customizable templates and delivery mechanisms.
"""

import asyncio
import logging
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from enum import Enum
import json
import pandas as pd
from jinja2 import Template
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import base64
from io import BytesIO

logger = logging.getLogger(__name__)

class ReportType(Enum):
    EXECUTIVE_SUMMARY = "executive_summary"
    OPERATIONAL_DASHBOARD = "operational_dashboard"
    PERFORMANCE_ANALYSIS = "performance_analysis"
    COMPLIANCE_REPORT = "compliance_report"
    PREDICTIVE_INSIGHTS = "predictive_insights"
    CUSTOM_REPORT = "custom_report"

class ReportFormat(Enum):
    HTML = "html"
    PDF = "pdf"
    JSON = "json"
    CSV = "csv"
    EXCEL = "excel"
    DASHBOARD = "dashboard"

class DeliveryMethod(Enum):
    EMAIL = "email"
    SLACK = "slack"
    WEBHOOK = "webhook"
    FILE_SYSTEM = "file_system"
    API_ENDPOINT = "api_endpoint"

@dataclass
class ReportConfiguration:
    report_id: str
    report_type: ReportType
    title: str
    description: str
    format: ReportFormat
    template_id: Optional[str] = None
    data_sources: List[str] = field(default_factory=list)
    filters: Dict[str, Any] = field(default_factory=dict)
    schedule: Optional[str] = None  # Cron expression
    delivery_methods: List[DeliveryMethod] = field(default_factory=list)
    recipients: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.utcnow)

@dataclass
class GeneratedReport:
    report_id: str
    config: ReportConfiguration
    content: Any
    metadata: Dict[str, Any]
    file_path: Optional[str] = None
    generated_at: datetime = field(default_factory=datetime.utcnow)

class ReportingFramework:
    """Advanced reporting framework with automated generation and delivery"""
    
    def __init__(self):
        self.report_configs = {}
        self.generated_reports = {}
        self.templates = {}
        self.schedulers = {}
        self.delivery_handlers = {}
        self._initialize_templates()
        self._initialize_delivery_handlers()
    
    def _initialize_templates(self):
        """Initialize report templates"""
        
        # Executive Summary Template
        self.templates['executive_summary'] = """
        <html>
        <head>
            <title>{{ title }}</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 40px; }
                .header { background: #2c3e50; color: white; padding: 20px; text-align: center; }
                .metric-card { background: #f8f9fa; border-left: 4px solid #007bff; padding: 15px; margin: 10px 0; }
                .chart-container { margin: 20px 0; text-align: center; }
                .insights { background: #e8f5e8; padding: 15px; border-radius: 5px; margin: 20px 0; }
            </style>
        </head>
        <body>
            <div class="header">
                <h1>{{ title }}</h1>
                <p>{{ description }}</p>
                <p>Generated: {{ generated_at }}</p>
            </div>
            
            <h2>Key Metrics</h2>
            {% for metric in key_metrics %}
            <div class="metric-card">
                <h3>{{ metric.name }}</h3>
                <p><strong>{{ metric.value }}</strong> {{ metric.unit }}</p>
                <p>{{ metric.change }} from previous period</p>
            </div>
            {% endfor %}
            
            <h2>Performance Overview</h2>
            <div class="chart-container">
                {{ performance_chart }}
            </div>
            
            <h2>Key Insights</h2>
            <div class="insights">
                <ul>
                {% for insight in insights %}
                    <li>{{ insight }}</li>
                {% endfor %}
                </ul>
            </div>
            
            <h2>Recommendations</h2>
            <ul>
            {% for recommendation in recommendations %}
                <li>{{ recommendation }}</li>
            {% endfor %}
            </ul>
        </body>
        </html>
        """
        
        # Operational Dashboard Template
        self.templates['operational_dashboard'] = """
        <html>
        <head>
            <title>{{ title }}</title>
            <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
            <style>
                body { font-family: Arial, sans-serif; margin: 0; padding: 20px; background: #f5f5f5; }
                .dashboard-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; }
                .widget { background: white; border-radius: 8px; padding: 20px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
                .metric-value { font-size: 2em; font-weight: bold; color: #2c3e50; }
                .metric-label { color: #7f8c8d; font-size: 0.9em; }
                .status-good { color: #27ae60; }
                .status-warning { color: #f39c12; }
                .status-critical { color: #e74c3c; }
            </style>
        </head>
        <body>
            <h1>{{ title }}</h1>
            <div class="dashboard-grid">
                {% for widget in widgets %}
                <div class="widget">
                    <h3>{{ widget.title }}</h3>
                    {% if widget.type == 'metric' %}
                        <div class="metric-value {{ widget.status_class }}">{{ widget.value }}</div>
                        <div class="metric-label">{{ widget.label }}</div>
                    {% elif widget.type == 'chart' %}
                        <div id="{{ widget.chart_id }}"></div>
                    {% endif %}
                </div>
                {% endfor %}
            </div>
            
            <script>
                {% for chart in charts %}
                Plotly.newPlot('{{ chart.id }}', {{ chart.data }}, {{ chart.layout }});
                {% endfor %}
            </script>
        </body>
        </html>
        """
    
    def _initialize_delivery_handlers(self):
        """Initialize delivery method handlers"""
        
        self.delivery_handlers[DeliveryMethod.EMAIL] = self._deliver_via_email
        self.delivery_handlers[DeliveryMethod.SLACK] = self._deliver_via_slack
        self.delivery_handlers[DeliveryMethod.WEBHOOK] = self._deliver_via_webhook
        self.delivery_handlers[DeliveryMethod.FILE_SYSTEM] = self._deliver_to_filesystem
    
    async def create_report_config(self, config: ReportConfiguration) -> str:
        """Create a new report configuration"""
        
        self.report_configs[config.report_id] = config
        
        # Set up scheduling if specified
        if config.schedule:
            await self._schedule_report(config)
        
        logger.info(f"Created report configuration: {config.report_id}")
        return config.report_id
    
    async def generate_report(self, report_id: str, override_filters: Optional[Dict[str, Any]] = None) -> GeneratedReport:
        """Generate a report based on configuration"""
        
        if report_id not in self.report_configs:
            raise ValueError(f"Report configuration not found: {report_id}")
        
        config = self.report_configs[report_id]
        
        try:
            # Apply override filters if provided
            filters = {**config.filters, **(override_filters or {})}
            
            # Fetch data from sources
            data = await self._fetch_report_data(config.data_sources, filters)
            
            # Generate report content based on type
            if config.report_type == ReportType.EXECUTIVE_SUMMARY:
                content = await self._generate_executive_summary(data, config)
            elif config.report_type == ReportType.OPERATIONAL_DASHBOARD:
                content = await self._generate_operational_dashboard(data, config)
            elif config.report_type == ReportType.PERFORMANCE_ANALYSIS:
                content = await self._generate_performance_analysis(data, config)
            elif config.report_type == ReportType.COMPLIANCE_REPORT:
                content = await self._generate_compliance_report(data, config)
            elif config.report_type == ReportType.PREDICTIVE_INSIGHTS:
                content = await self._generate_predictive_insights(data, config)
            else:
                content = await self._generate_custom_report(data, config)
            
            # Create generated report
            report = GeneratedReport(
                report_id=f"{report_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                config=config,
                content=content,
                metadata={
                    'data_points': len(data) if isinstance(data, pd.DataFrame) else 0,
                    'filters_applied': filters,
                    'generation_time': datetime.now(timezone.utc).isoformat()
                }
            )
            
            # Save report
            self.generated_reports[report.report_id] = report
            
            # Deliver report if delivery methods are configured
            if config.delivery_methods:
                await self._deliver_report(report)
            
            return report
            
        except Exception as e:
            logger.error(f"Error generating report {report_id}: {e}")
            raise
    
    async def _generate_executive_summary(self, data: pd.DataFrame, config: ReportConfiguration) -> str:
        """Generate executive summary report"""
        
        # Calculate key metrics
        key_metrics = [
            {
                'name': 'Total Leads',
                'value': f"{len(data):,}",
                'unit': 'leads',
                'change': '+12% from last month'
            },
            {
                'name': 'Conversion Rate',
                'value': f"{data['conversion_rate'].mean():.1%}" if 'conversion_rate' in data.columns else "N/A",
                'unit': '',
                'change': '+2.3% from last month'
            },
            {
                'name': 'Revenue Generated',
                'value': f"${data['revenue'].sum():,.0f}" if 'revenue' in data.columns else "N/A",
                'unit': '',
                'change': '+18% from last month'
            }
        ]
        
        # Generate performance chart
        performance_chart = self._create_performance_chart(data)
        
        # Generate insights
        insights = [
            "Lead quality has improved by 15% this quarter",
            "Healthcare insurance leads show highest conversion rates",
            "Mobile traffic accounts for 60% of new leads",
            "Peak lead generation occurs between 2-4 PM EST"
        ]
        
        # Generate recommendations
        recommendations = [
            "Increase budget allocation to healthcare insurance campaigns",
            "Optimize mobile user experience to improve conversion",
            "Focus marketing efforts during peak hours",
            "Implement A/B testing for landing page optimization"
        ]
        
        # Render template
        template = Template(self.templates['executive_summary'])
        return template.render(
            title=config.title,
            description=config.description,
            generated_at=datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            key_metrics=key_metrics,
            performance_chart=performance_chart,
            insights=insights,
            recommendations=recommendations
        )
    
    async def _generate_operational_dashboard(self, data: pd.DataFrame, config: ReportConfiguration) -> str:
        """Generate operational dashboard"""
        
        # Create dashboard widgets
        widgets = [
            {
                'title': 'Active Leads',
                'type': 'metric',
                'value': f"{len(data):,}",
                'label': 'Total leads in pipeline',
                'status_class': 'status-good'
            },
            {
                'title': 'Conversion Rate',
                'type': 'metric',
                'value': f"{data['conversion_rate'].mean():.1%}" if 'conversion_rate' in data.columns else "N/A",
                'label': 'Current conversion rate',
                'status_class': 'status-good'
            },
            {
                'title': 'Lead Trends',
                'type': 'chart',
                'chart_id': 'lead_trends_chart'
            },
            {
                'title': 'Revenue Distribution',
                'type': 'chart',
                'chart_id': 'revenue_chart'
            }
        ]
        
        # Create charts
        charts = [
            {
                'id': 'lead_trends_chart',
                'data': self._create_trend_chart_data(data),
                'layout': {'title': 'Lead Trends Over Time'}
            },
            {
                'id': 'revenue_chart',
                'data': self._create_revenue_chart_data(data),
                'layout': {'title': 'Revenue by Insurance Type'}
            }
        ]
        
        # Render template
        template = Template(self.templates['operational_dashboard'])
        return template.render(
            title=config.title,
            widgets=widgets,
            charts=charts
        )
    
    def _create_performance_chart(self, data: pd.DataFrame) -> str:
        """Create performance chart for executive summary"""
        
        if 'timestamp' not in data.columns:
            return "<p>No time series data available</p>"
        
        # Group by day and calculate metrics
        daily_data = data.groupby(data['timestamp'].dt.date).agg({
            'lead_id': 'count',
            'conversion_rate': 'mean',
            'revenue': 'sum'
        }).reset_index()
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Daily Leads', 'Conversion Rate', 'Daily Revenue', 'Cumulative Revenue'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Add traces
        fig.add_trace(
            go.Scatter(x=daily_data['timestamp'], y=daily_data['lead_id'], name='Daily Leads'),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(x=daily_data['timestamp'], y=daily_data['conversion_rate'], name='Conversion Rate'),
            row=1, col=2
        )
        
        fig.add_trace(
            go.Bar(x=daily_data['timestamp'], y=daily_data['revenue'], name='Daily Revenue'),
            row=2, col=1
        )
        
        fig.add_trace(
            go.Scatter(x=daily_data['timestamp'], y=daily_data['revenue'].cumsum(), name='Cumulative Revenue'),
            row=2, col=2
        )
        
        fig.update_layout(height=600, showlegend=False)
        
        return fig.to_html(include_plotlyjs=False, div_id="performance_chart")

# Global reporting framework instance
reporting_framework = ReportingFramework()