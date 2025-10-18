"""
Lead Performance Dashboard

Real-time dashboard for lead conversion tracking, ROI analysis, and performance monitoring.
Provides interactive visualizations and key performance indicators.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import streamlit as st
from dataclasses import asdict

from .advanced_analytics_engine import analytics_engine, MetricType, TimeGranularity

logger = logging.getLogger(__name__)

class LeadPerformanceDashboard:
    """Interactive lead performance dashboard"""
    
    def __init__(self):
        self.analytics_engine = analytics_engine
        self.default_time_range = "30d"
        
    async def render_dashboard(self):
        """Render the complete dashboard"""
        
        st.set_page_config(
            page_title="Lead Performance Dashboard",
            page_icon="📊",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        st.title("🎯 Lead Performance Dashboard")
        st.markdown("Real-time insights into lead generation, conversion, and revenue performance")
        
        # Sidebar controls
        await self._render_sidebar()
        
        # Main dashboard content
        col1, col2, col3, col4 = st.columns(4)
        
        # Key metrics cards
        await self._render_key_metrics(col1, col2, col3, col4)
        
        # Charts section
        st.markdown("---")
        
        # Lead volume and conversion trends
        await self._render_trend_charts()
        
        # Lead source performance
        await self._render_source_performance()
        
        # Sales team performance
        await self._render_sales_performance()
        
        # Predictive insights
        await self._render_predictive_insights()
    
    async def _render_sidebar(self):
        """Render sidebar controls"""
        
        st.sidebar.header("Dashboard Controls")
        
        # Time range selector
        time_range = st.sidebar.selectbox(
            "Time Range",
            ["24h", "7d", "30d", "90d"],
            index=2
        )
        
        # Auto-refresh toggle
        auto_refresh = st.sidebar.checkbox("Auto Refresh (30s)", value=True)
        
        # Refresh button
        if st.sidebar.button("🔄 Refresh Now"):
            st.experimental_rerun()
        
        # Filters
        st.sidebar.markdown("### Filters")
        
        # Lead source filter
        sources = await self._get_available_sources()
        selected_sources = st.sidebar.multiselect(
            "Lead Sources",
            sources,
            default=sources
        )
        
        # Sales rep filter
        reps = await self._get_available_reps()
        selected_reps = st.sidebar.multiselect(
            "Sales Representatives",
            reps,
            default=reps
        )
        
        # Store selections in session state
        st.session_state.time_range = time_range
        st.session_state.selected_sources = selected_sources
        st.session_state.selected_reps = selected_reps
        
        if auto_refresh:
            # Auto-refresh every 30 seconds
            await asyncio.sleep(30)
            st.experimental_rerun()
    
    async def _render_key_metrics(self, col1, col2, col3, col4):
        """Render key performance metrics cards"""
        
        time_range = st.session_state.get('time_range', '30d')
        metrics = await self.analytics_engine.get_real_time_metrics(time_range)
        
        # Total Leads
        with col1:
            st.metric(
                label="Total Leads",
                value=f"{metrics.total_leads:,}",
                delta=f"+{int(metrics.lead_velocity * 24)}/day"
            )
        
        # Conversion Rate
        with col2:
            st.metric(
                label="Conversion Rate",
                value=f"{metrics.conversion_rate:.1f}%",
                delta=f"{metrics.converted_leads} conversions"
            )
        
        # Total Revenue
        with col3:
            st.metric(
                label="Total Revenue",
                value=f"${metrics.total_revenue:,.0f}",
                delta=f"${metrics.revenue_per_lead:.0f}/lead"
            )
        
        # ROI
        with col4:
            roi_value = metrics.roi if hasattr(metrics, 'roi') else 0
            st.metric(
                label="ROI",
                value=f"{roi_value:.1f}%",
                delta="vs target"
            )
    
    async def _render_trend_charts(self):
        """Render lead volume and conversion trend charts"""
        
        st.subheader("📈 Lead Volume & Conversion Trends")
        
        # Get time series data
        time_range = st.session_state.get('time_range', '30d')
        trend_data = await self._get_trend_data(time_range)
        
        if not trend_data.empty:
            # Create subplot with secondary y-axis
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=('Lead Volume Over Time', 'Conversion Rate Trend', 
                              'Revenue Trend', 'Lead Velocity'),
                specs=[[{"secondary_y": True}, {"secondary_y": True}],
                       [{"secondary_y": True}, {"secondary_y": True}]]
            )
            
            # Lead volume chart
            fig.add_trace(
                go.Scatter(
                    x=trend_data['date'],
                    y=trend_data['leads'],
                    mode='lines+markers',
                    name='Leads',
                    line=dict(color='#1f77b4', width=2)
                ),
                row=1, col=1
            )
            
            # Conversion rate chart
            fig.add_trace(
                go.Scatter(
                    x=trend_data['date'],
                    y=trend_data['conversion_rate'],
                    mode='lines+markers',
                    name='Conversion Rate',
                    line=dict(color='#ff7f0e', width=2)
                ),
                row=1, col=2
            )
            
            # Revenue trend
            fig.add_trace(
                go.Scatter(
                    x=trend_data['date'],
                    y=trend_data['revenue'],
                    mode='lines+markers',
                    name='Revenue',
                    line=dict(color='#2ca02c', width=2)
                ),
                row=2, col=1
            )
            
            # Lead velocity
            fig.add_trace(
                go.Scatter(
                    x=trend_data['date'],
                    y=trend_data['lead_velocity'],
                    mode='lines+markers',
                    name='Lead Velocity',
                    line=dict(color='#d62728', width=2)
                ),
                row=2, col=2
            )
            
            fig.update_layout(
                height=600,
                showlegend=False,
                title_text="Performance Trends"
            )
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No trend data available for the selected time range.")
    
    async def _render_source_performance(self):
        """Render lead source performance analysis"""
        
        st.subheader("🎯 Lead Source Performance")
        
        source_metrics = await self.analytics_engine.get_lead_source_performance()
        
        if source_metrics:
            # Convert to DataFrame
            df = pd.DataFrame([asdict(metric) for metric in source_metrics])
            
            col1, col2 = st.columns(2)
            
            with col1:
                # ROI by source
                fig_roi = px.bar(
                    df.head(10),
                    x='source_name',
                    y='roi',
                    title='ROI by Lead Source',
                    color='roi',
                    color_continuous_scale='RdYlGn'
                )
                fig_roi.update_xaxes(tickangle=45)
                st.plotly_chart(fig_roi, use_container_width=True)
            
            with col2:
                # Conversion rate vs volume
                fig_scatter = px.scatter(
                    df,
                    x='total_leads',
                    y='conversion_rate',
                    size='total_revenue',
                    color='roi',
                    hover_name='source_name',
                    title='Conversion Rate vs Lead Volume',
                    labels={'total_leads': 'Total Leads', 'conversion_rate': 'Conversion Rate (%)'}
                )
                st.plotly_chart(fig_scatter, use_container_width=True)
            
            # Source performance table
            st.subheader("📊 Detailed Source Metrics")
            
            # Format the dataframe for display
            display_df = df.copy()
            display_df['conversion_rate'] = display_df['conversion_rate'].apply(lambda x: f"{x:.1f}%")
            display_df['roi'] = display_df['roi'].apply(lambda x: f"{x:.1f}%")
            display_df['total_revenue'] = display_df['total_revenue'].apply(lambda x: f"${x:,.0f}")
            display_df['average_ltv'] = display_df['average_ltv'].apply(lambda x: f"${x:,.0f}")
            
            st.dataframe(
                display_df[['source_name', 'total_leads', 'converted_leads', 'conversion_rate', 
                           'total_revenue', 'roi', 'average_ltv']],
                use_container_width=True
            )
        else:
            st.info("No lead source data available.")
    
    async def _render_sales_performance(self):
        """Render sales team performance analysis"""
        
        st.subheader("👥 Sales Team Performance")
        
        rep_metrics = await self.analytics_engine.get_sales_rep_performance()
        
        if rep_metrics:
            # Convert to DataFrame
            df = pd.DataFrame([asdict(metric) for metric in rep_metrics])
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Conversion rate by rep
                fig_conv = px.bar(
                    df.head(10),
                    x='rep_name',
                    y='conversion_rate',
                    title='Conversion Rate by Sales Rep',
                    color='conversion_rate',
                    color_continuous_scale='Blues'
                )
                fig_conv.update_xaxes(tickangle=45)
                st.plotly_chart(fig_conv, use_container_width=True)
            
            with col2:
                # Revenue by rep
                fig_revenue = px.bar(
                    df.head(10),
                    x='rep_name',
                    y='total_revenue',
                    title='Revenue by Sales Rep',
                    color='total_revenue',
                    color_continuous_scale='Greens'
                )
                fig_revenue.update_xaxes(tickangle=45)
                st.plotly_chart(fig_revenue, use_container_width=True)
            
            # Performance metrics table
            st.subheader("📊 Detailed Rep Performance")
            
            display_df = df.copy()
            display_df['conversion_rate'] = display_df['conversion_rate'].apply(lambda x: f"{x:.1f}%")
            display_df['total_revenue'] = display_df['total_revenue'].apply(lambda x: f"${x:,.0f}")
            display_df['average_response_time'] = display_df['average_response_time'].apply(lambda x: f"{x:.1f}h")
            
            st.dataframe(
                display_df[['rep_name', 'leads_assigned', 'leads_contacted', 'leads_converted', 
                           'conversion_rate', 'total_revenue', 'average_response_time']],
                use_container_width=True
            )
        else:
            st.info("No sales rep data available.")
    
    async def _render_predictive_insights(self):
        """Render predictive analytics insights"""
        
        st.subheader("🔮 Predictive Insights")
        
        try:
            insights = await self.analytics_engine.generate_predictive_insights(30)
            
            if "error" not in insights:
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric(
                        label="Predicted Leads (30 days)",
                        value=f"{insights['lead_volume_forecast']:,.0f}",
                        delta=f"±{(insights['confidence_intervals']['leads']['upper'] - insights['confidence_intervals']['leads']['lower'])/2:,.0f}"
                    )
                
                with col2:
                    st.metric(
                        label="Predicted Conversions (30 days)",
                        value=f"{insights['conversion_forecast']:,.0f}",
                        delta=f"±{(insights['confidence_intervals']['conversions']['upper'] - insights['confidence_intervals']['conversions']['lower'])/2:,.0f}"
                    )
                
                with col3:
                    st.metric(
                        label="Predicted Revenue (30 days)",
                        value=f"${insights['revenue_forecast']:,.0f}",
                        delta=f"±${(insights['confidence_intervals']['revenue']['upper'] - insights['confidence_intervals']['revenue']['lower'])/2:,.0f}"
                    )
                
                # Seasonal insights
                if 'seasonal_insights' in insights:
                    st.subheader("📅 Seasonal Patterns")
                    seasonal_data = insights['seasonal_insights']
                    
                    if seasonal_data:
                        st.json(seasonal_data)
            else:
                st.warning(insights['error'])
                
        except Exception as e:
            st.error(f"Error generating predictive insights: {e}")
    
    async def _get_trend_data(self, time_range: str) -> pd.DataFrame:
        """Get trend data for charts"""
        
        # This would typically fetch from your analytics engine
        # For now, return sample data structure
        
        end_date = datetime.utcnow()
        if time_range == "24h":
            start_date = end_date - timedelta(hours=24)
            freq = 'H'
        elif time_range == "7d":
            start_date = end_date - timedelta(days=7)
            freq = 'D'
        elif time_range == "30d":
            start_date = end_date - timedelta(days=30)
            freq = 'D'
        else:
            start_date = end_date - timedelta(days=90)
            freq = 'D'
        
        # Generate sample trend data
        date_range = pd.date_range(start=start_date, end=end_date, freq=freq)
        
        # This would be replaced with actual data from analytics engine
        np.random.seed(42)
        trend_data = pd.DataFrame({
            'date': date_range,
            'leads': np.random.poisson(50, len(date_range)),
            'conversions': np.random.poisson(8, len(date_range)),
            'revenue': np.random.normal(25000, 5000, len(date_range)),
            'lead_velocity': np.random.normal(2.1, 0.3, len(date_range))
        })
        
        trend_data['conversion_rate'] = (trend_data['conversions'] / trend_data['leads'] * 100).fillna(0)
        
        return trend_data
    
    async def _get_available_sources(self) -> List[str]:
        """Get available lead sources"""
        # This would fetch from your data
        return ["Website", "Google Ads", "Facebook", "LinkedIn", "Referral", "Email Campaign"]
    
    async def _get_available_reps(self) -> List[str]:
        """Get available sales reps"""
        # This would fetch from your data
        return ["John Smith", "Sarah Johnson", "Mike Wilson", "Lisa Chen", "David Brown"]

# Dashboard instance
dashboard = LeadPerformanceDashboard()