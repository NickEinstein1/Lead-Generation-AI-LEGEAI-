import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass, field
from enum import Enum
import uuid
import json
import scipy.stats as stats
from scipy.stats import chi2_contingency, ttest_ind
import hashlib

logger = logging.getLogger(__name__)

class TestStatus(Enum):
    """A/B test status"""
    DRAFT = "DRAFT"
    ACTIVE = "ACTIVE"
    PAUSED = "PAUSED"
    COMPLETED = "COMPLETED"
    CANCELLED = "CANCELLED"

class TestType(Enum):
    """Types of A/B tests"""
    MESSAGE_CONTENT = "MESSAGE_CONTENT"
    SUBJECT_LINE = "SUBJECT_LINE"
    CALL_TO_ACTION = "CALL_TO_ACTION"
    SEND_TIME = "SEND_TIME"
    PERSONALIZATION = "PERSONALIZATION"
    LANDING_PAGE = "LANDING_PAGE"
    LEAD_SCORING = "LEAD_SCORING"
    SEQUENCE_TIMING = "SEQUENCE_TIMING"

class MetricType(Enum):
    """Types of metrics to track"""
    CONVERSION_RATE = "CONVERSION_RATE"
    OPEN_RATE = "OPEN_RATE"
    CLICK_RATE = "CLICK_RATE"
    RESPONSE_RATE = "RESPONSE_RATE"
    REVENUE_PER_LEAD = "REVENUE_PER_LEAD"
    COST_PER_ACQUISITION = "COST_PER_ACQUISITION"
    LEAD_QUALITY_SCORE = "LEAD_QUALITY_SCORE"
    TIME_TO_CONVERSION = "TIME_TO_CONVERSION"

@dataclass
class TestVariant:
    """A/B test variant configuration"""
    variant_id: str
    name: str
    description: str
    traffic_allocation: float  # 0.0 to 1.0
    configuration: Dict[str, Any]
    is_control: bool = False
    
@dataclass
class TestConfiguration:
    """A/B test configuration"""
    test_id: str
    name: str
    description: str
    test_type: TestType
    variants: List[TestVariant]
    primary_metric: MetricType
    secondary_metrics: List[MetricType]
    target_sample_size: int
    confidence_level: float = 0.95
    minimum_detectable_effect: float = 0.05
    max_duration_days: int = 30
    traffic_allocation: Dict[str, float] = field(default_factory=dict)
    
@dataclass
class TestResult:
    """A/B test result"""
    variant_id: str
    sample_size: int
    conversions: int
    conversion_rate: float
    confidence_interval: Tuple[float, float]
    statistical_significance: bool
    p_value: float
    lift: float
    revenue: float = 0.0
    cost: float = 0.0

@dataclass
class TestAnalysis:
    """Complete A/B test analysis"""
    test_id: str
    status: TestStatus
    start_date: datetime
    end_date: Optional[datetime]
    duration_days: int
    results: List[TestResult]
    winner: Optional[str]
    statistical_power: float
    confidence_level: float
    recommendations: List[str]
    insights: Dict[str, Any]

class ABTestFramework:
    """Comprehensive A/B testing framework"""
    
    def __init__(self, config_path: str = None):
        self.active_tests = {}
        self.completed_tests = {}
        self.test_assignments = {}  # lead_id -> test_assignments
        self.test_results = {}
        self.config = self._load_config(config_path)
        
        # Statistical settings
        self.default_confidence_level = 0.95
        self.default_power = 0.8
        self.minimum_sample_size = 100
        
    def create_test(self, config: TestConfiguration) -> str:
        """Create a new A/B test"""
        logger.info(f"Creating A/B test: {config.name}")
        
        # Validate configuration
        self._validate_test_config(config)
        
        # Calculate required sample size
        required_sample_size = self._calculate_sample_size(
            config.minimum_detectable_effect,
            config.confidence_level,
            self.default_power
        )
        
        if config.target_sample_size < required_sample_size:
            logger.warning(f"Target sample size {config.target_sample_size} is below recommended {required_sample_size}")
        
        # Initialize test
        test_data = {
            'config': config,
            'status': TestStatus.DRAFT,
            'created_date': datetime.now(),
            'start_date': None,
            'end_date': None,
            'participants': {},
            'metrics': {variant.variant_id: {} for variant in config.variants},
            'daily_stats': []
        }
        
        self.active_tests[config.test_id] = test_data
        
        logger.info(f"A/B test created: {config.test_id}")
        return config.test_id
    
    def start_test(self, test_id: str) -> bool:
        """Start an A/B test"""
        if test_id not in self.active_tests:
            logger.error(f"Test {test_id} not found")
            return False
        
        test_data = self.active_tests[test_id]
        
        if test_data['status'] != TestStatus.DRAFT:
            logger.error(f"Test {test_id} is not in DRAFT status")
            return False
        
        # Start the test
        test_data['status'] = TestStatus.ACTIVE
        test_data['start_date'] = datetime.now()
        
        logger.info(f"A/B test started: {test_id}")
        return True
    
    def assign_variant(self, test_id: str, lead_id: str, 
                      lead_data: Dict[str, Any] = None) -> Optional[str]:
        """Assign a lead to a test variant"""
        if test_id not in self.active_tests:
            return None
        
        test_data = self.active_tests[test_id]
        
        if test_data['status'] != TestStatus.ACTIVE:
            return None
        
        # Check if lead already assigned
        if lead_id in test_data['participants']:
            return test_data['participants'][lead_id]['variant_id']
        
        # Assign variant based on traffic allocation
        variant_id = self._assign_variant_by_hash(test_id, lead_id, test_data['config'])
        
        # Record assignment
        test_data['participants'][lead_id] = {
            'variant_id': variant_id,
            'assigned_date': datetime.now(),
            'lead_data': lead_data or {},
            'events': []
        }
        
        # Track assignment in global registry
        if lead_id not in self.test_assignments:
            self.test_assignments[lead_id] = {}
        self.test_assignments[lead_id][test_id] = variant_id
        
        logger.debug(f"Lead {lead_id} assigned to variant {variant_id} in test {test_id}")
        return variant_id
    
    def track_event(self, test_id: str, lead_id: str, event_type: str, 
                   value: float = 1.0, metadata: Dict[str, Any] = None) -> bool:
        """Track an event for A/B test analysis"""
        if test_id not in self.active_tests:
            return False
        
        test_data = self.active_tests[test_id]
        
        if lead_id not in test_data['participants']:
            logger.warning(f"Lead {lead_id} not found in test {test_id}")
            return False
        
        # Record event
        event = {
            'event_type': event_type,
            'value': value,
            'timestamp': datetime.now(),
            'metadata': metadata or {}
        }
        
        test_data['participants'][lead_id]['events'].append(event)
        
        # Update metrics
        variant_id = test_data['participants'][lead_id]['variant_id']
        if event_type not in test_data['metrics'][variant_id]:
            test_data['metrics'][variant_id][event_type] = []
        
        test_data['metrics'][variant_id][event_type].append({
            'lead_id': lead_id,
            'value': value,
            'timestamp': datetime.now(),
            'metadata': metadata
        })
        
        logger.debug(f"Event tracked: {event_type} for lead {lead_id} in test {test_id}")
        return True
    
    def analyze_test(self, test_id: str) -> Optional[TestAnalysis]:
        """Analyze A/B test results"""
        if test_id not in self.active_tests and test_id not in self.completed_tests:
            logger.error(f"Test {test_id} not found")
            return None
        
        test_data = self.active_tests.get(test_id) or self.completed_tests.get(test_id)
        config = test_data['config']
        
        # Calculate results for each variant
        results = []
        for variant in config.variants:
            variant_result = self._calculate_variant_results(
                test_data, variant.variant_id, config.primary_metric
            )
            results.append(variant_result)
        
        # Determine statistical significance and winner
        winner, significance_results = self._determine_winner(results, config)
        
        # Calculate statistical power
        statistical_power = self._calculate_statistical_power(results)
        
        # Generate insights and recommendations
        insights = self._generate_insights(test_data, results, config)
        recommendations = self._generate_recommendations(results, significance_results, config)
        
        # Create analysis
        analysis = TestAnalysis(
            test_id=test_id,
            status=test_data['status'],
            start_date=test_data['start_date'],
            end_date=test_data.get('end_date'),
            duration_days=self._calculate_duration_days(test_data),
            results=results,
            winner=winner,
            statistical_power=statistical_power,
            confidence_level=config.confidence_level,
            recommendations=recommendations,
            insights=insights
        )
        
        return analysis
    
    def stop_test(self, test_id: str, reason: str = "Manual stop") -> bool:
        """Stop an active A/B test"""
        if test_id not in self.active_tests:
            logger.error(f"Test {test_id} not found")
            return False
        
        test_data = self.active_tests[test_id]
        
        if test_data['status'] != TestStatus.ACTIVE:
            logger.error(f"Test {test_id} is not active")
            return False
        
        # Stop the test
        test_data['status'] = TestStatus.COMPLETED
        test_data['end_date'] = datetime.now()
        test_data['stop_reason'] = reason
        
        # Move to completed tests
        self.completed_tests[test_id] = test_data
        del self.active_tests[test_id]
        
        logger.info(f"A/B test stopped: {test_id} - Reason: {reason}")
        return True
    
    def get_test_status(self, test_id: str) -> Optional[Dict[str, Any]]:
        """Get current test status and metrics"""
        test_data = self.active_tests.get(test_id) or self.completed_tests.get(test_id)
        
        if not test_data:
            return None
        
        config = test_data['config']
        
        # Calculate current metrics
        variant_stats = {}
        for variant in config.variants:
            stats = self._get_variant_stats(test_data, variant.variant_id)
            variant_stats[variant.variant_id] = stats
        
        # Check if test should auto-stop
        auto_stop_reason = self._check_auto_stop_conditions(test_data, variant_stats)
        
        return {
            'test_id': test_id,
            'name': config.name,
            'status': test_data['status'].value,
            'start_date': test_data.get('start_date'),
            'duration_days': self._calculate_duration_days(test_data),
            'total_participants': len(test_data['participants']),
            'variant_stats': variant_stats,
            'auto_stop_reason': auto_stop_reason,
            'progress': self._calculate_test_progress(test_data, config)
        }
    
    def _assign_variant_by_hash(self, test_id: str, lead_id: str, 
                               config: TestConfiguration) -> str:
        """Assign variant using consistent hashing"""
        # Create hash from test_id + lead_id for consistency
        hash_input = f"{test_id}:{lead_id}"
        hash_value = int(hashlib.md5(hash_input.encode()).hexdigest(), 16)
        
        # Normalize to 0-1
        normalized_hash = (hash_value % 10000) / 10000.0
        
        # Assign based on traffic allocation
        cumulative_allocation = 0.0
        for variant in config.variants:
            cumulative_allocation += variant.traffic_allocation
            if normalized_hash <= cumulative_allocation:
                return variant.variant_id
        
        # Fallback to control variant
        control_variant = next((v for v in config.variants if v.is_control), config.variants[0])
        return control_variant.variant_id
    
    def _calculate_variant_results(self, test_data: Dict, variant_id: str, 
                                 primary_metric: MetricType) -> TestResult:
        """Calculate results for a specific variant"""
        participants = [p for p in test_data['participants'].values() 
                       if p['variant_id'] == variant_id]
        
        sample_size = len(participants)
        
        if sample_size == 0:
            return TestResult(
                variant_id=variant_id,
                sample_size=0,
                conversions=0,
                conversion_rate=0.0,
                confidence_interval=(0.0, 0.0),
                statistical_significance=False,
                p_value=1.0,
                lift=0.0
            )
        
        # Calculate conversions based on primary metric
        conversions = 0
        total_revenue = 0.0
        total_cost = 0.0
        
        for participant in participants:
            events = participant['events']
            
            # Check for conversion events
            if primary_metric == MetricType.CONVERSION_RATE:
                if any(e['event_type'] == 'conversion' for e in events):
                    conversions += 1
            elif primary_metric == MetricType.OPEN_RATE:
                if any(e['event_type'] == 'email_open' for e in events):
                    conversions += 1
            elif primary_metric == MetricType.CLICK_RATE:
                if any(e['event_type'] == 'email_click' for e in events):
                    conversions += 1
            elif primary_metric == MetricType.RESPONSE_RATE:
                if any(e['event_type'] == 'response' for e in events):
                    conversions += 1
            
            # Calculate revenue and cost
            revenue_events = [e for e in events if e['event_type'] == 'revenue']
            total_revenue += sum(e['value'] for e in revenue_events)
            
            cost_events = [e for e in events if e['event_type'] == 'cost']
            total_cost += sum(e['value'] for e in cost_events)
        
        conversion_rate = conversions / sample_size if sample_size > 0 else 0.0
        
        # Calculate confidence interval
        confidence_interval = self._calculate_confidence_interval(
            conversions, sample_size, 0.95
        )
        
        return TestResult(
            variant_id=variant_id,
            sample_size=sample_size,
            conversions=conversions,
            conversion_rate=conversion_rate,
            confidence_interval=confidence_interval,
            statistical_significance=False,  # Will be calculated in comparison
            p_value=1.0,  # Will be calculated in comparison
            lift=0.0,  # Will be calculated relative to control
            revenue=total_revenue,
            cost=total_cost
        )
    
    def _determine_winner(self, results: List[TestResult], 
                         config: TestConfiguration) -> Tuple[Optional[str], Dict]:
        """Determine test winner and statistical significance"""
        if len(results) < 2:
            return None, {}
        
        # Find control variant
        control_result = None
        for result in results:
            variant = next(v for v in config.variants if v.variant_id == result.variant_id)
            if variant.is_control:
                control_result = result
                break
        
        if not control_result:
            control_result = results[0]  # Use first variant as control
        
        significance_results = {}
        best_variant = control_result
        
        # Compare each variant to control
        for result in results:
            if result.variant_id == control_result.variant_id:
                continue
            
            # Perform statistical test
            p_value = self._perform_statistical_test(control_result, result)
            is_significant = p_value < (1 - config.confidence_level)
            
            # Calculate lift
            if control_result.conversion_rate > 0:
                lift = (result.conversion_rate - control_result.conversion_rate) / control_result.conversion_rate
            else:
                lift = 0.0
            
            # Update result with significance data
            result.statistical_significance = is_significant
            result.p_value = p_value
            result.lift = lift
            
            significance_results[result.variant_id] = {
                'p_value': p_value,
                'is_significant': is_significant,
                'lift': lift
            }
            
            # Check if this is the best performing variant
            if (result.conversion_rate > best_variant.conversion_rate and 
                is_significant):
                best_variant = result
        
        winner = best_variant.variant_id if best_variant != control_result else None
        
        return winner, significance_results
    
    def _perform_statistical_test(self, control: TestResult, 
                                 variant: TestResult) -> float:
        """Perform statistical test between two variants"""
        # Chi-square test for conversion rates
        observed = np.array([
            [control.conversions, control.sample_size - control.conversions],
            [variant.conversions, variant.sample_size - variant.conversions]
        ])
        
        try:
            chi2, p_value, dof, expected = chi2_contingency(observed)
            return p_value
        except:
            return 1.0  # No significance if test fails
    
    def _calculate_confidence_interval(self, conversions: int, sample_size: int, 
                                     confidence_level: float) -> Tuple[float, float]:
        """Calculate confidence interval for conversion rate"""
        if sample_size == 0:
            return (0.0, 0.0)
        
        p = conversions / sample_size
        z_score = stats.norm.ppf(1 - (1 - confidence_level) / 2)
        
        margin_of_error = z_score * np.sqrt(p * (1 - p) / sample_size)
        
        lower_bound = max(0, p - margin_of_error)
        upper_bound = min(1, p + margin_of_error)
        
        return (lower_bound, upper_bound)
    
    def _calculate_sample_size(self, effect_size: float, confidence_level: float, 
                             power: float) -> int:
        """Calculate required sample size for A/B test"""
        alpha = 1 - confidence_level
        beta = 1 - power
        
        z_alpha = stats.norm.ppf(1 - alpha / 2)
        z_beta = stats.norm.ppf(power)
        
        # Assume baseline conversion rate of 5%
        p1 = 0.05
        p2 = p1 * (1 + effect_size)
        
        pooled_p = (p1 + p2) / 2
        
        sample_size = (
            (z_alpha * np.sqrt(2 * pooled_p * (1 - pooled_p)) + 
             z_beta * np.sqrt(p1 * (1 - p1) + p2 * (1 - p2))) ** 2
        ) / (p2 - p1) ** 2
        
        return max(self.minimum_sample_size, int(np.ceil(sample_size)))
    
    def _generate_insights(self, test_data: Dict, results: List[TestResult], 
                          config: TestConfiguration) -> Dict[str, Any]:
        """Generate insights from test results"""
        insights = {
            'test_duration': self._calculate_duration_days(test_data),
            'total_participants': len(test_data['participants']),
            'conversion_rates': {r.variant_id: r.conversion_rate for r in results},
            'sample_sizes': {r.variant_id: r.sample_size for r in results},
            'revenue_impact': {},
            'cost_efficiency': {},
            'segment_performance': {}
        }
        
        # Calculate revenue impact
        control_result = next((r for r in results if any(v.is_control and v.variant_id == r.variant_id for v in config.variants)), results[0])
        
        for result in results:
            if result.variant_id != control_result.variant_id:
                revenue_lift = result.revenue - control_result.revenue
                insights['revenue_impact'][result.variant_id] = revenue_lift
                
                if result.cost > 0:
                    roi = (result.revenue - result.cost) / result.cost
                    insights['cost_efficiency'][result.variant_id] = roi
        
        return insights
    
    def _generate_recommendations(self, results: List[TestResult], 
                                significance_results: Dict, 
                                config: TestConfiguration) -> List[str]:
        """Generate actionable recommendations"""
        recommendations = []
        
        # Check for clear winner
        significant_winners = [
            variant_id for variant_id, sig_data in significance_results.items()
            if sig_data['is_significant'] and sig_data['lift'] > 0
        ]
        
        if significant_winners:
            best_winner = max(significant_winners, 
                            key=lambda v: significance_results[v]['lift'])
            lift_pct = significance_results[best_winner]['lift'] * 100
            recommendations.append(
                f"Implement variant {best_winner} - shows {lift_pct:.1f}% improvement with statistical significance"
            )
        else:
            recommendations.append("No statistically significant winner found - consider running test longer or increasing sample size")
        
        # Sample size recommendations
        min_sample_per_variant = min(r.sample_size for r in results)
        if min_sample_per_variant < self.minimum_sample_size:
            recommendations.append(f"Increase sample size - current minimum is {min_sample_per_variant}, recommended minimum is {self.minimum_sample_size}")
        
        # Duration recommendations
        duration = self._calculate_duration_days({'start_date': datetime.now() - timedelta(days=7)})
        if duration < 7:
            recommendations.append("Run test for at least 7 days to account for weekly patterns")
        
        # Effect size recommendations
        max_lift = max((sig_data.get('lift', 0) for sig_data in significance_results.values()), default=0)
        if max_lift < 0.05:  # Less than 5% improvement
            recommendations.append("Consider testing more dramatic changes - current variations show minimal impact")
        
        return recommendations
    
    def _validate_test_config(self, config: TestConfiguration) -> None:
        """Validate A/B test configuration"""
        # Check traffic allocation sums to 1.0
        total_allocation = sum(v.traffic_allocation for v in config.variants)
        if abs(total_allocation - 1.0) > 0.01:
            raise ValueError(f"Traffic allocation must sum to 1.0, got {total_allocation}")
        
        # Check for control variant
        control_variants = [v for v in config.variants if v.is_control]
        if len(control_variants) != 1:
            raise ValueError("Exactly one variant must be marked as control")
        
        # Check minimum variants
        if len(config.variants) < 2:
            raise ValueError("At least 2 variants required for A/B test")
    
    def _load_config(self, config_path: str) -> Dict:
        """Load A/B testing configuration"""
        return {
            'default_confidence_level': 0.95,
            'default_power': 0.8,
            'minimum_sample_size': 100,
            'max_test_duration_days': 30,
            'auto_stop_conditions': {
                'min_sample_size': 100,
                'max_duration_days': 30,
                'significance_threshold': 0.95
            }
        }
    
    def get_active_tests(self) -> List[Dict[str, Any]]:
        """Get all active tests"""
        return [
            {
                'test_id': test_id,
                'name': test_data['config'].name,
                'status': test_data['status'].value,
                'start_date': test_data.get('start_date'),
                'participants': len(test_data['participants'])
            }
            for test_id, test_data in self.active_tests.items()
        ]

# A/B Test Manager for easy test creation and management
class ABTestManager:
    """High-level A/B test management"""
    
    def __init__(self):
        self.framework = ABTestFramework()
        self.test_templates = self._load_test_templates()
    
    def create_message_test(self, name: str, variants: List[Dict[str, Any]], 
                           target_sample_size: int = 1000) -> str:
        """Create a message content A/B test"""
        test_variants = []
        
        for i, variant_config in enumerate(variants):
            test_variants.append(TestVariant(
                variant_id=f"variant_{i}",
                name=variant_config.get('name', f'Variant {i}'),
                description=variant_config.get('description', ''),
                traffic_allocation=1.0 / len(variants),
                configuration=variant_config,
                is_control=(i == 0)
            ))
        
        config = TestConfiguration(
            test_id=str(uuid.uuid4()),
            name=name,
            description=f"Message content test with {len(variants)} variants",
            test_type=TestType.MESSAGE_CONTENT,
            variants=test_variants,
            primary_metric=MetricType.CONVERSION_RATE,
            secondary_metrics=[MetricType.OPEN_RATE, MetricType.CLICK_RATE],
            target_sample_size=target_sample_size
        )
        
        return self.framework.create_test(config)
    
    def create_timing_test(self, name: str, send_times: List[str], 
                          target_sample_size: int = 500) -> str:
        """Create a send timing A/B test"""
        test_variants = []
        
        for i, send_time in enumerate(send_times):
            test_variants.append(TestVariant(
                variant_id=f"time_{i}",
                name=f"Send at {send_time}",
                description=f"Send messages at {send_time}",
                traffic_allocation=1.0 / len(send_times),
                configuration={'send_time': send_time},
                is_control=(i == 0)
            ))
        
        config = TestConfiguration(
            test_id=str(uuid.uuid4()),
            name=name,
            description=f"Send timing test with {len(send_times)} time slots",
            test_type=TestType.SEND_TIME,
            variants=test_variants,
            primary_metric=MetricType.OPEN_RATE,
            secondary_metrics=[MetricType.CONVERSION_RATE],
            target_sample_size=target_sample_size
        )
        
        return self.framework.create_test(config)
    
    def _load_test_templates(self) -> Dict:
        """Load predefined test templates"""
        return {
            'message_content': {
                'primary_metric': MetricType.CONVERSION_RATE,
                'secondary_metrics': [MetricType.OPEN_RATE, MetricType.CLICK_RATE],
                'min_sample_size': 200
            },
            'subject_line': {
                'primary_metric': MetricType.OPEN_RATE,
                'secondary_metrics': [MetricType.CONVERSION_RATE],
                'min_sample_size': 500
            },
            'send_timing': {
                'primary_metric': MetricType.OPEN_RATE,
                'secondary_metrics': [MetricType.CONVERSION_RATE],
                'min_sample_size': 300
            }
        }

if __name__ == "__main__":
    # Example usage
    manager = ABTestManager()
    
    # Create a message content test
    message_variants = [
        {
            'name': 'Control - Standard Message',
            'description': 'Current standard message',
            'message_template': 'standard_template',
            'personalization_level': 'basic'
        },
        {
            'name': 'Variant A - High Urgency',
            'description': 'Message with urgency elements',
            'message_template': 'urgent_template',
            'personalization_level': 'high'
        },
        {
            'name': 'Variant B - Educational',
            'description': 'Educational approach',
            'message_template': 'educational_template',
            'personalization_level': 'medium'
        }
    ]
    
    test_id = manager.create_message_test(
        "Q1 Message Optimization Test",
        message_variants,
        target_sample_size=1500
    )
    
    print(f"Created A/B test: {test_id}")
    
    # Start the test
    manager.framework.start_test(test_id)
    
    # Simulate some assignments and events
    for i in range(100):
        lead_id = f"lead_{i}"
        variant = manager.framework.assign_variant(test_id, lead_id)
        
        # Simulate events
        if np.random.random() < 0.25:  # 25% open rate
            manager.framework.track_event(test_id, lead_id, 'email_open')
            
            if np.random.random() < 0.1:  # 10% conversion rate
                manager.framework.track_event(test_id, lead_id, 'conversion', value=1000)
    
    # Analyze results
    analysis = manager.framework.analyze_test(test_id)
    if analysis:
        print(f"Test Analysis:")
        print(f"Winner: {analysis.winner}")
        print(f"Statistical Power: {analysis.statistical_power:.2f}")
        print(f"Recommendations: {analysis.recommendations}")