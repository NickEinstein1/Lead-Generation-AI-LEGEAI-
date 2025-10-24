from fastapi import APIRouter, HTTPException, Query, Depends
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
import logging

from automation import workflow_engine
from automation.lead_routing import lead_router
from automation import task_manager
from automation.notification_system import notification_system
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from database.connection import session_dep
from models.lead import Lead
from models.score import Score
import os


router = APIRouter(prefix="/dashboard", tags=["Dashboard & Analytics"])

@router.get("/overview")
async def get_dashboard_overview():
    """Get overall dashboard metrics"""
    try:
        # Get current date for filtering
        today = datetime.now()
        week_ago = today - timedelta(days=7)

        # Workflow metrics
        workflow_stats = {}
        for workflow_id in workflow_engine.workflows.keys():
            stats = workflow_engine.get_workflow_stats(workflow_id)
            if stats:
                workflow_stats[workflow_id] = stats

        # Task metrics
        task_analytics = task_manager.get_task_analytics()
        overdue_tasks = task_manager.get_overdue_tasks()

        # Routing metrics
        routing_analytics = lead_router.get_routing_analytics()

        # Notification metrics
        notification_stats = notification_system.get_notification_stats()

        return {
            "status": "success",
            "timestamp": today.isoformat(),
            "overview": {
                "workflows": {
                    "total_workflows": len(workflow_engine.workflows),
                    "active_workflows": len([w for w in workflow_engine.workflows.values() if w.status.value == "active"]),
                    "total_executions": sum([w.execution_count for w in workflow_engine.workflows.values()]),
                    "success_rate": _calculate_workflow_success_rate(workflow_stats)
                },
                "tasks": {
                    "total_tasks": task_analytics.get("total_tasks", 0),
                    "completed_tasks": task_analytics.get("completed_tasks", 0),
                    "overdue_tasks": len(overdue_tasks),
                    "completion_rate": task_analytics.get("completion_rate", 0)
                },
                "routing": {
                    "total_assignments": routing_analytics.get("total_assignments", 0),
                    "active_reps": routing_analytics.get("active_reps", 0),
                    "average_assignments": routing_analytics.get("average_assignments_per_rep", 0)
                },
                "notifications": {
                    "total_sent": notification_stats.get("total_notifications", 0),
                    "success_rate": notification_stats.get("success_rate", 0),
                    "active_channels": len(notification_stats.get("registered_channels", []))
                }
            }
        }

    except Exception as e:
        logging.error(f"Error getting dashboard overview: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error getting dashboard overview: {str(e)}")

@router.get("/timeseries/leads")
async def timeseries_leads(days: int = Query(30, ge=1, le=180), session: AsyncSession = Depends(session_dep)):
    """Daily lead counts for the last N days."""
    try:
        now = datetime.now()
        start = (now - timedelta(days=days - 1)).replace(hour=0, minute=0, second=0, microsecond=0)
        use_db = os.getenv("USE_DB", "false").lower() == "true"
        buckets: Dict[str, int] = {}
        for i in range(days):
            d = (start + timedelta(days=i)).date().isoformat()
            buckets[d] = 0
        if use_db:
            rows = (await session.execute(select(Lead.created_at).where(Lead.created_at >= start))).scalars().all()
            for dt in rows:
                d = (dt.date().isoformat())
                if d in buckets:
                    buckets[d] += 1
        else:
            # Fallback to in-memory store
            try:
                from api.leads_api import LEADS_DB
                for lead in LEADS_DB.values():
                    created = lead.get("created_at")
                    d = (datetime.fromisoformat(created).date().isoformat()) if created else None
                    if d and d in buckets:
                        buckets[d] += 1
            except Exception:
                pass
        series = [{"date": d, "leads": buckets[d]} for d in sorted(buckets.keys())]
        return {"status": "success", "series": series}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error building lead timeseries: {e}")


@router.get("/timeseries/scores")
async def timeseries_scores(days: int = Query(30, ge=1, le=180), session: AsyncSession = Depends(session_dep)):
    """Daily average score and count for the last N days."""
    try:
        now = datetime.now()
        start = (now - timedelta(days=days - 1)).replace(hour=0, minute=0, second=0, microsecond=0)
        use_db = os.getenv("USE_DB", "false").lower() == "true"
        buckets: Dict[str, Dict[str, float]] = {}
        for i in range(days):
            d = (start + timedelta(days=i)).date().isoformat()
            buckets[d] = {"sum": 0.0, "count": 0}
        if use_db:
            rows = (await session.execute(select(Score.scored_at, Score.score).where(Score.scored_at >= start))).all()
            for scored_at, score in rows:
                d = (scored_at.date().isoformat())
                if d in buckets and score is not None:
                    buckets[d]["sum"] += float(score)
                    buckets[d]["count"] += 1
        else:
            try:
                from api.leads_api import SCORES_DB
                for entries in SCORES_DB.values():
                    for e in entries:
                        sa = e.get("scored_at")
                        d = (datetime.fromisoformat(sa).date().isoformat()) if sa else None
                        score = e.get("score")
                        if d and d in buckets and score is not None:
                            buckets[d]["sum"] += float(score)
                            buckets[d]["count"] += 1
            except Exception:
                pass
        series = []
        for d in sorted(buckets.keys()):
            s = buckets[d]
            avg = (s["sum"] / s["count"]) if s["count"] > 0 else 0
            series.append({"date": d, "avg_score": round(avg, 2), "count": s["count"]})
        return {"status": "success", "series": series}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error building score timeseries: {e}")


@router.get("/workflows/performance")
async def get_workflow_performance():
    """Get detailed workflow performance metrics"""
    try:
        performance_data = []

        for workflow_id, workflow in workflow_engine.workflows.items():
            stats = workflow_engine.get_workflow_stats(workflow_id)

            # Get recent executions
            recent_executions = [
                exec for exec in workflow_engine.executions.values()
                if exec.workflow_id == workflow_id and
                exec.started_at >= datetime.now() - timedelta(days=30)
            ]

            performance_data.append({
                "workflow_id": workflow_id,
                "name": workflow.name,
                "status": workflow.status.value,
                "total_executions": workflow.execution_count,
                "success_count": workflow.success_count,
                "failure_count": workflow.failure_count,
                "success_rate": stats.get("success_rate", 0),
                "recent_executions": len(recent_executions),
                "avg_execution_time": _calculate_avg_execution_time(recent_executions),
                "last_execution": max([e.started_at for e in recent_executions]).isoformat() if recent_executions else None
            })

        return {
            "status": "success",
            "workflow_performance": performance_data
        }

    except Exception as e:
        logging.error(f"Error getting workflow performance: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error getting workflow performance: {str(e)}")

@router.get("/tasks/analytics")
async def get_task_analytics(
    assignee_id: Optional[str] = Query(None),
    days: int = Query(30, description="Number of days to analyze")
):
    """Get detailed task analytics"""
    try:
        # Calculate date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)

        # Get analytics
        analytics = task_manager.get_task_analytics(
            assignee_id=assignee_id,
            date_range=(start_date, end_date)
        )

        # Get additional metrics
        overdue_tasks = task_manager.get_overdue_tasks()

        # Filter overdue tasks by assignee if specified
        if assignee_id:
            overdue_tasks = [t for t in overdue_tasks if t.assignee_id == assignee_id]

        # Calculate productivity metrics
        productivity_metrics = _calculate_productivity_metrics(analytics, overdue_tasks)

        return {
            "status": "success",
            "date_range": {
                "start_date": start_date.isoformat(),
                "end_date": end_date.isoformat(),
                "days": days
            },
            "assignee_id": assignee_id,
            "analytics": analytics,
            "productivity_metrics": productivity_metrics,
            "overdue_summary": {
                "count": len(overdue_tasks),
                "by_priority": _group_tasks_by_priority(overdue_tasks)
            }
        }

    except Exception as e:
        logging.error(f"Error getting task analytics: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error getting task analytics: {str(e)}")

@router.get("/routing/performance")
async def get_routing_performance():
    """Get lead routing performance metrics"""
    try:
        analytics = lead_router.get_routing_analytics()

        # Get detailed rep performance
        rep_performance = []
        for rep_id, rep in lead_router.sales_reps.items():
            # Calculate rep-specific metrics from available fields
            rep_assignments = [
                a for a in lead_router.assignment_history
                if a.get('rep_id') == rep_id
            ]
            last_assigned_at = rep_assignments[-1]['assigned_at'] if rep_assignments else None

            current_load = getattr(rep, 'assigned', 0)
            max_capacity = getattr(rep, 'capacity', 0)
            utilization = (current_load / max_capacity) if max_capacity > 0 else 0

            rep_performance.append({
                "rep_id": rep_id,
                "name": getattr(rep, 'name', rep_id),
                "status": "available" if current_load < max_capacity else "busy",
                "specialties": [],
                "territories": [getattr(rep, 'region', None)] if getattr(rep, 'region', None) else [],
                "current_load": current_load,
                "max_capacity": max_capacity,
                "utilization": utilization,
                "total_assignments": len(rep_assignments),
                "performance_score": 0,
                "last_assignment": last_assigned_at
            })

        return {
            "status": "success",
            "routing_analytics": analytics,
            "rep_performance": rep_performance,
            "routing_efficiency": _calculate_routing_efficiency(analytics, rep_performance)
        }

    except Exception as e:
        logging.error(f"Error getting routing performance: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error getting routing performance: {str(e)}")

@router.get("/notifications/analytics")
async def get_notification_analytics():
    """Get notification system analytics"""
    try:
        stats = notification_system.get_notification_stats()

        # Get recent notifications for trend analysis
        recent_notifications = [
            n for n in notification_system.notifications.values()
            if n.created_at >= datetime.now() - timedelta(days=7)
        ]

        # Calculate delivery metrics
        delivery_metrics = _calculate_delivery_metrics(recent_notifications)

        # Channel performance
        channel_performance = _calculate_channel_performance(recent_notifications)

        return {
            "status": "success",
            "notification_stats": stats,
            "delivery_metrics": delivery_metrics,
            "channel_performance": channel_performance,
            "recent_activity": {
                "last_7_days": len(recent_notifications),
                "avg_per_day": len(recent_notifications) / 7
            }
        }

    except Exception as e:
        logging.error(f"Error getting notification analytics: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error getting notification analytics: {str(e)}")

@router.get("/real-time/metrics")
async def get_real_time_metrics():
    """Get real-time system metrics"""
    try:
        current_time = datetime.now()

        # Active workflows
        active_workflows = len([
            w for w in workflow_engine.workflows.values()
            if w.status.value == "active"
        ])

        # Running executions
        running_executions = len([
            e for e in workflow_engine.executions.values()
            if e.status.value == "active"
        ])

        # Pending tasks
        pending_tasks = len([
            t for t in task_manager.tasks.values()
            if t.status.value == "pending"
        ])

        # Available reps (approximate: capacity not fully utilized)
        available_reps = len([
            r for r in lead_router.sales_reps.values()
            if getattr(r, 'assigned', 0) < getattr(r, 'capacity', 0)
        ])

        # Recent notifications (last hour)
        recent_notifications = len([
            n for n in notification_system.notifications.values()
            if n.created_at >= current_time - timedelta(hours=1)
        ])

        return {
            "status": "success",
            "timestamp": current_time.isoformat(),
            "real_time_metrics": {
                "active_workflows": active_workflows,
                "running_executions": running_executions,
                "pending_tasks": pending_tasks,
                "available_reps": available_reps,
                "notifications_last_hour": recent_notifications,
                "system_health": "healthy"  # Would implement actual health checks
            }
        }

    except Exception as e:
        logging.error(f"Error getting real-time metrics: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error getting real-time metrics: {str(e)}")

@router.get("/alerts/active")
async def get_active_alerts():
    """Get active system alerts"""
    try:
        alerts = []
        current_time = datetime.now()

        # Check for overdue tasks
        overdue_tasks = task_manager.get_overdue_tasks()
        if overdue_tasks:
            alerts.append({
                "type": "overdue_tasks",
                "severity": "high",
                "message": f"{len(overdue_tasks)} tasks are overdue",
                "count": len(overdue_tasks),
                "timestamp": current_time.isoformat()
            })

        # Check for failed workflows
        failed_workflows = [
            w for w in workflow_engine.workflows.values()
            if w.failure_count > 0 and w.failure_count / max(w.execution_count, 1) > 0.1
        ]
        if failed_workflows:
            alerts.append({
                "type": "workflow_failures",
                "severity": "medium",
                "message": f"{len(failed_workflows)} workflows have high failure rates",
                "count": len(failed_workflows),
                "timestamp": current_time.isoformat()
            })

        # Check for notification failures
        failed_notifications = [
            n for n in notification_system.notifications.values()
            if n.status.value == "failed"
        ]
        if len(failed_notifications) > 10:  # Threshold
            alerts.append({
                "type": "notification_failures",
                "severity": "medium",
                "message": f"{len(failed_notifications)} notifications have failed",
                "count": len(failed_notifications),
                "timestamp": current_time.isoformat()
            })

        # Check for rep availability
        available_reps = [
            r for r in lead_router.sales_reps.values()
            if getattr(r, 'assigned', 0) < getattr(r, 'capacity', 0)
        ]
        if len(available_reps) < 2:  # Minimum threshold
            alerts.append({
                "type": "low_rep_availability",
                "severity": "high",
                "message": f"Only {len(available_reps)} sales reps available",
                "count": len(available_reps),
                "timestamp": current_time.isoformat()
            })

        return {
            "status": "success",
            "active_alerts": alerts,
            "alert_count": len(alerts),
            "last_checked": current_time.isoformat()
        }

    except Exception as e:
        logging.error(f"Error getting active alerts: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error getting active alerts: {str(e)}")

# Helper functions
def _calculate_workflow_success_rate(workflow_stats: Dict[str, Any]) -> float:
    """Calculate overall workflow success rate"""
    total_executions = sum([stats.get("execution_count", 0) for stats in workflow_stats.values()])
    total_successes = sum([stats.get("success_count", 0) for stats in workflow_stats.values()])

    return total_successes / total_executions if total_executions > 0 else 0

def _calculate_avg_execution_time(executions: List[Any]) -> Optional[float]:
    """Calculate average execution time in seconds"""
    completed_executions = [
        e for e in executions
        if e.completed_at and e.started_at
    ]

    if not completed_executions:
        return None

    total_time = sum([
        (e.completed_at - e.started_at).total_seconds()
        for e in completed_executions
    ])

    return total_time / len(completed_executions)

def _calculate_productivity_metrics(analytics: Dict[str, Any], overdue_tasks: List[Any]) -> Dict[str, Any]:
    """Calculate productivity metrics"""
    total_tasks = analytics.get("total_tasks", 0)
    completed_tasks = analytics.get("completed_tasks", 0)

    return {
        "efficiency_score": completed_tasks / total_tasks if total_tasks > 0 else 0,
        "overdue_rate": len(overdue_tasks) / total_tasks if total_tasks > 0 else 0,
        "avg_completion_time": analytics.get("average_completion_time_hours", 0),
        "productivity_trend": "stable"  # Would calculate actual trend
    }

def _group_tasks_by_priority(tasks: List[Any]) -> Dict[str, int]:
    """Group tasks by priority"""
    priority_groups = {}
    for task in tasks:
        priority = task.priority.name
        priority_groups[priority] = priority_groups.get(priority, 0) + 1

    return priority_groups

def _calculate_routing_efficiency(analytics: Dict[str, Any], rep_performance: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Calculate routing efficiency metrics"""
    total_assignments = analytics.get("total_assignments", 0)
    active_reps = len([r for r in rep_performance if r["status"] == "available"])

    avg_utilization = sum([r["utilization"] for r in rep_performance]) / len(rep_performance) if rep_performance else 0

    return {
        "assignment_distribution_score": _calculate_distribution_score(rep_performance),
        "average_utilization": avg_utilization,
        "routing_balance": "good" if 0.3 <= avg_utilization <= 0.8 else "needs_attention"
    }

def _calculate_distribution_score(rep_performance: List[Dict[str, Any]]) -> float:
    """Calculate how evenly distributed assignments are"""
    if not rep_performance:
        return 0

    assignments = [r["total_assignments"] for r in rep_performance]
    if not assignments:
        return 1.0

    avg_assignments = sum(assignments) / len(assignments)
    variance = sum([(a - avg_assignments) ** 2 for a in assignments]) / len(assignments)

    # Lower variance = better distribution (score closer to 1)
    return 1 / (1 + variance / max(avg_assignments, 1))

def _calculate_delivery_metrics(notifications: List[Any]) -> Dict[str, Any]:
    """Calculate notification delivery metrics"""
    if not notifications:
        return {"delivery_rate": 0, "avg_delivery_time": 0}

    delivered = [n for n in notifications if n.status.value in ["sent", "delivered"]]
    delivery_rate = len(delivered) / len(notifications)

    # Calculate average delivery time
    delivery_times = []
    for n in delivered:
        if n.sent_at and n.created_at:
            delivery_times.append((n.sent_at - n.created_at).total_seconds())

    avg_delivery_time = sum(delivery_times) / len(delivery_times) if delivery_times else 0

    return {
        "delivery_rate": delivery_rate,
        "avg_delivery_time_seconds": avg_delivery_time,
        "total_notifications": len(notifications),
        "delivered_notifications": len(delivered)
    }

def _calculate_channel_performance(notifications: List[Any]) -> Dict[str, Dict[str, Any]]:
    """Calculate performance by notification channel"""
    channel_stats = {}

    for notification in notifications:
        channel = notification.notification_type.value

        if channel not in channel_stats:
            channel_stats[channel] = {
                "total": 0,
                "sent": 0,
                "failed": 0,
                "retry_count": 0
            }

        channel_stats[channel]["total"] += 1
        channel_stats[channel]["retry_count"] += notification.retry_count

        if notification.status.value in ["sent", "delivered"]:
            channel_stats[channel]["sent"] += 1
        elif notification.status.value == "failed":
            channel_stats[channel]["failed"] += 1

    # Calculate success rates
    for channel, stats in channel_stats.items():
        stats["success_rate"] = stats["sent"] / stats["total"] if stats["total"] > 0 else 0
        stats["avg_retries"] = stats["retry_count"] / stats["total"] if stats["total"] > 0 else 0

    return channel_stats