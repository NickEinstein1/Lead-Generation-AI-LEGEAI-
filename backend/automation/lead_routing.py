"""
Simple routing engine stub for MVP.
Provides analytics hooks used by dashboard_api and a basic route_lead method.
Replace with capacity-, skills-, and region-aware routing in later sprints.
"""
from dataclasses import dataclass
from typing import Dict, Any, List, Optional
from datetime import datetime, timezone

@dataclass
class Agent:
    id: str
    name: str
    region: Optional[str] = None
    capacity: int = 50
    assigned: int = 0
from enum import Enum
from dataclasses import dataclass

# Backward-compatibility types expected by automation.__init__
SalesRep = Agent

class SalesRepStatus(Enum):
    ACTIVE = "active"
    INACTIVE = "inactive"

class RoutingStrategy(Enum):
    CAPACITY_FIRST = "capacity_first"
    ROUND_ROBIN = "round_robin"

@dataclass
class RoutingRule:
    rule_id: str
    name: str
    strategy: RoutingStrategy = RoutingStrategy.CAPACITY_FIRST
    conditions: dict | None = None



class LeadRouter:
    def __init__(self):
        # Minimal in-memory agent pool
        self.sales_reps: Dict[str, Agent] = {
            "a1": Agent(id="a1", name="Alice", region="CA", capacity=50),
            "a2": Agent(id="a2", name="Bob", region="NY", capacity=40),
            "a3": Agent(id="a3", name="Carol", region="TX", capacity=45),
        }
        self.assignment_history: List[Dict[str, Any]] = []
        self._rr_order = list(self.sales_reps.keys())
        self._rr_index = 0

    def get_routing_analytics(self) -> Dict[str, Any]:
        total_capacity = sum(a.capacity for a in self.sales_reps.values())
        total_assigned = sum(a.assigned for a in self.sales_reps.values())
        active_reps = len([a for a in self.sales_reps.values() if a.assigned < a.capacity])
        total_assignments = len(self.assignment_history)
        avg_per_rep = (total_assignments / len(self.sales_reps)) if self.sales_reps else 0
        return {
            "agent_count": len(self.sales_reps),
            "total_capacity": total_capacity,
            "total_assigned": total_assigned,
            "assignments_last_24h": len(self.assignment_history[-1000:]),
            "total_assignments": total_assignments,
            "active_reps": active_reps,
            "average_assignments_per_rep": avg_per_rep,
        }

    def _next_agent_round_robin(self) -> Agent:
        agent_id = self._rr_order[self._rr_index % len(self._rr_order)]
        self._rr_index += 1
        return self.sales_reps[agent_id]

    def route_lead(self, lead: Dict[str, Any], score_band: Optional[str] = None) -> Dict[str, Any]:
        # Very naive policy: prefer agents with lowest utilization
        agents_sorted = sorted(self.sales_reps.values(), key=lambda a: (a.assigned / max(a.capacity, 1)))
        agent = agents_sorted[0] if agents_sorted else self._next_agent_round_robin()
        agent.assigned += 1
        assignment = {
            "lead_id": lead.get("id"),
            "agent_id": agent.id,
            "rep_id": agent.id,
            "assigned_at": datetime.now(timezone.utc).isoformat(),
            "policy": "capacity_first",
            "band": score_band,
        }
        self.assignment_history.append(assignment)
        return assignment


# Global instance expected by dashboard_api
lead_router = LeadRouter()

