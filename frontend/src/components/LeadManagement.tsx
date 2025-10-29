"use client";
import React from "react";

interface LeadActivity {
  id: string;
  type: "call" | "email" | "meeting" | "note" | "status_change";
  description: string;
  timestamp: string;
  user: string;
}

interface LeadManagementProps {
  activities?: LeadActivity[];
}

export default function LeadManagement({ activities }: LeadManagementProps) {
  const defaultActivities: LeadActivity[] = [
    {
      id: "1",
      type: "call",
      description: "Initial consultation call",
      timestamp: "2 hours ago",
      user: "John Smith",
    },
    {
      id: "2",
      type: "email",
      description: "Sent quote for auto insurance",
      timestamp: "4 hours ago",
      user: "Sarah Johnson",
    },
    {
      id: "3",
      type: "meeting",
      description: "Scheduled follow-up meeting",
      timestamp: "1 day ago",
      user: "Mike Davis",
    },
    {
      id: "4",
      type: "note",
      description: "Customer interested in bundled policies",
      timestamp: "2 days ago",
      user: "Emma Wilson",
    },
    {
      id: "5",
      type: "status_change",
      description: "Lead moved to Proposal stage",
      timestamp: "3 days ago",
      user: "System",
    },
  ];

  const leadActivities = activities || defaultActivities;

  const getActivityIcon = (type: string) => {
    switch (type) {
      case "call":
        return "📞";
      case "email":
        return "📧";
      case "meeting":
        return "📅";
      case "note":
        return "📝";
      case "status_change":
        return "✅";
      default:
        return "📌";
    }
  };

  const getActivityColor = (type: string) => {
    switch (type) {
      case "call":
        return "border-l-4 border-blue-500";
      case "email":
        return "border-l-4 border-purple-500";
      case "meeting":
        return "border-l-4 border-green-500";
      case "note":
        return "border-l-4 border-amber-500";
      case "status_change":
        return "border-l-4 border-emerald-500";
      default:
        return "border-l-4 border-slate-300";
    }
  };

  return (
    <div className="bg-white border-2 border-blue-200 rounded-lg p-6 shadow-md">
      <h2 className="text-xl font-bold text-slate-900 mb-6">Activity Timeline</h2>

      <div className="space-y-3">
        {leadActivities.map((activity) => (
          <div
            key={activity.id}
            className={`${getActivityColor(activity.type)} bg-slate-50 rounded p-4 hover:bg-slate-100 transition-colors`}
          >
            <div className="flex items-start gap-3">
              <span className="text-xl mt-1">{getActivityIcon(activity.type)}</span>
              <div className="flex-1 min-w-0">
                <div className="flex items-center justify-between gap-2">
                  <p className="font-semibold text-slate-900 text-sm">{activity.description}</p>
                  <span className="text-xs text-slate-500 font-medium whitespace-nowrap">{activity.timestamp}</span>
                </div>
                <p className="text-xs text-slate-600 mt-1">by {activity.user}</p>
              </div>
            </div>
          </div>
        ))}
      </div>

      <div className="mt-6 pt-6 border-t-2 border-blue-100">
        <button className="w-full bg-blue-600 hover:bg-blue-700 text-white font-semibold py-2 px-4 rounded transition-colors text-sm">
          + Add Activity
        </button>
      </div>
    </div>
  );
}

