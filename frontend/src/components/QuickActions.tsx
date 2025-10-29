"use client";
import React from "react";

interface Action {
  id: string;
  label: string;
  icon: string;
  description: string;
  color: string;
  onClick?: () => void;
}

interface QuickActionsProps {
  actions?: Action[];
}

export default function QuickActions({ actions }: QuickActionsProps) {
  const defaultActions: Action[] = [
    {
      id: "new-lead",
      label: "New Lead",
      icon: "➕",
      description: "Create a new lead",
      color: "bg-blue-500 hover:bg-blue-600",
    },
    {
      id: "import-leads",
      label: "Import Leads",
      icon: "📥",
      description: "Import leads from CSV",
      color: "bg-purple-500 hover:bg-purple-600",
    },
    {
      id: "send-campaign",
      label: "Send Campaign",
      icon: "📧",
      description: "Launch email campaign",
      color: "bg-green-500 hover:bg-green-600",
    },
    {
      id: "schedule-call",
      label: "Schedule Call",
      icon: "📞",
      description: "Schedule follow-up call",
      color: "bg-amber-500 hover:bg-amber-600",
    },
    {
      id: "generate-report",
      label: "Generate Report",
      icon: "📊",
      description: "Create performance report",
      color: "bg-red-500 hover:bg-red-600",
    },
    {
      id: "view-analytics",
      label: "View Analytics",
      icon: "📈",
      description: "View detailed analytics",
      color: "bg-cyan-500 hover:bg-cyan-600",
    },
  ];

  const quickActions = actions || defaultActions;

  return (
    <div className="bg-white border-2 border-blue-200 rounded-lg p-6 shadow-md">
      <h2 className="text-xl font-bold text-slate-900 mb-6">Quick Actions</h2>

      <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-3">
        {quickActions.map((action) => (
          <button
            key={action.id}
            onClick={action.onClick}
            className={`${action.color} text-white rounded-lg p-4 transition-all duration-200 hover:shadow-lg active:scale-95 flex flex-col items-center justify-center gap-2`}
          >
            <span className="text-2xl">{action.icon}</span>
            <span className="text-xs font-semibold text-center">{action.label}</span>
          </button>
        ))}
      </div>

      <div className="mt-6 pt-6 border-t-2 border-blue-100">
        <p className="text-xs text-slate-600 font-medium">
          💡 Tip: Use keyboard shortcuts for faster navigation. Press <kbd className="bg-slate-200 px-2 py-1 rounded text-xs">?</kbd> for help.
        </p>
      </div>
    </div>
  );
}

