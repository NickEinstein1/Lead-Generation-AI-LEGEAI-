"use client";
import React, { useState } from "react";

interface LeadActionsProps {
  leadId: string;
  onScore?: () => void;
  onCreateDoc?: () => void;
  onSendEmail?: () => void;
  onScheduleCall?: () => void;
  onAssign?: () => void;
  scoring?: boolean;
  creating?: boolean;
}

export default function LeadActions({
  leadId,
  onScore,
  onCreateDoc,
  onSendEmail,
  onScheduleCall,
  onAssign,
  scoring = false,
  creating = false,
}: LeadActionsProps) {
  const [showMenu, setShowMenu] = useState(false);

  const actions = [
    {
      id: "score",
      label: "Score Lead",
      icon: "🎯",
      color: "bg-blue-600 hover:bg-blue-700",
      onClick: onScore,
      loading: scoring,
    },
    {
      id: "document",
      label: "Create Document",
      icon: "📄",
      color: "bg-purple-600 hover:bg-purple-700",
      onClick: onCreateDoc,
      loading: creating,
    },
    {
      id: "email",
      label: "Send Email",
      icon: "📧",
      color: "bg-green-600 hover:bg-green-700",
      onClick: onSendEmail,
    },
    {
      id: "call",
      label: "Schedule Call",
      icon: "📞",
      color: "bg-amber-600 hover:bg-amber-700",
      onClick: onScheduleCall,
    },
    {
      id: "assign",
      label: "Assign Lead",
      icon: "👤",
      color: "bg-red-600 hover:bg-red-700",
      onClick: onAssign,
    },
  ];

  return (
    <div className="bg-white border-2 border-blue-200 rounded-lg p-6 shadow-md">
      <h2 className="text-xl font-bold text-slate-900 mb-4">Actions</h2>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
        {actions.map((action) => (
          <button
            key={action.id}
            onClick={action.onClick}
            disabled={action.loading}
            className={`${action.color} text-white font-semibold py-2 px-4 rounded-lg transition-all duration-200 hover:shadow-md active:scale-95 disabled:opacity-50 flex items-center justify-center gap-2`}
          >
            <span>{action.icon}</span>
            <span>{action.loading ? "Loading..." : action.label}</span>
          </button>
        ))}
      </div>

      <div className="mt-6 pt-6 border-t-2 border-blue-100">
        <div className="bg-blue-50 rounded-lg p-4">
          <p className="text-sm text-slate-700 font-medium">
            💡 <strong>Tip:</strong> Use the quick actions above to manage this lead efficiently. Score the lead to get AI-powered insights.
          </p>
        </div>
      </div>
    </div>
  );
}

