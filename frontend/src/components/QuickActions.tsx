"use client";
import React, { useState } from "react";
import { useRouter } from "next/navigation";

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
  const router = useRouter();
  const [showModal, setShowModal] = useState(false);
  const [modalContent, setModalContent] = useState<{ title: string; message: string } | null>(null);

  const handleAction = (actionId: string) => {
    switch (actionId) {
      case "new-lead":
        router.push("/dashboard/leads?action=new");
        break;
      case "import-leads":
        setModalContent({
          title: "Import Leads",
          message: "CSV import functionality coming soon! You'll be able to upload CSV files with lead data."
        });
        setShowModal(true);
        break;
      case "send-campaign":
        router.push("/dashboard/communications/campaigns");
        break;
      case "schedule-call":
        router.push("/dashboard/scheduler");
        break;
      case "generate-report":
        router.push("/dashboard/reports");
        break;
      case "view-analytics":
        router.push("/dashboard/analytics");
        break;
      default:
        console.log("Action not implemented:", actionId);
    }
  };

  const defaultActions: Action[] = [
    {
      id: "new-lead",
      label: "New Lead",
      icon: "➕",
      description: "Create a new lead",
      color: "bg-blue-500 hover:bg-blue-600",
      onClick: () => handleAction("new-lead"),
    },
    {
      id: "import-leads",
      label: "Import Leads",
      icon: "📥",
      description: "Import leads from CSV",
      color: "bg-purple-500 hover:bg-purple-600",
      onClick: () => handleAction("import-leads"),
    },
    {
      id: "send-campaign",
      label: "Send Campaign",
      icon: "📧",
      description: "Launch email campaign",
      color: "bg-green-500 hover:bg-green-600",
      onClick: () => handleAction("send-campaign"),
    },
    {
      id: "schedule-call",
      label: "Schedule Call",
      icon: "📞",
      description: "Schedule follow-up call",
      color: "bg-amber-500 hover:bg-amber-600",
      onClick: () => handleAction("schedule-call"),
    },
    {
      id: "generate-report",
      label: "Generate Report",
      icon: "📊",
      description: "Create performance report",
      color: "bg-red-500 hover:bg-red-600",
      onClick: () => handleAction("generate-report"),
    },
    {
      id: "view-analytics",
      label: "View Analytics",
      icon: "📈",
      description: "View detailed analytics",
      color: "bg-cyan-500 hover:bg-cyan-600",
      onClick: () => handleAction("view-analytics"),
    },
  ];

  const quickActions = actions || defaultActions;

  return (
    <>
      <div className="bg-slate-900/60 backdrop-blur-xl border border-blue-500/30 rounded-2xl p-4 sm:p-6 shadow-2xl hover:shadow-cyan-500/20 transition-all duration-300 relative overflow-hidden">
        {/* Background glow */}
        <div className="absolute inset-0 bg-gradient-to-br from-blue-500/10 via-orange-500/10 to-cyan-500/10"></div>

        <div className="flex items-center justify-between mb-4 sm:mb-6 relative z-10">
          <h2 className="text-lg sm:text-xl font-bold text-blue-100 flex items-center gap-2">
            <span className="text-2xl">⚡</span>
            <span className="bg-gradient-to-r from-cyan-300 to-blue-300 bg-clip-text text-transparent">Quick Actions</span>
          </h2>
          <span className="text-xs text-cyan-400 font-medium hidden sm:inline">Choose an action</span>
        </div>

        <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-6 gap-2 sm:gap-3 relative z-10">
          {quickActions.map((action) => (
            <button
              key={action.id}
              onClick={action.onClick}
              className={`group ${action.color} text-white rounded-xl p-3 sm:p-4 transition-all duration-300 hover:shadow-2xl hover:shadow-cyan-500/30 active:scale-95 hover:-translate-y-2 flex flex-col items-center justify-center gap-1.5 sm:gap-2 relative overflow-hidden border border-white/20 hover:border-cyan-300/50`}
              title={action.description}
            >
              {/* Shine effect on hover */}
              <div className="absolute inset-0 bg-gradient-to-r from-transparent via-white/30 to-transparent -translate-x-full group-hover:translate-x-full transition-transform duration-700"></div>

              {/* Glow effect */}
              <div className="absolute inset-0 bg-gradient-to-br from-cyan-500/0 to-cyan-500/20 opacity-0 group-hover:opacity-100 transition-opacity duration-300"></div>

              <span className="text-xl sm:text-2xl group-hover:scale-125 group-hover:rotate-12 transition-all duration-300 relative z-10 filter drop-shadow-lg">{action.icon}</span>
              <span className="text-[10px] sm:text-xs font-bold text-center leading-tight relative z-10">{action.label}</span>
            </button>
          ))}
        </div>

        <div className="mt-4 sm:mt-6 pt-4 sm:pt-6 border-t border-blue-500/30 bg-gradient-to-r from-blue-500/10 to-orange-500/10 -mx-4 sm:-mx-6 px-4 sm:px-6 py-3 sm:py-4 rounded-b-2xl relative z-10">
          <p className="text-[10px] sm:text-xs text-cyan-300 font-medium flex items-center gap-2">
            <span className="text-base">💡</span>
            <span>Tip: Click any action above to quickly navigate or perform common tasks.</span>
          </p>
        </div>
      </div>

      {/* Modal for notifications - Responsive */}
      {showModal && modalContent && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50 p-4" onClick={() => setShowModal(false)}>
          <div className="bg-white rounded-lg p-4 sm:p-6 max-w-md w-full shadow-xl" onClick={(e) => e.stopPropagation()}>
            <h3 className="text-lg sm:text-xl font-bold text-slate-900 mb-3 sm:mb-4">{modalContent.title}</h3>
            <p className="text-sm sm:text-base text-slate-700 mb-4 sm:mb-6">{modalContent.message}</p>
            <button
              onClick={() => setShowModal(false)}
              className="w-full bg-blue-600 hover:bg-blue-700 text-white font-semibold py-2 px-4 rounded-lg transition-all text-sm sm:text-base"
            >
              Got it!
            </button>
          </div>
        </div>
      )}
    </>
  );
}

