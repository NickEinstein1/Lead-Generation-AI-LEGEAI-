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
        setModalContent({
          title: "Schedule Call",
          message: "Call scheduling functionality coming soon! You'll be able to schedule follow-up calls with leads."
        });
        setShowModal(true);
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
      <div className="bg-white border-2 border-blue-200 rounded-lg p-6 shadow-md">
        <h2 className="text-xl font-bold text-slate-900 mb-6">Quick Actions</h2>

        <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-3">
          {quickActions.map((action) => (
            <button
              key={action.id}
              onClick={action.onClick}
              className={`${action.color} text-white rounded-lg p-4 transition-all duration-200 hover:shadow-lg active:scale-95 flex flex-col items-center justify-center gap-2`}
              title={action.description}
            >
              <span className="text-2xl">{action.icon}</span>
              <span className="text-xs font-semibold text-center">{action.label}</span>
            </button>
          ))}
        </div>

        <div className="mt-6 pt-6 border-t-2 border-blue-100">
          <p className="text-xs text-slate-600 font-medium">
            💡 Tip: Click any action above to quickly navigate or perform common tasks.
          </p>
        </div>
      </div>

      {/* Modal for notifications */}
      {showModal && modalContent && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50" onClick={() => setShowModal(false)}>
          <div className="bg-white rounded-lg p-6 max-w-md w-full mx-4 shadow-xl" onClick={(e) => e.stopPropagation()}>
            <h3 className="text-xl font-bold text-slate-900 mb-4">{modalContent.title}</h3>
            <p className="text-slate-700 mb-6">{modalContent.message}</p>
            <button
              onClick={() => setShowModal(false)}
              className="w-full bg-blue-600 hover:bg-blue-700 text-white font-semibold py-2 px-4 rounded-lg transition-all"
            >
              Got it!
            </button>
          </div>
        </div>
      )}
    </>
  );
}

