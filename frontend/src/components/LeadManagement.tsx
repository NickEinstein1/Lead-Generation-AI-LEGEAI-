"use client";
import React, { useState } from "react";

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
  const [showModal, setShowModal] = useState(false);
  const leadActivities = activities ?? [];

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

      {leadActivities.length === 0 ? (
        <div className="text-sm text-slate-600 font-medium">No recent activities.</div>
      ) : (
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
      )}

      <div className="mt-6 pt-6 border-t-2 border-blue-100">
        <button
          onClick={() => setShowModal(true)}
          className="w-full bg-blue-600 hover:bg-blue-700 text-white font-semibold py-2 px-4 rounded transition-colors text-sm active:scale-95"
        >
          + Add Activity
        </button>
      </div>

      {/* Add Activity Modal */}
      {showModal && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50" onClick={() => setShowModal(false)}>
          <div className="bg-white rounded-lg p-6 max-w-md w-full mx-4 shadow-xl" onClick={(e) => e.stopPropagation()}>
            <h3 className="text-xl font-bold text-slate-900 mb-4">Add Activity</h3>

            <div className="space-y-4">
              <div>
                <label className="block text-sm font-medium text-slate-900 mb-2">Activity Type</label>
                <select className="w-full px-4 py-2 border-2 border-blue-200 rounded-lg focus:outline-none focus:border-blue-600">
                  <option value="call">📞 Call</option>
                  <option value="email">📧 Email</option>
                  <option value="meeting">📅 Meeting</option>
                  <option value="note">📝 Note</option>
                </select>
              </div>

              <div>
                <label className="block text-sm font-medium text-slate-900 mb-2">Description</label>
                <textarea
                  className="w-full px-4 py-2 border-2 border-blue-200 rounded-lg focus:outline-none focus:border-blue-600"
                  rows={3}
                  placeholder="Enter activity details..."
                />
              </div>
            </div>

            <div className="flex gap-3 mt-6">
              <button
                onClick={() => setShowModal(false)}
                className="flex-1 bg-slate-200 hover:bg-slate-300 text-slate-700 font-semibold py-2 px-4 rounded-lg transition-all"
              >
                Cancel
              </button>
              <button
                onClick={() => {
                  setShowModal(false);
                  // In a real app, this would save the activity
                }}
                className="flex-1 bg-blue-600 hover:bg-blue-700 text-white font-semibold py-2 px-4 rounded-lg transition-all"
              >
                Add Activity
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

