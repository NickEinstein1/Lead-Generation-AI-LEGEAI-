"use client";
import React from "react";

interface LeadData {
  id: string;
  name?: string;
  email?: string;
  phone?: string;
  source?: string;
  channel?: string;
  product_interest?: string;
  status?: string;
  score?: number;
  created_at?: string;
  contact?: {
    email?: string;
    phone?: string;
    name?: string;
  };
}

interface LeadInfoProps {
  lead: LeadData;
}

export default function LeadInfo({ lead }: LeadInfoProps) {
  const getStatusColor = (status?: string) => {
    switch (status?.toLowerCase()) {
      case "new":
        return "bg-blue-100 text-blue-800";
      case "contacted":
        return "bg-cyan-100 text-cyan-800";
      case "qualified":
        return "bg-green-100 text-green-800";
      case "proposal":
        return "bg-amber-100 text-amber-800";
      case "closed":
        return "bg-emerald-100 text-emerald-800";
      default:
        return "bg-slate-100 text-slate-800";
    }
  };

  const getScoreColor = (score?: number) => {
    if (!score) return "text-slate-600";
    if (score >= 80) return "text-green-600";
    if (score >= 60) return "text-amber-600";
    return "text-red-600";
  };

  return (
    <div className="bg-white border-2 border-blue-200 rounded-lg p-6 shadow-md">
      <div className="flex items-start justify-between mb-6">
        <div>
          <h1 className="text-2xl font-bold text-slate-900">{lead.name || lead.contact?.name || "Lead"}</h1>
          <p className="text-sm text-slate-600 mt-1">ID: {lead.id}</p>
        </div>
        <div className="flex gap-2">
          {lead.status && (
            <span className={`px-3 py-1 rounded-full text-sm font-semibold ${getStatusColor(lead.status)}`}>
              {lead.status}
            </span>
          )}
          {lead.score && (
            <span className={`px-3 py-1 rounded-full text-sm font-semibold bg-slate-100 ${getScoreColor(lead.score)}`}>
              Score: {lead.score}
            </span>
          )}
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {/* Contact Information */}
        <div>
          <h3 className="text-sm font-bold text-slate-900 mb-4 uppercase tracking-wide">Contact Information</h3>
          <div className="space-y-3">
            {(lead.contact?.email || lead.email) && (
              <div>
                <p className="text-xs text-slate-600 font-medium">Email</p>
                <p className="text-sm text-slate-900 font-semibold">{lead.contact?.email || lead.email}</p>
              </div>
            )}
            {(lead.contact?.phone || lead.phone) && (
              <div>
                <p className="text-xs text-slate-600 font-medium">Phone</p>
                <p className="text-sm text-slate-900 font-semibold">{lead.contact?.phone || lead.phone}</p>
              </div>
            )}
          </div>
        </div>

        {/* Lead Details */}
        <div>
          <h3 className="text-sm font-bold text-slate-900 mb-4 uppercase tracking-wide">Lead Details</h3>
          <div className="space-y-3">
            {lead.source && (
              <div>
                <p className="text-xs text-slate-600 font-medium">Source</p>
                <p className="text-sm text-slate-900 font-semibold">{lead.source}</p>
              </div>
            )}
            {lead.channel && (
              <div>
                <p className="text-xs text-slate-600 font-medium">Channel</p>
                <p className="text-sm text-slate-900 font-semibold">{lead.channel}</p>
              </div>
            )}
            {lead.product_interest && (
              <div>
                <p className="text-xs text-slate-600 font-medium">Product Interest</p>
                <p className="text-sm text-slate-900 font-semibold">{lead.product_interest}</p>
              </div>
            )}
          </div>
        </div>
      </div>

      {lead.created_at && (
        <div className="mt-6 pt-6 border-t-2 border-blue-100">
          <p className="text-xs text-slate-600 font-medium">
            Created: {new Date(lead.created_at).toLocaleString()}
          </p>
        </div>
      )}
    </div>
  );
}

