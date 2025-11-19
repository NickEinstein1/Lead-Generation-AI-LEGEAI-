"use client";
import React from "react";

interface PipelineStage {
  name: string;
  count: number;
  percentage: number;
  color: string;
}

interface LeadPipelineProps {
  stages?: PipelineStage[];
}

export default function LeadPipeline({ stages }: LeadPipelineProps) {
  const defaultStages: PipelineStage[] = [
    { name: "New", count: 89, percentage: 18, color: "bg-blue-500" },
    { name: "Contacted", count: 127, percentage: 26, color: "bg-cyan-500" },
    { name: "Qualified", count: 98, percentage: 20, color: "bg-emerald-500" },
    { name: "Proposal", count: 112, percentage: 23, color: "bg-amber-500" },
    { name: "Closed", count: 64, percentage: 13, color: "bg-green-600" },
  ];

  const pipelineStages = stages || defaultStages;
  const totalLeads = pipelineStages.reduce((sum, s) => sum + s.count, 0);

  return (
    <div className="bg-white border-2 border-blue-200 rounded-lg p-6 shadow-md">
      <h2 className="text-xl font-bold text-slate-900 mb-6">Lead Pipeline</h2>
      
      <div className="space-y-4">
        {pipelineStages.map((stage, idx) => (
          <div key={idx} className="space-y-2">
            <div className="flex items-center justify-between">
              <span className="text-sm font-semibold text-slate-700">{stage.name}</span>
              <span className="text-sm font-bold text-slate-900">{stage.count} leads ({stage.percentage}%)</span>
            </div>
            <div className="w-full bg-slate-200 rounded-full h-3 overflow-hidden">
              <div
                className={`h-full ${stage.color} transition-all duration-300`}
                style={{ width: `${stage.percentage}%` }}
              />
            </div>
          </div>
        ))}
      </div>

      <div className="mt-6 pt-6 border-t-2 border-blue-100">
        <div className="grid grid-cols-2 gap-4">
          <div className="bg-blue-50 rounded p-3">
            <div className="text-xs text-slate-600 font-medium">Total Leads</div>
            <div className="text-2xl font-bold text-blue-700">{totalLeads}</div>
          </div>
          <div className="bg-green-50 rounded p-3">
            <div className="text-xs text-slate-600 font-medium">Conversion Rate</div>
            <div className="text-2xl font-bold text-green-700">
              {((pipelineStages[pipelineStages.length - 1].count / totalLeads) * 100).toFixed(1)}%
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

