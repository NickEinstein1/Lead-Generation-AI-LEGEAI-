"use client";
import React, { useState, useEffect } from "react";

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
  const [dynamicStages, setDynamicStages] = useState<PipelineStage[]>([]);

  // Generate dynamic values on component mount
  useEffect(() => {
    const generateDynamicStages = (): PipelineStage[] => {
      const randomInRange = (min: number, max: number) => Math.floor(Math.random() * (max - min + 1)) + min;

      // Generate random counts
      const newCount = randomInRange(75, 105);
      const contactedCount = randomInRange(110, 145);
      const qualifiedCount = randomInRange(85, 115);
      const proposalCount = randomInRange(95, 125);
      const closedCount = randomInRange(55, 75);

      const total = newCount + contactedCount + qualifiedCount + proposalCount + closedCount;

      return [
        { name: "New", count: newCount, percentage: Math.round((newCount / total) * 100), color: "bg-blue-500" },
        { name: "Contacted", count: contactedCount, percentage: Math.round((contactedCount / total) * 100), color: "bg-cyan-500" },
        { name: "Qualified", count: qualifiedCount, percentage: Math.round((qualifiedCount / total) * 100), color: "bg-emerald-500" },
        { name: "Proposal", count: proposalCount, percentage: Math.round((proposalCount / total) * 100), color: "bg-amber-500" },
        { name: "Closed", count: closedCount, percentage: Math.round((closedCount / total) * 100), color: "bg-green-600" },
      ];
    };

    setDynamicStages(generateDynamicStages());
  }, []); // Runs once on mount

  const pipelineStages = stages || dynamicStages;
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

