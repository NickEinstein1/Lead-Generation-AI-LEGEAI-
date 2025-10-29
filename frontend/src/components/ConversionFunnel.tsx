"use client";
import React from "react";

interface FunnelStage {
  name: string;
  count: number;
  conversionRate: number;
}

interface ConversionFunnelProps {
  stages?: FunnelStage[];
}

export default function ConversionFunnel({ stages }: ConversionFunnelProps) {
  const defaultStages: FunnelStage[] = [
    { name: "Leads", count: 500, conversionRate: 100 },
    { name: "Contacted", count: 380, conversionRate: 76 },
    { name: "Qualified", count: 285, conversionRate: 57 },
    { name: "Proposal", count: 156, conversionRate: 31.2 },
    { name: "Closed", count: 45, conversionRate: 9 },
  ];

  const funnelStages = stages || defaultStages;
  const maxCount = Math.max(...funnelStages.map((s) => s.count));

  return (
    <div className="bg-white border-2 border-blue-200 rounded-lg p-6 shadow-md">
      <h2 className="text-xl font-bold text-slate-900 mb-6">Conversion Funnel</h2>

      <div className="space-y-4">
        {funnelStages.map((stage, idx) => {
          const width = (stage.count / maxCount) * 100;
          const colors = [
            "from-blue-500 to-blue-600",
            "from-cyan-500 to-cyan-600",
            "from-emerald-500 to-emerald-600",
            "from-amber-500 to-amber-600",
            "from-green-500 to-green-600",
          ];

          return (
            <div key={idx} className="space-y-2">
              <div className="flex items-center justify-between mb-2">
                <span className="text-sm font-semibold text-slate-900">{stage.name}</span>
                <div className="flex gap-4 text-sm">
                  <span className="font-bold text-slate-900">{stage.count}</span>
                  <span className="text-slate-600 font-medium">{stage.conversionRate.toFixed(1)}%</span>
                </div>
              </div>
              <div className="w-full bg-slate-200 rounded-full h-8 overflow-hidden">
                <div
                  className={`h-full bg-gradient-to-r ${colors[idx % colors.length]} flex items-center justify-end pr-3 transition-all duration-300`}
                  style={{ width: `${width}%` }}
                >
                  {width > 15 && (
                    <span className="text-white text-xs font-bold">{stage.count}</span>
                  )}
                </div>
              </div>
            </div>
          );
        })}
      </div>

      <div className="mt-6 pt-6 border-t-2 border-blue-100">
        <div className="grid grid-cols-3 gap-4">
          <div className="bg-blue-50 rounded p-3 text-center">
            <p className="text-xs text-slate-600 font-medium">Total Leads</p>
            <p className="text-2xl font-bold text-blue-700">{funnelStages[0].count}</p>
          </div>
          <div className="bg-amber-50 rounded p-3 text-center">
            <p className="text-xs text-slate-600 font-medium">Conversion Rate</p>
            <p className="text-2xl font-bold text-amber-700">
              {((funnelStages[funnelStages.length - 1].count / funnelStages[0].count) * 100).toFixed(1)}%
            </p>
          </div>
          <div className="bg-green-50 rounded p-3 text-center">
            <p className="text-xs text-slate-600 font-medium">Closed Deals</p>
            <p className="text-2xl font-bold text-green-700">{funnelStages[funnelStages.length - 1].count}</p>
          </div>
        </div>
      </div>
    </div>
  );
}

