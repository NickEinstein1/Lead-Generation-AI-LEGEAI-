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
  const pipelineStages = stages ?? [];
  const totalLeads = pipelineStages.reduce((sum, s) => sum + s.count, 0);
  const conversionRate = totalLeads > 0
    ? ((pipelineStages[pipelineStages.length - 1]?.count || 0) / totalLeads) * 100
    : 0;

  return (
    <div className="bg-slate-900/60 backdrop-blur-xl border border-blue-500/30 rounded-2xl p-6 shadow-2xl hover:shadow-cyan-500/20 transition-all duration-300 relative overflow-hidden">
      {/* Background glow */}
      <div className="absolute inset-0 bg-gradient-to-br from-blue-500/10 via-orange-500/10 to-cyan-500/10"></div>

      <h2 className="text-xl font-bold text-blue-100 mb-6 flex items-center gap-2 relative z-10">
        <span className="text-2xl">📊</span>
        <span className="bg-gradient-to-r from-cyan-300 to-blue-300 bg-clip-text text-transparent">Lead Pipeline</span>
      </h2>

      {pipelineStages.length === 0 ? (
        <div className="relative z-10 text-sm text-cyan-200 font-medium">
          No pipeline data available.
        </div>
      ) : (
        <div className="space-y-4 relative z-10">
          {pipelineStages.map((stage, idx) => (
            <div key={idx} className="space-y-2 group">
              <div className="flex items-center justify-between">
                <span className="text-sm font-bold text-cyan-300 group-hover:text-cyan-200 transition-colors">{stage.name}</span>
                <span className="text-sm font-bold text-blue-100 bg-blue-500/20 backdrop-blur-sm px-3 py-1 rounded-lg border border-blue-400/30">{stage.count} leads ({stage.percentage}%)</span>
              </div>
              <div className="w-full bg-slate-800/50 backdrop-blur-sm rounded-full h-4 overflow-hidden shadow-inner border border-blue-500/20">
                <div
                  className={`h-full ${stage.color} transition-all duration-500 ease-out shadow-lg relative overflow-hidden`}
                  style={{ width: `${stage.percentage}%` }}
                >
                  {/* Animated shine effect */}
                  <div className="absolute inset-0 bg-gradient-to-r from-transparent via-white/40 to-transparent -translate-x-full animate-shimmer"></div>
                  {/* Glow effect */}
                  <div className="absolute inset-0 shadow-[inset_0_0_10px_rgba(255,255,255,0.3)]"></div>
                </div>
              </div>
            </div>
          ))}
        </div>
      )}

      <div className="mt-6 pt-6 border-t border-blue-500/30 relative z-10">
        <div className="grid grid-cols-2 gap-4">
          <div className="bg-blue-500/20 backdrop-blur-sm rounded-xl p-4 shadow-lg hover:shadow-cyan-500/20 transition-all border border-blue-400/30 hover:border-cyan-400/50 group">
            <div className="text-xs text-cyan-300 font-bold uppercase tracking-wider">Total Leads</div>
            <div className="text-2xl font-bold bg-gradient-to-r from-cyan-300 to-blue-300 bg-clip-text text-transparent mt-1">{totalLeads}</div>
          </div>
          <div className="bg-green-500/20 backdrop-blur-sm rounded-xl p-4 shadow-lg hover:shadow-green-500/20 transition-all border border-green-400/30 hover:border-green-300/50 group">
            <div className="text-xs text-green-300 font-bold uppercase tracking-wider">Conversion Rate</div>
            <div className="text-2xl font-bold bg-gradient-to-r from-green-300 to-emerald-300 bg-clip-text text-transparent mt-1">
              {totalLeads > 0 ? `${conversionRate.toFixed(1)}%` : "-"}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

