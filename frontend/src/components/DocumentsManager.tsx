"use client";
import React from "react";

interface Document {
  id: number;
  title: string;
  status: "pending" | "signed" | "declined";
  created_at?: string;
  signed_at?: string;
  signing_url?: string;
  provider?: string;
}

interface DocumentsManagerProps {
  documents: Document[];
  onCreateDoc?: () => void;
  onOpenSign?: (url: string) => void;
  onSimulateSign?: (docId: number) => void;
  creating?: boolean;
}

export default function DocumentsManager({
  documents,
  onCreateDoc,
  onOpenSign,
  onSimulateSign,
  creating = false,
}: DocumentsManagerProps) {
  const getStatusIcon = (status: string) => {
    switch (status) {
      case "signed":
        return "✅";
      case "declined":
        return "❌";
      case "pending":
        return "⏳";
      default:
        return "📄";
    }
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case "signed":
        return "bg-green-100 text-green-800";
      case "declined":
        return "bg-red-100 text-red-800";
      case "pending":
        return "bg-amber-100 text-amber-800";
      default:
        return "bg-slate-100 text-slate-800";
    }
  };

  return (
    <div className="bg-white border-2 border-blue-200 rounded-lg p-6 shadow-md">
      <div className="flex items-center justify-between mb-6">
        <h2 className="text-xl font-bold text-slate-900">Documents</h2>
        <button
          onClick={onCreateDoc}
          disabled={creating}
          className="bg-blue-600 hover:bg-blue-700 text-white font-semibold py-2 px-4 rounded-lg transition-all disabled:opacity-50 text-sm"
        >
          {creating ? "Creating..." : "+ Create Document"}
        </button>
      </div>

      {documents.length === 0 ? (
        <div className="text-center py-8">
          <p className="text-slate-600 font-medium mb-4">No documents yet</p>
          <button
            onClick={onCreateDoc}
            disabled={creating}
            className="bg-blue-600 hover:bg-blue-700 text-white font-semibold py-2 px-6 rounded-lg transition-all disabled:opacity-50"
          >
            Create First Document
          </button>
        </div>
      ) : (
        <div className="space-y-3">
          {documents.map((doc) => (
            <div
              key={doc.id}
              className="border-2 border-blue-100 rounded-lg p-4 hover:bg-blue-50 transition-colors"
            >
              <div className="flex items-start justify-between gap-4">
                <div className="flex-1">
                  <div className="flex items-center gap-2 mb-2">
                    <span className="text-xl">{getStatusIcon(doc.status)}</span>
                    <h3 className="font-bold text-slate-900">{doc.title}</h3>
                    <span className={`px-2 py-1 rounded text-xs font-semibold ${getStatusColor(doc.status)}`}>
                      {doc.status}
                    </span>
                  </div>
                  <div className="text-xs text-slate-600 space-y-1">
                    {doc.created_at && (
                      <p>Created: {new Date(doc.created_at).toLocaleString()}</p>
                    )}
                    {doc.signed_at && (
                      <p className="text-green-700 font-medium">
                        Signed: {new Date(doc.signed_at).toLocaleString()}
                      </p>
                    )}
                  </div>
                </div>

                <div className="flex gap-2">
                  {doc.status !== "signed" && doc.signing_url && (
                    <button
                      onClick={() => onOpenSign?.(doc.signing_url!)}
                      className="bg-blue-600 hover:bg-blue-700 text-white font-semibold py-1 px-3 rounded text-sm transition-all"
                    >
                      Sign
                    </button>
                  )}
                  {doc.provider === "internal" && doc.status !== "signed" && (
                    <button
                      onClick={() => onSimulateSign?.(doc.id)}
                      className="bg-slate-600 hover:bg-slate-700 text-white font-semibold py-1 px-3 rounded text-sm transition-all"
                    >
                      Simulate
                    </button>
                  )}
                </div>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

