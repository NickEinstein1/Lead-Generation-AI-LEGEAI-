"use client";
import { useState, useEffect } from "react";
import DashboardLayout from "@/components/DashboardLayout";

interface FileDocument {
  id: number;
  filename: string;
  original_filename: string;
  file_type: string;
  mime_type: string;
  file_size: number;
  title: string;
  description: string | null;
  category: string;
  tags: Record<string, any> | null;
  lead_id: number | null;
  uploaded_by: number | null;
  status: string;
  version: number;
  is_public: boolean;
  access_level: string;
  created_at: string;
  updated_at: string;
  last_accessed_at: string | null;
}

interface DocumentCategory {
  id: number;
  name: string;
  display_name: string;
  description: string | null;
  icon: string | null;
  color: string | null;
  sort_order: number;
  is_active: boolean;
}

interface DocumentStats {
  total_documents: number;
  total_size: number;
  by_category: Record<string, { count: number; size: number }>;
  by_type: Record<string, number>;
  recent_uploads: number;
}

export default function FileLibraryPage() {
  const [documents, setDocuments] = useState<FileDocument[]>([]);
  const [categories, setCategories] = useState<DocumentCategory[]>([]);
  const [stats, setStats] = useState<DocumentStats | null>(null);
  const [selectedCategory, setSelectedCategory] = useState<string | null>(null);
  const [searchQuery, setSearchQuery] = useState("");
  const [isUploading, setIsUploading] = useState(false);
  const [showUploadModal, setShowUploadModal] = useState(false);
  const [uploadProgress, setUploadProgress] = useState(0);

  const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://127.0.0.1:8000";

  useEffect(() => {
    fetchCategories();
    fetchStats();
    fetchDocuments();
  }, [selectedCategory, searchQuery]);

  const fetchCategories = async () => {
    try {
      const response = await fetch(`${API_BASE}/v1/file-management/categories`);
      if (response.ok) {
        const data = await response.json();
        setCategories(data);
      }
    } catch (error) {
      console.error("Failed to fetch categories:", error);
    }
  };

  const fetchStats = async () => {
    try {
      const response = await fetch(`${API_BASE}/v1/file-management/stats`);
      if (response.ok) {
        const data = await response.json();
        setStats(data);
      }
    } catch (error) {
      console.error("Failed to fetch stats:", error);
    }
  };

  const fetchDocuments = async () => {
    try {
      const params = new URLSearchParams();
      if (selectedCategory) params.append("category", selectedCategory);
      if (searchQuery) params.append("search", searchQuery);
      params.append("page", "1");
      params.append("page_size", "50");

      const response = await fetch(`${API_BASE}/v1/file-management/documents?${params}`);
      if (response.ok) {
        const data = await response.json();
        setDocuments(data.items || []);
      }
    } catch (error) {
      console.error("Failed to fetch documents:", error);
    }
  };

  const handleFileUpload = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const files = event.target.files;
    if (!files || files.length === 0) return;

    const file = files[0];
    setIsUploading(true);
    setUploadProgress(0);

    const formData = new FormData();
    formData.append("file", file);
    formData.append("title", file.name);
    formData.append("category", selectedCategory || "other");
    formData.append("access_level", "private");

    try {
      const response = await fetch(`${API_BASE}/v1/file-management/upload`, {
        method: "POST",
        body: formData,
      });

      if (response.ok) {
        setUploadProgress(100);
        setTimeout(() => {
          setIsUploading(false);
          setShowUploadModal(false);
          fetchDocuments();
          fetchStats();
        }, 500);
      } else {
        alert("Upload failed");
        setIsUploading(false);
      }
    } catch (error) {
      console.error("Upload error:", error);
      alert("Upload failed");
      setIsUploading(false);
    }
  };

  const formatFileSize = (bytes: number): string => {
    if (bytes === 0) return "0 B";
    const k = 1024;
    const sizes = ["B", "KB", "MB", "GB"];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return Math.round(bytes / Math.pow(k, i) * 100) / 100 + " " + sizes[i];
  };

  const getFileIcon = (fileType: string): string => {
    const icons: Record<string, string> = {
      pdf: "📄",
      doc: "📝",
      docx: "📝",
      xls: "📊",
      xlsx: "📊",
      csv: "📊",
      txt: "📃",
      png: "🖼️",
      jpg: "🖼️",
      jpeg: "🖼️",
    };
    return icons[fileType] || "📎";
  };

  return (
    <DashboardLayout>
      <div className="space-y-6">
        {/* Header */}
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-3xl font-bold text-slate-900">File Library</h1>
            <p className="text-slate-600 font-medium mt-1">
              Manage documents, spreadsheets, and files
            </p>
          </div>
          <button
            onClick={() => setShowUploadModal(true)}
            className="bg-blue-600 hover:bg-blue-700 text-white font-semibold py-2 px-6 rounded-lg transition-all shadow-md hover:shadow-lg"
          >
            📤 Upload File
          </button>
        </div>

        {/* Stats Cards */}
        {stats && (
          <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
            <div className="bg-white border-2 border-blue-200 rounded-lg p-4 shadow-md">
              <p className="text-slate-600 text-sm font-medium">Total Files</p>
              <p className="text-3xl font-bold text-blue-700 mt-2">
                {stats.total_documents.toLocaleString()}
              </p>
              <p className="text-xs text-slate-600 font-medium mt-2">
                {formatFileSize(stats.total_size)} total
              </p>
            </div>
            <div className="bg-white border-2 border-blue-200 rounded-lg p-4 shadow-md">
              <p className="text-slate-600 text-sm font-medium">Recent Uploads</p>
              <p className="text-3xl font-bold text-emerald-600 mt-2">
                {stats.recent_uploads}
              </p>
              <p className="text-xs text-slate-600 font-medium mt-2">Last 7 days</p>
            </div>
            <div className="bg-white border-2 border-blue-200 rounded-lg p-4 shadow-md">
              <p className="text-slate-600 text-sm font-medium">Categories</p>
              <p className="text-3xl font-bold text-purple-600 mt-2">
                {Object.keys(stats.by_category).length}
              </p>
              <p className="text-xs text-slate-600 font-medium mt-2">Active categories</p>
            </div>
            <div className="bg-white border-2 border-blue-200 rounded-lg p-4 shadow-md">
              <p className="text-slate-600 text-sm font-medium">File Types</p>
              <p className="text-3xl font-bold text-amber-600 mt-2">
                {Object.keys(stats.by_type).length}
              </p>
              <p className="text-xs text-slate-600 font-medium mt-2">Different formats</p>
            </div>
          </div>
        )}

        {/* Categories Filter */}
        <div className="bg-white border-2 border-blue-200 rounded-lg p-4 shadow-md">
          <h2 className="text-lg font-bold text-slate-900 mb-3">Categories</h2>
          <div className="flex flex-wrap gap-2">
            <button
              onClick={() => setSelectedCategory(null)}
              className={`px-4 py-2 rounded-lg font-medium transition-all ${
                selectedCategory === null
                  ? "bg-blue-600 text-white"
                  : "bg-slate-100 text-slate-700 hover:bg-slate-200"
              }`}
            >
              All Files
            </button>
            {categories.map((cat) => (
              <button
                key={cat.id}
                onClick={() => setSelectedCategory(cat.name)}
                className={`px-4 py-2 rounded-lg font-medium transition-all ${
                  selectedCategory === cat.name
                    ? "bg-blue-600 text-white"
                    : "bg-slate-100 text-slate-700 hover:bg-slate-200"
                }`}
                style={{
                  backgroundColor:
                    selectedCategory === cat.name ? cat.color || undefined : undefined,
                }}
              >
                {cat.icon} {cat.display_name}
              </button>
            ))}
          </div>
        </div>

        {/* Search Bar */}
        <div className="bg-white border-2 border-blue-200 rounded-lg p-4 shadow-md">
          <input
            type="text"
            placeholder="🔍 Search files by name, description..."
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            className="w-full px-4 py-2 border border-slate-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
          />
        </div>

        {/* Documents Table */}
        <div className="bg-white border-2 border-blue-200 rounded-lg shadow-md overflow-hidden">
          <div className="p-6 border-b border-blue-200">
            <h2 className="text-xl font-bold text-slate-900">Files</h2>
          </div>
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead className="bg-blue-50 border-b-2 border-blue-200">
                <tr>
                  <th className="text-left p-4 text-slate-900 font-bold">File</th>
                  <th className="text-left p-4 text-slate-900 font-bold">Title</th>
                  <th className="text-left p-4 text-slate-900 font-bold">Category</th>
                  <th className="text-left p-4 text-slate-900 font-bold">Size</th>
                  <th className="text-left p-4 text-slate-900 font-bold">Uploaded</th>
                  <th className="text-left p-4 text-slate-900 font-bold">Actions</th>
                </tr>
              </thead>
              <tbody>
                {documents.length === 0 ? (
                  <tr>
                    <td colSpan={6} className="p-8 text-center text-slate-500">
                      No files found. Upload your first file to get started!
                    </td>
                  </tr>
                ) : (
                  documents.map((doc) => (
                    <tr
                      key={doc.id}
                      className="border-t border-blue-100 hover:bg-blue-50 transition"
                    >
                      <td className="p-4 font-medium text-slate-900">
                        <div className="flex items-center gap-2">
                          <span className="text-2xl">{getFileIcon(doc.file_type)}</span>
                          <span className="text-xs text-slate-500 uppercase">
                            {doc.file_type}
                          </span>
                        </div>
                      </td>
                      <td className="p-4">
                        <div>
                          <p className="font-medium text-slate-900">{doc.title}</p>
                          <p className="text-xs text-slate-500">{doc.original_filename}</p>
                        </div>
                      </td>
                      <td className="p-4 text-slate-700">
                        <span className="px-2 py-1 bg-slate-100 rounded text-xs font-medium">
                          {doc.category}
                        </span>
                      </td>
                      <td className="p-4 text-slate-700">{formatFileSize(doc.file_size)}</td>
                      <td className="p-4 text-slate-700">
                        {new Date(doc.created_at).toLocaleDateString()}
                      </td>
                      <td className="p-4">
                        <div className="flex gap-2">
                          <a
                            href={`${API_BASE}/v1/file-management/documents/${doc.id}/download`}
                            className="text-blue-600 hover:text-blue-800 font-medium text-sm"
                            download
                          >
                            Download
                          </a>
                        </div>
                      </td>
                    </tr>
                  ))
                )}
              </tbody>
            </table>
          </div>
        </div>

        {/* Upload Modal */}
        {showUploadModal && (
          <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
            <div className="bg-white rounded-lg p-6 max-w-md w-full mx-4">
              <h3 className="text-xl font-bold text-slate-900 mb-4">Upload File</h3>
              <div className="space-y-4">
                <div>
                  <label className="block text-sm font-medium text-slate-700 mb-2">
                    Select File
                  </label>
                  <input
                    type="file"
                    onChange={handleFileUpload}
                    disabled={isUploading}
                    className="w-full px-3 py-2 border border-slate-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
                    accept=".pdf,.doc,.docx,.xls,.xlsx,.csv,.txt,.png,.jpg,.jpeg"
                  />
                  <p className="text-xs text-slate-500 mt-1">
                    Supported: PDF, Word, Excel, CSV, Images (Max 50MB)
                  </p>
                </div>

                {isUploading && (
                  <div className="space-y-2">
                    <div className="w-full bg-slate-200 rounded-full h-2">
                      <div
                        className="bg-blue-600 h-2 rounded-full transition-all"
                        style={{ width: `${uploadProgress}%` }}
                      ></div>
                    </div>
                    <p className="text-sm text-slate-600 text-center">
                      Uploading... {uploadProgress}%
                    </p>
                  </div>
                )}

                <div className="flex gap-2 justify-end">
                  <button
                    onClick={() => setShowUploadModal(false)}
                    disabled={isUploading}
                    className="px-4 py-2 bg-slate-200 text-slate-700 rounded-lg hover:bg-slate-300 disabled:opacity-50"
                  >
                    Cancel
                  </button>
                </div>
              </div>
            </div>
          </div>
        )}
      </div>
    </DashboardLayout>
  );
}

