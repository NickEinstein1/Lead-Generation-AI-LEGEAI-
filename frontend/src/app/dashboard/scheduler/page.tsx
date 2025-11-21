'use client';

import { useState, useEffect } from 'react';
import Link from 'next/link';

interface Meeting {
  id: string;
  title: string;
  description?: string;
  platform: string;
  platform_name: string;
  platform_icon: string;
  start_time: string;
  end_time: string;
  duration_minutes: number;
  attendees: string[];
  organizer_email: string;
  location?: string;
  meeting_url?: string;
  status: string;
  customer_name?: string;
}

interface Platform {
  id: string;
  name: string;
  icon: string;
  requires_auth: boolean;
  type: string;
}

export default function SchedulerPage() {
  const [meetings, setMeetings] = useState<Meeting[]>([]);
  const [platforms, setPlatforms] = useState<Platform[]>([]);
  const [loading, setLoading] = useState(true);
  const [view, setView] = useState<'list' | 'calendar' | 'today'>('list');
  const [showCreateModal, setShowCreateModal] = useState(false);
  const [filterPlatform, setFilterPlatform] = useState<string>('');
  const [filterStatus, setFilterStatus] = useState<string>('');
  const [todaysMeetings, setTodaysMeetings] = useState<Meeting[]>([]);
  const [statistics, setStatistics] = useState<any>(null);

  useEffect(() => {
    fetchMeetings();
    fetchPlatforms();
    fetchTodaysMeetings();
    fetchStatistics();
  }, [filterPlatform, filterStatus]);

  const fetchMeetings = async () => {
    try {
      const params = new URLSearchParams();
      if (filterPlatform) params.append('platform', filterPlatform);
      if (filterStatus) params.append('status', filterStatus);
      
      const response = await fetch(`http://localhost:8000/v1/scheduler?${params}`);
      const data = await response.json();
      setMeetings(data);
    } catch (error) {
      console.error('Error fetching meetings:', error);
    } finally {
      setLoading(false);
    }
  };

  const fetchPlatforms = async () => {
    try {
      const response = await fetch('http://localhost:8000/v1/scheduler/platforms/list');
      const data = await response.json();
      setPlatforms(data.platforms);
    } catch (error) {
      console.error('Error fetching platforms:', error);
    }
  };

  const fetchTodaysMeetings = async () => {
    try {
      const response = await fetch('http://localhost:8000/v1/scheduler/upcoming/today');
      const data = await response.json();
      setTodaysMeetings(data.meetings || []);
    } catch (error) {
      console.error('Error fetching today\'s meetings:', error);
    }
  };

  const fetchStatistics = async () => {
    try {
      const response = await fetch('http://localhost:8000/v1/scheduler/statistics/summary');
      const data = await response.json();
      setStatistics(data);
    } catch (error) {
      console.error('Error fetching statistics:', error);
    }
  };

  const formatDateTime = (isoString: string) => {
    const date = new Date(isoString);
    return date.toLocaleString('en-US', {
      month: 'short',
      day: 'numeric',
      hour: 'numeric',
      minute: '2-digit',
      hour12: true
    });
  };

  const formatDate = (isoString: string) => {
    const date = new Date(isoString);
    return date.toLocaleDateString('en-US', {
      weekday: 'short',
      month: 'short',
      day: 'numeric',
      year: 'numeric'
    });
  };

  const formatTime = (isoString: string) => {
    const date = new Date(isoString);
    return date.toLocaleTimeString('en-US', {
      hour: 'numeric',
      minute: '2-digit',
      hour12: true
    });
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'scheduled': return 'bg-blue-100 text-blue-800';
      case 'completed': return 'bg-green-100 text-green-800';
      case 'cancelled': return 'bg-red-100 text-red-800';
      case 'rescheduled': return 'bg-yellow-100 text-yellow-800';
      default: return 'bg-gray-100 text-gray-800';
    }
  };

  const isUpcoming = (startTime: string) => {
    return new Date(startTime) > new Date();
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center min-h-screen">
        <div className="text-center">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600 mx-auto"></div>
          <p className="mt-4 text-gray-600">Loading scheduler...</p>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 via-white to-purple-50 p-6">
      {/* Header */}
      <div className="max-w-7xl mx-auto mb-8">
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-4xl font-bold text-gray-900 mb-2">📅 Meeting Scheduler</h1>
            <p className="text-gray-600">Schedule meetings across Zoom, Google Meet, Teams, and more</p>
          </div>
          <div className="flex gap-3">
            <Link
              href="/dashboard"
              className="px-4 py-2 bg-gray-100 text-gray-700 rounded-lg hover:bg-gray-200 transition-colors"
            >
              ← Back to Dashboard
            </Link>
            <button
              onClick={() => setShowCreateModal(true)}
              className="px-6 py-2 bg-gradient-to-r from-blue-600 to-purple-600 text-white rounded-lg hover:from-blue-700 hover:to-purple-700 transition-all shadow-lg"
            >
              + Schedule Meeting
            </button>
          </div>
        </div>
      </div>

      {/* Statistics Cards */}
      {statistics && (
        <div className="max-w-7xl mx-auto mb-8 grid grid-cols-1 md:grid-cols-4 gap-6">
          <div className="bg-white rounded-xl shadow-lg p-6 border-l-4 border-blue-500">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm text-gray-600 mb-1">Total Meetings</p>
                <p className="text-3xl font-bold text-gray-900">{statistics.total_meetings}</p>
              </div>
              <div className="text-4xl">📊</div>
            </div>
          </div>

          <div className="bg-white rounded-xl shadow-lg p-6 border-l-4 border-green-500">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm text-gray-600 mb-1">Scheduled</p>
                <p className="text-3xl font-bold text-gray-900">{statistics.by_status?.scheduled || 0}</p>
              </div>
              <div className="text-4xl">✅</div>
            </div>
          </div>

          <div className="bg-white rounded-xl shadow-lg p-6 border-l-4 border-purple-500">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm text-gray-600 mb-1">Avg Duration</p>
                <p className="text-3xl font-bold text-gray-900">{statistics.average_duration_minutes}m</p>
              </div>
              <div className="text-4xl">⏱️</div>
            </div>
          </div>

          <div className="bg-white rounded-xl shadow-lg p-6 border-l-4 border-orange-500">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm text-gray-600 mb-1">Today</p>
                <p className="text-3xl font-bold text-gray-900">{todaysMeetings.length}</p>
              </div>
              <div className="text-4xl">📅</div>
            </div>
          </div>
        </div>
      )}

      {/* View Tabs */}
      <div className="max-w-7xl mx-auto mb-6">
        <div className="bg-white rounded-xl shadow-lg p-2 inline-flex gap-2">
          <button
            onClick={() => setView('today')}
            className={`px-6 py-2 rounded-lg transition-all ${
              view === 'today'
                ? 'bg-blue-600 text-white shadow-md'
                : 'text-gray-600 hover:bg-gray-100'
            }`}
          >
            📅 Today
          </button>
          <button
            onClick={() => setView('list')}
            className={`px-6 py-2 rounded-lg transition-all ${
              view === 'list'
                ? 'bg-blue-600 text-white shadow-md'
                : 'text-gray-600 hover:bg-gray-100'
            }`}
          >
            📋 All Meetings
          </button>
          <button
            onClick={() => setView('calendar')}
            className={`px-6 py-2 rounded-lg transition-all ${
              view === 'calendar'
                ? 'bg-blue-600 text-white shadow-md'
                : 'text-gray-600 hover:bg-gray-100'
            }`}
          >
            📆 Calendar View
          </button>
        </div>
      </div>

      {/* Filters */}
      <div className="max-w-7xl mx-auto mb-6">
        <div className="bg-white rounded-xl shadow-lg p-6">
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">Filter by Platform</label>
              <select
                value={filterPlatform}
                onChange={(e) => setFilterPlatform(e.target.value)}
                className="w-full px-4 py-2 border-2 border-gray-200 rounded-lg focus:outline-none focus:border-blue-500"
              >
                <option value="">All Platforms</option>
                {platforms.map((platform) => (
                  <option key={platform.id} value={platform.id}>
                    {platform.icon} {platform.name}
                  </option>
                ))}
              </select>
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">Filter by Status</label>
              <select
                value={filterStatus}
                onChange={(e) => setFilterStatus(e.target.value)}
                className="w-full px-4 py-2 border-2 border-gray-200 rounded-lg focus:outline-none focus:border-blue-500"
              >
                <option value="">All Statuses</option>
                <option value="scheduled">Scheduled</option>
                <option value="completed">Completed</option>
                <option value="cancelled">Cancelled</option>
                <option value="rescheduled">Rescheduled</option>
              </select>
            </div>

            <div className="flex items-end">
              <button
                onClick={() => {
                  setFilterPlatform('');
                  setFilterStatus('');
                }}
                className="w-full px-4 py-2 bg-gray-100 text-gray-700 rounded-lg hover:bg-gray-200 transition-colors"
              >
                Clear Filters
              </button>
            </div>
          </div>
        </div>
      </div>

      {/* Today's Meetings View */}
      {view === 'today' && (
        <div className="max-w-7xl mx-auto">
          <div className="bg-white rounded-xl shadow-lg p-6">
            <h2 className="text-2xl font-bold text-gray-900 mb-6">
              Today's Schedule - {new Date().toLocaleDateString('en-US', { weekday: 'long', month: 'long', day: 'numeric' })}
            </h2>

            {todaysMeetings.length === 0 ? (
              <div className="text-center py-12">
                <div className="text-6xl mb-4">📅</div>
                <p className="text-xl text-gray-600 mb-2">No meetings scheduled for today</p>
                <p className="text-gray-500">Enjoy your free day!</p>
              </div>
            ) : (
              <div className="space-y-4">
                {todaysMeetings.map((meeting) => (
                  <div
                    key={meeting.id}
                    className="border-2 border-gray-200 rounded-xl p-6 hover:border-blue-400 transition-all hover:shadow-lg"
                  >
                    <div className="flex items-start justify-between">
                      <div className="flex-1">
                        <div className="flex items-center gap-3 mb-2">
                          <span className="text-3xl">{meeting.platform_icon}</span>
                          <div>
                            <h3 className="text-xl font-bold text-gray-900">{meeting.title}</h3>
                            <p className="text-sm text-gray-600">{meeting.platform_name}</p>
                          </div>
                        </div>

                        <div className="grid grid-cols-2 gap-4 mt-4">
                          <div className="flex items-center gap-2 text-gray-700">
                            <span>🕐</span>
                            <span>{formatTime(meeting.start_time)} - {formatTime(meeting.end_time)}</span>
                            <span className="text-sm text-gray-500">({meeting.duration_minutes} min)</span>
                          </div>

                          <div className="flex items-center gap-2 text-gray-700">
                            <span>👥</span>
                            <span>{meeting.attendees.length} attendee{meeting.attendees.length !== 1 ? 's' : ''}</span>
                          </div>
                        </div>

                        {meeting.description && (
                          <p className="text-gray-600 mt-3">{meeting.description}</p>
                        )}

                        {meeting.meeting_url && (
                          <div className="mt-4">
                            <a
                              href={meeting.meeting_url}
                              target="_blank"
                              rel="noopener noreferrer"
                              className="inline-flex items-center gap-2 px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors"
                            >
                              🔗 Join Meeting
                            </a>
                          </div>
                        )}
                      </div>

                      <div className="flex flex-col items-end gap-2">
                        <span className={`px-3 py-1 rounded-full text-sm font-medium ${getStatusColor(meeting.status)}`}>
                          {meeting.status}
                        </span>
                        {isUpcoming(meeting.start_time) && (
                          <span className="px-3 py-1 bg-green-100 text-green-800 rounded-full text-sm font-medium">
                            Upcoming
                          </span>
                        )}
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>
        </div>
      )}

      {/* All Meetings List View */}
      {view === 'list' && (
        <div className="max-w-7xl mx-auto">
          <div className="bg-white rounded-xl shadow-lg overflow-hidden">
            <div className="overflow-x-auto">
              <table className="min-w-full divide-y divide-gray-200">
                <thead className="bg-gray-50">
                  <tr>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                      Meeting
                    </th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                      Platform
                    </th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                      Date & Time
                    </th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                      Duration
                    </th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                      Attendees
                    </th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                      Status
                    </th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                      Actions
                    </th>
                  </tr>
                </thead>
                <tbody className="bg-white divide-y divide-gray-200">
                  {meetings.length === 0 ? (
                    <tr>
                      <td colSpan={7} className="px-6 py-12 text-center">
                        <div className="text-6xl mb-4">📅</div>
                        <p className="text-xl text-gray-600 mb-2">No meetings found</p>
                        <p className="text-gray-500">Schedule your first meeting to get started</p>
                      </td>
                    </tr>
                  ) : (
                    meetings.map((meeting) => (
                      <tr key={meeting.id} className="hover:bg-gray-50 transition-colors">
                        <td className="px-6 py-4">
                          <div>
                            <div className="font-medium text-gray-900">{meeting.title}</div>
                            {meeting.customer_name && (
                              <div className="text-sm text-gray-500">Customer: {meeting.customer_name}</div>
                            )}
                          </div>
                        </td>
                        <td className="px-6 py-4">
                          <div className="flex items-center gap-2">
                            <span className="text-2xl">{meeting.platform_icon}</span>
                            <span className="text-sm text-gray-700">{meeting.platform_name}</span>
                          </div>
                        </td>
                        <td className="px-6 py-4">
                          <div className="text-sm text-gray-900">{formatDate(meeting.start_time)}</div>
                          <div className="text-sm text-gray-500">
                            {formatTime(meeting.start_time)} - {formatTime(meeting.end_time)}
                          </div>
                        </td>
                        <td className="px-6 py-4 text-sm text-gray-700">
                          {meeting.duration_minutes} min
                        </td>
                        <td className="px-6 py-4 text-sm text-gray-700">
                          {meeting.attendees.length}
                        </td>
                        <td className="px-6 py-4">
                          <span className={`px-3 py-1 rounded-full text-xs font-medium ${getStatusColor(meeting.status)}`}>
                            {meeting.status}
                          </span>
                        </td>
                        <td className="px-6 py-4">
                          <div className="flex items-center gap-2">
                            {meeting.meeting_url && (
                              <a
                                href={meeting.meeting_url}
                                target="_blank"
                                rel="noopener noreferrer"
                                className="text-blue-600 hover:text-blue-800 transition-colors"
                                title="Join Meeting"
                              >
                                🔗
                              </a>
                            )}
                            <Link
                              href={`/dashboard/scheduler/${meeting.id}`}
                              className="text-gray-600 hover:text-gray-800 transition-colors"
                              title="View Details"
                            >
                              👁️
                            </Link>
                          </div>
                        </td>
                      </tr>
                    ))
                  )}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      )}

      {/* Calendar View */}
      {view === 'calendar' && (
        <div className="max-w-7xl mx-auto">
          <div className="bg-white rounded-xl shadow-lg p-6">
            <div className="text-center py-12">
              <div className="text-6xl mb-4">📆</div>
              <h3 className="text-2xl font-bold text-gray-900 mb-2">Calendar View</h3>
              <p className="text-gray-600 mb-6">Interactive calendar view coming soon!</p>
              <p className="text-sm text-gray-500">
                This will display meetings in a monthly calendar grid with day/week/month views
              </p>
            </div>
          </div>
        </div>
      )}

      {/* Create Meeting Modal */}
      {showCreateModal && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50 p-4">
          <div className="bg-white rounded-xl shadow-2xl max-w-2xl w-full max-h-[90vh] overflow-y-auto">
            <div className="p-6 border-b border-gray-200">
              <div className="flex items-center justify-between">
                <h2 className="text-2xl font-bold text-gray-900">Schedule New Meeting</h2>
                <button
                  onClick={() => setShowCreateModal(false)}
                  className="text-gray-400 hover:text-gray-600 text-2xl"
                >
                  ×
                </button>
              </div>
            </div>

            <div className="p-6">
              <p className="text-center text-gray-600 py-8">
                Create meeting form will be implemented in the next step
              </p>
              <div className="text-center">
                <Link
                  href="/dashboard/scheduler/create"
                  className="inline-block px-6 py-3 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors"
                  onClick={() => setShowCreateModal(false)}
                >
                  Go to Create Meeting Page
                </Link>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

