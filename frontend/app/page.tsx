"use client";

import { useState, useEffect } from "react";

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

interface JobStatus {
  job_id: string;
  status: "pending" | "running" | "completed" | "failed";
  command_executed?: string;
  results?: any;
  error?: string;
}

export default function Home() {
  const [formData, setFormData] = useState({
    source_dir: "/data/CULane/train",
    target_dir: "/data/CULane/val",
    target_list_path: "/data/CULane/val/list.txt",
    sample_size: 100,
    num_runs: 5,
    gaussian_sigma: 0.0,
    command: "imagenet_weights",
  });

  const [activeJobId, setActiveJobId] = useState<string | null>(null);
  const [jobData, setJobData] = useState<JobStatus | null>(null);
  const [loading, setLoading] = useState(false);

  // Poll job status every 2 seconds when running
  useEffect(() => {
    if (!activeJobId || jobData?.status === "completed" || jobData?.status === "failed") return;

    const interval = setInterval(async () => {
      try {
        const res = await fetch(`${API_BASE}/api/v1/experiments/${activeJobId}`);
        const data = await res.json();
        setJobData(data);
      } catch (err) {
        console.error("Failed to fetch job status", err);
      }
    }, 2000);

    return () => clearInterval(interval);
  }, [activeJobId, jobData?.status]);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    setJobData(null);

    try {
      const res = await fetch(`${API_BASE}/api/v1/experiments`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(formData),
      });
      const data = await res.json();
      setActiveJobId(data.job_id);
    } catch (err) {
      alert("Error starting experiment");
    } finally {
      setLoading(false);
    }
  };

  return (
    <main className="min-h-screen bg-base-200 p-8">
      <div className="max-w-6xl mx-auto space-y-8">
        {/* Header */}
        <div className="navbar bg-base-100 rounded-box shadow-md px-6">
          <div className="flex-1">
            <h1 className="text-xl font-bold">Distribution Shift Detection Platform</h1>
          </div>
          <div className="badge badge-primary">v1.0.0</div>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
          {/* Experiment Form */}
          <div className="card bg-base-100 shadow-xl">
            <div className="card-body">
              <h2 className="card-title mb-4">Configure Experiment</h2>
              <form onSubmit={handleSubmit} className="space-y-4">
                <div className="form-control">
                  <label className="label"><span className="label-text">Source Directory</span></label>
                  <input
                    type="text"
                    className="input input-bordered w-full"
                    value={formData.source_dir}
                    onChange={(e) => setFormData({ ...formData, source_dir: e.target.value })}
                    required
                  />
                </div>

                <div className="form-control">
                  <label className="label"><span className="label-text">Target Directory</span></label>
                  <input
                    type="text"
                    className="input input-bordered w-full"
                    value={formData.target_dir}
                    onChange={(e) => setFormData({ ...formData, target_dir: e.target.value })}
                    required
                  />
                </div>

                <div className="form-control">
                  <label className="label"><span className="label-text">Target List Path</span></label>
                  <input
                    type="text"
                    className="input input-bordered w-full"
                    value={formData.target_list_path}
                    onChange={(e) => setFormData({ ...formData, target_list_path: e.target.value })}
                    required
                  />
                </div>

                <div className="grid grid-cols-2 gap-4">
                  <div className="form-control">
                    <label className="label"><span className="label-text">Sample Size</span></label>
                    <input
                      type="number"
                      className="input input-bordered"
                      value={formData.sample_size}
                      onChange={(e) => setFormData({ ...formData, sample_size: parseInt(e.target.value) })}
                    />
                  </div>
                  <div className="form-control">
                    <label className="label"><span className="label-text">Num Runs</span></label>
                    <input
                      type="number"
                      className="input input-bordered"
                      value={formData.num_runs}
                      onChange={(e) => setFormData({ ...formData, num_runs: parseInt(e.target.value) })}
                    />
                  </div>
                </div>

                <div className="form-control">
                  <label className="label"><span className="label-text">Gaussian Sigma Perturbation</span></label>
                  <input
                    type="number"
                    step="0.1"
                    className="input input-bordered"
                    value={formData.gaussian_sigma}
                    onChange={(e) => setFormData({ ...formData, gaussian_sigma: parseFloat(e.target.value) })}
                  />
                </div>

                <div className="form-control">
                  <label className="label"><span className="label-text">Weights Type</span></label>
                  <select
                    className="select select-bordered"
                    value={formData.command}
                    onChange={(e) => setFormData({ ...formData, command: e.target.value })}
                  >
                    <option value="imagenet_weights">ImageNet Weights</option>
                    <option value="random_weights">Random Weights</option>
                    <option value="custom_weights">Custom Weights</option>
                  </select>
                </div>

                <button
                  type="submit"
                  className={`btn btn-primary w-full ${loading ? "loading" : ""}`}
                  disabled={loading}
                >
                  Run Shift Test
                </button>
              </form>
            </div>
          </div>

          {/* Job Monitoring & Results */}
          <div className="card bg-base-100 shadow-xl">
            <div className="card-body">
              <h2 className="card-title mb-4">Execution Status</h2>

              {!activeJobId ? (
                <div className="text-center py-12 text-base-content/50">
                  Submit an experiment to view real-time status and output results.
                </div>
              ) : (
                <div className="space-y-6">
                  {/* Status Indicator */}
                  <div className="flex items-center justify-between p-4 bg-base-200 rounded-lg">
                    <div>
                      <p className="text-xs text-base-content/60">Job ID</p>
                      <p className="font-mono text-sm">{activeJobId}</p>
                    </div>
                    <div
                      className={`badge badge-lg ${
                        jobData?.status === "completed"
                          ? "badge-success"
                          : jobData?.status === "failed"
                          ? "badge-error"
                          : "badge-warning animate-pulse"
                      }`}
                    >
                      {jobData?.status || "initializing"}
                    </div>
                  </div>

                  {/* Executed Command */}
                  {jobData?.command_executed && (
                    <div>
                      <span className="text-xs text-base-content/60">Executed Binary Command</span>
                      <pre className="bg-neutral text-neutral-content p-3 rounded-md text-xs overflow-x-auto mt-1">
                        {jobData.command_executed}
                      </pre>
                    </div>
                  )}

                  {/* Output JSON Results */}
                  {jobData?.results && (
                    <div>
                      <span className="text-xs text-base-content/60">Experiment Output</span>
                      <pre className="bg-base-300 p-4 rounded-md text-xs overflow-x-auto max-h-96 mt-1">
                        {JSON.stringify(jobData.results, null, 2)}
                      </pre>
                    </div>
                  )}

                  {/* Error Trace */}
                  {jobData?.error && (
                    <div className="alert alert-error">
                      <span>{jobData.error}</span>
                    </div>
                  )}
                </div>
              )}
            </div>
          </div>
        </div>
      </div>
    </main>
  );
}