# Q3 2025 Operations Report (excerpt)

Fleet uptime for Q3 2025 was **99.92%** across all managed appliances, our
best quarter to date. We expanded the Kestrel Cloud control plane into two
new regions: Frankfurt and Osaka.

Latency benchmarks for the current hardware generation are summarized in the
figure `fig_latency_benchmark.png`; per-variant p95 numbers were collected on
ResNet-50 INT8 at batch size 1.

The quarterly revenue split across product lines is shown in
`fig_revenue_mix.png`. A 24-hour fleet GPU utilization sample, including the
daily peak and trough, is plotted in `fig_gpu_utilization.png`.

The end-to-end streaming pipeline that every appliance runs is diagrammed in
`fig_architecture.png`. Following the firmware 2.4 rollout, stream error
rates by firmware version are compared in `fig_error_rates.png`.
