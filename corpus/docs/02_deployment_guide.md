# VoltEdge Deployment Guide (excerpt)

## Host requirements

The management agent requires Ubuntu 22.04 LTS on the provisioning host.
Provisioning is done over the local network; each appliance exposes its
management API on **port 8443** (HTTPS, mutual TLS).

## Stream limits

Concurrent camera stream limits are enforced in firmware: 4 streams on
VoltEdge Nano, 16 streams on VoltEdge Pro, and 64 streams on VoltEdge Max.
Exceeding the limit returns HTTP 429 from the ingest gateway.

## Updates

Firmware updates are delivered over-the-air on a 6-week cadence. Appliances
download updates from the Kestrel Cloud CDN and apply them during the
configured maintenance window with automatic rollback on failed health checks.
