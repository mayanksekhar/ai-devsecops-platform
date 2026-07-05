# AI DevSecOps Platform

> Thinkwerke · Detect → Prevent → Observe

End-to-end AI security platform covering OWASP LLM Top 10 scanning,
an AI Security Gateway with OPA policy engine, and a full observability
stack with alerting.

**Status:** active build — documentation and architecture in progress.

## Structure

| Layer | Component | Status |
|---|---|---|
| Detect | OWASP LLM Top 10 scanner (pytest) | 🔨 building |
| Prevent | AI Security Gateway (Go + OPA) | 🔨 building |
| Observe | Prometheus + Grafana + Loki | 🔨 building |

## Docs

- [Threat Model](THREAT_MODEL.md)
- [MITRE ATLAS Mapping](docs/mitre-atlas-mapping.md)
- [Architecture Decision Records](docs/)
- [Incident Response](docs/INCIDENT_RESPONSE.md)
