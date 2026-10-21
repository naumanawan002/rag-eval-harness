# NIST SP 800-207 Zero Trust Architecture

Zero Trust Architecture (ZTA) is an enterprise cybersecurity strategy based on the fundamental tenet that no user or asset is inherently trusted, regardless of physical or network location.

Traditional perimeter-based network security models depend on firewalls and VPN tunnels. Once an attacker breaches the perimeter, lateral movement is unrestricted. In contrast, Zero Trust assumes the network is perpetually hostile and demands continuous verification.

The fundamental components of a Zero Trust Architecture comprise:
1. Policy Enforcement Point (PEP): The gateway or proxy that terminates client connections, enforces authentication, and intercepts incoming requests at the network edge.
2. Policy Decision Point (PDP): The central decision engine that computes access grants by evaluating subjects, target resources, device posture, and environmental signals.
3. Policy Administration Point (PAP): The control plane responsible for managing access control policies, rule definitions, and identity attributes.

Access decisions are strictly context-aware and dynamic. Short-lived cryptographic tokens (JWTs) and mutual TLS (mTLS) provide end-to-end encryption and prevent credential replay attacks across distributed internal microservices.
