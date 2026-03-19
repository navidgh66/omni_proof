# Security Policy

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| 0.0.x   | :white_check_mark: |

## Reporting a Vulnerability

If you discover a security vulnerability in OmniProof, please report it responsibly.

**Do NOT open a public GitHub issue for security vulnerabilities.**

### How to Report

1. **GitHub Private Advisory** (preferred): Go to the [Security Advisories](https://github.com/navidgh66/omni_proof/security/advisories) page and click "Report a vulnerability".
2. **Email**: Send details to **navidgh66@gmail.com** with the subject line `[SECURITY] OmniProof vulnerability report`.

### What to Include

- Description of the vulnerability
- Steps to reproduce
- Potential impact
- Suggested fix (if any)

### Response Timeline

- **Acknowledgment**: Within 48 hours
- **Initial assessment**: Within 1 week
- **Fix or mitigation**: Targeting 30 days for critical issues

### Disclosure Policy

- We follow coordinated disclosure. Please allow us reasonable time to address the issue before public disclosure.
- Credit will be given to reporters in the release notes (unless anonymity is requested).

## Security Hardening

OmniProof includes the following security measures:

- **CORS**: Restricted to localhost origins; methods limited to GET/POST
- **Path traversal protection**: `field_validator` rejects `..` and unsafe characters in asset paths
- **Filename sanitization**: Upload filenames are stripped of path components and unsafe characters
- **Input validation**: All API request models use `Field(min_length, max_length, pattern)` constraints
- **Exception handling**: Global handler prevents stack traces from leaking to clients
- **Causal route validation**: Treatment/outcome names validated with `^[a-zA-Z_][a-zA-Z0-9_]*$`
