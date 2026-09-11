# Security Policy

## Supported versions

This project is pre-1.0 research software. Security fixes are applied on the
`main` branch only.

## Reporting a vulnerability

Do **not** open a public GitHub issue for security reports.

Email the maintainer via the contact listed on the GitHub profile for
[RealTimElito/T1DA](https://github.com/RealTimElito/T1DA), or open a private
[GitHub security advisory](https://docs.github.com/en/code-security/security-advisories/guidance-on-reporting-and-writing/privately-reporting-a-security-vulnerability)
on the repository.

Please include:

- A description of the issue and its impact
- Steps to reproduce, or a proof of concept if available
- Affected commit or release if known

You should receive an acknowledgement within a few days when possible.

## Scope notes

- Do not commit clinical patient data, API tokens, or Hugging Face credentials.
- Dataset download scripts talk to third-party hosts; treat returned archives
  according to each provider's data use agreement.
