# Privacy and redaction

This release was scrubbed automatically before publication and the result was scanned with zero findings.

- Credentials, the model-gateway address and internal host names were removed. Live runs read the endpoint and key from the environment variables `API_BASE_URL` and `API_KEY`.
- Absolute local paths were replaced by relative paths or placeholders, and local serving hosts appear as `local_a` and `local_b` in the episode records.
- The episode records contain e-mail addresses, names and file paths that belong to the synthetic AgentDojo and InjecAgent environments or that the evaluated models wrote into tool calls. None of them refers to a real person involved in this work.
- Third-party benchmarks (AgentDojo, InjecAgent) are not redistributed; see README.md.
