"""Pydantic models for credential envelope and per-type credential structures.

Conforms to the credential_envelope and credential_structures schemas
defined in mcp-tool-interface.yaml section 3.
"""

from typing import Literal

from pydantic import BaseModel


class AwsCredentials(BaseModel):
    """AWS credentials for MCP tool invocation.

    In broker mode, includes a temporary session token from AWS STS.
    In direct mode, uses stored access key and secret key.

    Attributes:
        access_key_id: AWS access key ID.
        secret_access_key: AWS secret access key.
        session_token: Temporary session token (broker mode only).
        region: AWS region for API calls.
    """

    access_key_id: str
    secret_access_key: str
    session_token: str | None = None
    region: str


class GoogleWorkspaceCredentials(BaseModel):
    """Google Workspace credentials for MCP tool invocation.

    Attributes:
        service_account_json: Full service account key JSON as a string.
        delegated_email: Email to impersonate for domain-wide delegation.
    """

    service_account_json: str
    delegated_email: str | None = None


class GitHubCredentials(BaseModel):
    """GitHub credentials for MCP tool invocation.

    Attributes:
        personal_access_token: GitHub PAT with read-only permissions.
        organization: GitHub organization name to query against.
    """

    personal_access_token: str
    organization: str


class CredentialEnvelope(BaseModel):
    """Carries authentication credentials for a single MCP tool invocation.

    Conforms to the credential_envelope schema in mcp-tool-interface.yaml.
    Credentials exist in memory only for the duration of the invocation
    and MUST NOT be cached, logged, or persisted.

    Attributes:
        server_type: Identifies the credential format.
        credential_mode: Whether broker or direct credential transit is used.
        credential_data: The actual credential values, structure varies by server_type.
    """

    server_type: str
    credential_mode: Literal["broker", "direct"]
    credential_data: dict
