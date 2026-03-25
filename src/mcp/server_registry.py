"""MCP server registry.

Loads the MCP server configuration from a YAML file and provides
lookup by server_type. Used by the MCP client to determine how
to launch each server process.
"""

from dataclasses import dataclass, field
from pathlib import Path

import structlog
import yaml

logger = structlog.get_logger(__name__)


@dataclass
class MCPServerConfig:
    """Configuration for launching an MCP server process.

    Attributes:
        server_type: The type identifier (e.g., "github", "aws").
        command: The executable to run.
        args: Command-line arguments.
        cwd: Working directory for the process.
    """

    server_type: str
    command: str
    args: list[str] = field(default_factory=list)
    cwd: str | None = None


class ServerRegistry:
    """Registry of available MCP servers loaded from configuration.

    Loads the YAML configuration once at construction and provides
    lookup by server_type. This is an immutable, read-only registry.

    Args:
        config_path: Path to the mcp_servers.yaml configuration file.
    """

    def __init__(self, config_path: str) -> None:
        """Load MCP server configuration from YAML.

        Args:
            config_path: Path to the mcp_servers.yaml file.
        """
        self._servers: dict[str, MCPServerConfig] = {}

        path = Path(config_path)
        if not path.exists():
            logger.warn(
                "MCP servers config file not found",
                action="load_registry",
                config_path=config_path,
            )
            return

        with open(path) as f:
            config = yaml.safe_load(f)

        if not config or "servers" not in config:
            logger.warn(
                "MCP servers config is empty or malformed",
                action="load_registry",
                config_path=config_path,
            )
            return

        for key, server_data in config["servers"].items():
            server_config = MCPServerConfig(
                server_type=server_data.get("server_type", key),
                command=server_data["command"],
                args=server_data.get("args", []),
                cwd=server_data.get("cwd"),
            )
            self._servers[server_config.server_type] = server_config

        logger.info(
            "MCP server registry loaded",
            action="load_registry",
            server_count=len(self._servers),
            server_types=list(self._servers.keys()),
        )

    def get_server_config(self, server_type: str) -> MCPServerConfig | None:
        """Look up server configuration by type.

        Args:
            server_type: The server type to look up (e.g., "github").

        Returns:
            The server configuration, or None if not registered.
        """
        return self._servers.get(server_type)

    def get_all_server_types(self) -> list[str]:
        """Return all registered server types.

        Returns:
            List of server type identifiers.
        """
        return list(self._servers.keys())
