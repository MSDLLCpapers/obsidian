"""
HTTP client for executing OpenAI function calls against Obsidian API.

This module provides an optional helper class that maps OpenAI function calls
to HTTP requests against the Obsidian REST API.
"""

import os
import json
import requests
from typing import Any, Dict, Optional


class ObsidianToolExecutor:
    """Execute OpenAI function calls against Obsidian REST API.

    This class provides a convenient wrapper for executing tool calls returned
    by OpenAI-compatible LLMs. It maps function names to API endpoints and
    handles HTTP requests, error handling, and response formatting.

    Args:
        base_url: Base URL of the Obsidian API. If None, uses OBSIDIAN_API_URL
                  environment variable or defaults to http://localhost:8000/api/v1
        verbose: Print executor debug messages (default True for LLM agents)
        optimizer_verbose: Verbosity level for optimizer operations (0-3).
                          Automatically injected into fit/suggest/evaluate calls.
                          0=no output, 1=summary, 2=detailed, 3=debugging (default)

    Example:
        executor = ObsidianToolExecutor(verbose=True, optimizer_verbose=3)
        result = executor.execute_tool_call("create_optimization_session", {
            "name": "My Session",
            "parameters": [...],
            "targets": [...]
        })
    """

    def __init__(
        self,
        base_url: Optional[str] = None,
        verbose: bool = True,
        optimizer_verbose: int = 3
    ):
        """Initialize the tool executor.

        Args:
            base_url: Optional base URL override. If not provided, uses
                      OBSIDIAN_API_URL env var or defaults to localhost.
            verbose: Print executor's own debug messages (default True)
            optimizer_verbose: Optimizer verbosity level 0-3 (default 3 for max output)
        """
        self.base_url = base_url or os.getenv("OBSIDIAN_API_URL", "http://localhost:8000/api/v1")
        self.session = requests.Session()
        self.verbose = verbose
        self.optimizer_verbose = optimizer_verbose

    def execute_tool_call(self, function_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a tool call and return structured response.

        Args:
            function_name: Name of the function to call (e.g., "create_optimization_session")
            arguments: Dictionary of function arguments

        Returns:
            Dictionary containing either:
                - Successful response from the API
                - Error dict with {"success": False, "error": {...}}

        Example:
            result = executor.execute_tool_call("create_optimization_session", {
                "parameters": [{"type": "continuous", "name": "temp", "min": 0, "max": 100}],
                "targets": [{"name": "yield", "aim": "max"}]
            })
        """
        # Log function call if verbose
        if self.verbose:
            print(f"\n[Obsidian] → {function_name}")
            # Truncate long arguments for readability
            args_str = json.dumps(arguments, indent=2)
            if len(args_str) > 500:
                args_str = args_str[:500] + "\n  ... (truncated)"
            print(f"[Obsidian] Arguments: {args_str}")

        # Map function name to HTTP method and endpoint
        endpoint_map = {
            # Session Management
            "create_optimization_session": ("POST", "/sessions"),
            "list_optimization_sessions": ("GET", "/sessions"),
            "get_session_details": ("GET", "/sessions/{session_id}"),
            "delete_optimization_session": ("DELETE", "/sessions/{session_id}"),
            # Workflow Operations
            "initialize_experiments": ("POST", "/sessions/{session_id}/initialize"),
            "sample_parameter_space": ("POST", "/sessions/{session_id}/sample"),
            "add_experimental_data": ("POST", "/sessions/{session_id}/data"),
            "fit_surrogate_model": ("POST", "/sessions/{session_id}/fit"),
            "suggest_next_experiments": ("POST", "/sessions/{session_id}/suggest"),
            "evaluate_predictions": ("POST", "/sessions/{session_id}/evaluate"),
            # Analysis and Results
            "get_best_results": ("GET", "/sessions/{session_id}/best"),
            "get_campaign_data": ("GET", "/sessions/{session_id}/data"),
            "get_model_diagnostics": ("GET", "/sessions/{session_id}/diagnostics"),
            "get_optimization_history": ("GET", "/sessions/{session_id}/history"),
            "export_state_dictionary": ("GET", "/sessions/{session_id}/state_dict"),
            # Analysis
            "analyze_shap": ("POST", "/sessions/{session_id}/analysis/shap"),
            "analyze_sensitivity": ("POST", "/sessions/{session_id}/analysis/sensitivity"),
            # Informational
            "list_acquisition_functions": ("GET", "/acquisition-functions"),
        }

        if function_name not in endpoint_map:
            return {
                "success": False,
                "error": {
                    "type": "UnknownFunction",
                    "message": f"Unknown function: {function_name}",
                    "http_status": None,
                },
            }

        method, endpoint = endpoint_map[function_name]

        # Extract session_id if present and format endpoint
        arguments = arguments.copy()  # Don't modify original
        session_id = arguments.pop("session_id", None)
        if session_id and "{session_id}" in endpoint:
            endpoint = endpoint.format(session_id=session_id)

        # Auto-inject optimizer verbosity for optimizer operations (LLM-specific feature)
        optimizer_operations = ["fit_surrogate_model", "suggest_next_experiments", "evaluate_predictions"]
        if function_name in optimizer_operations and "verbose" not in arguments:
            arguments["verbose"] = self.optimizer_verbose
            if self.verbose:
                print(f"[Obsidian] Auto-injecting optimizer verbosity: {self.optimizer_verbose}")

        # Build full URL
        url = f"{self.base_url}{endpoint}"

        try:
            # Make HTTP request
            if method == "GET":
                response = self.session.get(url, params=arguments)
            elif method == "POST":
                response = self.session.post(url, json=arguments)
            elif method == "DELETE":
                response = self.session.delete(url)
            else:
                raise ValueError(f"Unsupported HTTP method: {method}")

            # Raise for HTTP errors
            response.raise_for_status()

            # Log successful response
            if self.verbose:
                print(f"[Obsidian] ← Status: {response.status_code}")

            # Handle 204 No Content (DELETE operations)
            if response.status_code == 204:
                if self.verbose:
                    print("[Obsidian] Result: Success (no content)")
                return {"success": True}

            # Return JSON response
            result = response.json()
            if self.verbose:
                result_str = json.dumps(result, indent=2)
                if len(result_str) > 300:
                    result_str = result_str[:300] + "\n  ... (truncated)"
                print(f"[Obsidian] Result: {result_str}")
            return result

        except requests.exceptions.HTTPError as e:
            # HTTP error response (4xx, 5xx)
            error_detail = "Unknown error"
            try:
                error_data = e.response.json()
                error_detail = error_data.get("detail", str(e))
            except Exception:
                error_detail = str(e)

            if self.verbose:
                print(f"[Obsidian] ✗ HTTP Error {e.response.status_code}: {error_detail}")

            return {
                "success": False,
                "error": {"type": "HTTPError", "message": error_detail, "http_status": e.response.status_code},
            }

        except requests.exceptions.RequestException as e:
            # Network error, timeout, connection error, etc.
            if self.verbose:
                print(f"[Obsidian] ✗ Network Error: {type(e).__name__}: {str(e)}")

            return {"success": False, "error": {"type": type(e).__name__, "message": str(e), "http_status": None}}

        except Exception as e:
            # Unexpected error
            if self.verbose:
                print(f"[Obsidian] ✗ Unexpected Error: {str(e)}")

            return {
                "success": False,
                "error": {"type": "UnexpectedError", "message": f"Unexpected error: {str(e)}", "http_status": None},
            }

    def close(self):
        """Close the HTTP session."""
        self.session.close()

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
