"""
Trek Export/Links API Client

This module provides a client for querying the TREK Web API Export/Links endpoint.
It handles authentication, request building, response parsing, and error handling.

Example:
    >>> client = TrekExportLinksClient()
    >>> response = client.get_links(
    ...     project_id=607,
    ...     campaign_id=102766584,
    ...     module_names=["SYT - Bus Communication"],
    ...     local_config_domain_type=4
    ... )
    >>> for link in response.links:
    ...     print(f"{link['source']} -> {link['target']}")
"""

import requests
from urllib.parse import quote
from typing import List, Dict, Optional, Tuple
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from requests_negotiate_sspi import HttpNegotiateAuth
from urllib3.exceptions import InsecureRequestWarning

# Suppress SSL warnings
requests.packages.urllib3.disable_warnings(category=InsecureRequestWarning)


class TrekExportLinksResponse:
    """
    Wrapper for TREK Export/Links API response.
    
    Attributes:
        raw_response (str): Raw text response from API
        links (List[Dict]): Parsed links from response
        status_code (int): HTTP status code
        success (bool): Whether request was successful
        error_message (str): Error message if request failed
    """
    
    def __init__(self, status_code: int, raw_response: str = "", error: str = ""):
        self.status_code = status_code
        self.raw_response = raw_response
        self.error_message = error
        self.success = 200 <= status_code < 300
        self.links = []
        
        if self.success:
            self._parse_response(raw_response)
    
    def _parse_response(self, response_text: str):
        """
        Parse the response text and extract links.
        
        The API typically returns plain text or JSON format.
        This method handles both formats.
        
        Args:
            response_text (str): Raw response from API
        """
        try:
            # Try parsing as JSON first (if response is JSON array/object)
            if response_text.strip().startswith('[') or response_text.strip().startswith('{'):
                self.links = json.loads(response_text)
                if isinstance(self.links, dict):
                    # If it's an object, try to extract 'links' or 'data' field
                    if 'links' in self.links:
                        self.links = self.links['links']
                    elif 'data' in self.links:
                        self.links = self.links['data']
                    else:
                        self.links = [self.links]
            else:
                # Parse as plain text - assuming line-by-line format
                self.links = self._parse_plain_text(response_text)
        except json.JSONDecodeError:
            # If JSON parsing fails, try plain text parsing
            self.links = self._parse_plain_text(response_text)
    
    def _parse_plain_text(self, text: str) -> List[Dict]:
        """
        Parse plain text response.
        
        Handles various text formats:
        - One link per line
        - Space-separated source and target
        - Arrow notation (source -> target)
        
        Args:
            text (str): Plain text response
            
        Returns:
            List[Dict]: Parsed links as dictionaries
        """
        links = []
        lines = text.strip().split('\n')
        
        for line in lines:
            line = line.strip()
            if not line or line.startswith('#'):  # Skip empty lines and comments
                continue
            
            # Try to parse different formats
            link_dict = self._parse_link_line(line)
            if link_dict:
                links.append(link_dict)
        
        return links
    
    def _parse_link_line(self, line: str) -> Optional[Dict]:
        """
        Parse a single line and extract link information.
        
        Supports formats:
        - "source -> target"
        - "source|target"
        - "source target"
        - JSON object
        
        Args:
            line (str): A single line from response
            
        Returns:
            Optional[Dict]: Parsed link or None
        """
        # Try arrow notation
        if '->' in line:
            parts = line.split('->')
            if len(parts) == 2:
                return {
                    'source': parts[0].strip(),
                    'target': parts[1].strip(),
                    'type': 'arrow'
                }
        
        # Try pipe notation
        if '|' in line:
            parts = line.split('|')
            if len(parts) >= 2:
                return {
                    'source': parts[0].strip(),
                    'target': parts[1].strip(),
                    'type': 'pipe'
                }
        
        # Try space-separated
        parts = line.split(None, 1)  # Split on first whitespace
        if len(parts) == 2:
            return {
                'source': parts[0].strip(),
                'target': parts[1].strip(),
                'type': 'space'
            }
        
        # Try as JSON object
        try:
            obj = json.loads(line)
            if isinstance(obj, dict) and ('source' in obj or 'from' in obj or 'id' in obj):
                return obj
        except json.JSONDecodeError:
            pass
        
        return None
    
    def to_dict(self) -> Dict:
        """Convert response to dictionary."""
        return {
            'success': self.success,
            'status_code': self.status_code,
            'error': self.error_message if not self.success else None,
            'link_count': len(self.links),
            'links': self.links
        }
    
    def to_json(self) -> str:
        """Convert response to JSON string."""
        return json.dumps(self.to_dict(), indent=2)
    
    def __str__(self) -> str:
        """String representation."""
        if self.success:
            return f"TrekExportLinksResponse(status={self.status_code}, links={len(self.links)})"
        else:
            return f"TrekExportLinksResponse(status={self.status_code}, error={self.error_message})"
    
    def __repr__(self) -> str:
        """Developer-friendly representation."""
        return self.__str__()


class TrekModulesResponse:
    """
    Wrapper for TREK Modules API response (/api/Modules).

    Attributes:
        raw_response (str): Raw JSON text from the API
        modules (List[Dict]): Parsed list of module dicts, each with keys:
                              'Id' (float), 'Name' (str), 'Total' (int, optional)
        status_code (int): HTTP status code
        success (bool): True when 200 <= status_code < 300
        error_message (str): Error description when success is False
    """

    def __init__(self, status_code: int, raw_response: str = "", error: str = ""):
        self.status_code   = status_code
        self.raw_response  = raw_response
        self.error_message = error
        self.success       = 200 <= status_code < 300
        self.modules: List[Dict] = []

        if self.success and raw_response:
            self._parse(raw_response)

    def _parse(self, text: str):
        try:
            data = json.loads(text)
            # API returns {"Data": [...]} or a bare list
            if isinstance(data, dict):
                items = data.get("Data", data.get("data", []))
            elif isinstance(data, list):
                items = data
            else:
                items = []
            self.modules = items
        except json.JSONDecodeError as exc:
            self.success       = False
            self.error_message = f"JSON parse error: {exc}"

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------

    def with_test_cases(self) -> List[Dict]:
        """Return only modules that have a 'Total' (test-case count) > 0."""
        return [m for m in self.modules if m.get("Total", 0) > 0]

    def names(self) -> List[str]:
        """Return a plain list of module names."""
        return [m["Name"] for m in self.modules]

    def to_dict(self) -> Dict:
        return {
            "success":      self.success,
            "status_code":  self.status_code,
            "error":        self.error_message if not self.success else None,
            "module_count": len(self.modules),
            "modules":      self.modules,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)

    def __str__(self) -> str:
        if self.success:
            return f"TrekModulesResponse(status={self.status_code}, modules={len(self.modules)})"
        return f"TrekModulesResponse(status={self.status_code}, error={self.error_message})"

    def __repr__(self) -> str:
        return self.__str__()


class TrekRequirementsResponse:
    """
    Wrapper for TREK Export/Requirements API response (/api/Export/Requirements).

    Mirrors TrekExportLinksResponse's parsing strategy (the endpoint follows
    the same {"Data": [...]} / bare-list JSON shape as Export/Links and
    Modules), but is kept as a distinct class since the payload shape for
    requirement objects (attributes, text fields, DOORS ids) is different
    from link edges.

    Attributes:
        raw_response (str): Raw text response from API
        requirements (List[Dict]): Parsed requirement objects from response
        status_code (int): HTTP status code
        success (bool): Whether request was successful
        error_message (str): Error message if request failed
    """

    def __init__(self, status_code: int, raw_response: str = "", error: str = ""):
        self.status_code   = status_code
        self.raw_response  = raw_response
        self.error_message = error
        self.success       = 200 <= status_code < 300
        self.requirements: List[Dict] = []

        if self.success and raw_response:
            self._parse(raw_response)

    def _parse(self, text: str):
        try:
            data = json.loads(text)
            if isinstance(data, dict):
                items = data.get("Data", data.get("data", []))
            elif isinstance(data, list):
                items = data
            else:
                items = []
            self.requirements = items
        except json.JSONDecodeError as exc:
            self.success       = False
            self.error_message = f"JSON parse error: {exc}"

    def to_dict(self) -> Dict:
        return {
            "success":      self.success,
            "status_code":  self.status_code,
            "error":        self.error_message if not self.success else None,
            "req_count":    len(self.requirements),
            "requirements": self.requirements,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)

    def __str__(self) -> str:
        if self.success:
            return f"TrekRequirementsResponse(status={self.status_code}, requirements={len(self.requirements)})"
        return f"TrekRequirementsResponse(status={self.status_code}, error={self.error_message})"

    def __repr__(self) -> str:
        return self.__str__()


class TrekExportLinksClient:
    """
    Client for TREK Export/Links API.
    
    Handles authentication, request building, and response parsing.
    Uses Integrated Windows Authentication (SSPI).
    
    Attributes:
        base_url (str): Base URL for TREK API
        verify_ssl (bool): Whether to verify SSL certificates
    """
    
    BASE_URL = "https://frz00052vma.automotive-wan.com:9621/api"
    EXPORT_ENDPOINT        = "/Export/Links"
    MODULES_ENDPOINT       = "/Modules"
    REQUIREMENTS_ENDPOINT  = "/Export/Requirements"

    def __init__(self, base_url: str = BASE_URL, verify_ssl: bool = False):
        """
        Initialize the TREK Export/Links client.
        
        Args:
            base_url (str): Base URL for TREK API
            verify_ssl (bool): Whether to verify SSL certificates (default: False)
        """
        self.base_url = base_url
        self.verify_ssl = verify_ssl
        self.session = self._create_session()
    
    def _create_session(self) -> requests.Session:
        """
        Create a requests session with proper authentication.
        
        Returns:
            requests.Session: Configured session object
        """
        session = requests.Session()
        try:
            session.auth = HttpNegotiateAuth()
        except Exception as e:
            print(f"Warning: Could not set up SSPI authentication: {e}")
            print("Continuing without authentication...")
        return session
    
    def get_links(
        self,
        project_id: int,
        campaign_id: int,
        module_names: Optional[List[str]] = None,
        local_config_domain_type: int = 4,
        timeout: int = 120
    ) -> TrekExportLinksResponse:
        """
        Fetch export links from TREK API.
        
        Args:
            project_id (int): TREK project ID (e.g., 607)
            campaign_id (int): TREK campaign ID (e.g., 102766584)
            module_names (Optional[List[str]]): Module names to export (e.g., ["SYT - Bus Communication"])
            local_config_domain_type (int): Local configuration domain type (default: 4)
            timeout (int): Request timeout in seconds (default: 120)
        
        Returns:
            TrekExportLinksResponse: Parsed API response
        
        Example:
            >>> client = TrekExportLinksClient()
            >>> response = client.get_links(
            ...     project_id=607,
            ...     campaign_id=102766584,
            ...     module_names=["SYT - Bus Communication"],
            ...     local_config_domain_type=4
            ... )
            >>> if response.success:
            ...     print(f"Retrieved {len(response.links)} links")
            ... else:
            ...     print(f"Error: {response.error_message}")
        """
        try:
            # Build query parameters
            params = self._build_params(
                project_id=project_id,
                campaign_id=campaign_id,
                module_names=module_names,
                local_config_domain_type=local_config_domain_type
            )
            
            # Construct URL
            url = f"{self.base_url}{self.EXPORT_ENDPOINT}"
            
            print(f"[TREK] Sending GET request to: {url}")
            print(f"[TREK] Parameters: {params}")
            
            # Make request
            response = self.session.get(
                url,
                params=params,
                timeout=timeout,
                verify=self.verify_ssl,
                headers={'accept': 'text/plain'}
            )
            
            # Handle response
            if response.status_code == 200:
                print(f"[TREK] Request successful (status {response.status_code})")
                return TrekExportLinksResponse(
                    status_code=response.status_code,
                    raw_response=response.text
                )
            else:
                error_msg = f"HTTP {response.status_code}: {response.reason}"
                print(f"[TREK] Request failed: {error_msg}")
                return TrekExportLinksResponse(
                    status_code=response.status_code,
                    error=error_msg
                )
        
        except requests.exceptions.Timeout:
            error_msg = f"Request timeout after {timeout} seconds"
            print(f"[TREK] {error_msg}")
            return TrekExportLinksResponse(
                status_code=0,
                error=error_msg
            )
        
        except requests.exceptions.ConnectionError as e:
            error_msg = f"Connection error: {str(e)}"
            print(f"[TREK] {error_msg}")
            return TrekExportLinksResponse(
                status_code=0,
                error=error_msg
            )
        
        except Exception as e:
            error_msg = f"Unexpected error: {str(e)}"
            print(f"[TREK] {error_msg}")
            return TrekExportLinksResponse(
                status_code=0,
                error=error_msg
            )
    
    def _build_params(
        self,
        project_id: int,
        campaign_id: int,
        module_names: Optional[List[str]] = None,
        local_config_domain_type: int = 4
    ) -> Dict[str, str]:
        """
        Build query parameters for the API request.
        
        Args:
            project_id (int): TREK project ID
            campaign_id (int): TREK campaign ID
            module_names (Optional[List[str]]): Module names to filter
            local_config_domain_type (int): Configuration domain type
        
        Returns:
            Dict[str, str]: Query parameters
        """
        params = {
            'projectId': str(project_id),
            'campaignId': str(campaign_id),
            'localConfigurationDomainType': str(local_config_domain_type)
        }
        
        # Add module names if provided
        if module_names:
            # Join multiple module names with comma (or use first if single)
            params['moduleNames'] = ','.join(module_names)
        
        return params
    
    def get_links_raw(
        self,
        project_id: int,
        campaign_id: int,
        module_names: Optional[List[str]] = None,
        local_config_domain_type: int = 4,
        timeout: int = 120
    ) -> Tuple[int, str]:
        """
        Fetch export links and return raw response (low-level).
        
        Args:
            project_id (int): TREK project ID
            campaign_id (int): TREK campaign ID
            module_names (Optional[List[str]]): Module names to export
            local_config_domain_type (int): Local configuration domain type
            timeout (int): Request timeout in seconds
        
        Returns:
            Tuple[int, str]: (status_code, response_text)
        """
        params = self._build_params(
            project_id=project_id,
            campaign_id=campaign_id,
            module_names=module_names,
            local_config_domain_type=local_config_domain_type
        )
        
        url = f"{self.base_url}{self.EXPORT_ENDPOINT}"
        response = self.session.get(
            url,
            params=params,
            timeout=timeout,
            verify=self.verify_ssl,
            headers={'accept': 'text/plain'}
        )
        
        return response.status_code, response.text

    def get_requirements(
        self,
        project_id: int,
        campaign_id: int,
        module_names: Optional[List[str]] = None,
        timeout: int = 120
    ) -> TrekRequirementsResponse:
        """
        Fetch requirement content from TREK API (/api/Export/Requirements).

        Returns the full requirement objects (text, attributes, DOORS ids)
        for every requirement in the given module(s) -- e.g. all SYR or SWR
        requirements in "SYR - Infrastructure". Unlike /Export/Links (which
        returns only link edges) or /AutoTestCases (which returns test case
        content), this endpoint returns the actual requirement text itself.

        Args:
            project_id (int): TREK project ID (e.g., 607)
            campaign_id (int): TREK campaign ID (e.g., 102766584)
            module_names (Optional[List[str]]): Module names to export
                                                 (e.g., ["SYR - Infrastructure"])
            timeout (int): Request timeout in seconds (default: 120)

        Returns:
            TrekRequirementsResponse: Parsed API response with .requirements list

        Example:
            >>> client = TrekExportLinksClient()
            >>> resp = client.get_requirements(
            ...     project_id=607,
            ...     campaign_id=102766584,
            ...     module_names=["SYR - Infrastructure"]
            ... )
            >>> if resp.success:
            ...     print(f"Retrieved {len(resp.requirements)} requirements")
        """
        params: Dict[str, str] = {
            'projectId':  str(project_id),
            'campaignId': str(campaign_id),
        }
        if module_names:
            params['moduleNames'] = ','.join(module_names)

        url = f"{self.base_url}{self.REQUIREMENTS_ENDPOINT}"

        print(f"[TREK] Sending GET request to: {url}")
        print(f"[TREK] Parameters: {params}")

        try:
            response = self.session.get(
                url,
                params=params,
                timeout=timeout,
                verify=self.verify_ssl,
                headers={'accept': 'text/plain'}
            )

            if response.status_code == 200:
                print(f"[TREK] Request successful (status {response.status_code})")
                return TrekRequirementsResponse(
                    status_code=response.status_code,
                    raw_response=response.text
                )
            else:
                error_msg = f"HTTP {response.status_code}: {response.reason}"
                print(f"[TREK] Request failed: {error_msg}")
                return TrekRequirementsResponse(
                    status_code=response.status_code,
                    error=error_msg
                )

        except requests.exceptions.Timeout:
            error_msg = f"Request timeout after {timeout} seconds"
            print(f"[TREK] {error_msg}")
            return TrekRequirementsResponse(status_code=0, error=error_msg)

        except requests.exceptions.ConnectionError as e:
            error_msg = f"Connection error: {str(e)}"
            print(f"[TREK] {error_msg}")
            return TrekRequirementsResponse(status_code=0, error=error_msg)

        except Exception as e:
            error_msg = f"Unexpected error: {str(e)}"
            print(f"[TREK] {error_msg}")
            return TrekRequirementsResponse(status_code=0, error=error_msg)

    def get_modules(
        self,
        project_id: int,
        config_id: int,
        local_config_domain_type: int = 4,
        timeout: int = 60
    ) -> TrekModulesResponse:
        """
        Fetch all modules for a TREK project configuration (/api/Modules).

        Args:
            project_id (int): TREK project ID (e.g., 607)
            config_id (int): TREK configuration ID (e.g., 8579)
            local_config_domain_type (int): Domain type (default: 4)
            timeout (int): Request timeout in seconds (default: 60)

        Returns:
            TrekModulesResponse: Parsed response with .modules list.

        Example:
            >>> client = TrekExportLinksClient()
            >>> resp = client.get_modules(project_id=607, config_id=8579)
            >>> for m in resp.modules:
            ...     print(m['Name'], m.get('Total', '-'))
        """
        url    = f"{self.base_url}{self.MODULES_ENDPOINT}"
        params = {
            "projectId":                   str(project_id),
            "configId":                    str(config_id),
            "localConfigurationDomainType": str(local_config_domain_type),
        }

        print(f"[TREK] Sending GET request to: {url}")
        print(f"[TREK] Parameters: {params}")

        try:
            response = self.session.get(
                url,
                params=params,
                timeout=timeout,
                verify=self.verify_ssl,
                headers={"accept": "text/plain"},
            )

            if response.status_code == 200:
                print(f"[TREK] Request successful (status {response.status_code})")
                return TrekModulesResponse(
                    status_code=response.status_code,
                    raw_response=response.text,
                )
            else:
                error_msg = f"HTTP {response.status_code}: {response.reason}"
                print(f"[TREK] Request failed: {error_msg}")
                return TrekModulesResponse(status_code=response.status_code, error=error_msg)

        except requests.exceptions.Timeout:
            error_msg = f"Request timeout after {timeout} seconds"
            print(f"[TREK] {error_msg}")
            return TrekModulesResponse(status_code=0, error=error_msg)

        except requests.exceptions.ConnectionError as exc:
            error_msg = f"Connection error: {exc}"
            print(f"[TREK] {error_msg}")
            return TrekModulesResponse(status_code=0, error=error_msg)

        except Exception as exc:
            error_msg = f"Unexpected error: {exc}"
            print(f"[TREK] {error_msg}")
            return TrekModulesResponse(status_code=0, error=error_msg)

    def get_requirements_keys_only(
        self,
        project_id: int,
        campaign_id: int,
        module_names: List[str],
        timeout: int = 300,
        retries: int = 3,
    ) -> List[Dict]:
        """Fast /Export/Requirements call asking ONLY for Key;ModuleName.

        Used by trek_index to build the object-id -> module map cheaply
        (attributeNames trims the payload; ModuleName is the real module
        name). Returns a list of {"Key": ..., "ModuleName": ...} rows, or
        an empty list on failure. Retries on transient timeouts because the
        Requirements endpoint can be slow.

        Args:
            project_id (int):  TREK project ID
            campaign_id (int): TREK campaign ID
            module_names (List[str]): modules to fetch (one batch)
            timeout (int): per-request timeout seconds
            retries (int): retry attempts on timeout/connection error
        """
        params = {
            "projectId":      str(project_id),
            "campaignId":     str(campaign_id),
            "attributeNames": "Key;ModuleName",
            "moduleNames":    ";".join(module_names),
        }
        url = f"{self.base_url}{self.REQUIREMENTS_ENDPOINT}"

        last_exc = None
        for attempt in range(retries):
            try:
                response = self.session.get(
                    url, params=params, timeout=timeout,
                    verify=self.verify_ssl, headers={"accept": "text/plain"},
                )
                if response.status_code == 200:
                    data = response.json()
                    if isinstance(data, dict):
                        return data.get("Data", data.get("data", [])) or []
                    if isinstance(data, list):
                        return data
                    return []
                # non-200: return empty (caller falls back)
                return []
            except (requests.exceptions.Timeout,
                    requests.exceptions.ConnectionError) as exc:
                last_exc = exc
                time.sleep(3 * (attempt + 1))
            except Exception:
                return []
        if last_exc:
            print(f"[TREK] get_requirements_keys_only failed: {last_exc}")
        return []

    def get_test_case_content(
        self,
        project_id: int,
        campaign_id: int,
        tc_ids: List[str],
        batch_size: int = 50,
        timeout: int = 60,
        verbose: bool = True,
        max_workers: int = 6,
    ) -> Dict[str, Dict]:
        """
        Fetch full test case content (Name, PreCondition, Procedure, Postcondition,
        Expected_result, Module_Path, Doors_Path) for a list of TC IDs.

        Uses /api/AutoTestCases with semicolon-separated keys (max ~50 per batch
        to stay within URL length limits).

        PERFORMANCE: batches are fetched CONCURRENTLY (up to max_workers at a
        time) instead of one-at-a-time, since each batch is an independent
        request that doesn't depend on any other batch's result. This is
        purely an I/O-bound wait-on-network operation, so overlapping several
        in-flight requests cuts wall-clock time roughly proportional to
        max_workers (e.g. a 1600-TC module needing ~33 sequential batches at
        ~1-2s each -- 40-60s total -- drops to roughly 40-60s / 6 ~= 7-10s
        with 6 concurrent workers, network/server permitting).

        Each worker uses its OWN requests.Session (via _create_session()),
        never sharing self.session across threads: requests_negotiate_sspi's
        HttpNegotiateAuth caches mutable per-instance state (self._host) on
        first use, so concurrent requests through a single shared session's
        auth object risk a data race. Windows SSPI credentials are ambient
        (tied to the logged-on user, not stored per-session), so creating a
        fresh session per worker is cheap and correct.

        Args:
            project_id (int):   TREK project ID (e.g., 607)
            campaign_id (int):  TREK campaign ID (e.g., 102766584)
            tc_ids (List[str]): List of TC object IDs (e.g., ['SYT_LGHT_1000', ...])
            batch_size (int):   Number of TCs per API call (default 50, max ~100)
            timeout (int):      Per-request timeout in seconds (default 60)
            verbose (bool):     Print progress (default True)
            max_workers (int):  Max concurrent in-flight batch requests (default 6)

        Returns:
            Dict[str, Dict]: Mapping TC ID -> content dict with keys:
                             Key, Name, PreCondition, Procedure, Postcondition,
                             Expected_result, Module_Path, Doors_Path, Modified_time
        """
        url = f"{self.base_url}/AutoTestCases"
        result: Dict[str, Dict] = {}
        ids = list(tc_ids)
        batches = [ids[start:start + batch_size] for start in range(0, len(ids), batch_size)]
        # IDs whose batch request FAILED (network error, timeout, non-200),
        # as opposed to IDs TREK answered for but did not return. Callers
        # read this after the call to tell "TREK unreachable / not
        # downloaded" apart from "not found in TREK".
        self.last_failed_ids: List[str] = []
        self.last_fetch_errors: List[str] = []
        total_batches = len(batches)

        if verbose:
            print(f"[TREK] Fetching content for {len(ids)} TCs "
                  f"({total_batches} batches of {batch_size}, "
                  f"up to {min(max_workers, total_batches) or 1} concurrent)...")

        def _fetch_one_batch(batch_num: int, batch: List[str]) -> Tuple[int, List[Dict], Optional[str]]:
            """Runs in a worker thread: its own session, no shared mutable
            state with any other in-flight batch fetch."""
            session = self._create_session()
            keys_param = ";".join(batch)
            params = {
                "projectId":        str(project_id),
                "campaignId":       str(campaign_id),
                "testCaseKeys":     keys_param,
                "modifiedTimeOnly": "false",
            }
            try:
                response = session.get(
                    url,
                    params=params,
                    timeout=timeout,
                    verify=self.verify_ssl,
                    headers={"accept": "text/plain"},
                )
                if response.status_code == 200:
                    data = json.loads(response.text)
                    return batch_num, data.get("Data", []), None
                return batch_num, [], f"HTTP {response.status_code}"
            except Exception as exc:
                return batch_num, [], str(exc)

        if total_batches == 0:
            return result

        with ThreadPoolExecutor(max_workers=max(1, min(max_workers, total_batches))) as executor:
            futures = {
                executor.submit(_fetch_one_batch, batch_num, batch): batch_num
                for batch_num, batch in enumerate(batches, 1)
            }
            for future in as_completed(futures):
                batch_num, items, error = future.result()
                if error is None:
                    for item in items:
                        key = item.get("Key", "")
                        if key:
                            result[key] = item
                    if verbose:
                        print(f"[TREK]   Batch {batch_num}/{total_batches}: "
                              f"{len(items)} records returned")
                else:
                    self.last_failed_ids.extend(batches[batch_num - 1])
                    self.last_fetch_errors.append(error)
                    if verbose:
                        print(f"[TREK]   Batch {batch_num}/{total_batches}: "
                              f"{error} - skipped")

        if verbose:
            print(f"[TREK] Content fetched: {len(result)}/{len(ids)} TCs resolved")

        return result


# =============================================================================
# Example Usage / Testing
# =============================================================================

if __name__ == "__main__":
    """
    Example usage of the TrekExportLinksClient.
    Run this script directly to test the API client.
    """
    
    # Initialize client
    print("=" * 70)
    print("TREK Export/Links API Client - Example Usage")
    print("=" * 70)
    
    client = TrekExportLinksClient(verify_ssl=False)
    
    # Make request
    print("\n[1] Making API request...")
    response = client.get_links(
        project_id=607,
        campaign_id=102766584,
        module_names=["SYT - Bus Communication"],
        local_config_domain_type=4,
        timeout=120
    )
    
    # Check success
    print(f"\n[2] Response: {response}")
    
    # Print results
    if response.success:
        print(f"\n[3] Success! Retrieved {len(response.links)} links")
        print(f"\n[4] Links (first 5):")
        for i, link in enumerate(response.links[:5], 1):
            print(f"  {i}. {link}")
        
        # Save to JSON
        output_file = "trek_export_links.json"
        with open(output_file, 'w') as f:
            f.write(response.to_json())
        print(f"\n[5] Full response saved to: {output_file}")
        
    else:
        print(f"\n[3] Error: {response.error_message}")
        print(f"\nRaw response:\n{response.raw_response}")
    
    print("\n" + "=" * 70)
