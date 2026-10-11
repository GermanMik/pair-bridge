"""Process-local PAIR mDNS observer; no broker, model changes or trust changes."""
import atexit
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout
import ipaddress
import threading
import time
import uuid

SERVICE = '_nvpair-node._tcp.local.'
_lock = threading.RLock()
_records = {}
_browser = None
_zeroconf = None
_status = 'not_started'
_refresh_at = 0.0
_lifecycle_lock = threading.Lock()
_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="PAIR-mDNS")


def _record(info, name):
    properties = {k.decode('utf-8'): v.decode('utf-8') for k, v in info.properties.items() if isinstance(k, bytes) and isinstance(v, bytes)}
    if properties.get('v') != '1':
        raise ValueError('Unsupported PAIR discovery schema')
    peer = str(uuid.UUID(properties['uuid']))
    candidates = [properties.get('ip', '')] + properties.get('ips', '').split(',') + info.parsed_addresses()
    addresses = []
    for candidate in candidates:
        try:
            address = ipaddress.ip_address(candidate)
        except ValueError:
            continue
        if address.version == 4 and address.is_private and not address.is_unspecified and not address.is_multicast:
            if str(address) not in addresses:
                addresses.append(str(address))
    if not addresses:
        raise ValueError('No local IPv4 PAIR address')
    ports = {}
    for key in ('ni', 'em', 'lm'):
        value = properties.get(key)
        if value is not None:
            port = int(value)
            if not 1 <= port <= 65535:
                raise ValueError('Invalid advertised PAIR service port')
            ports[key] = port
    return {'nodeUuid': peer, 'name': name.removesuffix('.' + SERVICE)[:128],
            'ipAddress': addresses[0], 'addresses': addresses[:4], 'ports': ports,
            'state': 'member', 'network_present': True, 'discovery_source': 'PAIR mDNS'}


class Listener:
    def add_service(self, zeroconf, service, name):
        try:
            info = zeroconf.get_service_info(service, name, timeout=1500)
            if info is None:
                return
            row = _record(info, name)
            with _lock:
                if len(_records) < 64 or name in _records:
                    _records[name] = row
        except (ValueError, KeyError, TypeError, UnicodeError, OSError):
            # Untrusted/malformed advertisements never change native trust.
            return

    def update_service(self, zeroconf, service, name):
        self.add_service(zeroconf, service, name)

    def remove_service(self, zeroconf, service, name):
        with _lock:
            _records.pop(name, None)


def close():
    global _browser, _zeroconf, _status
    with _lock:
        browser, zeroconf = _browser, _zeroconf
        _browser = _zeroconf = None
        _status = 'not_started'
        _records.clear()
    try:
        if browser:
            browser.cancel()
    finally:
        if zeroconf:
            zeroconf.close()


def _create_browser():
    from zeroconf import Zeroconf, ServiceBrowser, IPVersion, DNSQuestionType
    # A separate query socket avoids contention with the native PAIR scanner
    # and other MCP processes sharing UDP5353. Queries remain multicast;
    # standard QU replies return to this browser's own ephemeral port.
    zeroconf = Zeroconf(ip_version=IPVersion.V4Only, unicast=True)
    try:
        # Request standard unicast replies on every browse query. This also
        # works on LANs where peer multicast replies are filtered in transit.
        return zeroconf, ServiceBrowser(zeroconf, SERVICE, Listener(), question_type=DNSQuestionType.QU)
    except Exception:
        zeroconf.close()
        raise


def _discard_browser(future):
    try:
        zeroconf, browser = future.result()
    except Exception:
        return
    try:
        browser.cancel()
    finally:
        zeroconf.close()


def _open_browser():
    # FastMCP invokes sync tools on its event loop. Create Zeroconf off that loop
    # so incoming DNS is not stalled by our synchronous inventory API calls.
    future = _executor.submit(_create_browser)
    try:
        return future.result(timeout=10)
    except FutureTimeout:
        future.add_done_callback(_discard_browser)
        raise


def snapshot():
    """Live event state with bounded failed-start retry and interface rebrowse."""
    global _browser, _zeroconf, _status, _refresh_at
    first = False
    with _lifecycle_lock:
        now = time.monotonic()
        with _lock:
            start = (_status == 'not_started' or
                     (_status == 'unavailable' and now >= _refresh_at) or
                     (_status == 'active' and _browser is not None and now >= _refresh_at))
        if start:
            close()
            try:
                zeroconf, browser = _open_browser()
                with _lock:
                    _zeroconf, _browser = zeroconf, browser
                    _status = 'active'
                    _refresh_at = time.monotonic() + 30
                first = True
            except Exception:
                with _lock:
                    _status = 'unavailable'
                    _refresh_at = time.monotonic() + 30
        if first:
            time.sleep(2)
    with _lock:
        return {'status': _status, 'nodes': [{**row, 'addresses': list(row['addresses']), 'ports': dict(row['ports'])} for row in _records.values()],
                'notice': 'Live PAIR mDNS discovery; advertisements do not grant native trust.'}


atexit.register(close)
