"""Read-only existing NVIDIA PAIR paired roster; never starts a broker."""
import http.client
import ipaddress
import json
import os
from pathlib import Path
import ssl
import sys
import time
import uuid
import network_discovery


class InventoryHTTPError(ValueError):
    def __init__(self, status):
        self.status = status
        super().__init__('PAIR inventory HTTP ' + str(status))


def cluster_dir():
    override = os.environ.get('PAIR_CLUSTER_DIR')
    if override:
        path = Path(override)
        if not path.is_absolute():
            raise ValueError('PAIR_CLUSTER_DIR must be absolute')
        return path
    if sys.platform == 'win32':
        return Path(os.environ.get('LOCALAPPDATA', Path.home() / 'AppData/Local')) / 'Nvidia Corporation/Personal AI Router/cluster'
    if sys.platform == 'darwin':
        return Path.home() / 'Library/Application Support/Nvidia Corporation/Personal AI Router/cluster'
    return Path.home() / '.config/Nvidia Corporation/Personal AI Router/cluster'


def _read(path):
    if path.stat().st_size > 1024 * 1024:
        raise ValueError('PAIR inventory metadata exceeds size limit')
    return json.loads(path.read_text(encoding='utf-8'))


def peer_connection(host, port, root, peer, timeout=6):
    connection = None
    try:
        pin = _read(root / 'trusted' / (peer + '.json'))
        if pin.get('nodeUuid') != peer:
            raise ValueError('PAIR peer pin identity mismatch')
        der = ssl.PEM_cert_to_DER_cert(pin['certPem'])
        context = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
        context.check_hostname = False  # PAIR trusts exact UUID leaf pins, not DNS names.
        context.load_verify_locations(cadata=pin['certPem'])
        context.verify_flags |= ssl.VERIFY_X509_PARTIAL_CHAIN
        context.load_cert_chain(root / 'node.crt', root / 'node.key')
        connection = http.client.HTTPSConnection(host, port, timeout=timeout, context=context)
        connection.connect()
        if connection.sock.getpeercert(binary_form=True) != der:
            raise ValueError('PAIR peer certificate does not match the exact pin')
        return connection
    except Exception:
        if connection:
            connection.close()
        raise


def _get(host, port, route, root=None, peer=None):
    connection = None
    try:
        if peer:
            connection = peer_connection(host, port, root, peer)
        else:
            connection = http.client.HTTPConnection(host, port, timeout=6)
        connection.request('GET', route)
        response = connection.getresponse()
        if response.status != 200:
            raise InventoryHTTPError(response.status)
        payload = response.read(1024 * 1024 + 1)
        if len(payload) > 1024 * 1024:
            raise ValueError('PAIR inventory response exceeds size limit')
        data = json.loads(payload)
        if not isinstance(data, dict):
            raise ValueError('Invalid PAIR inventory response')
        return data
    finally:
        if connection:
            connection.close()


def inspect():
    try:
        root = cluster_dir()
        roster_available = (root / 'members.json').exists()
        roster = _read(root / 'members.json') if roster_available else []
        if not isinstance(roster, list) or len(roster) > 64:
            raise ValueError('Invalid PAIR paired roster')
        ports = {'node_info': 14318, 'models': 14322}
        overrides = root / 'bridge-inventory-ports.json'
        if overrides.exists():
            ports.update(_read(overrides))
        if any(type(ports[k]) is not int or not 1 <= ports[k] <= 65535 for k in ('node_info', 'models')):
            raise ValueError('Invalid PAIR inventory service ports')
    except (OSError, ValueError, TypeError):
        return {'status': 'invalid_metadata', 'nodes': [], 'notice': 'Native PAIR roster or service-port metadata is invalid.'}
    discovery = network_discovery.snapshot()
    merged = {}
    for member in roster:
        if isinstance(member, dict) and isinstance(member.get('nodeUuid'),str):
            merged[member['nodeUuid']] = dict(member)
    for discovered in discovery['nodes']:
        previous = merged.get(discovered['nodeUuid'], {})
        combined = {**previous, **discovered}
        try:
            if ipaddress.ip_address(previous.get('ipAddress','')).is_loopback:
                combined['ipAddress'] = previous['ipAddress']
        except ValueError:
            pass
        merged[discovered['nodeUuid']] = combined
    nodes = []
    deadline = time.monotonic() + 18
    for member in merged.values():
        row = None
        if not isinstance(member, dict) or member.get('state') != 'member':
            continue
        try:
            peer = str(uuid.UUID(str(member['nodeUuid'])))
            row = {'device': 'pair-' + peer, 'name': str(member.get('name', peer))[:128],
                   'physical_node_uuid': peer, 'source': member.get('discovery_source', 'NVIDIA PAIR paired roster'),
                   'network_present': member.get('network_present'),
                   'peer_pin_available': False,
                   'management_supported': False, 'inference_supported': False,
                   'models': [], 'model_inventory_status': 'unknown',
                   'service_ports_source': 'PAIR mDNS TXT' if 'ports' in member else ('native defaults' if not overrides.exists() else 'explicit native inventory override')}
            nodes.append(row)
            row['check_status'] = 'metadata_invalid'
            address = ipaddress.ip_address(member['ipAddress'])
            if not (address.is_loopback or address.is_private) or address.is_unspecified or address.is_multicast:
                raise ValueError('PAIR roster address is not a local unicast address')
            row['peer_pin_available'] = address.is_loopback or (root / 'trusted' / (peer + '.json')).exists()
            row['online'] = None
            if time.monotonic() >= deadline:
                row['error'] = 'Inventory deadline reached; node was not probed.'
                continue
            advertised = member.get('ports')
            ni_port = advertised.get('ni') if advertised is not None else ports['node_info']
            em_port = advertised.get('em') if advertised is not None else ports['models']
            if ni_port is None:
                row['error'] = 'PAIR node-info service is not advertised.'
                continue
            row['check_status'] = 'identity_unconfirmed'
            candidates = [str(address)] + member.get('addresses', [])
            last_error = None
            confirmed = False
            for candidate in list(dict.fromkeys(candidates))[:4]:
                if time.monotonic() >= deadline:
                    break
                try:
                    candidate_address = ipaddress.ip_address(candidate)
                    if not (candidate_address.is_private or candidate_address.is_loopback) or candidate_address.is_unspecified or candidate_address.is_multicast:
                        continue
                    info = _get(str(candidate_address), ni_port, '/v1/node-info')
                    if info.get('hostUuid') != peer:
                        raise ValueError('PAIR node identity mismatch')
                    address = candidate_address
                    confirmed = True
                    break
                except (OSError, ValueError, TypeError, http.client.HTTPException) as exc:
                    last_error = exc
            if not confirmed:
                if time.monotonic() >= deadline:
                    raise ValueError('PAIR inventory deadline reached before identity confirmation')
                if last_error is not None:
                    raise last_error
                raise ValueError('No valid reachable PAIR address')
            row['online'] = True
            row['check_status'] = 'ok'
            row['peer_pin_available'] = address.is_loopback or (root / 'trusted' / (peer + '.json')).exists()
            if time.monotonic() >= deadline:
                row['error'] = 'Inventory deadline reached before model probe.'
                continue
            try:
                if em_port is None:
                    raise ValueError('PAIR model service is not advertised')
                if not address.is_loopback and not row['peer_pin_available']:
                    row['model_access_status'] = 'native_pairing_required'
                    raise ValueError('Missing exact peer pin; no remote model probe sent')
                use_pin = not address.is_loopback
                data = _get(str(address), em_port, '/v1/models', root, peer if use_pin else None)
                by_engine, loaded = data.get('modelsByEngine', {}), data.get('loadedByEngine', {})
                if not isinstance(by_engine, dict) or not isinstance(loaded, dict):
                    raise ValueError('Invalid PAIR per-engine inventory')
                flat = data.get('models', [])
                if not isinstance(flat, list) or any(not isinstance(m, str) for m in flat):
                    raise ValueError('Invalid PAIR model inventory')
                for model in flat:
                    owners = [e for e, values in by_engine.items() if isinstance(values, list) and model in values]
                    row['models'].append({'id': model, 'device': row['device'], 'installed': True,
                                         'loaded': any(isinstance(loaded.get(e), list) and model in loaded[e] for e in owners) if owners else None,
                                         'engines': owners, 'type': 'unknown', 'max_context_length': None,
                                         'management_supported': False, 'inference_supported': False})
                row['model_inventory_status'] = 'confirmed'
            except (OSError, ValueError, KeyError, TypeError, http.client.HTTPException):
                row['models'] = []
                row['model_inventory_status'] = 'unavailable'
                row['error'] = 'Native model inventory unavailable; inspect PAIR membership, pins and service ports.'
        except (OSError, http.client.HTTPException):
            if row is not None:
                row['online'] = False
                row['check_status'] = 'unreachable'
                row['error'] = 'PAIR node-info service is unreachable.'
        except (ValueError, KeyError, TypeError):
            if row is not None:
                row['online'] = None
                if row.get('check_status') == 'metadata_invalid':
                    row['error'] = 'PAIR roster metadata is invalid; no node probe sent.'
                else:
                    row['check_status'] = 'identity_unconfirmed'
                    row['error'] = 'PAIR node identity could not be confirmed.'
    return {'status': 'available' if nodes or roster_available else 'not_available', 'nodes': nodes,
            'network_discovery_status': discovery['status'],
            'notice': 'Automatic PAIR mDNS plus native roster fallback. Native IDs are read-only; unpaired peers may require native pairing before their models can be read.'}
