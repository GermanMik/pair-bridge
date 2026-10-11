"""Stock publisher wire fixtures: no NVIDIA process, UDP5353 or LAN dependency."""
import ipaddress
import struct
import threading
import unittest
from unittest.mock import patch
from zeroconf import Zeroconf, IPVersion
from zeroconf._protocol.incoming import DNSIncoming
import network_discovery as nd


def name(labels):
    return b''.join(bytes([len(label.encode())])+label.encode() for label in labels)+b'\0'


def stock_packet(dotted=True, schema='1'):
    service=['_nvpair-node','_tcp','local']
    # Stock Go builds dotted hostname into two instance labels and appends local again.
    instance=(['Fixture-Mac','local'] if dotted else ['Fixture-PC'])+service
    host=['Fixture-Mac','local','local'] if dotted else ['Fixture-PC','local']
    txt=['v='+schema,'uuid=00000000-0000-4000-8000-000000000001','ip=10.10.10.10','ni=14318','em=14322']
    def rr(owner,kind,data):return name(owner)+struct.pack('!HHIH',kind,1,120,len(data))+data
    wire=struct.pack('!HHHHHH',0,0x8400,0,4,0,0)
    wire+=rr(service,12,name(instance))
    wire+=rr(instance,33,struct.pack('!HHH',0,0,14318)+name(host))
    wire+=rr(instance,16,b''.join(bytes([len(t)])+t.encode() for t in txt))
    wire+=rr(host,1,ipaddress.ip_address('10.10.10.10').packed)
    return wire,'.'.join(instance)+'.'


class StockDiscoveryTests(unittest.TestCase):
    def test_real_decoder_resolves_standard_and_stock_dotted_publisher(self):
        for dotted in (False,True):
            with self.subTest(dotted=dotted):
                z=Zeroconf(interfaces=['127.0.0.1'],ip_version=IPVersion.V4Only,unicast=True)
                try:
                    wire,instance=stock_packet(dotted)
                    incoming=DNSIncoming(wire);self.assertTrue(incoming.valid)
                    done=threading.Event()
                    def ingest():
                        z.record_manager.async_updates_from_response(incoming);done.set()
                    z.loop.call_soon_threadsafe(ingest)
                    self.assertTrue(done.wait(3))
                    # Callback consumes real decoded PTR/SRV/TXT/A from cache,
                    # including stock duplicate-local SRV target, without rewriting publisher.
                    with patch.object(nd,'_records',{}):
                        nd.Listener().add_service(z,nd.SERVICE,instance)
                        row=nd._records[instance]
                        self.assertEqual(row['nodeUuid'],'00000000-0000-4000-8000-000000000001')
                        self.assertEqual(row['ports'],{'ni':14318,'em':14322})
                        self.assertEqual(row['discovery_source'],'PAIR mDNS')
                finally:z.close()

    def test_invalid_schema_is_not_admitted(self):
        wire,instance=stock_packet(schema='99');msg=DNSIncoming(wire)
        self.assertTrue(msg.valid)
        # Existing listener's schema/UUID/address gates remain authoritative.
        from types import SimpleNamespace
        fake=SimpleNamespace(properties={b'v':b'99'},parsed_addresses=lambda:[])
        with self.assertRaises(ValueError):nd._record(fake,instance)


if __name__=='__main__':unittest.main()
