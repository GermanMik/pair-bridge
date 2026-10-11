import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import cluster_inventory as ci


class ClusterInventoryTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.addCleanup(patch.stopall)
        patch.object(ci, 'cluster_dir', return_value=self.root).start()
        patch.object(ci.network_discovery, 'snapshot', return_value={'status':'active','nodes':[]}).start()
        self.peer = '42590fa7-59fd-4480-8c59-bcfd94189eec'

    def roster(self, **kwargs):
        member = dict(nodeUuid=self.peer, name='MacBook Pro', ipAddress='192.168.50.67', state='member')
        member.update(kwargs)
        (self.root / 'members.json').write_text(json.dumps([member]))

    def test_missing_roster(self):
        self.assertEqual(ci.inspect()['status'], 'not_available')

    def test_remote_inventory_preserves_unknown_metadata(self):
        self.roster()
        (self.root / 'trusted').mkdir()
        (self.root / 'trusted' / (self.peer + '.json')).write_text('{}')
        with patch.object(ci, '_get', side_effect=[{'hostUuid': self.peer}, {'models': ['chat', 'embed'], 'modelsByEngine': {'lmstudio': ['chat','embed']}, 'loadedByEngine': {'lmstudio':['chat']}}]) as get:
            row = ci.inspect()['nodes'][0]
        self.assertTrue(row['online'])
        self.assertEqual([m['loaded'] for m in row['models']], [True, False])
        self.assertIsNone(row['models'][0]['max_context_length'])
        self.assertFalse(row['inference_supported'])
        self.assertEqual(get.call_args.args[-1], self.peer)

    def test_identity_mismatch_stops_model_request(self):
        self.roster()
        with patch.object(ci, '_get', return_value={'hostUuid':'different'}) as get:
            row = ci.inspect()['nodes'][0]
        self.assertIsNone(row['online'])
        self.assertEqual(row['check_status'],'identity_unconfirmed')
        self.assertEqual(get.call_count, 1)

    def test_model_failure_does_not_hide_live_node(self):
        self.roster()
        with patch.object(ci, '_get', side_effect=[{'hostUuid': self.peer}, OSError('private error')]):
            row = ci.inspect()['nodes'][0]
        self.assertTrue(row['online'])
        self.assertEqual(row['model_inventory_status'],'unavailable')
        self.assertNotIn('private error',str(row))

    def test_offline_node_is_retained(self):
        self.roster()
        with patch.object(ci, '_get', side_effect=OSError):
            self.assertFalse(ci.inspect()['nodes'][0]['online'])

    def test_invalid_and_public_roster_addresses_not_contacted(self):
        for host in ['8.8.8.8','0.0.0.0','224.0.0.251']:
            self.roster(ipAddress=host)
            with patch.object(ci, '_get') as get:
                ci.inspect()
                get.assert_not_called()

    def test_unknown_loaded_status_without_engine_attribution(self):
        self.roster()
        (self.root/'trusted').mkdir()
        (self.root/'trusted'/f'{self.peer}.json').write_text('{}')
        with patch.object(ci,'_get',side_effect=[{'hostUuid':self.peer},{'models':['chat']}]):
            self.assertIsNone(ci.inspect()['nodes'][0]['models'][0]['loaded'])

    def test_invalid_roster(self):
        (self.root/'members.json').write_text('{}')
        self.assertEqual(ci.inspect()['status'],'invalid_metadata')

    def test_pin_identity_mismatch_before_network(self):
        (self.root/'trusted').mkdir()
        (self.root/'trusted'/f'{self.peer}.json').write_text(json.dumps({'nodeUuid':'wrong'}))
        with patch.object(ci.http.client,'HTTPSConnection') as connection:
            with self.assertRaises(ValueError):
                ci._get('192.168.50.67',14322,'/v1/models',self.root,self.peer)
            connection.assert_not_called()

    def test_exact_certificate_mismatch_closes_before_http(self):
        from unittest.mock import MagicMock
        connection = MagicMock()
        connection.sock.getpeercert.return_value = b'different-certificate'
        with patch.object(ci, '_read', return_value={'nodeUuid': self.peer, 'certPem': 'public-pin'}), \
             patch.object(ci.ssl, 'PEM_cert_to_DER_cert', return_value=b'pinned-certificate'), \
             patch.object(ci.ssl, 'SSLContext'), \
             patch.object(ci.http.client, 'HTTPSConnection', return_value=connection):
            with self.assertRaises(ValueError):
                ci._get('192.168.50.67',14322,'/v1/models',self.root,self.peer)
        connection.request.assert_not_called()
        connection.close.assert_called_once()

    def test_discovered_node_without_roster_uses_advertised_ports(self):
        (self.root/'trusted').mkdir()
        (self.root/'trusted'/f'{self.peer}.json').write_text('{}')
        node=dict(nodeUuid=self.peer,name='MacBook',ipAddress='192.168.50.67',state='member',
                  ports={'ni':15318,'em':15322},discovery_source='PAIR mDNS',network_present=True)
        with patch.object(ci.network_discovery,'snapshot',return_value={'status':'active','nodes':[node]}), \
             patch.object(ci,'_get',side_effect=[{'hostUuid':self.peer},{'models':['new-model']}]) as get:
            row=ci.inspect()['nodes'][0]
        self.assertEqual(row['source'],'PAIR mDNS')
        self.assertTrue(row['peer_pin_available'])
        self.assertEqual(get.call_args_list[0].args[1],15318)
        self.assertEqual(get.call_args_list[1].args[1],15322)
        self.assertEqual(row['models'][0]['id'],'new-model')

    def test_discovered_address_supersedes_stale_roster(self):
        self.roster(ipAddress='192.168.50.66')
        node=dict(nodeUuid=self.peer,name='MacBook',ipAddress='192.168.50.68',state='member',ports={'ni':14318,'em':14322})
        with patch.object(ci.network_discovery,'snapshot',return_value={'status':'active','nodes':[node]}), \
             patch.object(ci,'_get',side_effect=[{'hostUuid':self.peer},{'models':[]}]) as get:
            self.assertEqual(len(ci.inspect()['nodes']),1)
        self.assertEqual(get.call_args_list[0].args[0],'192.168.50.68')

    def test_configured_pair_prefix_device_keeps_management_contract(self):
        import server
        with patch.object(server.management,'devices',return_value={'pair-lab':{'engine':'lmstudio'}}), \
             patch.object(server.management,'client'), \
             patch.object(server.management,'models',return_value=[]), \
             patch.object(server.cluster_inventory,'inspect',side_effect=AssertionError('Native inventory must not shadow configured IDs')):
            result=server.pair_list('pair-lab')
        self.assertEqual(result['device'],'pair-lab')
        self.assertEqual(result['source'],'lmstudio model API')

    def test_router_503_does_not_abort_catalog_or_disclose_error_body(self):
        import server
        with patch.object(server,'catalog',side_effect=ValueError('PAIR returned HTTP 503; private-token-detail')), \
             patch.object(server.management,'devices',return_value={}), \
             patch.object(server.cluster_inventory,'inspect',return_value={'status':'active','nodes':[]}):
            result=server.pair_list()
        self.assertEqual(result['router_status'],'unavailable')
        self.assertEqual(result['router_http_status'],503)
        self.assertNotIn('private-token-detail',str(result))

    def test_secondary_advertised_address_survives_stale_primary(self):
        self.roster(ipAddress='192.168.50.66',addresses=['192.168.50.67'])
        (self.root/'trusted').mkdir()
        (self.root/'trusted'/f'{self.peer}.json').write_text('{}')
        with patch.object(ci,'_get',side_effect=[OSError('old address'),{'hostUuid':self.peer},{'models':['chat']}]) as get:
            row=ci.inspect()['nodes'][0]
        self.assertTrue(row['online'])
        self.assertEqual(get.call_args.args[0],'192.168.50.67')

    def test_unpaired_remote_never_downgrades_to_plain_model_request(self):
        self.roster()
        with patch.object(ci,'_get',return_value={'hostUuid':self.peer}) as get:
            row=ci.inspect()['nodes'][0]
        self.assertTrue(row['online'])
        self.assertEqual(row['model_access_status'],'native_pairing_required')
        self.assertEqual(get.call_count,1)

    def test_malformed_roster_node_is_visible_without_probe(self):
        self.roster(ipAddress='broken')
        with patch.object(ci,'_get') as get:
            row=ci.inspect()['nodes'][0]
        get.assert_not_called()
        self.assertEqual(row['check_status'],'metadata_invalid')


if __name__ == '__main__':
    unittest.main()
