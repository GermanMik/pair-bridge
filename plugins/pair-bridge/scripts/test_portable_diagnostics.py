import unittest
from unittest.mock import patch
import diagnostics,server


class PortableDiagnosticTests(unittest.TestCase):
    def test_authorization_errors_are_unknown_online_and_redacted(self):
        for message in ('Device returned HTTP 401; private-secret http://private.example',
                        'Device returned HTTP 403',
                        'Configured device API token environment variable is missing',
                        'Configured Unsloth desktop credential file is unreadable'):
            result=diagnostics.device_failure(ValueError(message))
            self.assertIsNone(result['online']);self.assertEqual(result['auth_status'],'required')
            self.assertNotIn('private-secret',str(result));self.assertNotIn('private.example',str(result))

    def test_unknown_error_is_redacted_without_fake_offline(self):
        result=diagnostics.device_failure(ValueError('http://private.example token-secret invalid JSON'))
        self.assertIsNone(result['online']);self.assertNotIn('token-secret',str(result))

    def test_diagnose_redacts_discovery_and_keeps_auth_distinct(self):
        native={'status':'available','network_discovery_status':'active','nodes':[{
            'device':'pair-fixture','source':'PAIR mDNS','online':True,
            'ipAddress':'10.10.10.10','certPem':'private-material','models':[]}]}
        with patch.object(server.management,'devices',return_value={'fixture':{}}), \
             patch.object(server.management,'client',side_effect=ValueError('Device returned HTTP 401 secret-body')), \
             patch.object(server.cluster_inventory,'inspect',return_value=native), \
             patch.object(server,'catalog',return_value=[]), \
             patch.object(server.diagnostics,'recent_with_status',return_value=([],'not_created')):
            result=server.pair_diagnose()
        self.assertIsNone(result['devices'][0]['online'])
        self.assertEqual(result['discovery']['status'],'active')
        for value in ('secret-body','private-material','10.10.10.10'):self.assertNotIn(value,str(result))


if __name__=='__main__':unittest.main()
