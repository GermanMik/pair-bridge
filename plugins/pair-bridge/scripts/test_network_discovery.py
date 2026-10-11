from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
import network_discovery as nd


class NetworkDiscoveryTests(unittest.TestCase):
    def setUp(self):
        self.records = patch.object(nd, '_records', {}).start()
        self.addCleanup(patch.stopall)
        self.peer = b'42590fa7-59fd-4480-8c59-bcfd94189eec'
        self.name = 'MacBook.' + nd.SERVICE

    def info(self, **props):
        data = {b'v':b'1',b'uuid':self.peer,b'ip':b'192.168.50.67',b'ni':b'14318',b'em':b'14322'}
        data.update({k.encode():v.encode() for k,v in props.items()})
        return SimpleNamespace(properties=data,parsed_addresses=lambda:['192.168.50.67'])

    def test_add_update_remove_without_manual_device_configuration(self):
        listener=nd.Listener();z=Mock()
        z.get_service_info.return_value=self.info()
        listener.add_service(z,nd.SERVICE,self.name)
        self.assertEqual(self.records[self.name]['ports']['em'],14322)
        z.get_service_info.return_value=self.info(ip='192.168.50.68',em='14422')
        listener.update_service(z,nd.SERVICE,self.name)
        self.assertEqual(self.records[self.name]['ipAddress'],'192.168.50.68')
        self.assertEqual(self.records[self.name]['ports']['em'],14422)
        listener.remove_service(z,nd.SERVICE,self.name)
        self.assertEqual(self.records,{})

    def test_no_model_names_in_txt_are_assumed(self):
        row=nd._record(self.info(),self.name)
        self.assertNotIn('models',row)

    def test_invalid_uuid_port_and_schema_are_ignored(self):
        for props in [dict(uuid='bad'),dict(em='65536'),dict(em='0'),dict(v='99')]:
            z=Mock();z.get_service_info.return_value=self.info(**props)
            nd.Listener().add_service(z,nd.SERVICE,self.name)
            self.assertEqual(self.records,{})

    def test_public_address_is_filtered(self):
        info=self.info(ip='8.8.8.8');info.parsed_addresses=lambda:[]
        with self.assertRaises(ValueError):nd._record(info,self.name)

    def test_non_authoritative_srv_port_is_never_used(self):
        info=self.info();info.properties.pop(b'em');info.port=9999
        self.assertNotIn('em',nd._record(info,self.name)['ports'])

    def test_snapshot_returns_copies(self):
        self.records[self.name]=nd._record(self.info(),self.name)
        with patch.object(nd,'_status','active'):
            row=nd.snapshot()['nodes'][0];row['ports']['em']=1
        self.assertEqual(self.records[self.name]['ports']['em'],14322)

    def test_failed_browser_start_recovers_after_cooldown(self):
        clock=[0]
        z,b=Mock(),Mock()
        with patch.object(nd,'_status','not_started'),patch.object(nd,'_refresh_at',0), \
             patch.object(nd,'_open_browser',side_effect=[OSError('interface down'),(z,b)]) as opener, \
             patch.object(nd.time,'monotonic',side_effect=lambda:clock[0]),patch.object(nd.time,'sleep'):
            self.assertEqual(nd.snapshot()['status'],'unavailable')
            nd.snapshot();self.assertEqual(opener.call_count,1)
            clock[0]=31
            self.assertEqual(nd.snapshot()['status'],'active')
            self.assertEqual(opener.call_count,2)
            nd.close()

    def test_active_browser_reopens_after_network_refresh_interval(self):
        clock=[0];z1,b1,z2,b2=Mock(),Mock(),Mock(),Mock()
        with patch.object(nd,'_status','not_started'),patch.object(nd,'_refresh_at',0), \
             patch.object(nd,'_open_browser',side_effect=[(z1,b1),(z2,b2)]), \
             patch.object(nd.time,'monotonic',side_effect=lambda:clock[0]),patch.object(nd.time,'sleep'):
            nd.snapshot()
            clock[0]=31;nd.snapshot()
            b1.cancel.assert_called_once();z1.close.assert_called_once()
            nd.close()

    def test_browser_factory_does_not_inherit_mcp_async_loop(self):
        import asyncio
        observed=[]
        def factory():
            try:
                asyncio.get_running_loop()
                observed.append('active-loop')
            except RuntimeError:
                observed.append('no-loop')
            return Mock(),Mock()
        async def invoke_like_sync_mcp_tool():
            return nd._open_browser()
        with patch.object(nd,'_create_browser',side_effect=factory):
            asyncio.run(invoke_like_sync_mcp_tool())
        self.assertEqual(observed,['no-loop'])


if __name__ == '__main__':unittest.main()
