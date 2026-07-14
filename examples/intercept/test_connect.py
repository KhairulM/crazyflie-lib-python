import logging
import time

import cflib.crtp
from cflib.crazyflie import Crazyflie
from cflib.crazyflie.log import LogConfig
from cflib.crazyflie.syncCrazyflie import SyncCrazyflie
from cflib.crazyflie.syncLogger import SyncLogger
from cflib.utils import uri_helper

uri = uri_helper.uri_from_env(default='radio://0/40/2M/E7E7E7E702')
logging.basicConfig(level=logging.ERROR)


def param_stab_est_callback(name, value):
    print('The crazyflie has parameter ' + name + ' set at number: ' + value)


def simple_param_async(scf, groupstr, namestr):
    cf = scf.cf
    full_name = groupstr + '.' + namestr

    cf.param.add_update_callback(group=groupstr, name=namestr,
                                 cb=param_stab_est_callback)
    time.sleep(1)
    cf.param.set_value(full_name, 2)
    time.sleep(1)
    cf.param.set_value(full_name, 1)
    time.sleep(1)


def simple_log_async(scf, logconf):
    cf = scf.cf
    cf.log.add_config(logconf)

    def log_stab_callback(timestamp, data, logconf_name):
        print(f'{timestamp} [{logconf_name}]: {data}')

    logconf.data_received_cb.add_callback(log_stab_callback)
    logconf.start()
    time.sleep(5)
    logconf.stop()


def simple_logging(scf, logconf):
    with SyncLogger(scf, logconf) as logger:
        for logentry in logger:
            timestamp = logentry[0]
            data = logentry[1]
            logconf_name = logentry[2]

            print(f'{timestamp} [{logconf_name}]: {data}')


def simple_connect():
    print("Yeah, I'm connected! :D")
    time.sleep(3)
    print("Now I will disconnect :'(")


def main():
    cflib.crtp.init_drivers()
    lg_stab = LogConfig(name='Stabilizer', period_in_ms=10)
    lg_stab.add_variable('stabilizer.roll', 'float')
    lg_stab.add_variable('stabilizer.pitch', 'float')
    lg_stab.add_variable('stabilizer.yaw', 'float')

    print('Drivers initialized. Connecting...')

    with SyncCrazyflie(uri, cf=Crazyflie(rw_cache=None)) as scf:
        # simple_connect()
        # simple_logging(scf, lg_stab)
        simple_log_async(scf, lg_stab)
        # simple_param_async(scf, 'stabilizer', 'estimator')

    # def _connected(link_uri):
    #     print(f'SUCCESS! Connected to {link_uri}')

    # def _connection_failed(link_uri, msg):
    #     print(f'FAILED! {msg}')

    # cf.connected.add_callback(_connected)
    # cf.connection_failed.add_callback(_connection_failed)

    # cf.open_link(URI)

    # # Keep the script alive for a few seconds to let it connect
    # time.sleep(5)
    # cf.close_link()


if __name__ == '__main__':
    main()
