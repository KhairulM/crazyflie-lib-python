import time
import cflib.crtp
from cflib.crazyflie import Crazyflie

URI = 'radio://0/80/2M/E7E7E7E702'


def main():
    cflib.crtp.init_drivers()
    print("Drivers initialized. Connecting...")

    # Not using a cache here to force a fresh connection
    cf = Crazyflie(rw_cache=None)

    def _connected(link_uri):
        print(f"SUCCESS! Connected to {link_uri}")

    def _connection_failed(link_uri, msg):
        print(f"FAILED! {msg}")

    cf.connected.add_callback(_connected)
    cf.connection_failed.add_callback(_connection_failed)

    cf.open_link(URI)

    # Keep the script alive for a few seconds to let it connect
    time.sleep(5)
    cf.close_link()


if __name__ == '__main__':
    main()
