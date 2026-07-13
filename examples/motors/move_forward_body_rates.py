# -*- coding: utf-8 -*-
"""
Move a Crazyflie forward using low-level thrust + body-rate control.

This example bypasses the position/attitude controllers and commands the
crazyflie directly with:
    - collective thrust (integer 0..65535)
    - roll rate, pitch rate, yaw rate (deg/s)

It does so with Commander.send_setpoint(roll, pitch, yawrate, thrust) while the
firmware roll/pitch stabilization mode is switched to RATE mode through the
`flightmode.stabModeRoll` / `flightmode.stabModePitch` parameters.

WARNING
-------
This is *manual* flight with no feedback controller keeping the vehicle level.
The thrust and rate values below are conservative starting points and will very
likely need tuning for your specific vehicle / battery. Always test in a safe,
open area (or in simulation) and be ready to cut power.
"""
import time

import cflib.crtp
from cflib.crazyflie import Crazyflie
from cflib.crazyflie.syncCrazyflie import SyncCrazyflie
from cflib.utils import uri_helper

# URI to the Crazyflie to connect to
# uri = uri_helper.uri_from_env(default='radio://0/80/2M/E7E7E7E7E7')
uri = uri_helper.uri_from_env(default='udp://127.0.0.1:19850')

# Firmware stabilization modes for send_setpoint's roll/pitch fields
STAB_MODE_RATE = 0
STAB_MODE_ANGLE = 1

# Control loop rate. Keep this high (>=100 Hz) so the rate loop stays fed and
# the commander watchdog never trips.
CONTROL_RATE_HZ = 100.0
CONTROL_DT = 1.0 / CONTROL_RATE_HZ

# Thrust levels (integer 0..65535).
#
# For the CrazySim MuJoCo cf2x_L250 model the sim maps thrust linearly:
#   thrust_per_motor [N] = (pwm / 65535) * 0.12
# The L250 mass is 0.0319 kg, so hover needs 0.0319*9.81/4 = 0.0782 N/motor,
# i.e. a hover PWM of ~42700. Commands below this will NOT leave the ground.
THRUST_HOVER = 42700     # slightly above hover -> gentle hold/slow climb
THRUST_TAKEOFF = 50000   # clearly above hover -> climbs
THRUST_DESCEND = 40000   # just below hover -> gentle descent

# Body-rate command to pitch nose-down and translate forward (deg/s)
FORWARD_PITCH_RATE = 8.0


def set_rate_mode(cf):
    """Switch roll/pitch to body-rate control, keep yaw as rate."""
    cf.param.set_value('flightmode.stabModeRoll', STAB_MODE_RATE)
    cf.param.set_value('flightmode.stabModePitch', STAB_MODE_RATE)


def restore_angle_mode(cf):
    """Restore the default roll/pitch angle stabilization mode."""
    cf.param.set_value('flightmode.stabModeRoll', STAB_MODE_ANGLE)
    cf.param.set_value('flightmode.stabModePitch', STAB_MODE_ANGLE)


def send_for(cf, duration_s, roll_rate, pitch_rate, yaw_rate, thrust):
    """Stream a constant thrust + body-rate setpoint for `duration_s` seconds."""
    steps = int(duration_s * CONTROL_RATE_HZ)
    for _ in range(steps):
        cf.commander.send_setpoint(roll_rate, pitch_rate, yaw_rate, thrust)
        time.sleep(CONTROL_DT)


def fly_forward(scf):
    cf = scf.cf

    # Modern firmware (and CrazySim SITL) require an explicit arm request before
    # the motors will spin. Without this the drone connects but stays inert.
    cf.supervisor.send_arming_request(True)
    time.sleep(1.0)

    # The first setpoint must be a zero-thrust one to unlock the commander.
    cf.commander.send_setpoint(0.0, 0.0, 0.0, 0)
    time.sleep(0.1)

    set_rate_mode(cf)
    time.sleep(0.1)

    try:
        # 1) Spin up / take off with a short thrust pulse, wings level.
        send_for(cf, 1.0, 0.0, 0.0, 0.0, THRUST_TAKEOFF)

        # 2) Move forward: pitch nose-down at a fixed body rate while hovering.
        send_for(cf, 1.0, 0.0, FORWARD_PITCH_RATE, 0.0, THRUST_HOVER)

        # 3) Hold a level hover indefinitely until the user stops the script
        #    (Ctrl+C). Setpoints must keep streaming or the commander watchdog
        #    will cut the motors.
        print('Hovering. Press Ctrl+C to stop.')
        while True:
            send_for(cf, 0.5, 0.0, 0.0, 0.0, THRUST_HOVER)

    except KeyboardInterrupt:
        print('\nStopping.')

    finally:
        # Cut motors and hand back control cleanly.
        cf.commander.send_stop_setpoint()
        cf.commander.send_notify_setpoint_stop()
        restore_angle_mode(cf)
        # cf.supervisor.send_arming_request(False)


def main():
    cflib.crtp.init_drivers()

    with SyncCrazyflie(uri, cf=Crazyflie(rw_cache='./cache')) as scf:
        fly_forward(scf)


if __name__ == '__main__':
    main()
