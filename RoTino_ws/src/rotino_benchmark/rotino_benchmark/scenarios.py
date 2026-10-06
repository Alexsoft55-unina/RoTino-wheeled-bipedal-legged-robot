"""The benchmark scenarios: one place that says what is run, for how long, and what to look at.

Every scenario is run with the same world, URDF and launch arguments for the PID and the MPC; only the
controller package changes. `key` lists the metrics (compare.METRICS) that summarise the scenario.

Disturbances: `spinta` is the impulsive push applied by the controllers; `gradino` is a constant force
switched on and held (disturbance.py, same for both laws); `dossi`, `rampa` and `piastrelle` cross the
three uneven-ground platforms of rotino_world.sdf. The robot spawns in front of each platform
(spawn_x/spawn_y/spawn_yaw of robot.launch.py) and drives straight across it with the trapezoidal
profile; `zones` gives where the obstacles are along the path, in metres from the start.
"""

from collections import OrderedDict
from dataclasses import dataclass, field


@dataclass
class Scenario:
    title: str
    description: str
    args: list = field(default_factory=list)   # launch arguments of robot.launch.py
    duration: float = 20.0                      # s of logging after the logger starts
    key: tuple = ()
    disturbance: dict = None                    # step force: {'force': N (+ = backwards), 'start_time': s}
    zones: tuple = ()                           # obstacles along the path: (from m, to m, label)


# Trapezoid used on the platforms: same speed and acceleration for both laws, slow enough to keep the
# wheels on the obstacles for a few control periods but fast enough to load them dynamically.
PLATFORM_DRIVE = ['velocity_enable:=true', 'velocity_max:=0.5', 'accel_max:=0.6']


SCENARIOS = OrderedDict([
    ('equilibrio', Scenario(
        'Equilibrio da fermo', 'Rilascio dall\'ancora e equilibrio sul posto.',
        [], 15.0, ('pitch_rms_deg', 'pos_err_rms_m', 'wheel_tau_rms_Nm', 'chatter_Nm'))),
    ('spinta', Scenario(
        'Spinta sul torso', 'Impulso di 2,7 N s all\'indietro sul torso a 4 s dal rilascio.',
        ['push_enable:=true'], 20.0,
        ('pitch_peak_deg', 'recovery_s', 'pos_err_max_m', 'dist_energy_J', 'wheel_load_min_N'))),
    ('gradino', Scenario(
        'Disturbo a gradino', 'Forza costante di 3 N all\'indietro sul torso da 4 s dopo il rilascio, mantenuta '
        'fino alla fine: mostra il nuovo equilibrio e l\'errore di posizione a regime.',
        [], 20.0,
        ('pitch_peak_deg', 'settle_ss_s', 'pos_err_ss_m', 'pos_err_max_m', 'pitch_ss_deg'),
        disturbance={'force': 3.0, 'start_time': 4.0})),
    ('trapezio', Scenario(
        'Trapezio di velocita', '2 m in avanti con profilo trapezoidale: 1 m/s, 0,6 m/s^2.',
        ['velocity_enable:=true', 'velocity_max:=1.0'], 20.0,
        ('pos_err_rms_m', 'vel_err_rms_ms', 'pitch_peak_deg', 'recovery_s'))),
    ('va_e_vieni', Scenario(
        'Va e vieni', '1 m avanti e indietro in 8 s (riferimento sinusoidale).',
        ['drive_enable:=true'], 20.0,
        ('pos_err_rms_m', 'vel_err_rms_ms', 'pitch_peak_deg', 'wheel_tau_rms_Nm'))),
    ('altezza', Scenario(
        'Variazione di altezza', 'Altezza sinusoidale +-3 cm, periodo 2,2 s, da fermo.',
        ['height_enable:=true'], 15.0,
        ('pitch_rms_deg', 'pos_err_rms_m', 'height_rms_mm', 'wheel_tau_rms_Nm'))),
    ('curva_S', Scenario(
        'Curva a S', 'S di 3 m con scarto laterale di 0,6 m in 12 s.',
        ['planar_enable:=true'], 25.0,
        ('pos_err_rms_m', 'pitch_peak_deg', 'zmp_lat_ratio_pct', 'wheel_load_min_N'))),
    ('curva_S_veloce', Scenario(
        'Curva a S veloce', 'S di 3 m con scarto laterale di 1 m in 5 s.',
        ['planar_enable:=true', 'traj_duration:=5.0', 'traj_lateral:=1.0'], 15.0,
        ('pos_err_rms_m', 'pitch_peak_deg', 'zmp_lat_ratio_pct', 'wheel_load_min_N'))),
    ('dossi', Scenario(
        'Piattaforma: dossi', 'Tre dossi alti 2 cm (x = 6,0; 6,8; 7,6 m), attraversati in avanti a 0,5 m/s '
        'partendo da x = 4,5 m.',
        ['spawn_x:=4.5', *PLATFORM_DRIVE, 'velocity_distance:=4.5'], 22.0,
        ('zone_pitch_peak_deg', 'zone_tau_peak_Nm', 'zone_com_dev_mm', 'pitch_err_rms_deg', 'vel_err_rms_ms'),
        zones=((1.44, 1.56, 'dosso'), (2.24, 2.36, 'dosso'), (3.04, 3.16, 'dosso')))),
    ('rampa', Scenario(
        'Piattaforma: rampa', 'Rampa al 5 % fino a un ripiano alto 3 cm e discesa (x da -5,0 a -8,0 m), '
        'attraversata a 0,5 m/s partendo da x = -3,5 m in direzione -x.',
        ['spawn_x:=-3.5', 'spawn_yaw:=3.14159', *PLATFORM_DRIVE, 'velocity_distance:=5.5'], 22.0,
        ('zone_pitch_peak_deg', 'zone_tau_peak_Nm', 'pos_err_max_m', 'vel_err_rms_ms', 'pitch_err_rms_deg'),
        zones=((1.5, 2.1, 'salita'), (2.1, 3.9, 'ripiano'), (3.9, 4.5, 'discesa')))),
    ('piastrelle', Scenario(
        'Piattaforma: piastrelle', 'Campo di piastrelle 24x24 cm alte 5-19 mm (y da -4,1 a -6,1 m): le due ruote '
        'passano su colonne di piastrelle diverse. Attraversato a 0,5 m/s partendo da (4,875; -2,6) in direzione -y.',
        ['spawn_x:=4.875', 'spawn_y:=-2.6', 'spawn_yaw:=-1.5708', *PLATFORM_DRIVE, 'velocity_distance:=5.0'], 22.0,
        ('roll_peak_deg', 'yaw_dev_max_deg', 'zone_pitch_peak_deg', 'zone_com_dev_mm', 'pos_err_rms_m'),
        zones=((1.53, 3.52, 'piastrelle'),))),
    ('salto', Scenario(
        'Salto', 'Salto verticale a 4 s dal rilascio, poi equilibrio.',
        ['jump_enable:=true'], 15.0,
        ('com_rise_mm', 'pitch_peak_deg', 'recovery_s', 'pos_err_max_m'))),
])
