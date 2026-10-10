# RoTino — workspace ROS 2

Simulazione e controllo di **RoTino**, un robot bipede su ruote (wheeled biped), in ROS 2 Humble e Gazebo Fortress.
Il workspace contiene due leggi di controllo intercambiabili e gli strumenti per confrontarle sugli stessi scenari:

- **PID** — PID in cascata con lo Zero Moment Point come variabile di equilibrio (`rotino_pid`);
- **MPC** — MPC sulla parte superiore + TV-LQR sulle ruote + VMC sulle gambe, da Cui et al., *Micromachines* 2022 (`rotino_mpc`).



## Package

| Package | Contenuto |
|---|---|
| `rotino_description` | URDF/xacro, mondo Gazebo, configurazione `ros2_control`, launch comune e libreria condivisa (modello, cinematica, ZMP, traiettoria planare) |
| `rotino_pid` | Controllore PID in cascata sullo ZMP, con supervisore di salto |
| `rotino_mpc` | Controllore MPC + TV-LQR + VMC, con stimatore di stato (filtro di Kalman) |
| `rotino_dashboard` | Dashboard in tempo reale (PyQt5 + pyqtgraph) con comandi manuali |
| `rotino_benchmark` | Campagne di prova, logger CSV, confronto, grafici e analisi ZMP |

## Dipendenze

Ubuntu 22.04, ROS 2 Humble, Gazebo Fortress, Python 3.10.

```bash
cd RoTino_ws
rosdep install --from-paths src --ignore-src -r -y --rosdistro humble
pip install pyqtgraph        # non è dichiarato nei package.xml: rosdep non lo installa
```

`rosdep` installa `ros_gz_sim`, `ros_gz_bridge`, `ros_gz_interfaces`, `gz_ros2_control`, `controller_manager`,
`joint_state_broadcaster`, `effort_controllers`, `xacro`, `robot_state_publisher`, `urdfdom_py`, `numpy`,
`matplotlib`, `PyQt5` e `pytest`. Il controllore MPC usa solo `numpy`: non serve alcun solver esterno.

Opzionali:

- PlotJuggler, per aprire i CSV del logger: `sudo apt install ros-humble-plotjuggler-ros`

## Compilazione

```bash
cd rotino_ws
colcon build --symlink-install
source install/setup.bash        # con zsh: source install/setup.zsh
```

Il `source` va ripetuto in ogni nuovo terminale.

## Launch file

| Launch file | Cosa avvia |
|---|---|
| `rotino_pid/launch/rotino_pid.launch.py` | Simulazione + controllore PID |
| `rotino_mpc/launch/rotino_mpc.launch.py` | Simulazione + controllore MPC |
| `rotino_description/launch/robot.launch.py` | Launch comune incluso dagli altri due: Gazebo, spawn del robot, bridge ROS–Gazebo, controller `ros2_control` e il controllore indicato con `controller_pkg` / `controller_exe` |

```bash
# Equilibrio da fermo con il PID
ros2 launch rotino_pid rotino_pid.launch.py

# Equilibrio da fermo con l'MPC
ros2 launch rotino_mpc rotino_mpc.launch.py

# Elenco di tutti gli argomenti
ros2 launch rotino_pid rotino_pid.launch.py --show-args
```

Il robot parte bloccato; il controllore lo rilascia da solo all'avvio e i
movimenti programmati cominciano 2 s dopo il rilascio.

`robot.launch.py` non ha un controllore predefinito e va lanciato indicandolo:

```bash
ros2 launch rotino_description robot.launch.py controller_pkg:=rotino_pid controller_exe:=controller
```

### Esempi

Gli argomenti sono gli stessi per le due leggi: basta cambiare `rotino_pid` con `rotino_mpc`.

```bash
# Con la dashboard
ros2 launch rotino_pid rotino_pid.launch.py dashboard:=true

# Senza interfaccia grafica di Gazebo
ros2 launch rotino_mpc rotino_mpc.launch.py gui:=false

# Spinta sul torso: 2,7 N s a 4 s dal rilascio
ros2 launch rotino_pid rotino_pid.launch.py push_enable:=true

# Spinta più forte e anticipata
ros2 launch rotino_mpc rotino_mpc.launch.py push_enable:=true push_impulse:=4.0 push_time:=3.0

# Profilo di velocità trapezoidale: 2 m a 1 m/s
ros2 launch rotino_pid rotino_pid.launch.py velocity_enable:=true velocity_max:=1.0

# Va e vieni di 1 m in 8 s
ros2 launch rotino_mpc rotino_mpc.launch.py drive_enable:=true

# Variazione sinusoidale dell'altezza
ros2 launch rotino_pid rotino_pid.launch.py height_enable:=true

# Curva a S
ros2 launch rotino_mpc rotino_mpc.launch.py planar_enable:=true

# Curva a S veloce, con il PID che si inclina in curva
ros2 launch rotino_pid rotino_pid.launch.py planar_enable:=true traj_duration:=5.0 traj_lateral:=1.0 zmp_lateral:=true

# Salto a 4 s dal rilascio
ros2 launch rotino_pid rotino_pid.launch.py jump_enable:=true

# Dossi: partenza davanti alla piattaforma e attraversamento a 0,5 m/s
ros2 launch rotino_pid rotino_pid.launch.py spawn_x:=4.5 velocity_enable:=true velocity_max:=0.5 velocity_distance:=4.5

# Il movimento programmato aspetta un comando di avvio
ros2 launch rotino_pid rotino_pid.launch.py velocity_enable:=true wait_start:=true
ros2 topic pub --once /rotino/cmd_start std_msgs/msg/Empty     # da un altro terminale
```

## Argomenti di lancio

Generali:

| Argomento | Default | Significato |
|---|---|---|
| `gui` | `true` | Interfaccia grafica di Gazebo |
| `dashboard` | `false` | Apre la dashboard `rotino_dashboard` |
| `spawn_x`, `spawn_y` | `0.0` | Posizione iniziale [m] |
| `spawn_yaw` | `0.0` | Orientamento iniziale [rad] |
| `start_controller` | `true` | `false` per avviare il controllore a mano (gli serve il parametro `robot_description`) |
| `controller_pkg`, `controller_exe` | — | Package ed eseguibile del controllore; già impostati da `rotino_pid.launch.py` e `rotino_mpc.launch.py` |
| `controller_name` | `rotino_controller` | Nome del nodo del controllore |

Scenari:

| Argomento | Default | Significato |
|---|---|---|
| `push_enable` | `false` | Impulso orizzontale sul torso |
| `push_time` | `4.0` | Istante dell'impulso dopo il rilascio [s] |
| `push_impulse` | `2.7` | Impulso [N s], positivo all'indietro |
| `velocity_enable` | `false` | Profilo di velocità trapezoidale |
| `velocity_max` | `0.44` | Velocità di crociera [m/s] |
| `accel_max` | `0.6` | Accelerazione e decelerazione [m/s²] |
| `velocity_distance` | `2.0` | Distanza totale [m] |
| `drive_enable` | `false` | Va e vieni durante l'equilibrio |
| `drive_distance` | `1.0` | Corsa del va e vieni [m] |
| `drive_period` | `8.0` | Durata di un ciclo [s] |
| `height_enable` | `false` | Variazione sinusoidale dell'altezza |
| `height_amplitude` | `0.03` | Ampiezza [m] |
| `height_period` | `2.2` | Periodo [s] |
| `planar_enable` | `false` | Traiettoria planare a S (polinomiale) |
| `traj_length` | `3.0` | Lunghezza dell'arco [m] |
| `traj_lateral` | `0.6` | Scarto laterale della S [m] |
| `traj_duration` | `12.0` | Durata della traiettoria [s] |
| `jump_enable` | `false` | Salto (accovacciamento, spinta, volo, atterraggio) |
| `jump_start_time` | `4.0` | Istante del salto dopo il rilascio [s] |
| `jump_velocity` | `1.2` | Velocità verticale del CoM a fine spinta [m/s] |
| `wait_start` | `false` | Il movimento programmato aspetta un messaggio su `/rotino/cmd_start` |
| `zmp_lateral` | `false` | Solo PID: inclinazione in curva per tenere centrato lo ZMP laterale (`docs/PID_ZMP.md`) |

## Comandi manuali

Con la simulazione avviata, entrambi i controllori accettano comandi sui topic `/rotino/cmd_*`.
Al primo `cmd_vel` o `cmd_height` il movimento programmato viene sostituito dal riferimento manuale.

```bash
# Velocità lineare e di imbardata: va ripubblicato, dopo 0,5 s senza messaggi il robot si ferma
ros2 topic pub -r 20 /rotino/cmd_vel geometry_msgs/msg/Twist "{linear: {x: 0.3}, angular: {z: 0.5}}"

# Altezza: scostamento dalla posa nominale, tra -0,05 e +0,04 m
ros2 topic pub --once /rotino/cmd_height std_msgs/msg/Float64 "{data: 0.03}"

# Salto
ros2 topic pub --once /rotino/cmd_jump std_msgs/msg/Empty

# Spinta sul torso [N s], al massimo 6 in modulo
ros2 topic pub --once /rotino/cmd_push std_msgs/msg/Float64 "{data: 2.7}"

# Avvio del movimento programmato (solo con wait_start:=true)
ros2 topic pub --once /rotino/cmd_start std_msgs/msg/Empty
```

La velocità lineare è limitata a 1,5 m/s con il PID e a 0,6 m/s con l'MPC; quella di imbardata a 1,5 rad/s.

Topic utili da osservare:

| Topic | Contenuto |
|---|---|
| `/rotino/debug` | Tempo dal rilascio, beccheggio, posizione e velocità, comando alle ruote, altezza del CoM, forza di contatto |
| `/rotino/wbr_state` | Stati e riferimenti (posizione, beccheggio, direzione, velocità, altezza) e coppie comandate |
| `/rotino/tracking_error` | Errore di posizione x, y sulla traiettoria planare |
| `/rotino/jump_state` | Fase del supervisore di salto |
| `/rotino/planar` | Riferimento, posa ed errori lungo la traiettoria planare |
| `/rotino/zmp_ctrl` | Solo PID: ZMP desiderato e misurato, longitudinale e laterale |
| `/rotino/estimation_error` | Errore dello stimatore dell'MPC |
| `/rotino/odom`, `/rotino/imu`, `/joint_states` | Odometria, IMU e giunti |
| `/wheel_effort_controller/commands`, `/leg_effort_controller/commands` | Coppie comandate [Nm] |

## Dashboard

```bash
ros2 run rotino_dashboard dashboard
```

Oppure `dashboard:=true` in un launch file. Mostra LED di fase e di diagnostica, uno schema sagittale del robot
con le forze, i valori correnti e una griglia di grafici scorrevoli. Il pannello *Comandi manuali* pubblica su
`/rotino/cmd_vel`, `/rotino/cmd_height`, `/rotino/cmd_jump` e `/rotino/cmd_push`.

## Benchmark PID vs MPC

Ogni prova usa lo stesso mondo, lo stesso URDF e gli stessi argomenti di lancio: cambia solo il controllore.
Le simulazioni girano senza interfaccia grafica, una legge dopo l'altra.

> **Attenzione:** prima di ogni prova `campaign` e `suite` chiudono i processi rimasti aperti (Gazebo,
> controllori, dashboard, bridge, logger). Non lanciarli mentre è in corso una simulazione che vuoi tenere.

### Suite completa

```bash
ros2 run rotino_benchmark suite                          # tutti gli scenari, circa 10 minuti
ros2 run rotino_benchmark suite -- spinta trapezio       # solo alcuni scenari
ros2 run rotino_benchmark suite -- --list                # elenco degli scenari
ros2 run rotino_benchmark suite -- --controllers pid     # una sola legge
ros2 run rotino_benchmark suite -- --name mia_suite      # nome della cartella di uscita
ros2 run rotino_benchmark suite -- --analisi benchmark_runs/suite_20261006_1353   # rifà solo il confronto
```

### Singolo scenario

```bash
ros2 run rotino_benchmark campaign -- spinta                                   # scenario del catalogo
ros2 run rotino_benchmark campaign -- --scenario push_enable:=true push_impulse:=4.0 --duration 20
ros2 run rotino_benchmark campaign -- --gradino 5.0 --gradino-t 4.0            # forza costante di 5 N da 4 s
ros2 run rotino_benchmark campaign -- trapezio --topic-extra                   # anche l'errore dello stimatore MPC
ros2 run rotino_benchmark campaign -- salto --controllers mpc --no-analysis    # solo i CSV di una legge
```

Opzioni di `campaign`: `--scenario`, `--controllers`, `--gradino`, `--gradino-t`, `--duration`, `--run-name`,
`--out`, `--no-analysis`, `--topic-extra`.

`--topic-extra` aggiunge un topic al logger e carica sia il logger sia il controllore: conviene usarlo in una
prova separata e tenere quella normale per le metriche.

### Scenari del catalogo

| Nome | Durata | Prova |
|---|---|---|
| `equilibrio` | 15 s | Rilascio dall'ancora ed equilibrio sul posto |
| `spinta` | 20 s | Impulso di 2,7 N s all'indietro sul torso a 4 s dal rilascio |
| `gradino` | 20 s | Forza costante di 3 N all'indietro sul torso da 4 s, mantenuta fino alla fine |
| `trapezio` | 20 s | 2 m in avanti con profilo trapezoidale a 1 m/s |
| `va_e_vieni` | 20 s | 1 m avanti e indietro in 8 s |
| `altezza` | 15 s | Altezza sinusoidale di ±3 cm con periodo 2,2 s |
| `curva_S` | 25 s | S di 3 m con scarto laterale di 0,6 m in 12 s |
| `curva_S_veloce` | 15 s | S di 3 m con scarto laterale di 1 m in 5 s |
| `dossi` | 22 s | Tre dossi alti 2 cm attraversati a 0,5 m/s |
| `rampa` | 22 s | Rampa al 5 % fino a un ripiano alto 3 cm, poi discesa |
| `piastrelle` | 22 s | Campo di piastrelle alte 5–19 mm, diverse sotto le due ruote |
| `salto` | 15 s | Salto verticale a 4 s dal rilascio, poi equilibrio |

Gli scenari sono definiti in `src/rotino_benchmark/rotino_benchmark/scenarios.py`.

### Analisi di prove già fatte

Questi comandi leggono solo i CSV e non avviano la simulazione.

```bash
ros2 run rotino_benchmark compare -- <cartella_scenario>      # tabella PID | MPC | differenza | migliore
ros2 run rotino_benchmark compare -- <cartella_suite>         # tutti gli scenari e il riepilogo
ros2 run rotino_benchmark compare -- <cartella> --ricalcola   # ignora metriche.json, rifà grafici e ZMP
ros2 run rotino_benchmark compare -- <cartella> --no-plots    # solo le tabelle

ros2 run rotino_benchmark plot -- <cartella_scenario>         # grafici di confronto in plots/
ros2 run rotino_benchmark zmp -- <cartella_scenario>          # analisi dello ZMP
ros2 run rotino_benchmark export -- <cartella>                # CSV per tema in dati/
python3 -m rotino_benchmark.slides <cartella>                 # figure 16:9 per le slide in slide/
```

`slides` non è registrato come eseguibile di `ros2 run`: si lancia come modulo Python, dopo il `source`.

### Risultati

I risultati finiscono in `benchmark_runs/` (cartella ignorata da git):

```text
benchmark_runs/suite_<data>/
├── riepilogo.md / .csv / .png      metriche principali di ogni scenario e bilancio delle vittorie
└── <scenario>/
    ├── confronto.md / .csv         tabella completa dello scenario
    ├── metriche.json               metriche in cache
    ├── scenario.json               cosa è stato eseguito
    ├── rotino_<legge>_<data>.csv   log a 500 Hz, uno per legge
    ├── plots/                      figure di confronto (PID rosso, MPC blu)
    ├── dati/                       errori, riferimenti, stati e attuatori per tema
    └── logs/                       uscita della simulazione, un file per legge
```

Una prova singola di `campaign` crea `benchmark_runs/<scenario>_<data>/` con lo stesso contenuto di una
cartella di scenario.

### Logger e disturbo a mano

`campaign` li avvia da solo; servono solo per registrare una simulazione lanciata a mano.

```bash
# Logger: una riga CSV per passo di controllo (500 Hz)
ros2 run rotino_benchmark logger --ros-args -p use_sim_time:=true -p controller:=pid -p output_dir:=/tmp/prova

# Forza costante sul torso: 3 N all'indietro da 4 s dopo il rilascio
ros2 run rotino_benchmark disturbance --ros-args -p use_sim_time:=true -p force:=3.0 -p start_time:=4.0
```

Parametri del logger: `controller` (nome della legge nel file), `output_dir` (default `test_bench_logs/` nel
workspace), `extra_topics` (registra anche `/rotino/estimation_error`).
Parametri del disturbo: `force` [N], `start_time` [s], `duration` [s] (`0` = fino alla fine).

I CSV si aprono con PlotJuggler usando i layout già pronti:

```bash
ros2 run plotjuggler plotjuggler
```

- `src/rotino_benchmark/rotino_benchmark_csv_layout.xml` per i CSV del logger
- `src/rotino_description/config/rotino_plotjuggler_layout.xml` per i topic in tempo reale

## Altri strumenti

```bash
# Tabella dei parametri del modello (Tabella 1 di Cui et al.) calcolati dall'URDF
ros2 run rotino_description model

# URDF generato dallo xacro
xacro src/rotino_description/urdf/rotino.urdf.xacro \
    controllers_file:=$PWD/src/rotino_description/config/rotino_controllers.yaml

# Stato dei controller ros2_control, a simulazione avviata
ros2 control list_controllers
```

## Test

```bash
colcon test --packages-select rotino_pid rotino_benchmark
colcon test-result --verbose

# oppure direttamente con pytest, dopo il source
python3 -m pytest src/rotino_pid/test src/rotino_benchmark/test -q
```



## Problemi noti

- **Processi rimasti aperti.** Un Gazebo, un controllore o una dashboard di una prova precedente continuano a
  pubblicare su `/rotino/*` e falsano la prova successiva. Per ripulire:

  ```bash
  pkill -f "ign gazebo"; pkill -f parameter_bridge; pkill -f robot_state_publisher
  pkill -f rotino_pid/controller; pkill -f rotino_mpc/controller; pkill -f rotino_dashboard
  ```

- **`Package 'rotino_...' not found`.** Manca il `source install/setup.bash` (o `setup.zsh`) nel terminale.
- **Dashboard che non parte.** Manca `pyqtgraph`: `pip install pyqtgraph`.
- **Avviso `Unable to import Axes3D` di matplotlib.** Compare quando matplotlib è installato sia da `apt` sia
  da `pip`; i grafici del benchmark vengono prodotti lo stesso


  
