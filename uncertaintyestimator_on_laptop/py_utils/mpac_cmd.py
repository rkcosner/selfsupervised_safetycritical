import socket
import numpy as np
import atexit
import threading

UDP_IP = "10.8.56.225"#"10.8.38.125"#"10.8.54.172" #"192.168.12.227" #"127.0.0.1"
CTRL_UDP_PORT = 8081
ATNMY_UDP_PORT = 8082

print("UDP target IP:", UDP_IP)
print("UDP target port:", CTRL_UDP_PORT)
print("UDP receive port:", ATNMY_UDP_PORT)

sock = socket.socket(socket.AF_INET, # Internet
                     socket.SOCK_DGRAM) # UDP
# sock.bind((UDP_IP, ATNMY_UDP_PORT))

def hard_stop():
  dt = np.dtype([('mode', np.int)])
  x = np.array(0, dtype=dt)
  sock.sendto(x.tobytes(), (UDP_IP, CTRL_UDP_PORT))

def soft_stop():
  dt = np.dtype([('mode', np.int)])
  x = np.array(1, dtype=dt)
  sock.sendto(x.tobytes(), (UDP_IP, CTRL_UDP_PORT))

def lie():
  dt = np.dtype([('mode', np.int)])
  x = np.array(2, dtype=dt)
  sock.sendto(x.tobytes(), (UDP_IP, CTRL_UDP_PORT))
   
def stand_idqp(h=0.25, rx=0, ry=0, rz=0):
  dt = np.dtype([('mode', np.int),
                 ('h', np.double),
                 ('rx', np.double),
                 ('ry', np.double),
                 ('rz', np.double)])
  x = np.array((5,h,rx,ry,rz), dtype=dt)
  sock.sendto(x.tobytes(), (UDP_IP, CTRL_UDP_PORT))
   
def walk_mpc_idqp(h=0.25, vx=0, vy=0, vrz=0):
  dt = np.dtype([('mode', np.int),
                 ('h', np.double),
                 ('vx', np.double),
                 ('vy', np.double),
                 ('vrz', np.double)])
  x = np.array((8,h,vx,vy,vrz), dtype=dt)
  sock.sendto(x.tobytes(), (UDP_IP, CTRL_UDP_PORT))
  print("Sending info vx = ", vx, ' vrz = ', vrz)
   
def walk_quasi_idqp(h=0.25, vx=0, vy=0, vrz=0):
  dt = np.dtype([('mode', np.int),
                 ('h', np.double),
                 ('vx', np.double),
                 ('vy', np.double),
                 ('vrz', np.double)])
  x = np.array((9,h,vx,vy,vrz), dtype=dt)
  sock.sendto(x.tobytes(), (UDP_IP, CTRL_UDP_PORT))
   
def jump(z_vel=2):
  dt = np.dtype([('mode', np.int),
                 ('z_vel', np.double)])
  x = np.array((10,z_vel), dtype=dt)
  sock.sendto(x.tobytes(), (UDP_IP, CTRL_UDP_PORT))
   
def traj_track(filename = "../ctrl/traj_track/default.csv"):
  dt = np.dtype([('mode', np.int),
                 ('filename', 'S256')])
  x = np.array((6, filename), dtype=dt)
  sock.sendto(x.tobytes(), (UDP_IP, CTRL_UDP_PORT))
   
#run soft_stop when script is exited
atexit.register(soft_stop)

#order for q is: x, y, z,
#                rx, ry, rz,
#                fl1, fl2, fl3,
#                fr1, fr2, fr3,
#                bl1, bl2, bl3,
#                br1, br2, br3
#order for u is: fl1, fl2, fl3,
#                fr1, fr2, fr3,
#                bl1, bl2, bl3,
#                br1, br2, br3
# tlm_types = np.dtype([('start_time_sec', np.int64),
#                       ('start_time_nano', np.int64),
#                       ('cycle_duration', np.double),
#                       ('compute_duration', np.double),
#                       ('tictoc', np.double),
#                       ('q', np.double, (18,)),
#                       ('qd', np.double, (18,)),
#                       ('qdd_sim', np.double, (18,)),
#                       ('u', np.double, (12,)),
#                       ('act_mode', np.int, (12,)),
#                       ('u_des', np.double, (12,)),
#                       ('q_des', np.double, (18,)),
#                       ('qd_des', np.double, (18,)),
#                       ('f', np.double, (4,)),
#                       ('ctrl_curr', np.int, (9,)),
#                      ], align=True);

# lock = threading.Lock()
# tlm_data = None

# def get_tlm_data():
#   return tlm_data

# def tlm_read_thread():
#   global tlm_data
#   packets = True
#   buf = None
#   while True:
#     buf, addr = sock.recvfrom(1152) #size of tlm packet
#     if (buf):
#       with lock:
#         tlm_data = np.frombuffer(buf, dtype = tlm_types)[0]

# #start thread to read data
# t = threading.Thread(target=tlm_read_thread)
# t.daemon = True
# t.start()
