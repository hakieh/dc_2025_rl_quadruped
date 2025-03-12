import torch
import rospy
import time
import geometry_msgs.msg as gem
import gazebo_msgs.msg as gam
import unitree_legged_msgs.msg as unm
import sensor_msgs.msg as sem
import torch.nn as nn

class Estimator(nn.Module): 
    def __init__(self):
        super().__init__()
        self.estimator = nn.Sequential(
            nn.Linear(245,512),
            nn.ELU(),
            nn.Linear(512,256),
            nn.ELU(),
            nn.Linear(256,128),
            nn.ELU(),
            nn.Linear(128,64),
            nn.ELU(),
            nn.Linear(64,10),
        )
    def forward(self, x):
        return self.estimator(x)
class Actor(nn.Module): 
    def __init__(self):
        super(Actor, self).__init__()
        self.actor_backbone = nn.Sequential(
            nn.Linear(59,1024),
            nn.ELU(),
            nn.Linear(1024,512),
            nn.ELU(),
            nn.Linear(512,256),
            nn.ELU(),
            nn.Linear(256,128),
            nn.ELU(),
            nn.Linear(128,64),
            nn.ELU(),
            nn.Linear(64,32),
            nn.ELU(),
            nn.Linear(32,16)
        )
        self.actor_backbone2 = nn.Sequential(nn.Linear(16,12))
    def forward(self, x):
        x = self.actor_backbone(x)
        return self.actor_backbone2(x) 
class alg(nn.Module):
    def __init__(self):
        super(alg, self).__init__()
        self.actor = Actor()
    def forward(self, x):
        return self.actor(x)
    
def quat_rotate_inverse(q):
    shape=q.shape
    v=torch.tensor([0,0,-1],dtype=torch.float,device=mydiv).unsqueeze(0)
    q_w = q[:, -1]
    q_vec = q[:, :3]
    a = v * (2.0 * q_w ** 2 - 1.0).unsqueeze(-1)
    b = torch.cross(q_vec, v, dim=-1) * q_w.unsqueeze(-1) * 2.0
    c = q_vec * \
        torch.bmm(q_vec.view(shape[0], 1, 3), v.view(
            shape[0], 3, 1)).squeeze(-1) * 2.0
    return a - b + c

ang=gem.Vector3()
ori=gem.Quaternion()
j_pos=list()
j_vec=list()
puber=list()
mydiv='cuda'
dof_map= [1, 4, 7, 10, 2, 5, 8, 11, 0, 3, 6, 9]
rqt_plus=[-1.5,0,0.8,-1.5,0,0.8,-1.5,0.1,1,-1.5,-0.1,1]
model_plus=torch.tensor(list(rqt_plus[dof_map[i]] for i in range(0,12)),device=mydiv).unsqueeze(0)
total_obs=torch.zeros((1,294),device= mydiv, dtype= torch.float32)


FL_con=gem.Vector3()
FR_con=gem.Vector3()
RL_con=gem.Vector3()
RR_con=gem.Vector3()
def do_con_FL(msg_con):
    global FL_con
    FL_con=msg_con.wrench.force
def do_con_FR(msg_con):
    global FR_con
    FR_con=msg_con.wrench.force
def do_con_RL(msg_con):
    global RL_con
    RL_con=msg_con.wrench.force
def do_con_RR(msg_con):
    global RR_con
    RR_con=msg_con.wrench.force
def do_imu(msg_ang):
    global ang,ori
    ang=msg_ang.angular_velocity
    ori=msg_ang.orientation

def do_joint(msg_joi):
    global j_pos,j_vec,dof_map
    j_pos=list(msg_joi.position[dof_map[i]] for i in range(0,12))
    j_vec=list(msg_joi.velocity[dof_map[i]] for i in range(0,12))
    # print(j_pos)



def get_pub():
    puber=list()
    puber.append(rospy.Publisher("/go1_gazebo/FL_calf_controller/command",unm.MotorCmd,queue_size=1))
    puber.append(rospy.Publisher("/go1_gazebo/FL_hip_controller/command",unm.MotorCmd,queue_size=1))
    puber.append(rospy.Publisher("/go1_gazebo/FL_thigh_controller/command",unm.MotorCmd,queue_size=1))
    puber.append(rospy.Publisher("/go1_gazebo/FR_calf_controller/command",unm.MotorCmd,queue_size=1))
    puber.append(rospy.Publisher("/go1_gazebo/FR_hip_controller/command",unm.MotorCmd,queue_size=1))
    puber.append(rospy.Publisher("/go1_gazebo/FR_thigh_controller/command",unm.MotorCmd,queue_size=1))
    puber.append(rospy.Publisher("/go1_gazebo/RL_calf_controller/command",unm.MotorCmd,queue_size=1))
    puber.append(rospy.Publisher("/go1_gazebo/RL_hip_controller/command",unm.MotorCmd,queue_size=1))
    puber.append(rospy.Publisher("/go1_gazebo/RL_thigh_controller/command",unm.MotorCmd,queue_size=1))
    puber.append(rospy.Publisher("/go1_gazebo/RR_calf_controller/command",unm.MotorCmd,queue_size=1))
    puber.append(rospy.Publisher("/go1_gazebo/RR_hip_controller/command",unm.MotorCmd,queue_size=1))
    puber.append(rospy.Publisher("/go1_gazebo/RR_thigh_controller/command",unm.MotorCmd,queue_size=1))
    return puber
    
def get_model():
    act = alg()
    est=Estimator()
    weights=torch.load("./weights/shake_con/model_19750.pt")

    act.load_state_dict(weights['model_state_dict'],strict=False)
    act.eval()
    act=act.to(mydiv)

    est.load_state_dict(weights['estimator_state_dict'],strict=False)
    est.eval()
    est=est.to(mydiv)

    return act,est

def stand_policy():
    global j_pos,puber
    rate0=rospy.Rate(1000)
    targetpos=[-1.3,0.0,0.67,-1.3,0.0,0.67,-1.3,0.0,0.67,-1.3,0.0,0.67]
    kkp=[180,300,180,180,300,180,180,300,180,180,300,180]
    kkd=[8,15,8,8,15,8,8,15,8,8,15,8]
    time.sleep(0.5)
    # print(j_pos)
    startpos=j_pos[:]
    duaration=1000
    leg_msg=unm.LowCmd()
    for i in range(12):
        leg_msg.motorCmd[i].mode = 10
        leg_msg.motorCmd[i].Kp = 180
        leg_msg.motorCmd[i].Kd = 8
    percent=float(0)
    print(puber)
    for i in range(2000):
        percent+=float(1/duaration)
        if percent>1:
            percent=1
        for j in range(12):
            leg_msg.motorCmd[j].q=(1 - percent)*startpos[j] + percent*targetpos[dof_map[j]]
            puber[dof_map[j]].publish(leg_msg.motorCmd[j])
        rate0.sleep()

def get_con():
    global FL_con,FR_con,RL_con,RR_con
    f_1=(FL_con.x**2+FL_con.y**2+FL_con.z**2)>50
    f_2=(FR_con.x**2+FR_con.y**2+FR_con.z**2)>50
    f_3=(RL_con.x**2+RL_con.y**2+RL_con.z**2)>50
    f_4=(RR_con.x**2+RR_con.y**2+RR_con.z**2)>50
    return torch.tensor([f_1,f_2,f_3,f_4],device=mydiv).unsqueeze(0)

def get_obs(actions):
    global total_obs
    base_ang_vel=torch.tensor([ang.x,ang.y,ang.z],device=mydiv).unsqueeze(0)
    base_ang_vel*=0.25
    quaternion=torch.tensor([ori.x,ori.y,ori.z,ori.w],device=mydiv).unsqueeze(0)
    projected_gravity = quat_rotate_inverse(quaternion)
    velocity_commands=torch.tensor([0.5,0,0],device=mydiv).unsqueeze(0)
    joint_pos=torch.tensor(j_pos,device=mydiv).unsqueeze(0)-model_plus
    joint_vec=torch.tensor(j_vec,device=mydiv).unsqueeze(0)
    joint_vec*=0.05
    contact=get_con()
    obs_now=torch.cat([base_ang_vel,projected_gravity,velocity_commands,joint_pos,joint_vec,actions,contact],dim=-1)
    total_obs = torch.cat(( total_obs[:, 49:],obs_now[:, :]), dim=-1)
    return total_obs

def send(output):
    global puber
    leg_msg=unm.LowCmd()
    # print(output)
    for i in range(12):
        leg_msg.motorCmd[i].mode=10
        leg_msg.motorCmd[i].q=output[0][i]
        leg_msg.motorCmd[i].dq=0
        leg_msg.motorCmd[i].tau=0
        leg_msg.motorCmd[i].Kp=40
        leg_msg.motorCmd[i].Kd=1
        puber[dof_map[i]].publish(leg_msg.motorCmd[i])

if __name__ == "__main__":
    rospy.init_node("test_shake_con")

    puber=get_pub()
    act,est=get_model()
    
    # print(act,est)
    sub_con_FL=rospy.Subscriber("/visual/FL_foot_contact/the_force",gem.WrenchStamped,do_con_FL,queue_size=1)
    sub_con_FR=rospy.Subscriber("/visual/FR_foot_contact/the_force",gem.WrenchStamped,do_con_FR,queue_size=1)
    sub_con_RL=rospy.Subscriber("/visual/RL_foot_contact/the_force",gem.WrenchStamped,do_con_RL,queue_size=1)
    sub_con_RR=rospy.Subscriber("/visual/RR_foot_contact/the_force",gem.WrenchStamped,do_con_RR,queue_size=1)
    sub_imu=rospy.Subscriber("/trunk_imu",sem.Imu,do_imu,queue_size=1)
    sub_joint=rospy.Subscriber("/go1_gazebo/joint_states",sem.JointState,do_joint,queue_size=1)
    stand_policy()
    rate=rospy.Rate(50)
    actions=torch.tensor([0,0,0,0,0,0,0,0,0,0,0,0],device=mydiv).unsqueeze(0)
    
    aa=torch.ones((1,245),device=mydiv)
    print('est:',est(aa))
    bb=torch.ones((1,59),device=mydiv)
    print('act:',act(bb))
    while(not rospy.is_shutdown()):
        obs=get_obs(actions)
        his = obs[:,:-49]
        cur_obs = obs[:,-49:]
        latent = est(his)
        obs = torch.cat((cur_obs,latent),dim=1)
     
        actions = act(obs)
        
        output=actions[:]*0.25+model_plus
        send(output)
        
        #actions=output[:]
        rate.sleep()

'''
FLhip->FRhip->...->thigh->calf
dof_map= [1, 4, 7, 10,
         2, 5, 8, 11   ,
          0, 3, 6, 9
          ]

FLhip->FLthigh->FLcalf->...->FR
dof_map=[1,2,0,4,5,3,7,8,6,10,11,9]
'''