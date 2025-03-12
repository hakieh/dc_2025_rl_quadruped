import torch
import rospy
import geometry_msgs.msg as gem
import gazebo_msgs.msg as gam
import unitree_legged_msgs.msg as unm
import sensor_msgs.msg as sem
import torch.nn as nn

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

if __name__ == "__main__":
    rospy.init_node("test_topic")
    sub_con_FL=rospy.Subscriber("/visual/FL_foot_contact/the_force",gem.WrenchStamped,do_con_FL,queue_size=1)
    sub_con_FR=rospy.Subscriber("/visual/FR_foot_contact/the_force",gem.WrenchStamped,do_con_FR,queue_size=1)
    sub_con_RL=rospy.Subscriber("/visual/RL_foot_contact/the_force",gem.WrenchStamped,do_con_RL,queue_size=1)
    sub_con_RR=rospy.Subscriber("/visual/RR_foot_contact/the_force",gem.WrenchStamped,do_con_RR,queue_size=1)
    rr=rospy.Rate(50)
    while(not rospy.is_shutdown()):
        force_FL=(FL_con.x**2+FL_con.y**2+FL_con.z**2)>500
        force_FR=(FR_con.x**2+FR_con.y**2+FR_con.z**2)>500
        force_RL=(RL_con.x**2+RL_con.y**2+RL_con.z**2)>500
        force_RR=(RR_con.x**2+RR_con.y**2+RR_con.z**2)>500
        print(force_FL,force_FR,force_RL,force_RR)
        rr.sleep()