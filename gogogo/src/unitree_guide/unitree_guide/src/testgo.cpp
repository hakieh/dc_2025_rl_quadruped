#include"ros/ros.h"
#include "unitree_legged_msgs/LowCmd.h"
#include "unitree_legged_msgs/LowState.h"
#include "unitree_legged_msgs/MotorCmd.h"
#include "unitree_legged_msgs/MotorState.h"

int main(int argc,char **argv){
    ros::init(argc,argv,"testgo");
    ros::NodeHandle nh;
    ros::Publisher puber[12];
    unitree_legged_msgs::LowCmd msg;
    ROS_INFO("Success");
    puber[0] = nh.advertise<unitree_legged_msgs::MotorCmd>("/go1_gazebo/FR_hip_controller/command", 1);
    puber[1] = nh.advertise<unitree_legged_msgs::MotorCmd>("/go1_gazebo/FR_thigh_controller/command", 1);
    puber[2] = nh.advertise<unitree_legged_msgs::MotorCmd>("/go1_gazebo/FR_calf_controller/command", 1);
    puber[3] = nh.advertise<unitree_legged_msgs::MotorCmd>("/go1_gazebo/FL_hip_controller/command", 1);
    puber[4] = nh.advertise<unitree_legged_msgs::MotorCmd>("/go1_gazebo/FL_thigh_controller/command", 1);
    puber[5] = nh.advertise<unitree_legged_msgs::MotorCmd>("/go1_gazebo/FL_calf_controller/command", 1);
    puber[6] = nh.advertise<unitree_legged_msgs::MotorCmd>("/go1_gazebo/RR_hip_controller/command", 1);
    puber[7] = nh.advertise<unitree_legged_msgs::MotorCmd>("/go1_gazebo/RR_thigh_controller/command", 1);
    puber[8] = nh.advertise<unitree_legged_msgs::MotorCmd>("/go1_gazebo/RR_calf_controller/command", 1);
    puber[9] = nh.advertise<unitree_legged_msgs::MotorCmd>("/go1_gazebo/RL_hip_controller/command", 1);
    puber[10] = nh.advertise<unitree_legged_msgs::MotorCmd>("/go1_gazebo/RL_thigh_controller/command", 1);
    puber[11] = nh.advertise<unitree_legged_msgs::MotorCmd>("/go1_gazebo/RL_calf_controller/command", 1);
    for(int i(0); i < 4; ++i){
        msg.motorCmd[i*3+0].mode = 10;
        msg.motorCmd[i*3+0].Kp = 180;
        msg.motorCmd[i*3+0].Kd = 8;
        msg.motorCmd[i*3+1].mode = 10;
        msg.motorCmd[i*3+1].Kp = 180;
        msg.motorCmd[i*3+1].Kd = 8;
        msg.motorCmd[i*3+2].mode = 10;
        msg.motorCmd[i*3+2].Kp = 300;
        msg.motorCmd[i*3+2].Kd = 15;
    }
    ros::Rate r(10);
    while(ros::ok()){
        for(int m(0); m < 12; ++m){
        puber[m].publish(msg.motorCmd[m]);
        }
        r.sleep();
    }
    return 0;
}