import pybullet as p
import pybullet_envs
import pybullet_data
import time
p.connect(p.GUI)
p.resetSimulation()
p.setGravity(0,0,-9.8)
p.setRealTimeSimulation(0)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.loadURDF("plane.urdf",[0,0,0],[0,0,0,1])
testudogid = p.loadURDF("./urdf/go1.urdf",[0,0,0.4],[0,0,0,1])
focus,_ = p.getBasePositionAndOrientation(testudogid)
# p.resetDebugVisualizerCamera(cameraDistance=3,cameraYaw=90,cameraPitch=0,cameraTargetPosition=focus)
i=0
while True:
    i+=1
    p.stepSimulation()
    orientation = p.getLinkState(testudogid,0)[1]
    roll, pitch, yaw = p.getEulerFromQuaternion(orientation)
    print(roll, pitch)
    time.sleep(1/50)