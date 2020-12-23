# Big Data and Machine Intelligence
Learning Bipedal With Deep Reinforcement Learning
## Environment in this project
**numpy and cloudpickle version should be matched if you want to use the pretrained model**
* Stable-baselines Environment:
  * python==3.7.9
  * numpy==1.19.2   
  * cloudpickle==1.16.0
  * tensorflow==1.14
  * [pybulletgym](https://github.com/benelot/pybullet-gym)
  * [stable-baselines](https://stable-baselines.readthedocs.io/)
* rllib environment:
  * python==3.7.9
  * rllib
  * tensorflow==2.3.0
  * pytorch==1.7.0
  * numpy==1.19.2
  * cloudpickle==1.16.0
  * [pybulletgym](https://github.com/benelot/pybullet-gym)
## Parameters for training
We trained two environment(Walker2D & Humanoid) with SAC (stable-baselines) and PPO (rllib).

Walker2D is trained with default parameter.

Humanoid is trained with [recommended parameter](https://github.com/ray-project/ray/blob/master/rllib/tuned_examples/ppo/humanoid-ppo-gae.yaml).

## Patch for PybulletGym
Change the following code to fix the camera class in pybulletgym

```python
## pybulletgym/env/roboschool/envs/env_bases.py
## line 50 to 52
if self.isRender:
    self._p = bullet_client.BulletClient(connection_mode=pybullet.GUI)
    self.camera._p = self._p # this line is added to sync the pybullet server

## line 125 in class Camera
def __init__(self):
    self._p = None      # replace the return line with this
## line 131 in function move_and_look_at
    distance = 2        # change the distance between camera and robot 
                        # from 10 to 2 to get a close look
```
To enable camera following in **HumanoidPyBulletEnv-v0**, do the following changes:
```python
## in pybulletgym/envs/roboschool/envs/locomotion/walker_base_end.py
## line 111 in step()
    self.HUD(state, a, done)
    self.reward += sum(self.rewards)
    self.camera_adjust()        # add this line
## line 115 in function 'camera_adjust'
    x, y, z = self.robot.body_xyz
```
The modification above also works for **Walker2DPyBulletEnv-v0**.

To change the background in the scene, which makes it easier to view how fast the robot walk, do the following changes:
```python
## in pybulletgym/envs/roboschool/scenes/stadium.py
## line 27, replace the original line with this:
   filename = os.path.join(os.path.dirname(__file__), "..", "..", "assets", "scenes", "stadium", "stadium.sdf")
## in pybulletgym/envs/roboschool/envs/locomotion/walker_base_end.py
## line 49, replace the original line with this:
   foot_ground_object_names = set(["link_d0","link_d1","link_d1"])
```
You are all set!
