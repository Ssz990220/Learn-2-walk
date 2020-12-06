# Big Data and Machine Intelligence

___

### Learning Bipedal With Deep Reinforcement Learning

** Patch for PybulletGym
Change the following code to fix the camera in pybullet

```python
## env/roboschool/envs/env_bases.py
## line 50 to 52
if self.isRender:
    self._p = bullet_client.BulletClient(connection_mode=pybullet.GUI)
    self.camera._p = self._p # this line is added to sync the pybullet server

## line 125 in class Camera
    def __init__(self):
		self._p = None      # replace the return line with this
```
To enable camera following in **HumanoidPyBulletEnv-v0**, do the following changes:
```python
## in envs/roboschool/envs/locomotion/walker_base_end.py
## line 111 in step()
    self.HUD(state, a, done)
    self.reward += sum(self.rewards)
    self.camera_adjust()        # add this line
```
