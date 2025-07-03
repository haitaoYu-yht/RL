import gymnasium as gym
import numpy as np
import mujoco
import mujoco.viewer
from gymnasium import spaces
from typing import Optional, Tuple, Dict, Any
import os

class FingerPushEnv(gym.Env):
    """A Gym environment for a finger robot pushing a cube to a target position."""
    
    metadata = {"render_modes": ["rgb_array", "human"], "render_fps": 60}
    
    def __init__(self, 
                 xml_path: str = "../robot_description/finger_edu_description/xml/finger_edu_scene_cube.xml",
                 render_mode: Optional[str] = None):
        super().__init__()

        # get scene model path
        model_path = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), xml_path))
        
        # Load the MuJoCo model
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)
        
        # -- Action Space
        # TODO: Compute the action space range and number of DOF
        # Actions are joint position targets
        # Fetch robot joint limits from XML or from the mujoco model:
        self.num_dof = None # to be set
        self.action_space = spaces.Box(
            low=None, # to be set
            high=None, # to be set
            shape=(self.num_dof,),
            dtype=np.float32
        )
        self.prev_action = np.zeros(self.num_dof)
        
        # -- Observation Space
        # TODO: Compute the observation space dimensions
        # - Robot observations: Joint positions, Joint Velocities
        # - Object observations: Cube position (3D), Cube orientation (quaternion), Cube linear velocity (3D), Cube angular velocity (3D)
        # - Target position: Target position of the cube (2D)
        # - Previous action: Previous action taken by the robot
        obs_dim = None # to be set
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        
        # -- Task parameters
        # - Task: Push the cube to the target position
        # TODO: Define the target position for the cube (2D) to be 20cm away in the x-axis from the initial cube position
        self.cube_init_pos = np.array(self.data.qpos[3:5].copy()) # from XML
        self.target_pos_xy = None # To be set   
        # We use a timer to wait 100 simsteps at the target position before considering the goal reached
        self.reach_goal_timer = 0
        self.reach_goal_threshold = 100 # 100 simsteps
        
        # -- Controller parameters
        self.kp = 2.0  # Position gain
        self.kd = 0.1   # Velocity gain
        self.action_scale = 0.5  # Action scaling factor
        self.default_joint_pos = np.array([0.0, 0.5, -0.75])  # Default joint positions
        self.decimation = 4
        
        # -- Environment parameters
        self.episode_length = 0.5
        self.steps_per_episode = int(self.episode_length / self.model.opt.timestep)
        self.current_step = 0
        
        # -- Renderer
        self.render_mode = render_mode
        self.viewer = None
        if self.render_mode == "human":
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
        elif self.render_mode == 'rgb_array':
            self.viewer = mujoco.Renderer(self.model)

    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Step the environment forward. This function is called by the agent to take an action and get the 
        next observation, reward, done, and info.
        
        The action is processed and applied to the environment in this function. The decimation is used for smoother 
        tracking of the actions using the PD controller. Consequently, the PD controller and simulation are running
        at a higher frequency [1/(decimation * model.opt.timestep)] than the action taken by the agent [1/model.opt.timestep].
        
        The goal progress is also checked at every simulation time-step.

        The next observations, rewards and dones are computed after applying the action in this function.
                
        The info is a dictionary of additional information (here used to pass individual rewards for plotting).  
        
        Args:
            action: The action to be applied to the environment
        Returns:
            tuple: A tuple of (next_observation, reward, terminated, truncated, info)
            - observation: The current observation (np.ndarray)
            - reward: The reward for the current step (float)
            - terminated: Whether the episode is terminated (bool)
            - truncated: Whether the episode is truncated (bool)
            - info: A dictionary of additional information (dict)
        """
        # TODO: Check if the goal is reached (within 5cm of the target position)
        
        self.current_step += 1
        # TODO: Process action
        processed_action = self._process_action(action)
        
        for _ in range(self.decimation):        
            # TODO: Apply action
            self._apply_action(processed_action)
            # Step simulation
            mujoco.mj_step(self.model, self.data)
            # TODO: Check goal progress
            self._check_goal_progress()
            
            # Render (if needed)
            if self.render_mode == "human" and self.viewer is not None:
                self.render()
        
        # TODO: Get next observation
        next_obs = self._get_obs()
        # TODO: Get reward
        reward_dict = self._get_reward()
        # TODO: Check if episode is done
        terminated, truncated = self._get_dones()
        # Save previous action as the applied action
        self.prev_action = action.copy()

        return next_obs, sum(reward_dict.values()), terminated, truncated, {"reward_dict": reward_dict}
    

    def _get_obs(self) -> np.ndarray:
        """
        Get the current observation. Make sure that the observation contains all the information needed 
        to compute the reward and accomplish the task.
        
        Expected observations:
        - Robot states: Joint positions, Joint velocities
        - Object states: Cube position (3D), Cube orientation (Quaternion), Cube linear velocity, Cube angular velocity
        - Target position: Target position of the cube (2D)
        - Previous action: Previous action taken by the robot
        
        Args:
            None
        Returns:
            np.ndarray: The concatenated observations as a numpy array (dtype=np.float32)
        """
        # TODO: Implement the observation space
        
        return NotImplementedError # to be set
    
    def _get_reward(self) -> Dict[str, float]:
        """
        Calculate a reward based on distance to target (and other regularization rewards, if needed).
        Reward tuning is a very important part of the RL process.
        You can tune the reward function to encourage the agent to learn the task (e.g. reach the goal (sparse or dense reward(s))),
        and also add additional regularization rewards for the behaviours you want to encourage or discourage. 
        You can get creative here to get desired behaviours (robot/object behaviours) to achieve the task.

        Return a dictionary of reward (str) -> value (float).
        The key is the name of the reward, and the value is the reward value. Each will be logged in WandB.
        
        Args:
            None
        Returns:
            dict: A dictionary of reward (str) -> value (float)
        """
        # TODO: Implement the reward function
        return NotImplementedError # to be set
        
    
    def _get_dones(self) -> Tuple[bool, bool]:
        """
        Check if the episode is done.
        - Timeout: if the episode length is reached  (termination)
        - Goal reached: if the reach_goal_timer is greater than the reach_goal_threshold (truncation)
        
        Args:
            None
        Returns:
            tuple: A tuple of two booleans (timeout, goal_reached)
        """
        
        # TODO: Implement the done function        
        return NotImplementedError # to be set
    
    
    def _process_action(self, action: np.ndarray) -> np.ndarray:
        """
        Process the action to be applied to the environment.
        This function is used to scale the raw action from the policy (action_scale) to 
        a reasonable range and add to default position to get joint position targets.
        
        Args:
            action: Raw action from the policy in joint space (radians)
            
        Returns:
            Processed joint position targets
        """
        # TODO: Scale the action and add to default position to get joint position targets for the PD controller
        return NotImplementedError # to be set
        
    
    def _apply_action(self, joint_pos_target: np.ndarray) -> None:
        """
        Convert position targets to torques using PD control.
        This function is used to apply the joint position targets to the robot using PD control. 
        
        PD control law: τ = kp * (θd - θ) - kd * θ̇
        θd : Target joint position
        θ : Current joint position
        θ̇ : Current joint velocity
        τ : Torque
        
        Args:
            joint_pos_target: Target joint positions
        """

        # TODO: Implement the PD control law to get the torque
        torque = NotImplementedError # to be set
        
        # Apply torque to the robot
        self.data.ctrl[:self.num_dof] = torque
        
        
    def _check_goal_progress(self) -> None:
        """
        Check if the goal is reached (if the cube is within 5cm of the target position).
        If True, increment the reach_goal_timer.
        If False, reset the reach_goal_timer.
        """
        # TODO: Implement the goal progress check
        return NotImplementedError # to be set
    

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset the environment to initial state.
        
        Args:
            seed: Random seed for reproducibility
        Returns:
            np.ndarray: The initial observation
            dict: An empty dictionary
        """
        super().reset(seed=seed)
        
        # Reset timers and counters
        self.reach_goal_timer = 0
        self.current_step = 0
        
        # Reset action buffers
        self.prev_action = np.zeros(self.num_dof)

        # Update the target marker position in the model upon reset
        marker_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "target_marker")
        self.model.geom_pos[marker_id] = np.array([self.target_pos_xy[0], self.target_pos_xy[1], 0.05])
        
        # Reset MuJoCo
        mujoco.mj_resetData(self.model, self.data) 

        return self._get_obs(), {}
        
    
    def render(self):
        """
        Render the environment. 
        If render_mode is "human", this function will launch a viewer window.
        If render_mode is "rgb_array", this function will return a numpy array of the rendered image (useful for video recording).
        
        Args:
            None
        Returns:
            np.ndarray: The rendered image (if render_mode is "rgb_array")
            None: If render_mode is "human"
        """
        if self.render_mode == "human":
            if self.viewer is None:
                try:
                    self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
                except Exception:
                    return None
            if self.viewer is not None:
                try:
                    self.viewer.sync()
                except Exception:
                    pass
            return None
        elif self.render_mode == 'rgb_array':
            if self.viewer is None:
                try:
                    self.viewer = mujoco.Renderer(self.model)
                except Exception:
                    return None
            if self.viewer is not None:
                try:
                    self.viewer.update_scene(self.data)
                    return self.viewer.render()
                except Exception:
                    pass

        return None
    
    def close(self):
        """Close the environment and viewer."""
        if self.viewer is not None:
            try:
                self.viewer.close()
            except Exception:
                pass
            self.viewer = None