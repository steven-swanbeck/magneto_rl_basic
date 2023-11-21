#!/usr/bin/env python3
import os
import sys
import rospy
from std_msgs.msg import Bool
from std_srvs.srv import Trigger
from geometry_msgs.msg import Point, Pose
# from magneto_rl.srv import FootPlacement
from magneto_rl.srv import FootPlacement, FootPlacementResponse
from magneto_rl.srv import UpdateMagnetoAction, UpdateMagnetoActionResponse
from magneto_rl.srv import ReportMagnetoState, ReportMagnetoStateResponse
import roslaunch
import time
import pyautogui
# from seed_magnetism import MagnetismMapper
from magnetic_seeder import MagneticSeeder
import pygame
import numpy as np


class MagnetoRLPlugin (object):
    
    # def __init__(self, render_mode, render_fps, magnetic_seeds=10) -> None:
    def __init__(self, magnetic_seeds=10) -> None:
        rospy.init_node('magneto_rl_manager')
        
        self.next_step_service = rospy.Service('determine_next_step', FootPlacement, self.determine_next_step)
        self.test_action_service = rospy.Service('test_action_command', Trigger, self.test_action_command)
        self.set_magneto_action = rospy.ServiceProxy('set_magneto_action', UpdateMagnetoAction)
        self.command_sleep_duration = rospy.get_param('/magneto/simulation/resume_duration')
        self.test_state_service = rospy.Service('test_state_getter', Trigger, self.test_get_state)
        self.test_begin_episode = rospy.Service('test_episode_begin', Trigger, self.begin_episode_cb)
        self.test_end_episode = rospy.Service('test_episode_end', Trigger, self.end_episode_cb)
        self.test_reset_episode = rospy.Service('test_episode_reset', Trigger, self.reset_episode_cb)
        self.test_map = rospy.Service('test_map', Trigger, self.get_map)
        
        self.link_idx = {
            'AR':rospy.get_param('/magneto/simulation/link_idx/AR'),
            'AL':rospy.get_param('/magneto/simulation/link_idx/AL'),
            'BL':rospy.get_param('/magneto/simulation/link_idx/BL'),
            'BR':rospy.get_param('/magneto/simulation/link_idx/BR'),
        }
        self.naive_walk_order = ['AR', 'AL', 'BL', 'BR']
        self.last_foot_placed = None
        
        self.vertical_pixel_calibration_offset = rospy.get_param('/magneto/simulation/vertical_pixel_calibration_offset')
        
        self.mag_seeder = MagneticSeeder()
        self.num_seeds = magnetic_seeds
        
        self.wall_size = 5
        # self.fps = render_fps
        self.window = None
        self.clock = None
        self.wall_width = 5
        self.wall_height = 5
        self.im_width = 500
        self.im_height = 500
        self.window_size = 500
        self.scale = 500 / 5 #pixels/m
        self.heading_arrow_length = 0.2
        self.leg_length = 0.2
        self.body_radius = 0.08
        self.foot_radius = 0.03
        self.body_width = 0.2 #m
        self.body_width_pixels = self.scale * self.body_width
        self.body_height = 0.3 #m
        self.body_height_pixels = self.scale * self.body_height
        self.goal = np.array([1, 1]) # !
        self.heading = 0
    
    # . Testing functions
    # TODO validate the functionality of magnetism stuff
    def get_map (self, msg:Trigger):
        self.mag_map.create_map()
        value = self.mag_map.pixel_grab(1.5, 1)
        print(value)
        return True, ''
    
    def begin_episode_cb (self, msg:Trigger):
        success = self.begin_sim_episode()        
        return success, ''
    
    def end_episode_cb (self, msg:Trigger):
        success = self.end_sim_episode()
        return success, ''
    
    def reset_episode_cb (self, msg:Trigger):
        self.end_sim_episode()
        self.begin_sim_episode()
        
        test = Trigger()
        self.test_action_command(test)
        self.test_action_command(test)
        self.test_action_command(test)
        self.test_action_command(test)
        self.test_get_state(test)
        
        return True, ''
    
    # . Simulator Dynamics
    def provide_magnetic_force_modifier (self, msg):
        # TODO
        raise NotImplementedError
    
    def report_state (self):
        res = self.get_magneto_state()
        return res
    
    def update_goal (self, goal):
        self.goal = goal
    
    def update_action (self, link_id:str, pose:Pose) -> bool:
        res = self.set_magneto_action(self.link_idx[link_id], pose)
        rospy.sleep(self.command_sleep_duration)
        return res.success
    
    # WIP
    def begin_sim_episode (self) -> bool:
        node = roslaunch.core.Node('my_simulator', 
                            'magneto_ros') #,
        launch = roslaunch.scriptapi.ROSLaunch()
        launch.start()
        self.sim_process = launch.launch(node)
        
        time.sleep(3)
        
        self.set_magneto_action = rospy.ServiceProxy('set_magneto_action', UpdateMagnetoAction)
        self.get_magneto_state = rospy.ServiceProxy('get_magneto_state', ReportMagnetoState)
        
        pyautogui.doubleClick(1440 + 500/2, 10 + self.vertical_pixel_calibration_offset)
        pyautogui.click(1440 + 500/2, 500/2)
        pyautogui.press('space')
        time.sleep(1)
        
        pyautogui.press('s')
        
        start_state = self.get_magneto_state()
        self.ground_pose = start_state.ground_pose
        self.body_pose = start_state.body_pose
        
        self.foot_poses = [
            start_state.AR_state.pose,
            start_state.AL_state.pose,
            start_state.BL_state.pose,
            start_state.BR_state.pose,
        ]
        self.foot_mags = np.array([
            start_state.AR_state.magnetic_force,
            start_state.AL_state.magnetic_force,
            start_state.BL_state.magnetic_force,
            start_state.BR_state.magnetic_force,
        ])
        
        self.raw_map, self.seed_locations, self.single_channel_map = self.mag_seeder.generate_map(self.num_seeds)
        self.game_background = self.mag_seeder.transform_image_into_pygame(self.raw_map)
        
        return True
    
    def end_sim_episode (self) -> bool:
        pyautogui.click(1440 + 500/2, 500/2)
        pyautogui.click(1901, 21 + self.vertical_pixel_calibration_offset)
        self.sim_process.stop()
        time.sleep(1)
        return not self.sim_process.is_alive()

    def run (self):
        while not rospy.is_shutdown():
            rospy.spin()

    def _render_frame (self):
        if self.window is None:
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
        if self.clock is None:
            self.clock = pygame.time.Clock()
        
        canvas = pygame.Surface((self.window_size, self.window_size), pygame.SRCALPHA, 32)
        canvas = canvas.convert_alpha()

        body_center = self.cartesian_to_pygame_coordinates(np.array([self.body_pose.position.x, self.body_pose.position.y]))
        pygame.draw.circle(
            canvas,
            (150, 150, 150),
            center=body_center,
            radius=self.body_radius * self.scale * 2 / 3,
        )
        
        foot_pixel_positions = [self.cartesian_to_pygame_coordinates(np.array([self.foot_poses[ii].position.x, self.foot_poses[ii].position.y])) for ii in range(len(self.foot_poses))]
        for ii in range(len(self.foot_poses)):
            pygame.draw.circle(
                canvas,
                (150, 150, 150),
                center=foot_pixel_positions[ii],
                radius=self.foot_radius * self.scale,
            )
        
        heading_end = self.cartesian_to_pygame_coordinates(np.array([self.body_pose.position.x, self.body_pose.position.y]) + np.array([self.heading_arrow_length * np.cos(self.heading), self.heading_arrow_length * np.sin(self.heading)]))
        pygame.draw.line(
                canvas,
                (255, 255, 255),
                start_pos=body_center,
                end_pos=heading_end,
                width=3,
            )

        goal_center = self.cartesian_to_pygame_coordinates(self.goal)
        pygame.draw.circle(
            canvas,
            (0, 255, 0),
            center=goal_center,
            # radius=self.body_radius * self.scale / 2,
            radius=0.20 * self.scale, # & this should be set to the tolerance of the at_goal() function in the environment
        )
        
        self.window.blit(pygame.surfarray.make_surface(self.game_background), (0, 0))
        self.window.blit(canvas, canvas.get_rect())
        
        myfont = pygame.font.SysFont("monospace", 15)
        for ii in range(len(self.foot_poses)):
            label = myfont.render(str(ii), 1, (255,255,0))
            self.window.blit(label, foot_pixel_positions[ii])
    
        pygame.event.pump()
        pygame.display.update()
        # self.clock.tick(self.fps)
    
    def cartesian_to_pygame_coordinates (self, coords):
        output = np.array([
            coords[1] * (self.im_width / (2 * self.wall_width)) + self.im_width / 2,
            coords[0] * (self.im_height / (2 * self.wall_height)) + self.im_height / 2,
        ])
        return output

if __name__ == "__main__":
    magneto_rl = MagnetoRLPlugin()
    magneto_rl.run()
