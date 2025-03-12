import math
import os
import sys
from copy import copy
import yaml
from PIL import Image
from pygame import Rect
from visualization.utils import *


class Visualization:
    def __init__(self,
                 settings_yaml,
                 tractor_length: float,
                 trailer_length: float,
                 x_init=0,
                 y_init=0,  # meter
                 yaw_init=0,  # rad
                 gamma=0, # rad
                 display: bool = True,
                 fps=None,
                 ):
        assert os.path.isfile(settings_yaml), settings_yaml
        with open(settings_yaml, 'r') as stream:
            settings = yaml.safe_load(stream)
            self.a = tractor_length / 2
            self.b = settings['b']
            self.trailer_a = trailer_length / 2
            self.trailer_b = settings['trailer_b']
            self.rend_size = settings['rend_size']
            self.render = settings['render']
            self.env_size = settings['env_size']
        self.display = display
        
        if self.render:
            if not self.display: os.environ['SDL_VIDEODRIVER'] = 'dummy'
            pygame.init()
            pygame.display.set_caption("Episode Render")
            clock = pygame.time.Clock()
            clock.tick(fps)
            self.py_map = pygame.display.set_mode((self.rend_size, self.rend_size))
            self.rend_ratio = self.rend_size / self.env_size
            self.rend_truck(x=x_init, y=y_init, yaw_axis=yaw_init, gamma=gamma)
        
    def rend_truck(self, x, y, yaw_axis, gamma=None, steering_angle=None):
        """
        Render the truck-trailer simulation.
        Returns:
        """
        yaw_axis_x, yaw_axis_y = yaw_axis
        assert round((yaw_axis_x**2+yaw_axis_y**2), 3) == 1
        self.py_map.fill((255, 255, 255))
        
        # render truck
        if yaw_axis_x == 0:
            if yaw_axis_y == 1:
                yaw = np.pi/2
            elif yaw_axis_y == -1:
                yaw = np.pi/2
            else:
                raise NotImplementedError
        else:
            yaw = math.atan2(yaw_axis_y, yaw_axis_x)
        # truck body
        # points = rectangle_points(x, y, dx=self.a*2, dy=self.b*2, rotation=yaw)
        points = rectangle_points(x+self.a*yaw_axis_x, y+self.a*yaw_axis_y, dx=self.a*2, dy=self.b*2, rotation=yaw)
        points = self._sur_coord(points)
        pygame.draw.polygon(self.py_map, (125,0,0), points)
        
        # truck glass
        # points = rectangle_points(x+1.25*self.a*yaw_axis_x, y+1.25*self.a*yaw_axis_y, dx=self.a/2, dy=self.b*1.5, rotation=yaw)
        points = rectangle_points(x+2.25*self.a*yaw_axis_x, y+2.25*self.a*yaw_axis_y, dx=self.a/2, dy=self.b*1.5, rotation=yaw)
        points = self._sur_coord(points)
        pygame.draw.polygon(self.py_map, (0,0,200), points)
        
        
        steering_angle_ = steering_angle if steering_angle is not None else 0
        
        # truck wheels
        points = rectangle_points(x+self.a*yaw_axis_x-(self.b+self.a/8)*yaw_axis_y, y+self.a*yaw_axis_y+(self.b+self.a/8)*yaw_axis_x, 
                                  dx=self.a, dy=self.a/4, rotation=yaw+steering_angle_)
        points = self._sur_coord(points)
        pygame.draw.polygon(self.py_map, (0,0,0), points)
        points = rectangle_points(x+self.a*yaw_axis_x+(self.b+self.a/8)*yaw_axis_y, y+self.a*yaw_axis_y-(self.b+self.a/8)*yaw_axis_x, 
                                  dx=self.a, dy=self.a/4, rotation=yaw+steering_angle_)
        points = self._sur_coord(points)
        pygame.draw.polygon(self.py_map, (0,0,0), points)
        
        if steering_angle is not None:
            # arrow
            arrow_strat_x, arrow_strat_y = self._sur_coord(np.array([[x+self.a*yaw_axis_x, y+self.a*yaw_axis_y]]))[0]
            yaw_end_x, yaw_end_y = self._sur_coord(np.array([[x+5*self.a*yaw_axis_x, y+5*self.a*yaw_axis_y]]))[0]
            arrow_end_x, arrow_end_y = self._sur_coord(np.array([[x+self.a*yaw_axis_x+3*self.a*(yaw_axis_x*np.cos(steering_angle)-yaw_axis_y*np.sin(steering_angle)), 
                                                                y+self.a*yaw_axis_y+3*self.a*(yaw_axis_x*np.sin(steering_angle)+yaw_axis_y*np.cos(steering_angle))]]))[0]
            # yaw angle
            pygame.draw.line(self.py_map, (255,0,0), [arrow_strat_x, arrow_strat_y], [yaw_end_x, yaw_end_y], 2)
            # steering angle
            self.draw_arrow(self.py_map, 
                            pygame.Vector2(arrow_strat_x, arrow_strat_y), 
                            pygame.Vector2(arrow_end_x, arrow_end_y),
                            pygame.Color("dodgerblue"),
                            2, 20, 12)
        
        if gamma is not None:
            # print(f"yaw = {(yaw)*180/np.pi} yaw-gamma = {(yaw-gamma)*180/np.pi}")
            # points = trailer_rectangle_points(x-(self.a)*yaw_axis_x, y-(self.a)*yaw_axis_y, 
            #                                   dx=self.trailer_a*2, dy=self.trailer_b*2, rotation=yaw-gamma)
            # points = trailer_rectangle_points(x, y, 
            #                                   dx=self.trailer_a*2, dy=self.trailer_b*2, rotation=yaw-gamma)
            
            trailer_yaw_axis = np.array( [ [ cos(gamma), -sin(gamma) ], [sin(gamma), cos(gamma)] ])@yaw_axis
            
            trailer_heading_angle = yaw+gamma
            
            points = rectangle_points(x-self.trailer_a*trailer_yaw_axis[0], y-self.trailer_a*trailer_yaw_axis[1], 
                                  dx=self.trailer_a*2, dy=self.trailer_b*2, rotation=trailer_heading_angle)
            
            points = self._sur_coord(points)
            pygame.draw.polygon(self.py_map, (0,255,0), points)
            
            # trailer wheels
            front_axis = self.a*1.5
            cos_minus = yaw_axis_x*np.cos(gamma) - yaw_axis_y*np.sin(gamma)
            sin_minus = yaw_axis_y*np.cos(gamma) + yaw_axis_x*np.sin(gamma)
            points = rectangle_points(x-front_axis*cos_minus-(self.trailer_b+self.a/8)*sin_minus, 
                                      y-front_axis*sin_minus+(self.trailer_b+self.a/8)*cos_minus, 
                                    dx=self.a, dy=self.a/4, rotation=trailer_heading_angle)
            points = self._sur_coord(points)
            pygame.draw.polygon(self.py_map, (0,0,0), points)
            points = rectangle_points(x-front_axis*cos_minus+(self.trailer_b+self.a/8)*sin_minus, 
                                      y-front_axis*sin_minus-(self.trailer_b+self.a/8)*cos_minus, 
                                    dx=self.a, dy=self.a/4, rotation=trailer_heading_angle)
            points = self._sur_coord(points)
            pygame.draw.polygon(self.py_map, (0,0,0), points)
            
            tail_axis = self.a*4.5
            points = rectangle_points(x-tail_axis*cos_minus-(self.trailer_b+self.a/8)*sin_minus, 
                                      y-tail_axis*sin_minus+(self.trailer_b+self.a/8)*cos_minus, 
                                    dx=self.a, dy=self.a/4, rotation=trailer_heading_angle)
            points = self._sur_coord(points)
            pygame.draw.polygon(self.py_map, (0,0,0), points)
            points = rectangle_points(x-tail_axis*cos_minus+(self.trailer_b+self.a/8)*sin_minus, 
                                      y-tail_axis*sin_minus-(self.trailer_b+self.a/8)*cos_minus, 
                                    dx=self.a, dy=self.a/4, rotation=trailer_heading_angle)
            points = self._sur_coord(points)
            pygame.draw.polygon(self.py_map, (0,0,0), points)
        
        # goal box
        goal_x = 25
        goal_y = 9.5
        points = rectangle_points(goal_x, goal_y, dx=self.b*2.2, dy=(self.a + self.trailer_a)*2.2)
        points = self._sur_coord(points)
        pygame.draw.polygon(self.py_map, (255,0,0), points, 5)
        
        pygame.display.update()
        image_str = pygame.image.tobytes(self.py_map, 'RGB')
        image = Image.frombytes('RGB', self.py_map.get_size(), image_str)
        return np.array(image)
    
    def _sur_coord(self, point):
        """
        transform to pygame surface coordinate
        :param point, point in world coordinate
        :return:
        """
        if type(point) is np.ndarray:
            point[:, 0] = (point[:, 0]*self.rend_ratio).copy()
            point[:, 1] = ((self.rend_size-point[:, 1]*self.rend_ratio)).copy()
            return point
        else:
            raise NotImplementedError

    def draw_arrow(
            self,
            surface: pygame.Surface,
            start: pygame.Vector2,
            end: pygame.Vector2,
            color: pygame.Color,
            body_width: int = 2,
            head_width: int = 4,
            head_height: int = 2,
        ):
        """Draw an arrow between start and end with the arrow head at the end.

        Args:
            surface (pygame.Surface): The surface to draw on
            start (pygame.Vector2): Start position
            end (pygame.Vector2): End position
            color (pygame.Color): Color of the arrow
            body_width (int, optional): Defaults to 2.
            head_width (int, optional): Defaults to 4.
            head_height (float, optional): Defaults to 2.
        """
        arrow = start - end
        angle = arrow.angle_to(pygame.Vector2(0, -1))
        body_length = arrow.length() - head_height

        # Create the triangle head around the origin
        head_verts = [
            pygame.Vector2(0, head_height / 2),  # Center
            pygame.Vector2(head_width / 2, -head_height / 2),  # Bottomright
            pygame.Vector2(-head_width / 2, -head_height / 2),  # Bottomleft
        ]
        # Rotate and translate the head into place
        translation = pygame.Vector2(0, arrow.length() - (head_height / 2)).rotate(-angle)
        for i in range(len(head_verts)):
            head_verts[i].rotate_ip(-angle)
            head_verts[i] += translation
            head_verts[i] += start

        pygame.draw.polygon(surface, color, head_verts)

        # Stop weird shapes when the arrow is shorter than arrow head
        if arrow.length() >= head_height:
            # Calculate the body rect, rotate and translate into place
            body_verts = [
                pygame.Vector2(-body_width / 2, body_length / 2),  # Topleft
                pygame.Vector2(body_width / 2, body_length / 2),  # Topright
                pygame.Vector2(body_width / 2, -body_length / 2),  # Bottomright
                pygame.Vector2(-body_width / 2, -body_length / 2),  # Bottomleft
            ]
            translation = pygame.Vector2(0, body_length / 2).rotate(-angle)
            for i in range(len(body_verts)):
                body_verts[i].rotate_ip(-angle)
                body_verts[i] += translation
                body_verts[i] += start

            pygame.draw.polygon(surface, color, body_verts)