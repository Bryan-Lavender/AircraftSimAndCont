#Copy and paste draw_Mav.py here and then edit to draw a UAV instead of the Mav.
"""
Mavsim_python: drawing tools
    - Beard & McLain, PUP, 2012
    - Update history:
        1/13/2021 - TWM
        7/13/2023 - RWB
        1/16/2024 - RWB
"""
import numpy as np
import pyqtgraph.opengl as gl
from tools.rotations import euler_to_rotation
from tools.drawing import rotate_points, translate_points, points_to_mesh


class DrawMav:
    def __init__(self, state, window, scale=10):
        """
        Draw the Mav.

        The input to this function is a (message) class with properties that define the state.
        The following properties are assumed:
            state.north  # north position
            state.east  # east position
            state.altitude   # altitude
            state.phi  # roll angle
            state.theta  # pitch angle
            state.psi  # yaw angle
        """
        self.unit_length = scale
        sc_position = np.array([[state.north], [state.east], [-state.altitude]])  # NED coordinates
        # attitude of Mav as a rotation matrix R from body to inertial
        R_bi = euler_to_rotation(state.phi, state.theta, state.psi)
        # convert North-East Down to East-North-Up for rendering
        self.R_ned = np.array([[0, 1, 0], [1, 0, 0], [0, 0, -1]])
        # get points that define the non-rotated, non-translated Mav and the mesh colors
        self.sc_points, self.sc_index, self.sc_meshColors = self.get_sc_points()
        self.sc_body = self.add_object(
            self.sc_points,
            self.sc_index,
            self.sc_meshColors,
            R_bi,
            sc_position)
        window.addItem(self.sc_body)  # add Mav to plot     

    def update(self, state):
        sc_position = np.array([[state.north], [state.east], [-state.altitude]])  # NED coordinates
        # attitude of Mav as a rotation matrix R from body to inertial
        R_bi = euler_to_rotation(state.phi, state.theta, state.psi)
        self.sc_body = self.update_object(
            self.sc_body,
            self.sc_points,
            self.sc_index,
            self.sc_meshColors,
            R_bi,
            sc_position)

    def add_object(self, points, index, colors, R, position):
        rotated_points = rotate_points(points, R)
        translated_points = translate_points(rotated_points, position)
        translated_points = self.R_ned @ translated_points
        # convert points to triangular mesh defined as array of three 3D points (Nx3x3)
        mesh = points_to_mesh(translated_points, index)
        object = gl.GLMeshItem(
            vertexes=mesh,  # defines the triangular mesh (Nx3x3)
            vertexColors=colors,  # defines mesh colors (Nx1)
            drawEdges=True,  # draw edges between mesh elements
            smooth=False,  # speeds up rendering
            computeNormals=False)  # speeds up rendering
        return object

    def update_object(self, object, points, index, colors, R, position):
        rotated_points = rotate_points(points, R)
        translated_points = translate_points(rotated_points, position)
        translated_points = self.R_ned @ translated_points
        # convert points to triangular mesh defined as array of three 3D points (Nx3x3)
        mesh = points_to_mesh(translated_points, index)
        object.setMeshData(vertexes=mesh, vertexColors=colors)
        return object

    def get_sc_points(self):
        """"
            Points that define the Mav, and the colors of the triangular mesh
            Define the points on the Mav following information in Appendix C.3
        """
        # points are in XYZ coordinates
        #   define the points on the Mav according to Appendix C.3
        fuse_l1 = .5
        fuse_l2 = .3
        fuse_l3 = 1

        points = self.unit_length * np.array([
            
            [.5, 0, .1],  # point 1 [0]
            [.3, .2, -.2],  # point 2 [1]
            [.3, -.2, -.2],  # point 3 [2]
            [.3, -.2, .3],  # point 4 [3] z axis for now
            [.3, .2, .3],  # point 5 [4]

            
            [-1, 0, 0],  # point 6 [5]

            [0, .5, 0],  # point 7 [6]
            [-.4, .5, 0],  # point 8 [7]
            [-.4, -.5, 0],  # point 9 [8]
            [0, -.5, 0],  # point 10 [9]


            [-.7, .3, 0],  # point 11 [10]
            [-1, .3, 0],  # point 12 [11]
            [-1, -.3, 0],  # point 13 [12]
            [-.7, -.3, 0],  # point 14 [13]

            [-.7, 0, 0],  # point 15 [14]
            [-1, 0, -.4],  # point 16 [15]
            ]).T
        # point index that defines the mesh
        index = np.array([
            [0, 1, 2],  # Mesh 0 Top Nose Face
            [0, 2, 3],  # left face from x axis
            [0, 3, 4],  # bottom face
            [0, 4, 1],  # right face from x axis
            
            [1, 2, 5], #top
            [2, 3, 5], #left side from x axis
            [4, 1, 5], # right face form x axis
            [3, 4, 5], #bottom

            [6, 7, 9],  # wing 1
            [7, 8, 9],  # wing 2

            [10,11,13],  # tail wing 1
            [11,12,13],  # tail wing 2
            [5,15,14],  # tail wing top rutter
            # [1, 5, 6],  # left 1
            # [1, 6, 2],  # left 2
            # [4, 5, 6],  # top 1
            # [4, 6, 7],  # top 2
            # [8, 9, 10],  # bottom 1
            # [8, 10, 11],  # bottom 2  
            ])
        #   define the colors for each face of triangular mesh
        red = np.array([1., 0., 0., 1])
        green = np.array([0., 1., 0., 1])
        blue = np.array([0., 0., 1., 1])
        yellow = np.array([1., 1., 0., 1])
        meshColors = np.empty((13, 3, 4), dtype=np.float32)
        meshColors[0] = yellow  # Top Nose Face
        meshColors[1] = yellow  # front 2
        meshColors[2] = yellow  # back 1
        meshColors[3] = yellow  # back 2

        meshColors[4] = blue  # right 1
        meshColors[5] = blue  # right 2
        meshColors[6] = blue  # left 1
        meshColors[7] = blue  # left 2
        
        meshColors[8] = red  # top 1
        meshColors[9] = red  # top 2
        
        meshColors[10] = green  # bottom 1
        meshColors[11] = green  # bottom 2
        meshColors[12] = red
        return points, index, meshColors

