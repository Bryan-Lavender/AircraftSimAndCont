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


class DrawB2:
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

        points = [
            [4,0,0],       #0 
            [3,-1,0],      #1 
            [3,0,-.5],     #2
            [3,0,0],       #3
            
            [0, 0,-2],     #4
            [0,-2.5,-.5],  #5

            [-4,-1.5,-1.5],#6
            [-6,0,-1.5],   #7

            [-6, -1.5, -1.5],  #8
            [-16,-20,0],   #9
            [-17,-20,-.5], #10
            [-17, 0, -0.5],#11

            
            [-6, -1.75, -.5],#12
            [-16, -20,0], # 13
            [-16, 0, 0], # 14

            [-5, -6, -.5], # 15
            [-5, -5.5, -1.5], # 16
            [-5, -4.5, -1.5], # 17
            [-5, -3.5, -1.5], # 18
            [-5, -3, -.5], # 19
            [-5.5, -4, -1.5], # 20
            [-5.5, -5, -1.5], # 21
            [-11, -6, -.5], # 22
            [-11, -5.5, -.5], # 23
            [-11, -3.5, -.5], # 24
            [-11, -3, -.5], # 25


            [-5.5,-5.5,-1.5], #26
            [-5.5,-3.5,-1.5], #27
            ]
        static_len = len(points)
        for i in range(0, len(points)):
            
            points.append([points[i][0], -1 *points[i][1], points[i][2]])
        
        points = self.unit_length * 10/10 * np.array(points).T
        # point index that defines the mesh
        index = [
            [0, 1, 2],  #1  beak top left
            [0,1,3],     #2  beak bottom left
    
            [4,5,2], #3  front left panal

            [6,4,5], #4 pit left panal
            [6,7,4], #5 pit right panal

            [6,7,8], #6 pit back detail

            [1,9,10],#wing edge
            [1,10,2], #wing edge

        
            [2,11,10], #big upper triangle
            [5,8,6],

            [7,8,12],
            [11,12,7],
            [5,12,8],
            [0,13,14],

            [9,10,11],
            [11,14,9],

            #ENGINE BEGINNING AT 16
            [16,21,26],
            [21,20,17],
            [20,27,18],
            [23,24,26],
            [24,26,27],
            [23,15,16],
            [16,23,26],
            
            [24,25,18],
            [18,19,25]
            ]
        static_len_2 = len(index)
        for i in range(len(index)):
            index.append([index[i][0] + static_len, index[i][1] + static_len, index[i][2] + static_len])
        print(index)
        index = np.array(index)
        #   define the colors for each face of triangular mesh
        red = np.array([1., 0., 0., 1])
        green = np.array([0., 1., 0., 1])
        blue = np.array([0., 0., 1., 1])
        yellow = np.array([1., 1., 0., 1])
        meshColors = np.empty((index.shape[0], 3, 4), dtype=np.float32)

        meshColors[0] = yellow  # Top Nose Face
        meshColors[1] = yellow    # Bottom Nose Face
        meshColors[2] = blue
        meshColors[3] = green
        meshColors[4] = blue
        meshColors[5] = green
        meshColors[6] = yellow
        meshColors[7] = yellow
        meshColors[8] = yellow
        meshColors[9] = blue
        meshColors[10] = green
        meshColors[11] = blue
        meshColors[12] = green
        meshColors[13] = yellow
        meshColors[14] = yellow
        meshColors[15] = yellow

        for i in range(16, static_len_2):
            meshColors[i]= red
        for i in range(static_len_2, len(index)):
            meshColors[i] = meshColors[i - static_len_2]
        return points, index, meshColors

