'''
Program written by Jackson Vanderkooy for use in AER1403 2026
Absolutely no AI was used in the writing of this program.
Intended to be run in a terminal with python version 3.11.5
'''

from math import cos, sin, pi
import matplotlib.pyplot as plt
import numpy as np

class laminae:
    def __init__(self,E1,E2,G12,v12,t=0):
        self.E1 = E1
        self.E2 = E2
        self.G12 = G12
        self.v12 = v12
        self.v21 = self.v12*(self.E2 / self.E1)
        self.global_axis_stress = np.array([0, 0, 0],dtype='d')
        self.local_axis_stress = np.array([0, 0, 0],dtype='d')
        self.t = t
        self.T = np.array([[0]*3]*3,dtype='d')
        self.calc_stiff_local()

        self.is_global_aligned = False

    def calc_stiff_local(self):
        E1 = self.E1
        E2 = self.E2
        v12 = self.v12
        v21 = self.v21
        G12 = self.G12

        #Formulas from notes chapter 1 page 17
        self.S = np.array([[0,0,0],[0,0,0],[0,0,0]], dtype='d')
        self.S[0,0] = (E1/(1-(v12*v21)))
        self.S[0,1] = ((v12*E2)/(1-(v12*v21)))
        self.S[1,0] = self.S[0,1]
        self.S[1,1] = (E2/(1-(v12*v21)))
        self.S[2,2] = G12

    def global_align(self, theta=0):
        S = self.S
        self.Sbar = np.array([[0,0,0],[0,0,0],[0,0,0]], dtype='d')
        # self.SbarT = np.array([[0,0,0],[0,0,0],[0,0,0]], dtype='d')
        theta = theta*(pi/180.0)
        self.theta = theta

        self.T[0,0] = cos(theta)**2
        self.T[0,1] = sin(theta)**2
        self.T[0,2] = 2*cos(theta)*sin(theta)
        self.T[1,0] = sin(theta)**2
        self.T[1,1] = cos(theta)**2
        self.T[1,2] = -2*cos(theta)*sin(theta)
        self.T[2,0] = -cos(theta)*sin(theta)
        self.T[2,1] = cos(theta)*sin(theta)
        self.T[2,2] = ((cos(theta)**2) - (sin(theta)**2))

        # print(self.T)

        # self.Sbar[0,0] = S[0,0]*(cos(theta)**4) + S[1,1]*(sin(theta)**4) + 2*(S[0,1] + (2*S[2,2]))*(sin(theta)**2)*(cos(theta)**2)
        # self.Sbar[0,1] = (S[0,0] + S[1,1] - (4*S[2,2]))*(cos(theta)**2)*(sin(theta)**2) + S[0,1]*((cos(theta)**4) + (sin(theta)**4))
        # self.Sbar[1,0] = self.Sbar[0,1]
        # self.Sbar[1,1] = S[0,0]*(sin(theta)**4) + S[1,1]*(cos(theta)**4) + 2*(S[0,1] + 2*S[2,2])*(sin(theta)**2)*(cos(theta)**2)
        # self.Sbar[0,2] = (S[0,0] - S[0,1] - 2*S[2,2])*(cos(theta)**3)*(sin(theta)) - (S[1,1] - S[0,1] - 2*S[2,2])*(cos(theta))*(sin(theta)**3)
        # self.Sbar[2,0] = self.Sbar[0,2]
        # self.Sbar[1,2] = (S[0,0] - S[0,1] - 2*S[2,2])*(cos(theta))*(sin(theta)**3) - (S[1,1] - S[0,1] - 2*S[2,2])*(cos(theta)**3)*(sin(theta))
        # self.Sbar[2,1] = self.Sbar[1,2]
        # self.Sbar[2,2] = (S[0,0] + S[1,1] - 2*S[0,1] - 2*S[2,2])*(cos(theta)**2)*(sin(theta)**2) + S[2,2]*((cos(theta)**4) + (sin(theta)**4))
        

        self.Sbar = np.matmul(np.linalg.inv(self.T),np.matmul(self.S,np.linalg.inv(np.transpose(self.T))))

        print(f'Sbar Layer 2={self.Sbar}')
        # print(self.SbarT-self.Sbar)

        self.is_global_aligned = True

    def get_local_stiffness(self):
        return self.S

    def get_global_stiffness(self):
        if not self.is_global_aligned:
            return None
        return self.Sbar

class laminate:
    def __init__(self,layers=[]):
        self.layers = layers
        self.numLayers = len(layers)
        self.build_z()
        self.build_stiffness()
        # self.build_plate_strains()
        # self.build_layer_stresses()

    def build_z(self):
        self.z_interfaces = np.array([0]*(len(self.layers)+1), dtype='d')
        self.z_midplanes = np.array([0]*len(self.layers), dtype='d')
        self.totalThickness = 0
        
        for i in range(len(self.z_interfaces)-1):
            self.totalThickness += self.layers[i].t
            self.z_interfaces[i+1] = self.totalThickness
            self.z_midplanes[i] = (self.z_interfaces[i]+self.z_interfaces[i+1])/2
        self.z_interfaces = self.z_interfaces - (self.totalThickness/2)
        self.z_midplanes = self.z_midplanes - (self.totalThickness/2)
        # print(f'interface:{self.z_interfaces}')
        # print(f'midplanes:{self.z_midplanes}')

    def build_stiffness(self):
        ## Building A
        A = np.array([[0,0,0],[0,0,0],[0,0,0]], dtype='d')
        for i in range(3):
            for j in range(3):
                for k in range(self.numLayers):
                    layer = self.layers[k]
                    A[i,j] = A[i,j] + layer.Sbar[i,j]*(self.z_interfaces[k+1]-self.z_interfaces[k])
                # A[i,j] = A[i,j] *

        ## Building B
        B = np.array([[0,0,0],[0,0,0],[0,0,0]], dtype='d')
        for i in range(3):
            for j in range(3):
                for k in range(self.numLayers):
                    layer = self.layers[k]
                    B[i,j] = B[i,j] + ((1/2)*layer.Sbar[i,j]*(((self.z_interfaces[k+1])**2)-(self.z_interfaces[k]**2)))
                # B[i,j] = B[i,j] *1000

        ## Building D
        D = np.array([[0,0,0],[0,0,0],[0,0,0]], dtype='d')
        for i in range(3):
            for j in range(3):
                for k in range(self.numLayers):
                    layer = self.layers[k]
                    D[i,j] = D[i,j] + ((1/3)*layer.Sbar[i,j]*(((self.z_interfaces[k+1])**3)-(self.z_interfaces[k]**3)))
                # D[i,j] = D[i,j] *1000

        self.plate_S = np.block([[A,B],[B,D]])
        self.plate_S = self.plate_S
        print(f'A: {A}')
        print(f'B: {B}')
        print(f'D: {D}')
        # print(self.plate_S)

    def get_stackup_symbolic(self):
        stackupString = '['
        for i in range(len(self.layers)):
            stackupString = f'{stackupString}{round(self.layers[i].theta / (pi/180),2)}'
            if not i == (len(self.layers)-1):
                stackupString = f'{stackupString}\\'
            else:
                stackupString = f'{stackupString}]'
        return stackupString

    def build_plate_strains(self,Nx,Ny,Nxy,Mx,My,Mxy):
        self.force_vec = np.array([Nx,Ny,Nxy,Mx,My,Mxy],dtype='d')
        invert_plate_S = np.linalg.inv(self.plate_S)
        self.plate_strains = np.matmul(invert_plate_S,self.force_vec)
        # self.plate_strains[0:3] = self.plate_strains[0:3]/1000 #Accounting for unit different between kN and GPa
        # print(self.plate_strains)

    def build_layer_stresses(self):
        for i in range(len(self.layers)):
            layer_strain = np.array([0]*3,dtype='d')
            layer_strain[0] = self.plate_strains[0]+(self.z_midplanes[i]*self.plate_strains[3])
            layer_strain[1] = self.plate_strains[1]+(self.z_midplanes[i]*self.plate_strains[4])
            layer_strain[2] = self.plate_strains[2]+(self.z_midplanes[i]*self.plate_strains[5])
            self.layers[i].local_axis_stress = np.matmul(self.layers[i].Sbar,layer_strain)
            # self.layers[i].local_axis_stress = np.matmul(self.layers[i].T,self.layers[i].global_axis_stress)
            # print(layer_strain)
            

def buildLayer():
    E1 = float(input("E1 [GPa]:"))
    E2 = float(input("E2 [GPa]:"))
    G12 = float(input("G12 [GPa]:"))
    v12 = float(input("v12:"))
    return laminae(E1,E2,G12,v12)

def buildLaminate():
    print('How many layers?')
    numLayer = int(input('->'))
    layers = []
    matchingLayers = False
    for i in range(numLayer):
        print(f'Layer #{i+1}')
        if not matchingLayers:
            E1 = float(input("E1 [GPa]:"))
            E2 = float(input("E2 [GPa]:"))
            G12 = float(input("G12 [GPa]:"))
            v12 = float(input("v12:"))
            t = float(input("t:"))
            if (input('Are all layers identical? (y/n) ->') == 'y') : matchingLayers = True
        theta = float(input("Theta:"))
        layers.append(laminae(E1,E2,G12,v12,t))
        layers[i].global_align(theta*(pi/180))

    return laminate(layers)

def max_stress_failure_check(LayerStrengths,LayerStresses,print_failures=False):
    sigma_L_minus,sigma_L_plus,sigma_T_minus,sigma_T_plus,tau_LT = LayerStrengths
    sigma_11,sigma_22,tau_12 = LayerStresses
    
    sigma_L_minus = abs(sigma_L_minus)
    sigma_T_minus = abs(sigma_T_minus)
    sigma_L_plus = abs(sigma_L_plus)
    sigma_T_plus = abs(sigma_T_plus)
    tau_LT = abs(tau_LT)

    failures = [0,0,0] # Longitudinal, Transverse, Shear
    failures[0] = -1*(sigma_11 <= -sigma_L_minus) + 1*(sigma_11 >= sigma_L_plus)
    failures[1] = -1*(sigma_22 <= -sigma_T_minus) + 1*(sigma_22 >= sigma_T_plus)
    failures[2] = 1*(abs(tau_12) >= tau_LT)

    if print_failures:
        if failures[0] > 0: print(f'Failure: sigma_11={sigma_11} > sigma_L_plus={sigma_L_plus}')
        if failures[0] < 0: print(f'Failure: sigma_11={sigma_11} < -sigma_L_minus={sigma_L_minus}')
        if failures[1] > 0: print(f'Failure: sigma_22={sigma_22} > sigma_T_plus={sigma_T_plus}')
        if failures[1] < 0: print(f'Failure: sigma_22={sigma_22} < -sigma_T_minus={sigma_T_minus}')
        if failures[2] > 0: print(f'Failure: |tau_12|={abs(sigma_22)} > tau_LT={tau_LT}')
        if not(failures[0] or failures[1] or failures[2]): print(f'No failures.')

    return failures

def tsaihill(LayerStrengths,LayerStresses,print_failures=False):

    sigma_L_minus,sigma_L_plus,sigma_T_minus,sigma_T_plus,tau_LT = LayerStrengths
    sigma_11,sigma_22,tau_12 = LayerStresses

    term1 = (((sigma_11**2)/(sigma_L_plus**2))*(sigma_11 > 0) + ((sigma_11**2)/(sigma_L_minus**2))*(sigma_11 < 0))
    term2 = ((sigma_11*sigma_22)/(sigma_L_plus**2))*(sigma_11 > 0) + ((sigma_11*sigma_22)/(sigma_L_minus**2))*(sigma_11 < 0)
    term3 = (((sigma_22**2)/(sigma_T_plus**2))*(sigma_22 > 0) + ((sigma_22**2)/(sigma_T_minus**2))*(sigma_22 < 0))
    term4 = (tau_12**2)/(tau_LT**2)

    result = term1 - term2 + term3 + term4
    failed = result >= 1

    if print_failures:
        if failed: print(f'Failure: {result} >= 1')
        else: print(f'No failues.')
        print(result)

    return result

def tsaiwu(LayerStrengths,LayerStresses,F12=None,print_failures=False):
    sigma_L_minus,sigma_L_plus,sigma_T_minus,sigma_T_plus,tau_LT = LayerStrengths
    sigma_11,sigma_22,tau_12 = LayerStresses
    F11 = 1/abs(sigma_L_plus*sigma_L_minus)
    F22 = 1/abs(sigma_T_plus*sigma_T_minus)
    if F12 == None: #Using assumed F12
        F12 = -((F11*F22)**(1/2))/2
    F1 = (1/sigma_L_plus)-(1/abs(sigma_L_minus))
    F2 = (1/sigma_T_plus)-(1/abs(sigma_T_minus))
    F66 = 1/(tau_LT**2)

    result = F11*(sigma_11**2) + F22*(sigma_22**2) + F66*(tau_12**2) + F1*sigma_11 + F2*sigma_22 + 2*F12*sigma_11*sigma_22
    failed = result >= 1

    if print_failures:
            if failed: print(f'Failure: {result} >= 1')
            else: print(f'No failues.')
            print(result)

    return result

layer1 = laminae(E1=125.0,E2=8.8,G12=6.8,v12=0.24,t=0.125)
layer3 = laminae(E1=125.0,E2=8.8,G12=6.8,v12=0.24,t=0.125)

layer1.global_align(30.0)
layer3.global_align(60.0)

LayerStrengths = [750.0,900.0,180.0,50.0,80.0] # sigmaL-, sigmaL+, sigmaT-, sigmaT+, tauLT

angles = []
ms_failed = []
th_failed = []
tw_failed = []

for i in range(-90,91):   
    print(f'\nAngle i:{i}')
    layer2 = laminae(E1=125.0,E2=8.8,G12=6.8,v12=0.24,t=0.125)
    i = float(i)
    layer2.global_align(i)

    plate = laminate([layer1,layer2,layer3])
    plate.build_plate_strains(3,-2,0,2,2,0)
    plate.build_layer_stresses()

    print(f'Layer 1 Stress [MPa]: {plate.layers[0].local_axis_stress}')
    print(f'Layer 2 Stress [MPa]: {plate.layers[1].local_axis_stress}')
    print(f'Layer 3 Stress [MPa]: {plate.layers[2].local_axis_stress}')
    angles.append(i)

    th1 = tsaihill(LayerStrengths,plate.layers[0].local_axis_stress,print_failures=False)
    th2 = tsaihill(LayerStrengths,plate.layers[1].local_axis_stress,print_failures=False)
    th3 = tsaihill(LayerStrengths,plate.layers[2].local_axis_stress,print_failures=False)
    tw1 = tsaiwu(LayerStrengths,plate.layers[0].local_axis_stress,print_failures=False)
    tw2 = tsaiwu(LayerStrengths,plate.layers[1].local_axis_stress,print_failures=False)
    tw3 = tsaiwu(LayerStrengths,plate.layers[2].local_axis_stress,print_failures=False)

    print(f'Tsai-Hill Values for layers [1,2,3]={th1,th2,th3}')
    print(f'Tsai-Wu Values for layers [1,2,3]={tw1,tw2,tw3}')
    
    th = th1 >= 1.0 or th2 >= 1.0 or th3 >= 1.0
    tw = tw1 >= 1.0 or tw2 >= 1.0 or tw3 >= 1.0
    if th: th_failed.append(i)
    if tw: tw_failed.append(i)


print(f'Tsai-Hill predicts failure at the following angles A={th_failed}')
print(f'Tsai-Wu predicts failure at the following angles A={tw_failed}')
