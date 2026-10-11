'''
Program written by Jackson Vanderkooy for use in AER1403 2026
Absolutely no AI was used in the writing of this program.
Intended to be run in a terminal with python version 3.11.5
'''

from math import cos, sin, pi
import matplotlib.pyplot as plt
import numpy as np

def deg2rad(angle):
    return angle*(pi/180.0)

def rad2deg(angle):
    return angle/(pi/180.0)

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

        #Formulas from notes chapter 1 pg 21
        # self.Sbar[0,0] = S[0,0]*(cos(theta)**4) + S[1,1]*(sin(theta)**4) + 2*(S[0,1] + (2*S[2,2]))*(sin(theta)**2)*(cos(theta)**2)
        # self.Sbar[0,1] = (S[0,0] + S[1,1] - (4*S[2,2]))*(cos(theta)**2)*(sin(theta)**2) + S[0,1]*((cos(theta)**4) + (sin(theta)**4))
        # self.Sbar[1,0] = self.Sbar[0,1]
        # self.Sbar[1,1] = S[0,0]*(sin(theta)**4) + S[1,1]*(cos(theta)**4) + 2*(S[0,1] + 2*S[2,2])*(sin(theta)**2)*(cos(theta)**2)
        # self.Sbar[0,2] = (S[0,0] - S[0,1] - 2*S[2,2])*(cos(theta)**3)*(sin(theta)) - (S[1,1] - S[0,1] - 2*S[2,2])*(cos(theta))*(sin(theta)**3)
        # self.Sbar[2,0] = self.Sbar[0,2]
        # self.Sbar[1,2] = (S[0,0] - S[0,1] - 2*S[2,2])*(cos(theta))*(sin(theta)**3) - (S[1,1] - S[0,1] - 2*S[2,2])*(cos(theta)**3)*(sin(theta))
        # self.Sbar[2,1] = self.Sbar[1,2]
        # self.Sbar[2,2] = (S[0,0] + S[1,1] - 2*S[0,1] - 2*S[2,2])*(cos(theta)**2)*(sin(theta)**2) + S[2,2]*((cos(theta)**4) + (sin(theta)**4))

        # print(self.Sbar)
        self.Sbar = np.matmul(np.linalg.inv(self.T),np.matmul(self.S,np.linalg.inv(np.transpose(self.T))))

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

    def build_stiffness(self):
        ## Building A
        A = np.array([[0,0,0],[0,0,0],[0,0,0]], dtype='d')
        for i in range(3):
            for j in range(3):
                for k in range(self.numLayers):
                    layer = self.layers[k]
                    A[i,j] = A[i,j] + layer.Sbar[i,j]*(self.z_interfaces[k+1]-self.z_interfaces[k])

        ## Building B
        B = np.array([[0,0,0],[0,0,0],[0,0,0]], dtype='d')
        for i in range(3):
            for j in range(3):
                for k in range(self.numLayers):
                    layer = self.layers[k]
                    B[i,j] = B[i,j] + layer.Sbar[i,j]*(((self.z_interfaces[k+1])**2)-(self.z_interfaces[k]**2))
                B[i,j] = B[i,j] / 2

        ## Building D
        D = np.array([[0,0,0],[0,0,0],[0,0,0]], dtype='d')
        for i in range(3):
            for j in range(3):
                for k in range(self.numLayers):
                    layer = self.layers[k]
                    D[i,j] = D[i,j] + layer.Sbar[i,j]*(((self.z_interfaces[k+1])**3)-(self.z_interfaces[k]**3))
                D[i,j] = D[i,j] / 3

        self.plate_S = np.block([[A,B],[B,D]])

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

    def build_layer_stresses(self):
        for i in range(len(self.layers)):
            layer_strain = np.array([0]*3,dtype='d')
            layer_strain[0] = self.plate_strains[0]+(self.z_midplanes[i]*self.plate_strains[3])
            layer_strain[1] = self.plate_strains[1]+(self.z_midplanes[i]*self.plate_strains[4])
            layer_strain[2] = self.plate_strains[2]+(self.z_midplanes[i]*self.plate_strains[5])
            self.layers[i].global_axis_stress = np.matmul(self.layers[i].Sbar,layer_strain)
            self.layers[i].local_axis_stress = np.matmul(self.layers[i].T,self.layers[i].global_axis_stress)
            

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

def max_stress_failure_check(sigma_L_minus,sigma_L_plus,sigma_T_minus,sigma_T_plus,tau_LT,sigma_11,sigma_22,tau_12,print_failures=False):
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

def tsaihill(sigma_L_minus,sigma_L_plus,sigma_T_minus,sigma_T_plus,tau_LT,sigma_11,sigma_22,tau_12,print_failures=False):

    term1 = (((sigma_11**2)/(sigma_L_plus**2))*(sigma_11 > 0) + ((sigma_11**2)/(sigma_L_minus**2))*(sigma_11 < 0))
    term2 = ((sigma_11*sigma_22)/(sigma_L_plus**2))*(sigma_11*sigma_22 > 0) + ((sigma_11*sigma_22)/(sigma_L_minus**2))*(sigma_11*sigma_22 < 0)
    term3 = (((sigma_22**2)/(sigma_T_plus**2))*(sigma_22 > 0) + ((sigma_22**2)/(sigma_T_minus**2))*(sigma_22 < 0))
    term4 = (tau_12**2)/(tau_LT**2)

    result = term1 - term2 + term3 + term4
    failed = result >= 1

    if print_failures:
        if failed: print(f'Failure: {result} >= 1')
        else: print(f'No failues.')
        print(result)

    return failed

def tsaiwu_factors(sigma_L_minus,sigma_L_plus,sigma_T_minus,sigma_T_plus,tau_LT,F12=None):
    F11 = 1/abs(sigma_L_plus*sigma_L_minus)
    F22 = 1/abs(sigma_T_plus*sigma_T_minus)
    if F12 == None: #Using assumed F12
        F12 = -((F11*F22)**(1/2))/2
    F1 = (1/sigma_L_plus)-(1/abs(sigma_L_minus))
    F2 = (1/sigma_T_plus)-(1/abs(sigma_T_minus))
    F66 = 1/(tau_LT**2)

    return [F11,F22,F12,F1,F2,F66]



[F11,F22,F12,F1,F2,F66] = tsaiwu_factors(700,850,130,35,60)

x_vals = []
y_vals_pos = []
y_vals_neg = []

for i in range(0,900):
    i = i/10.0
    x_vals.append(i)
    t = deg2rad(i)
    A = (F11*(cos(t)**4) + F22*(sin(t)**4) + 2*F12*(sin(t)**2)*(cos(t)**2) + F66*(cos(t)*sin(t))**2)
    B = (F1*(cos(t)**2) + F2*(sin(t)**2))
    C = -1.0

    y_vals_pos.append((-B + ((B**2) - (4*A*C))**(1/2))/(2*A))
    y_vals_neg.append((-B - ((B**2) - (4*A*C))**(1/2))/(2*A))

plt.plot(x_vals,y_vals_pos)
plt.plot(x_vals,y_vals_neg)
plt.title("Max and min uniaxial stresses in the xx direction for range of layer angles.")
plt.xlabel("Theta [deg]")
plt.ylabel("Max Stress [MPa]")
plt.grid()
plt.show()