# -*- coding: utf-8 -*-
"""
Created on: Tuesday 29 september 
@author   : Awanit 
"""

# this file is the extension of Ar_512_without_feasible
#isme hm ek sath ek se jada pont random starting pint leke kre skte h he na majedar 


import numpy  as np
import pandas as pd                        
from   math   import radians as DegToRad       # Degrees to radians Conversion

from shapely.geometry import Point             # Used in constraint checking
from shapely.geometry.polygon import Polygon

import warnings
warnings.filterwarnings("ignore")
import csv
import two_turbine as tt
import time


#-------------------------------------------------------------------------------------------------------------


def loadPowerCurve(power_curve_file_name):
    """
    -**-THIS FUNCTION SHOULD NOT BE MODIFIED-**-
    
    Returns a 2D numpy array with information about
    turbine thrust coeffecient and power curve of the 
    turbine for given wind speed
    
    :called_from
        main function
    
    :param
        power_curve_file_name - power curve csv file location
        
    :return
        Returns a 2D numpy array with cols Wind Speed (m/s), 
        Thrust Coeffecient (non dimensional), Power (MW)
    """
    powerCurve = pd.read_csv(power_curve_file_name, sep=',', dtype = np.float32)
    powerCurve = powerCurve.to_numpy(dtype = np.float32)
    return(powerCurve)
    

def binWindResourceData(wind_data_file_name):
    r"""
    -**-THIS FUNCTION SHOULD NOT BE MODIFIED-**-
    
    Loads the wind data. Returns a 2D array with shape (36,15). 
    Each cell in  array is a wind direction and speed 'instance'. 
    Values in a cell correspond to probability of instance
    occurence.  
    
    :Called from
        main function
        
    :param
        wind_data_file_name - Wind Resource csv file  
        
    :return
        1-D flattened array of the 2-D array shown below. Values 
        inside cells, rough probabilities of wind instance occurence. 
        Along: Row-direction (drct), Column-Speed (s). Array flattened
        for vectorization purpose. 
        
                      |0<=s<2|2<=s<4| ...  |26<=s<28|28<=s<30|
        |_____________|______|______|______|________|________|
        | drct = 360  |  --  |  --  |  --  |   --   |   --   |
        | drct = 10   |  --  |  --  |  --  |   --   |   --   |
        | drct = 20   |  --  |  --  |  --  |   --   |   --   |
        |   ....      |  --  |  --  |  --  |   --   |   --   |
        | drct = 340  |  --  |  --  |  --  |   --   |   --   |
        | drct = 350  |  --  |  --  |  --  |   --   |   --   |        
    """
    
    # Load wind data. Then, extracts the 'drct', 'sped' columns
    df = pd.read_csv(wind_data_file_name)
    wind_resource = df[['drct', 'sped']].to_numpy(dtype = np.float32)
    
    # direction 'slices' in degrees
    slices_drct   = np.roll(np.arange(10, 361, 10, dtype=np.float32), 1)
    ## slices_drct   = [360, 10.0, 20.0.......340, 350]
    n_slices_drct = slices_drct.shape[0]
    
    # speed 'slices'
    slices_sped   = [0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 
                        18.0, 20.0, 22.0, 24.0, 26.0, 28.0, 30.0]
    n_slices_sped = len(slices_sped)-1

    
    # placeholder for binned wind
    binned_wind = np.zeros((n_slices_drct, n_slices_sped), 
                           dtype = np.float32)
    
    # 'trap' data points inside the bins. 
    for i in range(n_slices_drct):
        for j in range(n_slices_sped):     
            
            # because we already have drct in the multiples of 10
            foo = wind_resource[(wind_resource[:,0] == slices_drct[i])] 

            foo = foo[(foo[:,1] >= slices_sped[j]) 
                          & (foo[:,1] <  slices_sped[j+1])]
            
            binned_wind[i,j] = foo.shape[0]  
    
    wind_inst_freq   = binned_wind/np.sum(binned_wind)
    wind_inst_freq   = wind_inst_freq.ravel()
    
    return(wind_inst_freq)


def searchSorted(lookup, sample_array):
    """
    -**-THIS FUNCTION SHOULD NOT BE MODIFIED-**-
    
    Returns lookup indices for closest values w.r.t sample_array elements
    
    :called_from
        preProcessing, getAEP
    
    :param
        lookup       - The lookup array
        sample_array - Array, whose elements need to be matched
                       against lookup elements. 
        
    :return
        lookup indices for closest values w.r.t sample_array elements 
    """
    lookup_middles = lookup[1:] - np.diff(lookup.astype('f'))/2
    idx1 = np.searchsorted(lookup_middles, sample_array)
    indices = np.arange(lookup.shape[0])[idx1]
    return indices

   

def preProcessing(power_curve,k):
    """
    -**-THIS FUNCTION SHOULD NOT BE MODIFIED-**-
    
    Doing preprocessing to avoid the same repeating calculations.
    Record the required data for calculations. Do that once.
    Data are set up (shaped) to assist vectorization. Used later in
    function totalAEP. 
    
    :called_from
        main function
    
    :param
        power_curve - 2D numpy array with cols Wind Speed (m/s), 
                      Thrust Coeffecient (non dimensional), Power (MW)
        
    :return
        n_wind_instances  - number of wind instances (int)
        cos_dir           - For coordinate transformation 
                            2D Array. Shape (n_wind_instances,1)
        sin_dir           - For coordinate transformation 
                            2D Array. Shape (n_wind_instances,1)
        wind_sped_stacked - column staked all speed instances n_turb times. 
        C_t               - 3D array with shape (n_wind_instances, n_turbs, n_turbs)
                            Value changing only along axis=0. C_t, thrust coeff.
                            values for all speed instances. 
    """
    # number of turbines
    n_turbs       =   k
    
    # direction 'slices' in degrees
    slices_drct   = np.roll(np.arange(10, 361, 10, dtype=np.float32), 1)
    ## slices_drct   = [360, 10.0, 20.0.......340, 350]
    n_slices_drct = slices_drct.shape[0]
    
    # speed 'slices'
    slices_sped   = [0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 
                        18.0, 20.0, 22.0, 24.0, 26.0, 28.0, 30.0]
    n_slices_sped = len(slices_sped)-1
    
    # number of wind instances
    n_wind_instances = (n_slices_drct)*(n_slices_sped)
    
    # Create wind instances. There are two columns in the wind instance array
    # First Column - Wind Speed. Second Column - Wind Direction
    # Shape of wind_instances (n_wind_instances,2). 
    # Values [1.,360.],[3.,360.],[5.,360.]...[25.,350.],[27.,350.],29.,350.]
    wind_instances = np.zeros((n_wind_instances,2), dtype=np.float32)
    counter = 0
    for i in range(n_slices_drct):
        for j in range(n_slices_sped): 
            
            wind_drct =  slices_drct[i]
            wind_sped = (slices_sped[j] + slices_sped[j+1])/2
            
            wind_instances[counter,0] = wind_sped
            wind_instances[counter,1] = wind_drct
            counter += 1

    # So that the wind flow direction aligns with the +ve x-axis.           
    # Convert inflow wind direction from degrees to radians
    wind_drcts =  np.radians(wind_instances[:,1] - 90)
    # For coordinate transformation 
    cos_dir = np.cos(wind_drcts).reshape(n_wind_instances,1)
    sin_dir = np.sin(wind_drcts).reshape(n_wind_instances,1)
    
    # create copies of n_wind_instances wind speeds from wind_instances
    wind_sped_stacked = np.column_stack([wind_instances[:,0]]*n_turbs)
   
    # Pre-prepare matrix with stored thrust coeffecient C_t values for 
    # n_wind_instances shape (n_wind_instances, n_turbs, n_turbs). 
    # Value changing only along axis=0. C_t, thrust coeff. values for all 
    # speed instances.
    # we use power_curve data as look up to estimate the thrust coeff.
    # of the turbine for the corresponding closest matching wind speed
    indices = searchSorted(power_curve[:,0], wind_instances[:,0])
    C_t     = power_curve[indices,1]
    # stacking and reshaping to assist vectorization
    C_t     = np.column_stack([C_t]*(n_turbs*n_turbs))
    C_t     = C_t.reshape(n_wind_instances, n_turbs, n_turbs)
    
    return(n_wind_instances, cos_dir, sin_dir, wind_sped_stacked, C_t)


def getAEP(turb_rad, turb_coords, power_curve, wind_inst_freq,n_wind_instances, cos_dir, sin_dir, wind_sped_stacked, C_t,p):
    
    """
    -**-THIS FUNCTION SHOULD NOT BE MODIFIED-**-
    
    Calculates AEP of the wind farm. Vectorised version.
    
    :called from
        main
        
    :param
        turb_diam         - Radius of the turbine (m)
        turb_coords       - 2D array turbine euclidean x,y coordinates
        power_curve       - For estimating power. 
        wind_inst_freq    - 1-D flattened with rough probabilities of 
                            wind instance occurence.
                            n_wind_instances  - number of wind instances (int)
        cos_dir           - For coordinate transformation 
                            2D Array. Shape (n_wind_instances,1)
        sin_dir           - For coordinate transformation 
                            2D Array. Shape (n_wind_instances,1)
        wind_sped_stacked - column staked all speed instances n_turb times. 
        C_t               - 3D array with shape (n_wind_instances, n_turbs, n_turbs)
                            Value changing only along axis=0. C_t, thrust coeff.
                            values for all speed instances. 
    
    :return
        wind farm AEP in Gigawatt Hours, GWh (float)
    """
    # number of turbines
    n_turbs        =   turb_coords.shape[0]
    #print("n_turbs from getAEP is :",n_turbs)
    #assert n_turbs ==  50, "Error! Number of turbines is not 50."    #this is commented by     Ar512
    
    # Prepare the rotated coordinates wrt the wind direction i.e downwind(x) & crosswind(y) 
    # coordinates wrt to the wind direction for each direction in wind_instances array
    rotate_coords   =  np.zeros((n_wind_instances, n_turbs, 2), dtype=np.float32)
    # Coordinate Transformation. Rotate coordinates to downwind, crosswind coordinates
    rotate_coords[:,:,0] =  np.matmul(cos_dir, np.transpose(turb_coords[:,0].reshape(n_turbs,1))) - \
                           np.matmul(sin_dir, np.transpose(turb_coords[:,1].reshape(n_turbs,1)))
    rotate_coords[:,:,1] =  np.matmul(sin_dir, np.transpose(turb_coords[:,0].reshape(n_turbs,1))) +\
                           np.matmul(cos_dir, np.transpose(turb_coords[:,1].reshape(n_turbs,1)))
 
    
    # x_dist - x dist between turbine pairs wrt downwind/crosswind coordinates)
    # for each wind instance
    x_dist = np.zeros((n_wind_instances,n_turbs,n_turbs), dtype=np.float32)
    for i in range(n_wind_instances):
        tmp = rotate_coords[i,:,0].repeat(n_turbs).reshape(n_turbs, n_turbs)
        x_dist[i] = tmp - tmp.transpose()
    

    # y_dist - y dist between turbine pairs wrt downwind/crosswind coordinates)
    # for each wind instance    
    y_dist = np.zeros((n_wind_instances,n_turbs,n_turbs), dtype=np.float32)
    for i in range(n_wind_instances):
        tmp = rotate_coords[i,:,1].repeat(n_turbs).reshape(n_turbs, n_turbs)
        y_dist[i] = tmp - tmp.transpose()
    y_dist = np.abs(y_dist) 
     

    # Now use element wise operations to calculate speed deficit.
    # kw, wake decay constant presetted to 0.05
    # use the jensen's model formula. 
    # no wake effect of turbine on itself. either j not an upstream or wake 
    # not happening on i because its outside of the wake region of j
    # For some values of x_dist here RuntimeWarning: divide by zero may occur
    # That occurs for negative x_dist. Those we anyway mark as zeros. 


    

    sped_deficit = (1-np.sqrt(1-C_t))*( (turb_rad/(turb_rad + 0.05*x_dist))**2) 
    sped_deficit[((x_dist <= 0) | ((x_dist > 0) & (y_dist > (turb_rad + 0.05*x_dist))))] = 0.0
    if(p==1):
        print("//////////////////////////////")

    
    
    # Calculate Total speed deficit from all upstream turbs, using sqrt of sum of sqrs
    sped_deficit_eff  = np.sqrt(np.sum(np.square(sped_deficit), axis = 2))

    
    # Element wise multiply the above with (1- sped_deficit_eff) to get
    # effective windspeed due to the happening wake
    wind_sped_eff     = wind_sped_stacked*(1.0-sped_deficit_eff)

    
    # Estimate power from power_curve look up for wind_sped_eff
    indices = searchSorted(power_curve[:,0], wind_sped_eff.ravel())
    power   = power_curve[indices,2]
    power   = power.reshape(n_wind_instances,n_turbs)
    
    # Farm power for single wind instance 
    power   = np.sum(power, axis=1)
    
    # multiply the respective values with the wind instance probabilities 
    # year_hours = 8760.0
    AEP = 8760.0*np.sum(power*wind_inst_freq)
    
    # Convert MWh to GWh
    AEP = AEP/1e3
    
    return(AEP)
    

    
def checkConstraints(turb_coords, turb_diam):
    """
    -**-THIS FUNCTION SHOULD NOT BE MODIFIED-**-
    
    Checks if the turbine configuration satisfies the two
    constraints:(i) perimeter constraint,(ii) proximity constraint 
    Prints which constraints are violated if any. Note that this 
    function does not quantifies the amount by which the constraints 
    are violated if any. 
    
    :called from
        main 
        
    :param
        turb_coords - 2d np array containing turbine x,y coordinates
        turb_diam   - Diameter of the turbine (m)
    
    :return
        None. Prints messages.   
    """
    bound_clrnc      = 50
    prox_constr_viol = False
    peri_constr_viol = False
    k=turb_coords.shape[0]
    
    # create a shapely polygon object of the wind farm
    farm_peri = [(0, 0), (0, 4000), (4000, 4000), (4000, 0)]
    farm_poly = Polygon(farm_peri)
    
    # checks if for every turbine perimeter constraint is satisfied. 
    # breaks out if False anywhere
    for turb in turb_coords:
        turb = Point(turb)
        inside_farm   = farm_poly.contains(turb)
        correct_clrnc = farm_poly.boundary.distance(turb) >= bound_clrnc
        if (inside_farm == False or correct_clrnc == False):
            peri_constr_viol = True
            print(turb)
            break
    
    # checks if for every turbines proximity constraint is satisfied. 
    # breaks out if False anywhere
    for i,turb1 in enumerate(turb_coords):
        for turb2 in np.delete(turb_coords, i, axis=0):
            if  np.linalg.norm(turb1 - turb2) < 4*turb_diam:
                prox_constr_viol = True
                break
    
    # print messages
    if  peri_constr_viol  == True  and prox_constr_viol == True:
          print('Somewhere both perimeter constraint and proximity constraint are violated\n')
    elif peri_constr_viol == True  and prox_constr_viol == False:
          print('Somewhere perimeter constraint is violated\n')
    elif peri_constr_viol == False and prox_constr_viol == True:
          print('Somewhere proximity constraint is violated\n')
    else: print('Both perimeter and proximity constraints are satisfied !!\n')
        
    return()





#--------------------------------------------------------------------------------------------------------------





















#S



def getTurbLoc(turb_loc_file_name):
    """ 
    -**-THIS FUNCTION SHOULD NOT BE MODIFIED-**-
    
    Returns x,y turbine coordinates
    
    :Called from
        main function
    
    :param
        turb_loc_file_name - Turbine Loc csv file location
        
    :return
        2D array
    """
    
    df = pd.read_csv(turb_loc_file_name, sep=',')
    turb_coords = df.to_numpy(dtype = np.float32)
    return(turb_coords)








def feasible(rr,cc,XI_x,XI_y,not_consider):
   # if((rr+cc)%100==0):
       # print('faesible hu kya?')
    x=rr
    y=cc
    t=0
    for i,j in zip(XI_x,XI_y):
        if t==not_consider:
            t+=1
            continue
        t+=1
        x1=i
        y1=j
        if (x-x1)*(x-x1) + (y-y1)*(y-y1) < 400*400:
            return False
    return True
    






# ALGO-1
#here r_x and r_ynare the points in G but not its actual coordinate the actula coordinate are x=G_r_x[r_x][r_y] and y=G_c_y[r_x][r_y]
def greedy_F(MAX_TILL_NOW,t,dis,turb_rad,power_curve,XI_x,XI_y,dc,not_consider,n_wind_instances,cos_dir,sin_dir,wind_sped_stacked,C_t):
    print('Greedy-F')
    # XI_x=[]   #possible points solution set having $x coordinates  x=G_r_x[$x][$y]
    # XI_y=[]   #possible points solution set having $y coordinates where y=G_c_y[$x][$y]
    ppp=1
    #arbitray point is X1_x-->r_x and X1_y-->r_y

    # XI_x.append(r_x-1)
    # XI_y.append(r_y-1)
    k=1
    xx=XI_x[not_consider]
    yy=XI_y[not_consider]
    x1=max(50,xx-dis)
    print("x1",x1)
    x2=min(3950,xx+dis)
    print("x2",x2)
    y1=max(50,yy-dis)
    print("y1",y1)
    y2=min(3950,yy+dis)
    print("y2",y2)


    
    i=0
    while i<k:
        print('in while loop trying to optmized for point number->',not_consider+1)
        
        p_x=0
        p_y=0
        print('point under consideraion Now is (',xx,yy,')')
        
        flag=0
        max_power=0.0
        
        


        #Select unchosen point p such that  objective function is maximaizied 
        for rr in np.arange(x1,x2+0.001,t):
            for cc in np.arange(y1,y2+0.001,t):
                
                # condition 1 chosen p is feasible or not

                if(feasible(rr,cc,XI_x,XI_y,not_consider)):
                    #Cp=Construct_Cp(rs_start,rows,cs_start,columns,G_r_x,G_c_y,rr,cc)
                    #if F(Cp,rr,cc,XI_x,XI_y,G_r_x,G_c_y) >= offset+k-i-1 :
                    if True:
                        flag=1

                        #here we have to check for greedeness function -->
                        #Select p ∈ S i such that u(X i ) − u(X i ∪ p) is maximized, subject to:
                        #philal ke liye rhne de wake  effect
                        turb_coordss = []

                        for pt in range(50):
                            if(pt==not_consider):
                                turb_coordss.append([rr,cc])
                            else:
                                turb_coordss.append([XI_x[pt],XI_y[pt]])


                        
                        
                        turb_coordss=np.array(turb_coordss,dtype=np.float)
                        # print("I am in Greedy")
                        # print(turb_coords)
                        #print('no of turbines = '+str(i))
                        #print(turb_coords.shape[0])
                        #print(turb_coords.shape[1])
                        # print("I am out from Greedy")
                        #break
                        #n_wind_instances,cos_dir,sin_dir,wind_sped_stacked,C_t = preProcessing(power_curve,turb_coords.shape[0])
                        AEP=0.00
                        for wind_inst_freq in dc:
                            #print("DDDDDDDDDD")
                            A=getAEP(turb_rad, turb_coordss, power_curve, dc[wind_inst_freq],n_wind_instances, cos_dir, sin_dir, wind_sped_stacked,C_t,ppp)
                            AEP+=A



                        
                        ppp+=1
                        if (max_power<=AEP):
                            #print('\nhmm changed occurs for p_x,p_y',p_x,p_y)
                            #print('hmm AEP=',AEP/3)
                            #print('\n')
                            p_x=rr
                            p_y=cc
                            max_power=AEP
                            if(ppp<10):
                                print(max_power)




                    else:
                        continue
                else:
                    continue

        if flag==1: 
            print('The point '+str(xx)+" "+str(yy)+'has changed  to (',p_x,p_y,')')
            XI_x[not_consider]=p_x
            XI_y[not_consider]=p_y

            i+=1
        else:
            print('that point itself is its optimized point')
            i+=1


    #print('AT the end before return we have offset-',offset)
    return(XI_x,XI_y)




def save(XI_x,XI_y,s):
    
    with open(s+'.csv','w') as csv_file:
        csv_writer=csv.writer(csv_file,delimiter=',')
        d=[]
        d.append('x')
        d.append('y')
        csv_writer.writerow(d)
        for i,j in zip(XI_x,XI_y):
            d=[]
            d.append(i)
            d.append(j)
            csv_writer.writerow(d)
    print('file saved succesfully')



if __name__ == "__main__":

    # Turbine Specifications.
    # -**-SHOULD NOT BE MODIFIED-**-
    turb_specs    =  {   
                         'Name': 'AR',
                         'Vendor': 'AR',
                         'Type': 'Anon Type',
                         'Dia (m)': 100,
                         'Rotor Area (m2)': 7853,
                         'Hub Height (m)': 100,
                         'Cut-in Wind Speed (m/s)': 3.5,
                         'Cut-out Wind Speed (m/s)': 25,
                         'Rated Wind Speed (m/s)': 15,
                         'Rated Power (MW)': 3
                     }
    turb_diam      =  turb_specs['Dia (m)']
    turb_rad       =  turb_diam/2 
  
    


    
    # Turbine x,y coordinates
    #turb_coords   =  getTurbLoc(r'../Shell_Hackathon Dataset/turbine_loc_tests_1.csv')
    
    #Load the power curve
    power_curve   =  loadPowerCurve('../Shell_Hackathon Dataset/power_curve.csv')
    
    # Pass wind data csv file location to function binWindResourceData.
    # Retrieve probabilities of wind instance occurence.
    wind_inst_freq1 = binWindResourceData(r'../Shell_Hackathon Dataset/Wind Data/wind_data_2017.csv')
    # wind_inst_freq2 = binWindResourceData(r'../Shell_Hackathon Dataset/Wind Data/wind_data_2015.csv')  
    # wind_inst_freq3 = binWindResourceData(r'../Shell_Hackathon Dataset/Wind Data/wind_data_2014.csv')
    #wind_inst_freq4 = binWindResourceData(r'../Shell_Hackathon Dataset/Wind Data/wind_data_2013.csv')
    wind_inst_freq5 = binWindResourceData(r'../Shell_Hackathon Dataset/Wind Data/wind_data_2008.csv')
    wind_inst_freq6 = binWindResourceData(r'../Shell_Hackathon Dataset/Wind Data/wind_data_2007.csv')
    #wind_inst_freq7 = binWindResourceData(r'../Shell_Hackathon Dataset/Wind Data/wind_data_2009.csv')
    dc={}
    dc['1']=wind_inst_freq1
    # dc['2']=wind_inst_freq2
    # dc['3']=wind_inst_freq3
    #dc['4']=wind_inst_freq4
    dc['5']=wind_inst_freq5
    dc['6']=wind_inst_freq6
    #dc['7']=wind_inst_freq7
    n_wind_instances,cos_dir,sin_dir,wind_sped_stacked,C_t = preProcessing(power_curve,50)


    start=time.time()




    


    cs_start=0.0
    rs_start=0.0



    start_x=50.0
    start_y=50.0
    end_x=3950.0
    end_y=3950.0

    collect=[]
    #cho=int(input('Give the no of random starting point u want to have  : --> :'))
    cho=1
    v=1
    g=0
    XI_y=[]
    XI_x=[]
    XII_y=[]
    XII_y=[]
    print('   * * * * * * * * * *')
    print('   * * * * * * * * * * ')
    print('   * * * * * * * * * * ')
    print('   * * * * * * * * * * ')
    print('   * * * * * * * * * * ')
    print('   * * * * * * * * * *')
    print('   * * * * * * * * * *')
    print('   * * * * * * * * * *')
    print('   * * * * * * * * * *')
    print('\n\n\n\n')
    while v<=cho:
        # dis=int(input('Enter min-distance between adjacents turbines for point no: '+str(v)+' (keep it 50 or less [yr wish] :'))
        # r=int(3900/dis)+1
        # x_c=int(input('Enter x coordinate of point: '+ str(v)+' (Note coordinate should be less than '+str(r)+'):'))
        # y_c=int(input('Enter y coordinate of point: '+str(v)+' (Note coordinate should be less than '+str(r)+'):'))
        # collect.append([x_c,y_c,dis])
        print('Hey I hope You are prepaared with file of 50 points')
        g=input("Enter file name without extension to optimized rest points :")
        dis=float(input("Enter min distance to be taken as a grid (Generally +10 -10 is preferred.):"))
        ttt=float(input("Enter min distance to be taken within that grid prefered is (1m or 0.5m):"))
        opt=int(input('how much points ke bad aap code se poinst save krna chate h <= 50'))


        if dis==0:
            while dis<=0:
                print('soory min distance cannot be 0 so enter again min distance')
                dis=float(input("Enter min distance to be taken as a grid (Generally +10 -10 is preferred.):"))
        #print('So your x1,x2,y1,y2 are respectively :'+str(x1)+" "+str(x2)+" "+str(y1)+" "+str(y2))
        #print('Hello User your new grid search area is '+str(x2-x1)+' X '+str(y2-y1)+' having points: '+str(int(((x2-x1)*(y2-y1))/dis)))

        v+=1




    v=0
    while v<cho:

        
        print("Now lets start: for optiimization of the points...."+"\n\n")
        turb_coords=getTurbLoc(g+'.csv')
        print('Calculating AEP......')
        AEP1 = getAEP(turb_rad, turb_coords, power_curve, wind_inst_freq1,n_wind_instances, cos_dir, sin_dir, wind_sped_stacked, C_t,1)
        AEP2 = getAEP(turb_rad, turb_coords, power_curve, wind_inst_freq5,n_wind_instances, cos_dir, sin_dir, wind_sped_stacked, C_t,1)
        AEP3 = getAEP(turb_rad, turb_coords, power_curve, wind_inst_freq6,n_wind_instances, cos_dir, sin_dir, wind_sped_stacked, C_t,1)
        AEP=(AEP1+AEP2+AEP3)
        print('Total power produced mAX TILL NOW FOR YOUR GIVEN FILE by the wind farm is: ', "%.12f"%(AEP/3), 'GWh')
        MAX_TILL_NOW=AEP
        print("***************************************************************************************")

        for i in range(50):
            XI_x.append(turb_coords[i][0])
            XI_y.append(turb_coords[i][1])
            #print(XI_x[i],XI_y[i])
        




        itr=0
        while True:
            itr+=1
            print('\n\n\niterration',itr)

            for i in range(opt):
                print('for point no',i+1)
                u=time.time()
                XI_x,XI_y = greedy_F(MAX_TILL_NOW,ttt,dis,turb_rad,power_curve,XI_x,XI_y,dc,i,n_wind_instances,cos_dir,sin_dir,wind_sped_stacked,C_t)
                print('Time take for updtion of this point->>',i+1,' is'+str(time.time()-u)+' sec.')
                

        # XI_x,XI_y = greedy_F(k,12,G_r_x,G_c_y,40-40,79,40-40,79,S,r_x,r_y,turb_rad,power_curve,XI_x,XI_y,dc)



            t=1
            turb_coordss = []
            for i,j in zip(XI_x,XI_y):
                turb_coordss.append([i,j])
                print(t," [",i,j,"]")
                t+=1
                
            turb_coords=np.array(turb_coordss,dtype=np.float)





    #------------------------------------------------------------ ...............---------------------------

            pin=time.time()-start
            start=time.time()
            if(pin>3600):
                s=str(int(pin/3600))+' h '+str(int((pin%3600)/60))+' min'
            else:
                s=str(int((pin/60)))+' min'

        #print('\ntime-taken'+str(pin))
            print('Actual time in min approx--<>:',s)
            print("\n\n")
            v+=1


            checkConstraints(turb_coords, turb_diam)
    
            print('Calculating AEP......')
            AEP1 = getAEP(turb_rad, turb_coords, power_curve, wind_inst_freq1,n_wind_instances, cos_dir, sin_dir, wind_sped_stacked, C_t,1)
            AEP2 = getAEP(turb_rad, turb_coords, power_curve, wind_inst_freq5,n_wind_instances, cos_dir, sin_dir, wind_sped_stacked, C_t,1)
            AEP3 = getAEP(turb_rad, turb_coords, power_curve, wind_inst_freq6,n_wind_instances, cos_dir, sin_dir, wind_sped_stacked, C_t,1)
            AEP=(AEP1+AEP2+AEP3)
            print('Total power produced by the wind farm is: ', "%.12f"%(AEP/3), 'GWh')
            if(AEP>MAX_TILL_NOW):
                MAX_TILL_NOW=AEP
                print('Calculated coordinates of turbine now saving in file......')
                save(XI_x,XI_y,g+'_itr_'+str(itr))

