import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize
# from scipy.interpolate import interp2d, bisplev
from scipy.interpolate import RectBivariateSpline, bisplrep, bisplev, griddata
import glob
import os
import re
from scipy.optimize import curve_fit
import sys
from matplotlib import patches, ticker
import matplotlib.image as mpimg



def load_data(path):
    return np.loadtxt(path, dtype=np.double, delimiter=",", skiprows=5)

def get_data_sweep(data, ien=0):
    ## 1 GS, 2: exited state
    data_map = np.zeros([len(data),1])
    det_range = np.zeros([len(data),1])
    data_map = data[:,ien+1]
    det_range = data[:,0]
    
    return data_map,det_range
 
def get_data_map(data):
    x_labels, y_labels = get_xy(data)
    Nx = len(x_labels)
    Ny = len(y_labels)

    data_map = np.zeros([Ny,Nx])

    # Maybe it's better to be safe than smart?
    # indicies = np.zeros([np.shape(data)[0], 2], dtype=int)
    # indicies[:, 0] = data[:, 0] / np.min(x_labels) - 1
    # indicies[:, 1] = data[:, 1] / np.min(y_labels) - 1

    # for ij, val in zip(indicies, data[:, 2]):
    #     i, j = ij
    #     print(i,j,val)
    #     data_map[j, i] = val  # here we have a transposition
        # Why? Image is like: constant x = consant row, but first index changes row, so unintuitive.

    # Being smart = errros
    data_map = np.zeros([Nx,Ny])
    for i in range(Ny):
        data_map[:, i] = data[Nx*i:Nx*(i+1), 2]
    
    return data_map

def get_xy(data):
    x = sorted(list(set(data[:, 0])))
    y = sorted(list(set(data[:, 1])))
    
    return x, y

def plot_sweep(ax, detuning, energy, title, ticks_every=2, scale=1, color='r', label = ''):
    plt.plot(detuning,energy*scale,marker='o',color=color,label=label) 
    
    plt.xlabel("detuning (V)")
    plt.ylabel(r"Energy ($\mu$eV)")
    plt.title(title)


    
    # ax.setxlabel("detuning (V)")
    # plt.ylabel(r"Energy ($\mu$eV)")
    # plt.title(title)

import matplotlib.colors as mcolors

def plot_map(ax, data_map, xlabels, ylabels, title, ticks_every=2, scale=1, cbar_label='',cmap='viridis',gates=False, vmin=None, vmax=None,NormCenter=False,nbins=None,log_scale=False, cbar=True):
    
    # Use TwoSlopeNorm if centering around zero
    if NormCenter:
        norm = mcolors.TwoSlopeNorm(vcenter=0)  # Adjust vcenter for better contrast
    elif log_scale:
        norm = mcolors.LogNorm()  # Log scale if values vary greatly
    else:
        norm = None

    plt.imshow(data_map*scale, origin='lower',cmap=cmap, vmin=vmin, vmax=vmax,norm=norm )
    plt.xticks(np.arange(0, len(xlabels), ticks_every), labels=np.array(xlabels[::ticks_every]).astype(int))
    plt.yticks(np.arange(0, len(ylabels), ticks_every), labels=np.array(ylabels[::ticks_every]).astype(int))
    if gates:    
        draw_gates(xlabels, ylabels)
    
    if cbar==True:
        cb = plt.colorbar(fraction=0.046, pad=0.04, label= cbar_label)

    if nbins is not None:
        if cbar==True:
            cb.locator = ticker.MaxNLocator(nbins=nbins)  # Adjust number of colorbar ticks
            cb.update_ticks()
    
    plt.xlabel("x (nm)")
    plt.ylabel("y (nm)")
    plt.title(title)



# def get_interpolated_map(data_map, xlabels, ylabels, dx=0, dy=0, Nx=200, Ny=200, return_func=False, scale=1):
#     # interpolated_func = interp2d(xlabels, ylabels, data_map*scale, kind="cubic")  # interp2d is reprected
#     interpolated_func = RectBivariateSpline(xlabels, ylabels, data_map*scale)
#     xspace = np.linspace(xlabels[0], xlabels[-1], Nx)
#     yspace = np.linspace(ylabels[0], ylabels[-1], Ny)
#     if not return_func:
#         interpolated = interpolated_func(xspace, yspace, dx=dx, dy=dy)
#         return xspace, yspace, interpolated
#     else:
#         return xspace, yspace, interpolated_func

def length_real2plot(labels, val):
    # Labels indicate the axis
    return val / (max(labels) - min(labels)) * len(labels)  # note that length will scale without shifting val


def coord_real2plot(xlabels, ylabels, xy):
    x, y = xy
    # Relative is in px (but can be float)
    # Here
    relative_x = (x - min(xlabels)) / (max(xlabels) - min(xlabels)) * len(xlabels) 
    relative_y = (y - min(ylabels)) / (max(ylabels) - min(ylabels)) * len(ylabels)
    # This weird substraction is related to the fact that each point corresponds to the middle of a pixel, so it shifts
    #   coordinate system by half of pixel.
    # return np.array([relative_x, relative_y]) 
    return np.array([relative_x, relative_y]) - np.array([0.5, 0.5])

def reflect_xy(xy, width=10, height=10, ref_point=0, axis='x'):
    """xy = Bottom-left corner """
    if axis=='x':
        (2 * ref_point - (xy[0] + width), xy[1])
    elif axis=='y':
        reflected_xy = (xy[0], 2 * ref_point - (xy[1] + height))        
    return reflected_xy


def draw_gates(xlabels, ylabels):
    d_dot = 110
    w_met = 45
    d_ch = 140
    d_scr = 2*d_dot
    w_block = 6*d_dot
    d_block=2*d_scr + d_ch
    d_gate_top = 20
    h_ox=10
    col="k"
    
    barrier = patches.Rectangle(coord_real2plot(xlabels, ylabels, [w_block/2-w_met/2, 0]),
                              width=length_real2plot(xlabels, w_met),
                              height=length_real2plot(ylabels, d_scr+h_ox),
                             fill=False,
                             ls="--",
                             color=col,
                             lw=1)
    plt.gca().add_patch(barrier)

    barrier = patches.Rectangle(coord_real2plot(xlabels, ylabels, [w_block/2-w_met/2, d_scr+h_ox]),
                              width=length_real2plot(xlabels, w_met),
                              height=length_real2plot(ylabels, d_ch-2*h_ox),
                             fill=False,
                             ls="--",
                             color=col,
                             lw=1)
    plt.gca().add_patch(barrier)

    barrier = patches.Rectangle(coord_real2plot(xlabels, ylabels, [w_block/2-w_met/2, d_scr+d_ch-h_ox]),
                              width=length_real2plot(xlabels, w_met),
                              height=length_real2plot(ylabels, d_gate_top),
                             fill=False,
                             ls="--",
                             color=col,
                             lw=1)
    plt.gca().add_patch(barrier)

    
    ## Plunger Right
    reflected = reflect_xy((w_block/2-w_met/2 +d_dot/2,0), width = w_met, height = d_scr+h_ox, ref_point = d_block/2, axis='y')
    Plunger_R = patches.Rectangle(coord_real2plot(xlabels, ylabels, [ reflected[0], reflected[1] ]  ),
                              width=length_real2plot(xlabels, w_met),
                              height=length_real2plot(ylabels, d_scr+h_ox),
                             fill=False,
                             ls="--",
                             color=col,
                             lw=1)
    plt.gca().add_patch(Plunger_R)


    reflected = reflect_xy((w_block/2-w_met/2 +d_dot/2,d_scr+h_ox), width = w_met, height = d_ch-2*h_ox, ref_point = d_block/2, axis='y')
    Plunger_R = patches.Rectangle(coord_real2plot(xlabels, ylabels, [ reflected[0], reflected[1] ]  ),
                              width=length_real2plot(xlabels, w_met),
                              height=length_real2plot(ylabels, d_ch-2*h_ox),
                             fill=False,
                             ls="--",
                             color=col,
                             lw=1)
    plt.gca().add_patch(Plunger_R)

    reflected = reflect_xy((w_block/2-w_met/2 +d_dot/2,d_scr+d_ch-h_ox), width = w_met, height = d_gate_top, ref_point = d_block/2, axis='y')
    Plunger_R = patches.Rectangle(coord_real2plot(xlabels, ylabels, [ reflected[0], reflected[1] ]  ),
                              width=length_real2plot(xlabels, w_met),
                              height=length_real2plot(ylabels, d_gate_top),
                             fill=False,
                             ls="--",
                             color=col,
                             lw=1)
    plt.gca().add_patch(Plunger_R)


    ## Plunger Left
    reflected = reflect_xy((w_block/2-w_met/2 -d_dot/2,0), width = w_met, height = d_scr+h_ox, ref_point = d_block/2, axis='y')
    Plunger_L = patches.Rectangle(coord_real2plot(xlabels, ylabels, [ reflected[0], reflected[1] ]  ),
                              width=length_real2plot(xlabels, w_met),
                              height=length_real2plot(ylabels, d_scr+h_ox),
                             fill=False,
                             ls="--",
                             color=col,
                             lw=1)
    plt.gca().add_patch(Plunger_L)


    reflected = reflect_xy((w_block/2-w_met/2 -d_dot/2,d_scr+h_ox), width = w_met, height = d_ch-2*h_ox, ref_point = d_block/2, axis='y')
    Plunger_L = patches.Rectangle(coord_real2plot(xlabels, ylabels, [ reflected[0], reflected[1] ]  ),
                              width=length_real2plot(xlabels, w_met),
                              height=length_real2plot(ylabels, d_ch-2*h_ox),
                             fill=False,
                             ls="--",
                             color=col,
                             lw=1)
    plt.gca().add_patch(Plunger_L)

    reflected = reflect_xy((w_block/2-w_met/2 -d_dot/2,d_scr+d_ch-h_ox), width = w_met, height = d_gate_top, ref_point = d_block/2, axis='y')
    Plunger_L = patches.Rectangle(coord_real2plot(xlabels, ylabels, [ reflected[0], reflected[1] ]  ),
                              width=length_real2plot(xlabels, w_met),
                              height=length_real2plot(ylabels, d_gate_top),
                             fill=False,
                             ls="--",
                             color=col,
                             lw=1)
    plt.gca().add_patch(Plunger_L)
    




def draw_gates2(xlabels, ylabels, ax=plt.gca(), lw=1):
    d_dot = 110
    w_met = 45
    d_ch = 140
    d_scr = 2*d_dot
    w_block = 6*d_dot
    d_block=2*d_scr + d_ch
    d_gate_top = 20
    h_ox=10
    col="k"
    
    barrier = patches.Rectangle(coord_real2plot(xlabels, ylabels, [w_block/2-w_met/2, 0]),
                              width=length_real2plot(xlabels, w_met),
                              height=length_real2plot(ylabels, d_scr+h_ox),
                             fill=False,
                             ls="--",
                             color=col,
                             lw=lw)
    ax.add_patch(barrier)

    barrier = patches.Rectangle(coord_real2plot(xlabels, ylabels, [w_block/2-w_met/2, d_scr+h_ox]),
                              width=length_real2plot(xlabels, w_met),
                              height=length_real2plot(ylabels, d_ch-2*h_ox),
                             fill=False,
                             ls="--",
                             color=col,
                             lw=lw)
    ax.add_patch(barrier)

    barrier = patches.Rectangle(coord_real2plot(xlabels, ylabels, [w_block/2-w_met/2, d_scr+d_ch-h_ox]),
                              width=length_real2plot(xlabels, w_met),
                              height=length_real2plot(ylabels, d_gate_top),
                             fill=False,
                             ls="--",
                             color=col,
                             lw=lw)
    ax.add_patch(barrier)

    
    ## Plunger Right
    reflected = reflect_xy((w_block/2-w_met/2 +d_dot/2,0), width = w_met, height = d_scr+h_ox, ref_point = d_block/2, axis='y')
    Plunger_R = patches.Rectangle(coord_real2plot(xlabels, ylabels, [ reflected[0], reflected[1] ]  ),
                              width=length_real2plot(xlabels, w_met),
                              height=length_real2plot(ylabels, d_scr+h_ox),
                             fill=False,
                             ls="--",
                             color=col,
                             lw=lw)
    ax.add_patch(Plunger_R)


    reflected = reflect_xy((w_block/2-w_met/2 +d_dot/2,d_scr+h_ox), width = w_met, height = d_ch-2*h_ox, ref_point = d_block/2, axis='y')
    Plunger_R = patches.Rectangle(coord_real2plot(xlabels, ylabels, [ reflected[0], reflected[1] ]  ),
                              width=length_real2plot(xlabels, w_met),
                              height=length_real2plot(ylabels, d_ch-2*h_ox),
                             fill=False,
                             ls="--",
                             color=col,
                             lw=lw)
    ax.add_patch(Plunger_R)

    reflected = reflect_xy((w_block/2-w_met/2 +d_dot/2,d_scr+d_ch-h_ox), width = w_met, height = d_gate_top, ref_point = d_block/2, axis='y')
    Plunger_R = patches.Rectangle(coord_real2plot(xlabels, ylabels, [ reflected[0], reflected[1] ]  ),
                              width=length_real2plot(xlabels, w_met),
                              height=length_real2plot(ylabels, d_gate_top),
                             fill=False,
                             ls="--",
                             color=col,
                             lw=lw)
    ax.add_patch(Plunger_R)


    ## Plunger Left
    reflected = reflect_xy((w_block/2-w_met/2 -d_dot/2,0), width = w_met, height = d_scr+h_ox, ref_point = d_block/2, axis='y')
    Plunger_L = patches.Rectangle(coord_real2plot(xlabels, ylabels, [ reflected[0], reflected[1] ]  ),
                              width=length_real2plot(xlabels, w_met),
                              height=length_real2plot(ylabels, d_scr+h_ox),
                             fill=False,
                             ls="--",
                             color=col,
                             lw=lw)
    ax.add_patch(Plunger_L)


    reflected = reflect_xy((w_block/2-w_met/2 -d_dot/2,d_scr+h_ox), width = w_met, height = d_ch-2*h_ox, ref_point = d_block/2, axis='y')
    Plunger_L = patches.Rectangle(coord_real2plot(xlabels, ylabels, [ reflected[0], reflected[1] ]  ),
                              width=length_real2plot(xlabels, w_met),
                              height=length_real2plot(ylabels, d_ch-2*h_ox),
                             fill=False,
                             ls="--",
                             color=col,
                             lw=lw)
    ax.add_patch(Plunger_L)

    reflected = reflect_xy((w_block/2-w_met/2 -d_dot/2,d_scr+d_ch-h_ox), width = w_met, height = d_gate_top, ref_point = d_block/2, axis='y')
    Plunger_L = patches.Rectangle(coord_real2plot(xlabels, ylabels, [ reflected[0], reflected[1] ]  ),
                              width=length_real2plot(xlabels, w_met),
                              height=length_real2plot(ylabels, d_gate_top),
                             fill=False,
                             ls="--",
                             color=col,
                             lw=lw)
    ax.add_patch(Plunger_L)

# Function to extract the numeric suffix from directory names
def extract_number_from_name(name):
    match = re.search(r'([-+]?\d*\.\d+|\d+)$', name)
    return float(match.group(1)) if match else float('inf')




def get_subdir_paths(data_root):
  
  all_subdirs = os.listdir(data_root)
  subdir_paths_notsorted = [os.path.join(data_root, item) for item in all_subdirs if os.path.isdir(os.path.join(data_root, item))]
    
  subdir_paths = sorted(subdir_paths_notsorted, key=extract_number_from_name)
  
  det_paths = []
    
    # Loop over all subdirectories in the main directory
  for sub_dir in subdir_paths:
           
      double_defect_paths_notsorted = glob.glob(sub_dir+"\\8_1*")
      single_defect_paths_notsorted = glob.glob(sub_dir+"\\pert*")
      det_paths.append(double_defect_paths_notsorted)
  N_det = len(det_paths)
  return det_paths, subdir_paths, N_det

def custom_linear(x, a, Eav):
    return 0.5 * a * (x) + Eav



def get_error(a,b):
    return 100*(abs(a)-abs(b))/(abs(a))
    
def choose_data(x,y,xinds=[0,10],yinds=[0,10]):
    x2 = x[ xinds[0]:xinds[1] ] 
    y2 = y[ yinds[0]:yinds[1] ] 
    return x2,y2

def get_inds(x,y):
    # Calculate the first derivative (numerical slope) 
    dy_dx = np.gradient(y, x)
    
    # Find where the slope changes sign
    slope_change_indices = np.where(np.diff(np.sign(dy_dx)))[0]
    # print(slope_change_indices)
    
    if len(slope_change_indices) > 0:
        # Index where slope changes
        index_of_slope_change = slope_change_indices[-1]
        # print(f"Index of slope change: {index_of_slope_change}")
    else:
        index_of_slope_change=10
        print("No slope change detected.")
    max_index = np.argmax(y)        


    return index_of_slope_change, max_index


def get_Eav_tc_eps0(eps, E):
    
    """
    this function will calculate:
        
        - Energy avarage            
        - tc : the tunncel coupling 
        - eps0: the detuning voltage that tc occurs
        
    """
    branch_lower = E[:,0]
    branch_upper = E[:,1]
    indx_L = np.argwhere((branch_lower == max(branch_lower) ))[0][0]
    indx_U = np.argwhere((branch_upper == min(branch_upper) ))[0][0]
    if indx_L != indx_U:
        print(f" indx_L ({indx_L}) and indx_U ({indx_U}).")

        # sys.exit()
    
    E_avg = ( max(branch_lower)+min(branch_upper) )/2
    tc = (  min(branch_upper) - max(branch_lower))
    eps0 = eps[indx_L]
    
    return E_avg, tc, eps0 
    
    
    # return E_avg, tc, eps0


def calculate_lever_arm(det_voltage, Energy, i=18):

    f=i-2
    
    Delta_E = Energy[i] - Energy[f]
    Delta_V = det_voltage[i] - det_voltage[f]
       
    # alpha =  2*Delta_E/Delta_V
    alpha =  Delta_E/Delta_V
    print("lever arm = ", alpha, "\mu eV/V")
    
    
    plt.plot(det_voltage[i], Energy[i], color="black", marker='o',ms=10)
    plt.plot(det_voltage[f], Energy[f], color="black", marker='o',ms=10)
    


    return alpha, i, f

# Function to extract numerical values from filenames using regex
def extract_numbers(filename):
    return list(map(int, re.findall(r'\d+', filename)))

def calculate_error(a, b):
    """
    This function calculates the element-wise error between two 2D arrays a and b.
    
    For each element:
    1. error_element = (a[i,j] - b[i,j]) / a[i,j] if a[i,j] > b[i,j]
    2. error_element = (a[i,j] - b[i,j]) / b[i,j] if a[i,j] <= b[i,j]
    
    Args:
    a (numpy.ndarray): 2D array of values.
    b (numpy.ndarray): 2D array of values.
    
    Returns:
    numpy.ndarray: 2D array of calculated errors.
    """
    # Initialize an array to store the selected error for each element
    error_matrix = np.zeros_like(a, dtype=float)

    # Loop through each element in the 2D arrays
    for i in range(a.shape[0]):
        for j in range(a.shape[1]):
            
            error_matrix[i, j] = (abs(a[i, j]) - abs(b[i, j])) / abs(a[i, j])
            
    return error_matrix
    
def custom_linear(x, a, Eav):
    return 0.5 * a * (x) + Eav

def parabola_func_p(x, tc, a, en):
    # return  a*np.sqrt((x-x0)**2) + b*x + c
    sign = 1    
    return  0.5*sign * np.sqrt((a**2)*(x-eps_0)**2+4*tc**2) +en

def parabola_func_m(x, tc, a, en):
    # return  a*np.sqrt((x-x0)**2) + b*x + c
    sign = -1    
    return   0.5*sign * np.sqrt((a**2)*(x-eps_0)**2+4*tc**2) +en


def get_fit(x,y):
    
    
    indx, max_index = get_inds(x,y[:,0])
    inds_upper = get_inds(x,y[:,1])
    
    indx_upper, extremum_upper = inds_upper[0], inds_upper[-1]
    
    indxi = 0
    
    if inds_upper[-1]==0:
        
        indxf = len(x)
        
    else:
        indxf = extremum_upper-1
        
    
    indx=indx+1
    
    # The avoided crossoing is fittes with linear funtion y = a1x+b1 & y = a2x+b2
    # where a1 = -a2 
    # for y = a1x+b1: we get data from lower and upper branch which mimics as a straight line

    xdata1, ydata1 = choose_data(x,y[:,0],xinds=[0,indx],yinds=[0,indx])
    xdata2, ydata2 = choose_data(x,y[:,1],xinds=[indx,indxf],yinds=[indx,indxf])
    xdata = np.concatenate((xdata1, xdata2))
    ydata = np.concatenate((ydata1, ydata2))
    params_L, covariance = curve_fit(custom_linear, xdata ,ydata  )

    # for y = a2x+b2: we get data from lower and upper branch which mimics as a straight line
    xdata1, ydata1 = choose_data(x,y[:,1],xinds=[0,indx],yinds=[0,indx])
    xdata2, ydata2 = choose_data(x,y[:,0],xinds=[indx,len(x)],yinds=[indx,len(x)])
    xdata = np.concatenate((xdata1, xdata2))
    ydata = np.concatenate((ydata1, ydata2))
    params_R, covariance = curve_fit(custom_linear, xdata ,ydata )
    
    # paramerts for linear function after fitting
    a1, b1 = params_L
    a2, b2 = params_R
    
    # Create the fitted curve
    fitted_y_L = custom_linear(x, a1, b1)
    fitted_y_R = custom_linear(x, a2, b2)
    
    global eps_0, E_av
    # obtaing intersection of two lines which gives the energy avarage    
    eps_0 = 2*(b2 - b1) / (a1 - a2)
    E_av = custom_linear(eps_0, a1, b1)
    
    
    
    eps_dens = np.linspace(min(x),max(x),300)
    

    """p0: initial guess of tc: tunnel coupling, a:lever arm and E: enery avg offset
        bounds: it's a tuple of array.  
            ([0,   5000,   -1e6], [100,   1e6,   0])
             [tc_min,a_min,Eavg_min],[tc_max,a_max,Eavg_max]
        this helps to sweep around the data in the plot and avoid to use the default values
        of function which make the guess of fit more unreasonable.
    """
    
    p0 = [18, 10000, -13800]  # Larger initial guess for a
    bounds = ([0, 5000, -1e6], [100, 1e6, 0])  # Allow a larger range for 'a' and limit 'en'
    
    
    params_parabola, _ = curve_fit(parabola_func_m, x, y[:,0], p0=p0, bounds=bounds)
    tc_L, a_L, en_L = params_parabola  
    
        
    params_parabola, _ = curve_fit(parabola_func_p, x, y[:,1], p0=p0, bounds=bounds)
    tc_U, a_U, en_U = params_parabola  
    
    
    fitted_parabola_y_lower = parabola_func_m(eps_dens, tc_L, a_L, en_L )
    fitted_parabola_y_upper = parabola_func_p(eps_dens, tc_U, a_U, en_U)
    
    # ## plot after fitting
    # if plot_fit:

    #     ax.plot(eps_dens, fitted_parabola_y_lower, color='r', linestyle='-', linewidth=3)
        
    #     ax.plot(eps_dens, fitted_parabola_y_upper, color='r', linestyle='--', linewidth=3)

    y2 = np.zeros((len(eps_dens),2), dtype=float)
    y2[:,0] = fitted_parabola_y_lower
    y2[:,1] = fitted_parabola_y_upper
    
    E_avg, tc, eps0 = get_Eav_tc_eps0(eps_dens, y2)

    Dense_detuning, Dense_Energy = eps_dens, y2

    return E_avg, tc, eps0, Dense_detuning, Dense_Energy

def subplot_sweep(ax, detuning, energy, title, ticks_every=2, scale=1, color='r', label = '', marker='none'):

    ax.plot(detuning,energy,marker=marker,color=color,label=label) 
    ax.set_xlabel("detuning (V)")
    ax.set_ylabel(r"Energy ($\mu$eV)")
    ax.set_title(title)

def lineplot_sweep(ax, detuning, energy, title, color='r', xlabel='',ylabel=''):
    ax.plot(detuning, energy, color=color, linestyle='-', linewidth=3)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)

def del_empty_axes(i,j,fig, axes):
    plt.subplots_adjust(wspace=0.4,hspace=0.5)
    for n in range(i, j):
        fig.delaxes(axes[n])   

def get_DQDparams(file_paths, _plot_=True, _fit_=True):
    
    N_files = len(file_paths)

    pert_eps0 = np.zeros( (N_files,3),dtype=float)
    pert_E_avg = np.zeros_like(pert_eps0)
    pert_tc = np.zeros_like(pert_eps0)


    # Determine dimensions (assume a square grid for simplicity)
    grid_size = int(np.sqrt(N_files))

    if _plot_:
        fig, axes = plt.subplots(9,5, figsize=(25,25))
        axes = axes.flatten()


    for i in range(N_files):

        
        fname = file_paths[i]
        data = load_data(fname)

        # x_str, y_str = fname.strip(".csv").split("_")[-3:-1]
        filename_numbers = [int(x) for x in re.findall(r'\d+', fname)]
        x_str, y_str =  filename_numbers[-3:-1]

        x0 = int(x_str)
        y0 = int(y_str)
        detuning = data[:,0]
        energy = data[:,1:]
        
        E_avg, tc, eps0 =  get_Eav_tc_eps0(detuning, energy)
        if _plot_:
            subplot_sweep(axes[i], detuning, energy,f"(x0,y0)={x0,y0}",color='k',marker='o' )

        if _fit_:

            E_avg, tc, eps0, Dense_detuning, Dense_Energy = get_fit(detuning,energy)
            if _plot_:
                lineplot_sweep(axes[i], Dense_detuning, Dense_Energy, "",color='r')  
        
        
        

        pert_eps0[i,:] = [x0,y0, eps0]
        pert_tc[i,:] = [x0,y0, tc]
        pert_E_avg[i,:] = [x0,y0,  E_avg]

    if _plot_:
        del_empty_axes(N_files,len(axes), fig, axes)
        # plt.savefig(figure_root + "sweep_%d.png"%i, dpi=100)
    return pert_E_avg, pert_tc, pert_eps0         