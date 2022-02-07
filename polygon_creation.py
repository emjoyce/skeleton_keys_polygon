import trimesh
from neuroglancer_scripts.mesh import read_precomputed_mesh
from meshparty import trimesh_io, skeleton, mesh_filters
import nglui

import json 
from scipy import spatial
KDTree = spatial.cKDTree
import numpy as np
import math
pi = math.pi
from collections import OrderedDict
import os

# load the up vector dictionary and x and z ranges
# x and z ranges have been extended to span entire dataset 
with open('./up_vecs_ranges.json') as json_file:
    json_dict = json.load(json_file)
    up_vec_dict = json_dict['up vecs'][0]
    x_ranges = json_dict['x ranges']
    z_ranges = json_dict['z_ranges']

# load the layer meshes 
# I don't like this out of a function. but it will be necessary for polygon creation 
# so maybe this is ok? but what if someone just wants to use find_up_vec 
# or some other function that does not need the layer meshes?



# make this a function with paths as input and 
# 

def load_meshes(mesh_folder_path):
    mesh_dict={}
    
    for f in os.listdir(mesh_folder_path):
        if 'pia' in f:
            # pia has to be loaded differently 
            pia_mesh = trimesh.load_mesh('layer_meshes/pia3.ply')
            mesh_dict['pia'] = pia_mesh
            mesh_dict['pia'] = trimesh_io.Mesh(mesh_dict['pia'].vertices * 1_000_000, mesh_dict['pia'].faces)
       
        elif not f.startswith('.'):
            filepath = os.path.join(mesh_folder_path, f)
            with open(filepath,'rb') as fp:
                verts,faces = read_precomputed_mesh(fp)
            mesh = trimesh_io.Mesh(verts, faces)
            filename = filepath.split('/')[1]
            mesh_dict[filename] = mesh
    
    return mesh_dict

# this should probably be some object that needs to be initiated - like client?
#mesh_folder_path = input('what is the local path to the meshes you wish to use to create your polygons? ')
mesh_dict = load_meshes(mesh_folder_path = 'layer_meshes')
# kwarg and give it a default, or an environment variable with a default
# class like client and all these funcs are methods on class
# class that 

def find_up_vec(x_pos, z_pos, up_vec_dictionary = up_vec_dict):
    
    '''
    takes x and y position (nm) of a soma and returns the corresponding up 
    vector for the xz column that it falls into 

    the x and z positions are used to find the column the position falls into
    example of what these columns look like here:
    _________________________________________________________
    |       |       |       |       |       |       |       |  
    | [0,0] | [1,0] | [2,0] | [3,0] | [4,0] | [5,0] | [6,0] | 
    |_______|_______|_______|_______|_______|_______|_______|
    |       |       |       |       |       |       |       |  
    | [0,1] | [1,1] | [2,1] | [3,1] | [4,1] | [5,1] | [6,1] | 
    |_______|_______|_______|_______|_______|_______|_______|

    a list such as [2,1], is a key with a corresponding up vector as the value 
    in up_vec_dictionary 

    **possible future problem - it's possible that my xz range does not encompass  
    all possible soma locations that I will want to pass through this function.
    One solution I can think of is to change the lowest and highest value in 
    x and z ranges**

    Parameters
    ----------
    x_pos: int
        x value(nm) of soma location 
    z_pos: int
        z value(nm) of soma location 
    up_vec_dictionary: dict
        the dictionary that contains the row,column pair and the corresponding 
        up vector for that column 

    Returns
    -------
    up vector: (3, 1) np.array 
        indicates the up vector for the column in which the x_pos and z_pos lie 
    '''

    # find row/column combo in the grid of the xz spread of the dataset 
    xz_bin = []
    # find x_pos column
    for i in range(len(x_ranges)):
        if x_ranges[i][0] <= x_pos <= x_ranges[i][1]:
            xz_bin.append(i)
        else:
            continue
    
    # find z_pos row
    for ii in range(len(z_ranges)):
        if z_ranges[ii][0] <= z_pos <= z_ranges[ii][1]:
            xz_bin.append(ii)
        else:
            continue

    return up_vec_dictionary[str(xz_bin)]


# following two functions from Forrest, I just tailored to meet my needs 

# functions to create the layer meshes 
def make_layer_poly(mesh_top, mesh_bottom, soma_pos, offset=1000, up_vec = np.array([0,-1,0])):
    # get the vec perpendicular to the plane of up vec and x : 
    verts_top = get_mesh_line(mesh_top, soma_pos, up_vec)
    verts_bot = get_mesh_line(mesh_bottom, soma_pos, up_vec)
    verts_top = verts_top[::-1,:] + np.array([0,offset,0])
    verts_bot = verts_bot - np.array([0,offset,0])
    poly_verts= np.concatenate([verts_top, verts_bot])
    # add the first vert back to the end of the array to close the loop
    poly_verts = np.vstack([poly_verts, poly_verts[0]])
    return poly_verts

def build_layer_dict(layer_name, up_vec, verts_top, verts_bottom, soma_pos, res):
    '''
    Creates layer polygon at a specific resoluton showing layers accross x. 
    Returns only x and y coordinates. 
    Adds that layer polygon and other information to a dictionary and returns that dictionary

    Parameters
    ----------
    layer_name: str
        the name of the layer. If minnie, will be "Layer1", "Layer2/3", "Layer4", "Layer5", "Layer6a", "Layer6b". Pia and white matter are handled with get_mesh_line function
    up_vec: 3x1 array
        indicates the up vector for the column in which the soma lies. use find_up_vec to find in minnie.
    verts_top: nx3 np array 
        array of the top mesh vertices
    verts_bottom: nx3 np array 
        array of the bottom mesh verts 
    soma_pos: 3, array
        soma position (todo: coordinates?)
    res: float
        resolution of the voxels (todo ?)

    Returns
    -------
    layer_dict: dictionary 
        dictionary formatted to be used in creation of all layer bounds for individual cells 
    
    '''

    # create the subdict that we will later add to the full layer_dict
    layer_dict = {}
    # create and insert the poly 

    # create and insert the poly
    layer_poly = np.around(make_layer_poly(verts_top, verts_bottom, soma_pos = soma_pos, up_vec = up_vec)[:,0:2]/[1000*res,1000*res])
    layer_dict['path'] = layer_poly.tolist()
    layer_dict['name'] = layer_name    
    layer_dict['resolution'] = res
    return layer_dict

def get_mesh_line(mesh, soma_pos, up_vec = np.array([0,-1,0])):
    up_vec = up_vec.reshape((3,))
    uv_norm = np.cross(up_vec, np.array([1,0,0]).T)
    lines=trimesh.intersections.mesh_plane(mesh, uv_norm, soma_pos)
    lines_con = lines.reshape(lines.shape[0]*lines.shape[1],3)
    
    verts,inv = np.unique(lines_con, axis=0, return_inverse=True)
    lines=inv.reshape(len(inv)//2,2)
    
    tree = KDTree(verts)
    pairs = tree.query_pairs(1)
    for pair in pairs:
        lines[lines==pair[1]]=pair[0]
    
    sk=skeleton.Skeleton(verts,lines,root=np.argmin(verts[:,0]))
    sk=sk.apply_mask(mesh_filters.filter_largest_component(sk))
    
    cp = sk.cover_paths[0]
    verts = sk.vertices[cp]
    return verts

def build_meshline_dict(layer_name, up_vec, mesh_verts, soma_pos, res, specimen_id):
    '''
    Creates mesh line at a specific resoluton showing mesh line cut accross x. 
    Returns only x and y coordinates. 
    returns a dictionary ready to be inserted into layer polygon

    Parameters
    ----------
    layer_name: str
        the name of the layer. If minnie, will be "Pia" or "White Matter"
    up_vec: 3x1 array
        indicates the up vector for the column in which the soma lies. use find_up_vec to find in minnie.
    mesh_verts: nx3 np array 
        array of the layer mesh vertices
    soma_pos: 3, array
        soma position (todo: coordinates?)
    res: float
        resolution of the voxels (todo ?)
    id: int
        numerical string that identifies the neuron for which the poly file is being made 

    Returns
    -------
    layer_dict: dictionary 
        dictionary formatted to be used in creation of all layer bounds for individual cells 
    '''
    # find the cut of mesh through line 
    layer_poly_line = np.around(get_mesh_line(mesh_verts, soma_pos, up_vec=up_vec)[:,0:2]/[1000*res,1000*res])

    # create dict to fill in 
    layer_dict = {}
    layer_dict['name'] = layer_name
    layer_dict['path'] = layer_poly_line.tolist()
    layer_dict['resolution'] = res
    layer_dict['biospecimen_id'] = specimen_id

    return layer_dict


def calculate_soma_poly(soma_loc, n, radius):
    '''
    returns n number of points evenly spaced around radius r circle from soma location
    this data does not really matter much for the EM application of skeleton keys 

    Parameters
    ----------
    soma_loc: (3,) array like object 
        soma coordinates of a given neuron
    n: int
        number of points to place around the circle 
    radius: int
        the radius of the circle 

    Returns
    -------
    vertices: (n, 2) np.array 
        vertices of the circle drawn around the given soma point
    '''
    
    return np.array([(math.cos(2*pi/n*x)*radius + soma_loc[0], math.sin(2*pi/n*x)*radius + soma_loc[2]) for x in range(0,n)])
    
  
def make_poly_file(mesh_dict, soma_pos, specimen_id, n_soma_circ_pts = 65, soma_rad = 2500, res = 0.3603):
    
    '''
    creates the polygon file that is needed to create layer aligned neurons in the skeleton keys repository 

    **this needs loops! I am not sure exactly what the best structure of the loop should be**

    Parameters
    ----------
    mesh_dict: dict
        dictionary contianing the meshes to be used to create layer bounds
    soma_pos: (3,) array like object 
        soma coordinates of a given neuron
    specimen_id: int
        numerical identification of the given neuron 
    n_soma_circ_pts: int (optional)
        number of points to place around the soma circle 
    soma_rad: int (optional)
        the radius of the circle drawn around the soma
    res: float
        resolution of the voxels (?)

    Returns
    -------
    poly_dict: dict
        polygon file needed to create layer aligned neurons using skeleton keys repository. contains information 
        on the layer locations of the neuron in question.
    '''
    # create the empty poly dict to be filled
    poly_dict = {}
    # create empty sub dicts/lists which will be filled then entered into poly dict

    # find the up vector for the given soma location  
    up_vec = np.array(find_up_vec(soma_pos[0], soma_pos[2]))
    
    # calculate layer vertices for the given soma location, taking into account the up vector for that area
    # start with l1, l23, l4, l5, l6a, l6b
    layers = ['pia', 'l23', 'l4', 'l5', 'l6a','l6b','wm']
    layer_names = ['Layer1', 'Layer2/3', 'Layer4', 'Layer5', 'Layer6a', 'Layer6b']
    # we will put these dicts into a list
    layer_polygons_list = []
    for i in range(len(layer_names)):
        # insert the poly info into poly_dict['layer_polygons']
        layer_polygon = build_layer_dict(layer_names[i], up_vec, mesh_dict[layers[i]], mesh_dict[layers[i+1]], soma_pos, res)
        layer_polygons_list.append(layer_polygon)
    # this polygon list will be appended to the poly_dict at the end!

    # now to create mesh lines for pia and white matter and insert them into poly_dict
    # redefine layers and layer names 
    layers = ['pia', 'wm']
    layer_names = ['Pia', 'White Matter']
    for i in range(len(layers)):
        dict_key = layers[i] + '_path'
        poly_dict[dict_key] = build_meshline_dict(layer_names[i], up_vec, mesh_dict[layers[i]], soma_pos, res, specimen_id)
    
    # draw a circle around the soma and add that to poly_dict
    soma_dict = {}
    soma_dict['name'] = "Soma"
    soma_dict['path'] = np.around(calculate_soma_poly(soma_pos, n_soma_circ_pts, soma_rad)/[1000*res,1000*res]).tolist()
    soma_dict['resolution'] = res
    soma_dict['biospecimen_id'] = specimen_id 
    soma_dict['center'] =  [soma_pos[0], soma_pos[2]]
    poly_dict['soma_path'] = soma_dict

    # finally insert layer_polygons_list to poly_dict
    poly_dict['layer_polygons'] = layer_polygons_list

    return poly_dict
    