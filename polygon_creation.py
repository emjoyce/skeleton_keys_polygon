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

# here are the x and z ranges of the dataset 
# maybe I should stick these in the json as well 
x_ranges = [(570112.0, 675958.8571428572),
 (675958.8571428572, 781805.7142857143),
 (781805.7142857143, 887652.5714285714),
 (887652.5714285714, 993499.4285714286),
 (993499.4285714286, 1099346.2857142857),
 (1099346.2857142857, 1205193.1428571427),
 (1205193.1428571427, 1311040.0)]

z_ranges = [(748880.0, 842260.0), (842260.0, 935640.0)]

# load the up vector dictionary 
with open('/Users/emily.joyce/Work/Repos/code_review/skeleton_keys_files/skeleton_keys_excitatory_features/polygon_creation/up_vecs.json') as json_file:
    up_vec_dict = json.load(json_file)

# load the layer meshes 
# I don't like this out of a function. but it will be necessary for polygon creation 
# so maybe this is ok? but what if someone just wants to use find_up_vec 
# or some other function that does not need the layer meshes?
mesh_dict={}
for filepath in ["layer_meshes/l23","layer_meshes/l4","layer_meshes/l5","layer_meshes/l6a","layer_meshes/l6b","layer_meshes/wm"]:
    with open(filepath,'rb') as fp:
        verts,faces = read_precomputed_mesh(fp)
    mesh = trimesh_io.Mesh(verts, faces)
    filename = filepath.split('/')[1]
    mesh_dict[filename]=mesh
# load pia separately 
pia_mesh = trimesh.load_mesh('layer_meshes/pia3.ply')
mesh_dict['pia'] = pia_mesh
mesh_dict['pia'] = trimesh_io.Mesh(mesh_dict['pia'].vertices * 1_000_000, mesh_dict['pia'].faces)


def find_up_vec(x_pos, z_pos, up_vec_dictionary = up_vec_dict):
    
    '''
    takes x and y position (nm) of a soma and returns the corresponding up 
    vector for the xz column that it falls into 

    the x and z positiona are used to find the column the position falls into
    example of what these columns look like here:
    _________________________________________________________
    |       |       |       |       |       |       |       |  
    | [0,0] | [1,0] | [2,0] | [3,0] | [4,0] | [5,0] | [6,0] | 
    |_______|_______|_______|_______|_______|_______|_______|
    |       |       |       |       |       |       |       |  
    | [0,1] | [1,1] | [2,1] | [3,1] | [4,1] | [5,1] | [6,1] | 
    |_______|_______|_______|_______|_______|_______|_______|

    a list such as [2,1], is a key with a correstponding up vector as the value 
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
    return poly_verts


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
    
  
def make_poly_file(soma_pos, specimen_id, n_soma_circ_pts = 65, soma_rad = 2500, res = 0.3603):
    
    '''
    creates the polygon file that is needed to create layer aligned neurons in the skeleton keys repository 

    **this needs loops! I am not sure exactly what the best structure of the loop should be**

    Parameters
    ----------
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
    pia_dict = {}
    wm_dict = {}
    soma_dict = {}
    layer_polygons_list = []
    # create the layer polygons subdicts 
    layer1_dict = {}
    layer2_3_dict = {}
    layer4_dict = {}
    layer5_dict = {}
    layer6a_dict = {}
    layer6b_dict = {}

    # find the up vector for the given soma location  
    up_vec = np.array(find_up_vec(soma_pos[0], soma_pos[2]))
    
    # calculate layer vertices for the given soma location, taking into account the up vector for that area
    # this could be a loop... pia and wm in a loop and all layers in another 
    pia_verts = np.around(get_mesh_line(mesh_dict['pia'], soma_pos, up_vec=up_vec)[:,0:2]/[1000*res,1000*res])
    lay1verts = np.around(make_layer_poly(mesh_dict['pia'], mesh_dict['l23'], up_vec=up_vec, soma_pos = soma_pos)[:,0:2]/[1000*res,1000*res])
    lay23verts = np.around(make_layer_poly(mesh_dict['l23'], mesh_dict['l4'], up_vec=up_vec, soma_pos = soma_pos)[:,0:2]/[1000*res,1000*res])
    lay4verts = np.around(make_layer_poly(mesh_dict['l4'], mesh_dict['l5'], up_vec=up_vec, soma_pos = soma_pos)[:,0:2]/[1000*res,1000*res])
    lay5verts = np.around(make_layer_poly(mesh_dict['l5'], mesh_dict['l6a'], up_vec=up_vec, soma_pos = soma_pos)[:,0:2]/[1000*res,1000*res])
    lay6averts = np.around(make_layer_poly(mesh_dict['l6a'], mesh_dict['l6b'], up_vec=up_vec, soma_pos = soma_pos)[:,0:2]/[1000*res,1000*res])
    lay6bverts = np.around(make_layer_poly(mesh_dict['l6b'], mesh_dict['wm'], up_vec=up_vec, soma_pos = soma_pos)[:,0:2]/[1000*res,1000*res])
    wm_verts = np.around(get_mesh_line(mesh_dict['wm'], soma_pos, up_vec=up_vec)[:,0:2]/[1000*res,1000*res])
    
    # these could also be loops 
    # fill out pia dict
    pia_dict['name'] = "Pia"
    pia_dict['path'] = pia_verts.tolist()
    pia_dict['resolution'] = res
    pia_dict['biospecimen_id'] = specimen_id
    # insert pia dict into main poly dict 
    poly_dict['pia_path'] = pia_dict
    
    # fill out white matter dict 
    wm_dict['name'] = "White Matter"
    wm_dict['path'] = wm_verts.tolist()
    wm_dict['resolution'] = res
    wm_dict['biospecimen_id'] = specimen_id
    # insert wm_dict to poly_dict
    poly_dict['wm_path'] = pia_dict
    
    # now do the soma path 
    soma_dict['name'] = "Soma"
    soma_dict['path'] = np.around(calculate_soma_poly(soma_pos, n_soma_circ_pts, soma_rad)/[1000*res,1000*res]).tolist()
    soma_dict['resolution'] = res
    soma_dict['biospecimen_id'] = specimen_id 
    soma_dict['center'] =  [soma_pos[0], soma_pos[2]]
    # insert to larger dict
    poly_dict['soma_path'] = soma_dict
    
    # now fill out the larger layer_polygons_dict
    # layer by layer 
    # these should be loops too 
    layer1_dict['path'] = lay1verts.tolist()
    layer1_dict['name'] = 'Layer1'
    layer1_dict['resolution'] = res
    
    layer2_3_dict['path'] = lay23verts.tolist()
    layer2_3_dict['name'] = 'Layer2/3'
    layer2_3_dict['resolution'] = res
    
    layer4_dict['path'] = lay4verts.tolist()
    layer4_dict['name'] = 'Layer4'
    layer4_dict['resolution'] = res

    layer5_dict['path'] = lay5verts.tolist()
    layer5_dict['name'] = 'Layer5'
    layer5_dict['resolution'] = res

    layer6a_dict['path'] = lay6averts.tolist()
    layer6a_dict['name'] = 'Layer6a'
    layer6a_dict['resolution'] = res
    
    layer6b_dict['path'] = lay6bverts.tolist()
    layer6b_dict['name'] = 'Layer6b'
    layer6b_dict['resolution'] = res
    
    #insert all of the above into layer_polygons_list
    layer_polygons_list.append(layer1_dict)
    layer_polygons_list.append(layer2_3_dict)
    layer_polygons_list.append(layer4_dict)
    layer_polygons_list.append(layer5_dict)
    layer_polygons_list.append(layer6a_dict)
    layer_polygons_list.append(layer6b_dict)

    # insert the layer_polygons_list into poly_dict
    poly_dict['layer_polygons'] = layer_polygons_list

    return poly_dict
    
    
    
    
