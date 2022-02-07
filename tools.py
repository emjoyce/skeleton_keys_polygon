import pandas as pd
import numpy as np



def extend_vertices(vertices, minx = 193344.0, maxx = 1737536.0, minz = 593040.0, maxz = 1115160.0):

    '''
    takes in an array of 3d vertices that are organized in a grid-like manner and extends the edge of those vertices to a specified x min, x max, z min, z max. Extention is done by creating a best fit line on each row and column and placing the new extended vertices at the min and max x or z point. Default mins/maxes are pulled from min and max soma location in minnie dataset. Note that the mesh extension is missing its corners. The vertices added will be like a border at z min, max and x min and max

                              .  .  .  .  .   
                                              
    .  .  .  .  .         .   .  .  .  .  .   .                
    .  .  .  .  .         .   .  .  .  .  .   .
    .  .  .  .  .   ->    .   .  .  .  .  .   .
    .  .  .  .  .         .   .  .  .  .  .   .
    .  .  .  .  .         .   .  .  .  .  .   .

                              .  .  .  .  .  
    
    Parameters
    ----------
    vertices: nx3 array
        vertices arranged in a grid-like manner as shown above on the left 
    minx: int/float
        the minimum x value you wish to extend your vertices to 
    maxx: int/float
        the maximum x value you wish to extend your vertices to 
    minz: int/float
        the minimum z value you wish to extend your vertices to 
    maxz: int/float
        the maximum z value you wish to extend your vertices to 

    Returns
    -------
    extended_vertices: nx3 array like 
        array containing the old vertices as well as new vertices representing the x and z extension 
    
    '''
    
    extended_vertices = vertices
    
    # put verts into a df 
    verts_df = pd.DataFrame(vertices, columns = ['x','y','z'])

    # group by x to iterate through columns, group by z to navigate through rows
    xverts_df = verts_df.sort_values(['x', 'z']).groupby(['x'])
    zverts_df = verts_df.sort_values(['z', 'x']).groupby(['z'])
    # this dict will help us to navigate through the rows and columns
    xz_navigation_dict = {'x': [xverts_df, 'z', minz, maxz], 
                        'z': [zverts_df, 'x', minx, maxx]}

    for key, list_values in xz_navigation_dict.items():
        for x, group_df in list_values[0]: # go group by group in xverts_df or zverts_df
            group_df = group_df.reset_index(drop = True)
            # get the slope of the best fit line for this row/column
            m, b = np.polyfit(x = group_df[list_values[1]], y = group_df['y'], deg = 1)
            
            # now create a new vertex point at the specified minimum x or z value
            # use the slope from best fit line to find what the y value should be at this point   
            delta_xorz_min = group_df.loc[0, list_values[1]] - list_values[2]
            delta_y_min = m * delta_xorz_min
            additional_y_min = group_df.loc[0, 'y'] - delta_y_min

            # now repeat but for the max x or z
            delta_xorz_max = group_df.loc[len(group_df)-1, list_values[1]] - list_values[3]
            delta_y_max = m * delta_xorz_max
            additional_y_max = group_df.loc[len(group_df)-1, 'y'] - delta_y_max
            
            # add them to the extended verts array. done differently if dealing with x or z groups. 
            if key == 'x':
                minz_coords = np.array([x, additional_y_min, list_values[2]])
                maxz_coords = np.array([x, additional_y_max, list_values[3]]) 
                
                extended_vertices = np.vstack((extended_vertices, minz_coords))
                extended_vertices = np.vstack((extended_vertices, maxz_coords))

            if key == 'z':
                minx_coords = np.array([list_values[2], additional_y_min, x])
                maxx_coords = np.array([list_values[3], additional_y_max, x])
                
                extended_vertices = np.vstack((extended_vertices, minx_coords))
                extended_vertices = np.vstack((extended_vertices, maxx_coords))

    return extended_vertices


def create_faces(vertices):
    '''
    creates triangular faces for grid like vertices that have been extended by the extend_vertices function.

                            1----2
                          / |  / | \\
                         /  | /  |  \       
                       3----4----5----6    
      [[1,2],          |  / |  / |  / |
    [3,4,5,6],    ->   | /  | /  | /  |
    [7,8,9,10],        7----8----9----10
     [11, 12]]           \  |  / |  /
                          \ | /  | /
                            11---12
            with each triangle represented as a list as such:
        [[3,1,4], [4,1,2], [4,2,5], [5,2,6], [7,3,4]... [12,9,10]]


    Parameters
    ----------
    vertices: nx3 array
        vertices arranged in a grid-like manner that have been extended by extend_vertices function  

    Returns
    -------
    faces: nx3 array  
        array containing the indices of each triangle covering the grid
    
    '''
    
    faces = np.empty((0,3), int)
    # put mesh vertices into dataframe and group by the z parameter so we can go row by row 
    mesh_vert_df = verts_to_df(vertices).groupby('z')

    # go z value by z value (row by row)
    previous_indices = []
    for z, zdf in mesh_vert_df:

        # if this is the first row, continue 
        if previous_indices == []:
            previous_indices = list(zdf['index'])
            continue

        current_indices = list(zdf['index'])
        # iterate through verts 
        # it it's the first row:
        if len(previous_indices) < len(zdf):
            # draw the first triangle 
            faces = np.vstack((faces, [current_indices[0], previous_indices[0], current_indices[1]]))
            # draw the middle triangles 
            middle_faces = triangulate_grid_rows(previous_indices, current_indices[1:-1])
            faces = np.vstack((faces, middle_faces))
            # draw the final triangles 
            faces = np.vstack((faces, [current_indices[-1], current_indices[-2], previous_indices[-1]]))

        # if it's the last row: 
        elif len(zdf) < len(previous_indices):
            # first triangle
            faces = np.vstack((faces, [current_indices[0], previous_indices[0], previous_indices[0]]))
            # middle triangles 
            middle_faces = triangulate_grid_rows(previous_indices[1:-1], current_indices)
            faces = np.vstack((faces, middle_faces))    
            # last triangle 
            faces = np.vstack((faces, [current_indices[-1], previous_indices[-1], previous_indices[-2]]))
        
        # otherwise, it's a middle row 
        else:
            faces = np.vstack((faces, triangulate_grid_rows(previous_indices, current_indices)))

        # set the previous indices to these ones for the next iteration 
        previous_indices = list(zdf['index'])

    return faces

def triangulate_grid_rows(row_1_idxs, row_2_idxs):
    
    '''
    creates triangle faces for two rows with equal number of indices 
    
                            1----2----3
    [1,2,3], [4,5,6]   ->   |   /|   /|   with each triangle face represented as 
                            |  / |  / |   list as such:
                            | /  | /  |   [[4,1,2], [4,2,5], [5,2,3], [5,3,6]]
                            4----5----6
    Parameters
    ----------
    row_1_idxs: list/array
        indices of first row. num indices row 1 = num indices row 2
    row_2_idxs: list/array
        indices of second row. num indices row 2 = num indices row 1

    Returns
    -------
    faces: nx3 array like 
        array containing the indices of each triangular face 
    '''
    faces = []
    # start at the bottom row, make 2 clockwise triangles from each idx
    # stop at the last idx - no indices to the right of that one to create triangles with 
    for i in range(len(row_2_idxs[:-1])):
        faces.append([row_2_idxs[i], row_1_idxs[i], row_1_idxs[i+1]])
        faces.append([row_2_idxs[i], row_1_idxs[i+1], row_2_idxs[i+1]])
    return np.array(faces)


def verts_to_df(vertices):
    '''
    takes vertices that are distributed in a grid like manner and puts them into a dataframe 
    that is sorted by z and x components 
    
    Parameters
    ----------
    vertices: array-like nx3
        xyz vertices 

    Returns
    -------
    dataframe: np.DataFrame
        dataframe contianing the x, y, z components of each vertex in 'x', 'y', 'z' labeled columns
        sorted primarily by z, then by x. 
        'index' column pulled out indicating the order of each row 
    '''
    return pd.DataFrame(vertices, columns = ['x', 'y', 'z']).sort_values(['z', 'x']).reset_index(drop = True).reset_index()
    