import pandas as pd
import numpy as np

mesh_dict = {}


new_mesh_dict = {}

def extend_meshes(minx = 193344.0, maxx = 1737536.0, minz = 593040.0, maxz = 1115160.0):
    for mesh_key in mesh_dict.keys():
        
        new_mesh_dict[mesh_key] = mesh_dict[mesh_key].vertices
        
        layer_verts = mesh_dict[mesh_key].vertices

        verts_df = pd.DataFrame()
        verts_df['x'] = layer_verts[:,0]
        verts_df['y'] = layer_verts[:,1]
        verts_df['z'] = layer_verts[:,2]

        xverts_df = verts_df.sort_values('x')
        xverts_df = verts_df.groupby(['x'])
        
        zverts_df = verts_df.sort_values('z')
        zverts_df = verts_df.groupby(['z'])
        
        if mesh_key == 'pia':
            break

        # this dict will help us to navigate through extending the x and then extanding the z 
        # mesh rows 
        xz_navigation_dict = {'x': [xverts_df, 'z', minz, maxz], 
                            'z': [zverts_df, 'x', minx, maxx]}
        min_zs = np.empty((0,3), int)
        max_zs = np.empty((0,3), int)
        min_xs = np.empty((0,3), int)
        max_xs = np.empty((0,3), int)
        for key, list_values in xz_navigation_dict.items():

            
            for x, yz_df in list_values[0]: # group by group in xverts_df or zverts_df
                
                m, b = np.polyfit(x = yz_df[list_values[1]], y = yz_df['y'], deg = 1)
                yz_df = yz_df.sort_values([list_values[1]]).reset_index()
                
                # plot
                fig, ax = pyplot.subplots()
                fig.set_size_inches(10.5, 2.5)
                ax.scatter(x = yz_df[list_values[1]], y = yz_df['y'])

                # now draw a line from the first point to min_x w/ the slope

                # find the y value at x = 48336
                # find delta z for min
                dz1 = yz_df.loc[0, list_values[1]] - list_values[2]
                dy1 = m*dz1
                additional_y1 = yz_df.loc[0, 'y']-dy1

                # find delta z for max
                dz2 = yz_df.loc[len(yz_df)-1, list_values[1]] - list_values[3]
                dy2 = m*dz2
                additional_y2 = yz_df.loc[len(yz_df)-1, 'y']-dy2

                ax.plot([yz_df.loc[0, list_values[1]], list_values[2]], [yz_df.loc[0, 'y'], additional_y1])
                ax.plot([yz_df.loc[len(yz_df)-1, list_values[1]], list_values[3]], [yz_df.loc[len(yz_df)-1, 'y'], additional_y2])
                
                
                # these all look good! add them to the mesh verts 
                # if mesh dict = x:
                if key == 'x':
                    minz_coords = np.array([x, additional_y1, list_values[2]])
                    maxz_coords = np.array([x, additional_y2, list_values[3]]) 
                    
                    # add min z pt
                    new_mesh_dict[mesh_key] = np.vstack((new_mesh_dict[mesh_key], minz_coords))
                    # add max z pt 
                    new_mesh_dict[mesh_key] = np.vstack((new_mesh_dict[mesh_key], maxz_coords))
                    
                    # finally, keep track of the min and max coordinates. 
                    min_zs = np.vstack((min_zs, minz_coords))
                    max_zs = np.vstack((max_zs, maxz_coords))
                    
                if key == 'z':
                    minx_coords = np.array([list_values[2], additional_y1, x])
                    maxx_coords = np.array([list_values[3], additional_y2, x])
                    
                    # add min z pt
                    new_mesh_dict[mesh_key] = np.vstack((new_mesh_dict[mesh_key], minx_coords))
                    # add max z pt 
                    new_mesh_dict[mesh_key] = np.vstack((new_mesh_dict[mesh_key], maxx_coords))
                    
                    # finally, keep track of the min and max coordinates. 
                    min_xs = np.vstack((min_xs, minx_coords))
                    max_xs = np.vstack((max_xs, maxx_coords))



