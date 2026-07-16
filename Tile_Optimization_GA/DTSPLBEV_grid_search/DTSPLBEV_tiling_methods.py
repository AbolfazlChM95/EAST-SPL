# Tiling Methods
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from time import time as tick

def Tiling(Method, # See the list below
           img_shape, # you could just call img.shape or equally mask.shape
           initial_tile_size: int = 1024,  # for static one this is the length
           mask = None, # the masked image to verfy the tiles
           Tiles2FLOPs = None, # Need this function to get the FLOPs for tile configuration
           margin: int = 0, # the margin to add to each tile for receptive field 
           minimum_length: int = 50, # The minimum length to stop dividing, specificly the only critical option in DynMinL method
           outlier_pixels_threshold = 1, # the limit for including tiles for number of pixels in the field, extrem case is 1
           Full_process_verbos = False # Return All the Tilings during the process of making and merging
           ):
    # Merging all of the tiling methods together as one function to call

    match Method:
        case 'JT':                 # Just Tile whole 4K
            Tiles = JustTileIt(img_shape, initial_tile_size)
        case 'NS':                 # Naive Static
            Tiles = Naive_Static(img_shape, initial_tile_size, mask, outlier_pixels_threshold = outlier_pixels_threshold)
        case 'CS':                 # Cropped Static
            Tiles = Cropped_Static(initial_tile_size, mask, outlier_pixels_threshold = outlier_pixels_threshold)
        case 'DynMinL':            # Dynamic Stop with Min Length
            Tiles = Dynamic_min_length(mask, initial_tile_size = initial_tile_size, minimum_length = minimum_length, margin = margin, Full_process_verbos = Full_process_verbos, outlier_pixels_threshold = outlier_pixels_threshold)
        case 'DynFlopsA':          # Dynamic Stop with Flops
            Tiles = Dynamic_FLOPs_Stop(mask, Tiles2FLOPs, initial_tile_size = initial_tile_size, margin = margin, Full_process_verbos = Full_process_verbos, outlier_pixels_threshold = outlier_pixels_threshold)
        case 'DynFlopsNest':       # Dynamic Flops check in chain
            Tiles = Dynamic_FLOPs_Nested(mask, Tiles2FLOPs, initial_tile_size = initial_tile_size, minimum_length = minimum_length, margin = margin, Full_process_verbos = Full_process_verbos, outlier_pixels_threshold = outlier_pixels_threshold)
        case 'DynFlopsConsist':    # Dynamic Flops check in chain, repeat untill converge
            Tiles = Dynamic_FLOPs_Nested_Consistency(mask, Tiles2FLOPs, initial_tile_size = initial_tile_size, minimum_length = minimum_length, margin = margin, Full_process_verbos = Full_process_verbos, outlier_pixels_threshold = outlier_pixels_threshold)
        case 'DynFlopsExhaust':    # Dynamic Flops Tree exhaustive search
            Tiles = Dynamic_FLOPs_Exhaustive(mask, Tiles2FLOPs, initial_tile_size = initial_tile_size, minimum_length = minimum_length, margin = margin, Full_process_verbos = Full_process_verbos, outlier_pixels_threshold = outlier_pixels_threshold)
        

    outside_pixels_count = Count_Outside_Pixels(Tiles, mask)
    Num_tile_for_pixels = Count_NumTiles_for_Pixels(mask, Tiles, margin)

    return Tiles, outside_pixels_count, Num_tile_for_pixels


def JustTileIt(img_shape,
               tile_length):
    # This is Niave tiling without using mask. this is minimum requirement for memory consideration if we want to replace 4K
    # This function was added for second time submittion of DTSPL-BEV!
    x_s = np.arange(1, img_shape[0], tile_length)
    y_s = np.arange(1, img_shape[1], tile_length)
    X, Y = np.meshgrid(x_s, y_s)
    Tile_boxes = np.concatenate((np.expand_dims(X.flatten(), axis=1),
                           np.expand_dims(Y.flatten(), axis=1)) , axis=1)
    Tile_boxes = np.hstack( (Tile_boxes, tile_length * np.ones( (Tile_boxes.shape[0],2), dtype=np.int64) ))
    Tile_boxes = refine_out_image(Tile_boxes, img_shape)
    return Tile_boxes

def Naive_Static(img_shape, 
                tile_length, # Lenght (= width = height) of squre tiles
                mask, 
                outlier_pixels_threshold = 1,
                ):
    # Simple Tiling, start from top left corner of image and continue with tile_length
    # assume square tiles, height = width as tile_length
    # img_shape is dimension of image
    # tile_length is a scalar
    # mask is array with same shape as img_shape
    # If mask is given, return only tiles with at least 1 pixel in the mask

    # output is a Nx4 array each row has x,y,w,h here (w=h) for box of tilings

    x_s = np.arange(1, img_shape[0], tile_length)
    y_s = np.arange(1, img_shape[1], tile_length)
    X, Y = np.meshgrid(x_s, y_s)
    Tile_boxes = np.concatenate((np.expand_dims(X.flatten(), axis=1),
                           np.expand_dims(Y.flatten(), axis=1)) , axis=1)
    Tile_boxes = np.hstack( (Tile_boxes, tile_length * np.ones( (Tile_boxes.shape[0],2), dtype=np.int64) ))

    Tile_boxes = refine_out_image(Tile_boxes, img_shape)

    Tile_boxes_filtered = []
    # Loop over boxes
    for box in Tile_boxes:
        in_pixels, out_pixels = in_out_count(box, mask)
        # check if even 1 pixel is 1
        if in_pixels >= outlier_pixels_threshold :
            Tile_boxes_filtered.append(box)
            
    # replace filtered boxes
    Tile_boxes = np.array(Tile_boxes_filtered)        

    return Tile_boxes


def Cropped_Static(tile_length, # Lenght (= width = height) of squre tiles
                    mask ,
                    outlier_pixels_threshold = 1, # the limit for including tiles for number of pixels in the field, extrem case is 1
                    ):
    # Simple Tiling, start from top left corner of image and continue with tile_length
    # assume square tiles, height = width as tile_length
    # tile_length is a scalar
    # mask is 2D array
    # If mask is given, return only tiles with at least 1 pixel in the mask
    # This version (Cropped) of naive tiling just limit the x_s and y_s to the box arount the field not the whole image
    # output is a Nx4 array each row has x,y,w,h here (w=h) for box of tilings
    
    ones_x, ones_y = np.where(mask == 1)
    x_field_min, x_field_max = ones_x.min(), ones_x.max()
    y_field_min, y_field_max = ones_y.min(), ones_y.max()

    x_s = np.arange(x_field_min, x_field_max, tile_length)
    y_s = np.arange(y_field_min, y_field_max, tile_length)

    X, Y = np.meshgrid(x_s, y_s)

    Tile_boxes = np.concatenate((np.expand_dims(X.flatten(), axis=1),
                           np.expand_dims(Y.flatten(), axis=1)) , axis=1)
    Tile_boxes = np.hstack( (Tile_boxes, tile_length * np.ones( (Tile_boxes.shape[0],2), dtype=np.int64) ))
    
    Tile_boxes = refine_out_image(Tile_boxes, mask.shape)
    
    Tile_boxes_filtered = []

    # Loop over boxes
    for box in Tile_boxes:
        in_pixels, out_pixels = in_out_count(box, mask)
        # check number of pixels in field is at least as threshold
        if in_pixels >= outlier_pixels_threshold :
            Tile_boxes_filtered.append(box)
    # replace filtered boxes
    Tile_boxes = np.array(Tile_boxes_filtered)        


    return Tile_boxes


def Dynamic_min_length(
        mask,
        initial_tile_size: int = 1024,
        minimum_length: int = 50,
        margin: int = 0,
        outlier_pixels_threshold: int = 1,
        Full_process_verbos = False, # Return All the Tilings during the process of making and merging
        ):
    
    # This method uses method similar to the cropped naive method to get the initial list of Tilings!
    # Then uses divide and select method to throw out or devide some tiles.
    # This method doesn't rely on any smart stopping method and only stops when tile length reached a given minimum length
    # for deviding by 2, it's better that the initial tile size is power of 2, or at least to the point that it gets less than minimum it's devidable by 2 multiple time
    # The above caution (power of 2) is not checked or flagged here! that the code could be more flexible

    ## Note!!!!
    # Making Full_process_verbos True will change the output Tile_boxes data structure, it will be a dictionary. the last item will be the final ordinary tiling

    #%% Initialization with cropped static method
    Tile_boxes = Cropped_Static(initial_tile_size, mask, outlier_pixels_threshold= outlier_pixels_threshold)

    if Full_process_verbos:
        Tiling_hist = {"Initial Tiles" : Tile_boxes.copy()}

    #%% Dynamic Loop for dividing the tiles
    Divide_flag = True
    divide_step = 1
    while Divide_flag:

        Tile_boxes_filtered = []

        Divide_flag = False

        for tile_idx in range(Tile_boxes.shape[0]):
            _, _, w, h = Tile_boxes[tile_idx,:]
            tile_length = min(w, h)

            # Remove Trivial Tiles
            in_pixels, out_pixels = in_out_count(Tile_boxes[tile_idx,:], mask)
            if out_pixels <= outlier_pixels_threshold:
                Tile_boxes_filtered.append(Tile_boxes[tile_idx,:].astype(int))
                
            # Dividing
            else:

                if (tile_length/2)< minimum_length:
                    Tile_boxes_filtered.append(Tile_boxes[tile_idx,:].astype(int))
                
                else:
                    
                    # Following few Lines, are based on the Idea that I have the number of outside Pixels and I can look to the future
                    #       and see there's no way that I will throw even one tile out, so I'll skip the whole dividing here
                    # Look at the future subtiles and check if even in most extreme didive we throw something out!
                    final_minimum_length = tile_length / 2 ** (np.ceil(np.log2(tile_length / minimum_length)))
                    if (final_minimum_length **2 - out_pixels) > outlier_pixels_threshold:
                        # This means that in final divides we're not throwing out anything so don't devide it at the first place
                        Tile_boxes_filtered.append(Tile_boxes[tile_idx,:].astype(int))
                        continue

                    # This means that we have more than outlier_pixels_threshold pixels outside of the field on tile and also 
                    # we can still divide and not reach the minumum length
                    Divide_flag = True

                    subtiles = Divide_Tile(Tile_boxes[tile_idx,:])

                    for subtile in subtiles:

                        in_pixels, out_pixels = in_out_count(subtile, mask)
                        if in_pixels >= outlier_pixels_threshold:
                            Tile_boxes_filtered.append(subtile.astype(int))
        
        if Divide_flag: # If still false keep the previous
            Tile_boxes = np.array(Tile_boxes_filtered)

            if Full_process_verbos:
                Tiling_hist[f'step {divide_step} of Dividing'] = Tile_boxes.copy()
                divide_step += 1

    if Full_process_verbos:
        # If full history is wanted I am returning the whole history as tiles
        Tile_boxes = Tiling_hist

    return Tile_boxes


def Dynamic_FLOPs_Stop(
        mask,
        Tiles2FLOPs, # Need this function to get the FLOPs for tile configuration
        initial_tile_size: int = 1024,
        margin: int = 0,
        outlier_pixels_threshold: int = 1,
        Full_process_verbos = False, # Return All the Tilings during the process of making and merging
        ):
    
    # This code is updated version of Dynamic_min_lenght but based on using FLOPs on one step for Dividing criteria Step
    # In the Dynamic_min_length I wasn't initializing the tiles, now I am throwing completely outside fields out

    # This method uses method similar to the cropped naive method to get the initial list of Tilings!
    # Then uses divide and select method to throw out or devide some tiles.
    # for deviding by 2, it's better that the initial tile size is power of 2, or at least to the point that it gets less than minimum it's devidable by 2 multiple time
    # The above caution (power of 2) is not checked or flagged here! that the code could be more flexible

    ## Note!!!!
    # Making Full_process_verbos True will change the output Tile_boxes data structure, it will be a dictionary. the last item will be the final ordinary tiling

    #%% Initialization with cropped static method
    Tile_boxes = Cropped_Static(initial_tile_size, mask, outlier_pixels_threshold= outlier_pixels_threshold)

    if Full_process_verbos:
        Tiling_hist = {"Initial Tiles" : Tile_boxes.copy()}

    #%% Dynamic Loop for dividing the tiles

    Divide_flag = True
    divide_step = 1

    minimum_length = 20

    while Divide_flag:

        Tile_boxes_filtered = []

        Divide_flag = False

        for tile_idx in range(Tile_boxes.shape[0]):
            
            # Get the tile
            _, _, w, h = Tile_boxes[tile_idx,:].astype(int)
            
            # Remove Trivial Tiles
            # If tile has less than the threshold from the field!
            in_pixels, out_pixels = in_out_count(Tile_boxes[tile_idx,:], mask)
            if out_pixels <= outlier_pixels_threshold: # Condition that there Tile contain outside of field (Up to a threshold)
                Tile_boxes_filtered.append(Tile_boxes[tile_idx,:].astype(int))
            
            elif min(w,h) <= minimum_length:
                Tile_boxes_filtered.append(Tile_boxes[tile_idx,:].astype(int))
            
            # Dividing
            else:
                # Calculate Pre-Dividing FLOPs
                FLOPS_pre = Tiles2FLOPs(Tile_boxes[[tile_idx],:].astype(int), margin)

                subtiles = Divide_Tile(Tile_boxes[tile_idx,:])

                subtiles_filtered = []
                for subtile in subtiles:
                    in_pixels, out_pixels = in_out_count(subtile, mask)
                    if in_pixels >= outlier_pixels_threshold:
                        subtiles_filtered.append(subtile.astype(int))

                # Calculate Post-Dividing FLOPs
                FLOPS_post = Tiles2FLOPs(np.array(subtiles_filtered).astype(int), margin)

                if FLOPS_post < FLOPS_pre:
                    Divide_flag = True # The configuration changed
                    for subtile in subtiles_filtered:
                        Tile_boxes_filtered.append(subtile.astype(int))
                else:
                    Tile_boxes_filtered.append(Tile_boxes[tile_idx,:].astype(int))


        if Divide_flag: # If still false keep the previous
            Tile_boxes = np.array(Tile_boxes_filtered)

            if Full_process_verbos:
                Tiling_hist[f'step {divide_step} of Dividing'] = Tile_boxes.copy()
                divide_step += 1

    if Full_process_verbos:
        # If full history is wanted I am returning the whole history as tiles
        Tile_boxes = Tiling_hist

    return Tile_boxes


def Dynamic_FLOPs_Nested(
        mask,
        Tiles2FLOPs, # Need this function to get the FLOPs for tile configuration
        initial_tile_size: int = 1024,
        minimum_length: int = 64, # Here I am using this for how much I want to go finer and calculate the FLOPs
        margin: int = 0,
        outlier_pixels_threshold: int = 1,
        Full_process_verbos = False, # Return All the Tilings during the process of making and merging
        ):
    
    # This code is updated version of Dynamic_FLOPs_Stop
    # Here I am proceding the Division of Tiles without caring about FLOPs at the begining and keep tracks of FLOPs for different stages
    # But at the end selecting the one division that has the best FLOPs!
    # So Instead of While loop on division process and then loop through the tiles, I will loop through the Tiles and do the division untill possible!

    # This method uses method similar to the cropped naive method to get the initial list of Tilings!
    # Then uses divide and select method to throw out or devide some tiles.
    # for deviding by 2, it's better that the initial tile size is power of 2, or at least to the point that it gets less than minimum it's devidable by 2 multiple time
    # The above caution (power of 2) is not checked or flagged here! that the code could be more flexible

    ## Note!!!!
    # Making Full_process_verbos True will change the output Tile_boxes data structure, it will be a dictionary. the last item will be the final ordinary tiling

    #%% Initialization with cropped static method
    Tile_boxes = Cropped_Static(initial_tile_size, mask, outlier_pixels_threshold= outlier_pixels_threshold)

    if Full_process_verbos:
        Tiling_hist = {"Initial Tiles" : Tile_boxes.copy()}

    #%% Dynamic Loop for dividing the tiles

    Tile_boxes_filtered = []
    
    for tile_idx in range(Tile_boxes.shape[0]):
        Dynamic_Tile_boxes = Tile_boxes[[tile_idx],:]
        Dynamic_Tile_hist = [Dynamic_Tile_boxes]
        FLOPS_list = [Tiles2FLOPs(Tile_boxes[[tile_idx],:].astype(int), margin)]
        
        # Remove Trivial Tiles
        # If tile has less than the threshold from the field!
        in_pixels, out_pixels = in_out_count(Tile_boxes[tile_idx,:], mask)
        if out_pixels <= outlier_pixels_threshold: # Condition that there Tile contain outside of field (Up to a threshold)
            Divide_flag = False
        else: # Then I need to look at Dividing
            Divide_flag = True

        while Divide_flag:

            Divide_flag = False # To control If the list of tile changed, not stuck in loop without dividing
            Dynamic_Tile_boxes_filtered = []
            for tile_idx_inner in range(Dynamic_Tile_boxes.shape[0]):
                    # Get the tile
                    x, y, w, h = Dynamic_Tile_boxes[tile_idx_inner,:].astype(int)
                    in_pixels, out_pixels = in_out_count(Dynamic_Tile_boxes[tile_idx_inner,:], mask)
                    if out_pixels <= outlier_pixels_threshold:
                        # Check if I need to divide this subtile
                        Dynamic_Tile_boxes_filtered.append(Dynamic_Tile_boxes[tile_idx_inner,:].astype(int))
                    elif max(w,h) < minimum_length:
                        # Check if this subtile reached minimum length
                        Dynamic_Tile_boxes_filtered.append(Dynamic_Tile_boxes[tile_idx_inner,:].astype(int))
                    else:
                        Divide_flag = True
                        # Divide the subtile
                        subtiles = Divide_Tile(Dynamic_Tile_boxes[tile_idx_inner,:])
                        for subtile in subtiles:
                            in_pixels, out_pixels = in_out_count(subtile, mask)
                            if in_pixels >= outlier_pixels_threshold:
                                Dynamic_Tile_boxes_filtered.append(subtile.astype(int))
            Dynamic_Tile_boxes = np.array(Dynamic_Tile_boxes_filtered).astype(int)
            Dynamic_Tile_hist.append(Dynamic_Tile_boxes)

            FLOPS_list.append(Tiles2FLOPs(Dynamic_Tile_boxes, margin))
            
        # Select the optimal level of Dividing based on FLOPs        
        FLOPS_list = np.array(FLOPS_list)
        FLOPs_argmin_idx = np.argmin(FLOPS_list)
        Optimal_Tile_config = Dynamic_Tile_hist[FLOPs_argmin_idx]
        for subtile in Optimal_Tile_config:
            Tile_boxes_filtered.append(subtile)

    Tile_boxes = np.array(Tile_boxes_filtered)

    if Full_process_verbos:
        # If full history is wanted I am returning the whole history as tiles
        Tile_boxes = Tiling_hist

    return Tile_boxes

def Dynamic_FLOPs_Nested_Consistency(
        mask,
        Tiles2FLOPs, # Need this function to get the FLOPs for tile configuration
        initial_tile_size: int = 1024,
        minimum_length: int = 64, # Here I am using this for how much I want to go finer and calculate the FLOPs
        margin: int = 0,
        outlier_pixels_threshold: int = 1,
        Full_process_verbos = False, # Return All the Tilings during the process of making and merging
        ):
    
    # This code is updated version of Dynamic_FLOPs_Stop
    # Here I am proceding the Division of Tiles without caring about FLOPs at the begining and keep tracks of FLOPs for different stages
    # But at the end selecting the one division that has the best FLOPs!
    # So Instead of While loop on division process and then loop through the tiles, I will loop through the Tiles and do the division untill possible!

    # This method uses method similar to the cropped naive method to get the initial list of Tilings!
    # Then uses divide and select method to throw out or devide some tiles.
    # for deviding by 2, it's better that the initial tile size is power of 2, or at least to the point that it gets less than minimum it's devidable by 2 multiple time
    # The above caution (power of 2) is not checked or flagged here! that the code could be more flexible

    ## Note!!!!
    # Making Full_process_verbos True will change the output Tile_boxes data structure, it will be a dictionary. the last item will be the final ordinary tiling
        
    #%% Initialization with cropped static method
    Tile_boxes = Cropped_Static(initial_tile_size, mask, outlier_pixels_threshold= outlier_pixels_threshold)

    if Full_process_verbos:
        Tiling_hist = {"Initial Tiles" : Tile_boxes.copy()}

    #%% Dynamic Loop for dividing the tiles

    Divide_consistency_flag = True
    Dividing_step = 1

    while Divide_consistency_flag:

        Tile_boxes_pre = Tile_boxes.copy()
        Tile_boxes_filtered = []
        
        for tile_idx in range(Tile_boxes.shape[0]):
            Dynamic_Tile_boxes = Tile_boxes[[tile_idx],:]
            Dynamic_Tile_hist = [Dynamic_Tile_boxes]
            FLOPS_list = [Tiles2FLOPs(Tile_boxes[[tile_idx],:].astype(int), margin)]

            # Get the tile
            x, y, w, h = Tile_boxes[tile_idx,:].astype(int)
            
            # Remove Trivial Tiles
            # If tile has less than the threshold from the field!
            in_pixels, out_pixels = in_out_count(Tile_boxes[tile_idx,:], mask)
            if out_pixels <= outlier_pixels_threshold: # Condition that there Tile contain outside of field (Up to a threshold)
                # Tile_boxes_filtered.append(Tile_boxes[tile_idx,:].astype(int))
                Divide_flag = False
            elif max(w,h) < minimum_length:
                Divide_flag = False
            else: # Then I need to look at Dividing
                Divide_flag = True


            while Divide_flag:

                Divide_flag = False # To control If the list of tile changed, not stuck in loop without dividing
                Dynamic_Tile_boxes_filtered = []

                for tile_idx_inner in range(Dynamic_Tile_boxes.shape[0]):
                    x, y, w, h = Dynamic_Tile_boxes[tile_idx_inner,:]
                    in_pixels, out_pixels = in_out_count(Dynamic_Tile_boxes[tile_idx_inner,:], mask)
                    if out_pixels <= outlier_pixels_threshold:
                        # Check if I need to divide this subtile
                        Dynamic_Tile_boxes_filtered.append(Dynamic_Tile_boxes[tile_idx_inner,:].astype(int))
                    elif max(w,h) < minimum_length:
                        Dynamic_Tile_boxes_filtered.append(Dynamic_Tile_boxes[tile_idx_inner,:].astype(int))
                    else:
                        Divide_flag = True
                        # Divide the subtile
                        subtiles = Divide_Tile(Dynamic_Tile_boxes[tile_idx_inner,:])
                        for subtile in subtiles:
                            in_pixels, out_pixels = in_out_count(subtile, mask)
                            if in_pixels >= outlier_pixels_threshold:
                                Dynamic_Tile_boxes_filtered.append(subtile.astype(int))
                    
                Dynamic_Tile_boxes = np.array(Dynamic_Tile_boxes_filtered).astype(int)
                Dynamic_Tile_hist.append(Dynamic_Tile_boxes)

                FLOPS_list.append(Tiles2FLOPs(Dynamic_Tile_boxes, margin))
            
            # Select the optimal level of Dividing based on FLOPs        
            FLOPS_list = np.array(FLOPS_list)
            FLOPs_argmin_idx = np.argmin(FLOPS_list)
            Optimal_Tile_config = Dynamic_Tile_hist[FLOPs_argmin_idx]
            for subtile in Optimal_Tile_config:
                Tile_boxes_filtered.append(subtile)
        Tile_boxes = np.array(Tile_boxes_filtered)

        Divide_consistency_flag = not np.array_equal(Tile_boxes, Tile_boxes_pre)

        if Full_process_verbos:
            Tiling_hist = {f"Dividing Step {Dividing_step}" : Tile_boxes.copy()}
            Dividing_step += 1


    if Full_process_verbos:
        # If full history is wanted I am returning the whole history as tiles
        Tile_boxes = Tiling_hist

    return Tile_boxes

def Dynamic_FLOPs_Exhaustive(
        mask,
        Tiles2FLOPs, # Need this function to get the FLOPs for tile configuration
        initial_tile_size: int = 1024,
        minimum_length: int = 64, # Here I am using this for how much I want to go finer and calculate the FLOPs
        margin: int = 0,
        outlier_pixels_threshold: int = 1,
        Full_process_verbos = False, # Return All the Tilings during the process of making and merging
        ):
    
    # This code is update based on FLOPs Nested (not with consistency loop, better results)

    #%% Initialization with cropped static method
    Tile_boxes = Cropped_Static(initial_tile_size, mask, outlier_pixels_threshold= outlier_pixels_threshold)

    if Full_process_verbos:
        Tiling_hist = {"Initial Tiles" : Tile_boxes.copy()}

    #%% Dynamic Loop for dividing the tiles

    Tile_boxes_filtered = []
    
    for tile_idx in range(Tile_boxes.shape[0]):
        
        Super_Tile_boxes = [Tile_boxes[[tile_idx],:]]

        # Skip Tiles with less than threshold pixels outside of the field
        in_pixels, out_pixels = in_out_count(Tile_boxes[tile_idx, :], mask)
        if out_pixels <= outlier_pixels_threshold: # Condition that there Tile contain outside of field (Up to a threshold)
            Divide_flag = False
        else: # Then I need to look at Dividing
            Divide_flag = True

        checked_index = 0 # Using this to skip going through already processed configurations again

        while Divide_flag:

            Divide_flag = False

            previous_num_tiling_configs = len(Super_Tile_boxes)

            Current_Super_Tile_boxes = Super_Tile_boxes.copy()

            # loop throguh Super_Tile_boxes
            for Tiling in Current_Super_Tile_boxes[ checked_index : ]:
                
                num_tiles = Tiling.shape[0]
                for tile_idx in range(num_tiles):
                    
                    tile = Tiling[tile_idx, :]
                    in_pixels, out_pixels = in_out_count(tile, mask)
                    if out_pixels > outlier_pixels_threshold: # check if we potentially need to divide it
                        _, _, w, h = tile
                        if min(w,h) > minimum_length: # check if it's possible to divide it based on reaching minimum length

                            # here we need to and we can divide it
                            # making new tiling by dividing only this tile
                            new_Tiling = np.delete(Tiling, tile_idx, 0)
                            subtiles = Divide_Tile(tile)
                            for subtile in subtiles:
                                in_pixels, out_pixels = in_out_count(subtile, mask)
                                if in_pixels >= outlier_pixels_threshold:
                                    new_Tiling = np.append(new_Tiling, subtile.astype(int).reshape((1,4)), axis = 0)
                                    # new_Tiling.append(subtile.astype(int))
                            new_Tiling = np.array(new_Tiling)
                            
                            # check if this new tiling exist!
                            exist = False
                            for Tiling_check in Super_Tile_boxes:
                                if np.array_equal(Tiling_check, new_Tiling):
                                    exist = True
                                    break
                            if not exist:
                                Super_Tile_boxes.append(new_Tiling)
                                Divide_flag = True
                
                checked_index += 1 # never divide this tiling config again :)

            print(f"num of tiling config: {len(Super_Tile_boxes)}")
            # if len(Super_Tile_boxes) == previous_num_tiling_configs:
            #     # we added all possible dividings
            #     Divide_flag = False

        # Loop through all configurations to select the best one in terms of FLOPs
        best_sub_config, best_flops = Super_Tile_boxes[0], Tiles2FLOPs(Super_Tile_boxes[0], margin)     
        for tile_config in Super_Tile_boxes[1:]:
            this_flops = Tiles2FLOPs(tile_config, margin)
            if this_flops < best_flops:
                best_sub_config = tile_config
                best_flops = this_flops

        # add best Tiling config for this subtile in initial tiling to final tilings :)
        for tile in best_sub_config:
            Tile_boxes_filtered.append(tile)

    
    Tile_boxes = np.array(Tile_boxes_filtered)

    # if Full_process_verbos:
    #     # If full history is wanted I am returning the whole history as tiles
    #     Tile_boxes = Tiling_hist

    return Tile_boxes

'''
Some Functions here to use inside the tiling methods
'''


def Count_Outside_Pixels(Tiles, mask):
    # Given Tiles set and the mask
    # It will count how many pixels are outside the field in this tiling config
    # I had this in every step of my codes but I am not using this data anymore, so moved it here
    # If I need it anywhere I could just call this now :)
    outside_pixels_count = 0
    for box in Tiles:
        x, y, w, h = box.astype(int)
        x_end = min(x + w, mask.shape[0])
        y_end = min(y + h, mask.shape[1])
        tile_mask = mask[x:x_end, y:y_end]
        outside_pixels_count += np.sum(tile_mask == 0)
    return outside_pixels_count

def Count_NumTiles_for_Pixels(mask, Tiles, margin):
    # Given the size (mask.shape) here, but I put mask inside the inputs just not to change the codes
    # and Tiles set and margin
    # This will return an array with same size as margin (and image) that the value of each pixel is number of tiles that pixel is in them
    # This might be usefull if you care about minimizing double (triple or quarople) counting of pixels!
    # Similar to count outside pixels I had this code everywhere, and now it's just a function to call
    Num_tiles_for_Pixels = np.zeros_like(mask)
    for box in Tiles:
        x, y, w, h = box.astype(int)
        x_hollow = max(0, x - margin)
        y_hollow = max(0, y - margin)
        x_end_hollow = min(mask.shape[0], x + margin)
        y_end_hollow = min(mask.shape[1], y + margin)
        Num_tiles_for_Pixels[x_hollow:x_end_hollow, y_hollow:y_end_hollow] += 1
    return Num_tiles_for_Pixels


def Divide_Tile(Tile):
    # I wrote this code to have the same Tile division code everywhere in my codes
    # Here I also solved a bug that I had with my previous codes!
    # The problem with dividing by 2 and making the final tiling integer was that the subtiles might have not covered the whole Initial Tile area!
    # So here I first int-divide the width and height and for the other subtiles instead of this length I consider the remaining integers is width and height
    # This function also works with dividing rectangular tiles, although I never had this problem since all the initial tiles were square and the division was always to square subtiles
    x, y, w, h = Tile.astype(int)

    w_sub_1 = w // 2
    w_sub_2 = w - w_sub_1
    h_sub_1 = h // 2
    h_sub_2 = h - h_sub_1

    subtiles = np.int32([
        [x          , y          , w_sub_1, h_sub_1],
        [x + w_sub_1, y          , w_sub_2, h_sub_1],
        [x          , y + h_sub_1, w_sub_1, h_sub_2],
        [x + w_sub_1, y + h_sub_1, w_sub_2, h_sub_2],
    ])
    return subtiles


def in_out_count(tile,mask):
    # Calculating number of pixels of tile inside and outside of the field
    # Since I am using this everywhere in the tilings I made one function for it
    
    x, y, w, h = tile.astype(int)
    x_end = min(x + w, mask.shape[0])
    y_end = min(y + h, mask.shape[1])

    tile_mask = mask[x : x_end, y : y_end]
    in_pixels = np.sum(tile_mask == 1)
    out_pixels = np.sum(tile_mask == 0)

    return in_pixels, out_pixels

def refine_out_image(Tile_boxes, img_shape):
    # In this function I am correcting width and heights of tiles that are outside of the image array!
    # the correction is in the way that x+w could become exactly equal to img_shape[0]
    # since the final index of image is img_shape[0]-1 and the tile should cover that!
    x_out_field_idxs = Tile_boxes[:,0] + Tile_boxes[:,2] >= img_shape[0]
    Tile_boxes[x_out_field_idxs , 2] = img_shape[0] - Tile_boxes[x_out_field_idxs, 0]
    y_out_field_idxs = Tile_boxes[:,1] + Tile_boxes[:,3] >= img_shape[1]
    Tile_boxes[y_out_field_idxs , 3] = img_shape[1] - Tile_boxes[y_out_field_idxs, 1] 
    return Tile_boxes

def Merge_Tiles(Tile_boxes, 
                Merge_max_length: int,
                method = None,
                num_permut = 1000,
                Tiles2FLOPs = None,
                margin = None,
                Full_process_verbos = False,
                print_flag = False):
    
    # I did try to merge more than one tile to tile_1, the result were'nt that different, or different at all! -> next idea was to collect all possible mergings
    #                   and then select best one with merging aspect ratio closer to 1! :)

    # Remember to consider margin in merge max length
    if print_flag:
        print(f"Num of Tiles before merge : {Tile_boxes.shape[0]}")
    if Full_process_verbos:
        Tiling_hist = {"step 0 of merging" : Tile_boxes.copy()}

    change_flag = True
    merge_step = 1

    if not method == "Permutation":
        num_permut = 1
    else:
        assert Tiles2FLOPs!=None , "For Permutation method, Tiles2FLOPs should be given"
        assert margin!=None, "For Permutation method, margin should be given"
        Initial_Tile_boxes = Tile_boxes.copy()
        Best_Tile_boxes = Tile_boxes.copy()
        

    for run in range(num_permut):

        if method == "Permutation":
            Tile_boxes = Initial_Tile_boxes[np.random.permutation(Initial_Tile_boxes.shape[0]), :]
            # print(f"permutation number {run}")

        while change_flag:

            if (method == 'SortBySize'):
                Tile_boxes = Tile_boxes[np.argsort(np.max(Tile_boxes[:,2:4], axis=1))[::-1], :]

            change_flag = False
            merged = []

            used = set()

            for i, tile_1 in enumerate(Tile_boxes):
                if i in used:
                    continue

                x1, y1, w1, h1 = tile_1
                
                possible_mergings = [] # merged x,y,w,h, then index, then aspect ratio

                for j, tile_2 in enumerate(Tile_boxes):
                    
                    if (j==i) or (j in used):
                        continue
                        
                    x2, y2, w2, h2 = tile_2

                    if (x1 == x2) and (w1 == w2) and (y2 == y1+h1) and (h1 + h2 <= Merge_max_length):
                        possible_mergings.append([x1, y1, w1, h1+h2, j, abs(1-(h1+h2)/w1)])

                    if (x1 == x2) and (w1 == w2) and (y1 == y2+h2) and (h1+h2 <= Merge_max_length):
                        possible_mergings.append([x1, y2, w1, h1+h2, j, abs(1-(h1+h2)/w1)])
                        
                    if (y1 == y2) and (h1 == h2) and (x2 == x1+w1) and (w1 + w2 <= Merge_max_length):
                        possible_mergings.append([x1, y1, w1+w2, h1, j, abs(1-h1/(w1+w2))])
                        
                    if (y1 == y2) and (h1 == h2) and (x1 == x2+w2) and (w1 + w2 <= Merge_max_length):
                        possible_mergings.append([x2, y1, w1+w2, h1, j, abs(1-h1/(w1+w2))])

                possible_mergings = np.array(possible_mergings)
                if len(possible_mergings) > 0:
                    aspect_ratios = possible_mergings[:, -1]
                    best_idx = np.argmin(aspect_ratios)
                    x_m, y_m, w_m, h_m = possible_mergings[best_idx, :4]
                    merged_idx = int(possible_mergings[best_idx, 4])
                    merged.append([int(x_m), int(y_m), int(w_m), int(h_m)])
                    used.update([i,merged_idx])
                    change_flag = True

            for i, tile in enumerate(Tile_boxes):
                if not i in used:
                    merged.append(tile)

            Tile_boxes = np.array(merged)
            
            merge_step += 1
        
            if Full_process_verbos:
                if not method == "Permutation":
                    Tiling_hist[f'step {merge_step} of merging'] = Tile_boxes.copy()
                else:
                    Tiling_hist[f'run: {run}, step {merge_step} of merging'] = Tile_boxes.copy()

        if method == "Permutation":
            # print(f"Tile FLOPs = {Tiles2FLOPs(Tile_boxes, margin=margin)}")
            if Tiles2FLOPs(Tile_boxes, margin=margin) < Tiles2FLOPs(Best_Tile_boxes, margin=margin):
                Best_Tile_boxes = Tile_boxes.copy()

    if method == "Permutation":
        Tile_boxes = Best_Tile_boxes

    if print_flag:
        print(f"Num of Tiles after merge : {Tile_boxes.shape[0]}")

    if Full_process_verbos:
        Tile_boxes = Tiling_hist

    return Tile_boxes




def merge_grid_collection(Tile_boxes,
                        Merge_max_length: int,
                        img_shape):

    
    x_col = np.unique(np.concatenate((Tile_boxes[:,0],Tile_boxes[:,0] + Tile_boxes[:,2]), axis=0))
    y_col = np.unique(np.concatenate((Tile_boxes[:,1],Tile_boxes[:,1] + Tile_boxes[:,3]), axis=0))

    # create union of tiles as a mask
    mask = np.zeros(img_shape)
    for tile in Tile_boxes:
        x, y, w, h = tile
        mask[x:x+w, y:y+h] = 1
    # t_0 = tick()
    All_possible_tiles = []
    for idx_x_0 , x_0 in enumerate(x_col):
        for idx_y_o, y_0 in enumerate(y_col):
            for x_e in x_col[idx_x_0+1:]:
                for y_e in y_col[idx_y_o+1:]:
                    # Filter on size
                    if (x_e - x_0 <= Merge_max_length) and (y_e - y_0 <= Merge_max_length):
                        tile = np.array([x_0, y_0, x_e-x_0, y_e-y_0])
                        _ , out_pixels = in_out_count(tile, mask) # mask here is union of tiles not the field!
                        # Filter on being in union of tiles
                        if out_pixels == 0:
                            All_possible_tiles.append(tile)
    # print(f"time {tick() - t_0:0.2f}")
    # print(len(All_possible_tiles))
    All_possible_tiles = np.array(All_possible_tiles)
    # Tile_boxes = All_possible_tiles
    # Covered = np.zeros((len(x_col)-1, len(y_col)-1))

    Tile_boxes_new = []
    while True:
        biggest_tile_idx = np.argmax(np.prod(All_possible_tiles[:, 2:4], axis=1))
        biggest_tile = All_possible_tiles[biggest_tile_idx, :]
        All_possible_tiles = np.delete(All_possible_tiles, biggest_tile_idx, axis=0)

        Tile_boxes_new.append(biggest_tile)

        x_s, y_s, w_s, h_s = biggest_tile

        # Remove all the tiles with intersection!
        All_possible_tiles_filtered = []
        for box in All_possible_tiles:

            x, y, w, h = box

            # Cond_x = ( (x>x_s) and (x<x_s+w_s) ) or ( (x+w>x_s) and (x+w<x_s+w_s) )
            # Cond_y = ( (y>y_s) and (y<y_s+h_s) ) or ( (y+h>y_s) and (y+h<y_s+h_s) )
            # Cond = Cond_x and Cond_y
            # if not Cond:
            #     All_possible_tiles_filtered.append(box)

            Cond = (x+w <= x_s) or (x >= x_s + w_s) or (y+h <= y_s) or (y >= y_s + h_s)
            if Cond:
                All_possible_tiles_filtered.append(box)

        All_possible_tiles = np.array(All_possible_tiles_filtered)

        # print(f"num remaining: {All_possible_tiles.shape[0]}")
        if len(All_possible_tiles) == 0:
            break
    
    Tile_boxes = np.array(Tile_boxes_new)

    return Tile_boxes

def Merge_Tiles_older(Tile_boxes, 
                Merge_max_length: int,
                method = None,
                num_permut = 1000,
                Tiles2FLOPs = None,
                margin = None,
                Full_process_verbos = False,
                print_flag = False):
    # Remember to consider margin in merge max length
    if print_flag:
        print(f"Num of Tiles before merge : {Tile_boxes.shape[0]}")
    if Full_process_verbos:
        Tiling_hist = {"step 0 of merging" : Tile_boxes.copy()}

    change_flag = True
    merge_step = 1

    if not method == "Permutation":
        num_permut = 1
    else:
        assert Tiles2FLOPs!=None , "For Permutation method, Tiles2FLOPs should be given"
        assert margin!=None, "For Permutation method, margin should be given"
        Initial_Tile_boxes = Tile_boxes.copy()
        Best_Tile_boxes = Tile_boxes.copy()
        

    for run in range(num_permut):

        print("Test")

        if method == "Permutation":
            Tile_boxes = Initial_Tile_boxes[np.random.permutation(Initial_Tile_boxes.shape[0]), :]
            # print(f"permutation number {run}")

        while change_flag:

            if (method == 'SortBySize'):
                Tile_boxes = Tile_boxes[np.argsort(np.max(Tile_boxes[:,2:4], axis=1))[::-1], :]

            change_flag = False
            merged = []

            used = set()

            for i, tile_1 in enumerate(Tile_boxes):
                if i in used:
                    continue

                x1, y1, w1, h1 = tile_1
                
                for j, tile_2 in enumerate(Tile_boxes):
                    
                    if (j==i) or (j in used):
                        continue
                        
                    x2, y2, w2, h2 = tile_2

                    if (x1 == x2) and (w1 == w2) and (y2 == y1+h1) and (h1 + h2 <= Merge_max_length):
                        merged.append([x1, y1, w1, h1+h2])
                        used.update([i, j])
                        change_flag = True
                        break

                    if (y1 == y2) and (h1 == h2) and (x2 == x1+w1) and (w1 + w2 <= Merge_max_length):
                        merged.append([x1, y1, w1+w2, h1])
                        used.update([i,j])
                        change_flag = True
                        break
                
            for i, tile in enumerate(Tile_boxes):
                if not i in used:
                    merged.append(tile)

            Tile_boxes = np.array(merged)
            
            merge_step += 1
        
            if Full_process_verbos:
                if not method == "Permutation":
                    Tiling_hist[f'step {merge_step} of merging'] = Tile_boxes.copy()
                else:
                    Tiling_hist[f'run: {run}, step {merge_step} of merging'] = Tile_boxes.copy()

        if method == "Permutation":
            # print(f"Tile FLOPs = {Tiles2FLOPs(Tile_boxes, margin=margin)}")
            if Tiles2FLOPs(Tile_boxes, margin=margin) < Tiles2FLOPs(Best_Tile_boxes, margin=margin):
                Best_Tile_boxes = Tile_boxes.copy()

    if method == "Permutation":
        Tile_boxes = Best_Tile_boxes

    if print_flag:
        print(f"Num of Tiles after merge : {Tile_boxes.shape[0]}")

    if Full_process_verbos:
        Tile_boxes = Tiling_hist

    return Tile_boxes