'''
Functional toolbox for optimization process
Mostly related to calculation of FLOPs

Methods:
    Tile2FLOPs: Calculating FLOPs for a given tile and tabular flops data
    Tiles2FLOPs: Calculating FLOPs for a given tiles set (configuration) and tabular flops data
    Tiles2TotalFLOPs: wrapper to sum TIles2FLOPs (needed to seperat them for indivisual result)
    P2Tiles: Collocated points -> Tile configuration for given mask
    Tiles2Prob: Tile configuration to vector of probability for given projected heatmap
    in_out_count: Count num of pixels inside and outside of the mask for a given tile 
    RejNetFLOPs: constant FLOPs for given tile configuration related to non-shared block
    Merge: Merge loop for a given tile configuration
    PlotTilesProbOnImage
    PlotTilesOnImage

Objects:
    FLOPsCalculators: Non-statistical and statistical Flops for given tile config or directly from collocated points
    
Project:
    EAST-SPL
    https://github.com/AbolfazlChM95/EAST-SPL
'''

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches


def Tile2FLOPs(FLOPs_array, shift, w, h, margin=0):
    """Return FLOPs for one or more tile dimensions from a tabular FLOPs as numpy array."""
    i, j = w - shift + 2*margin, h - shift + 2 * margin
    if np.any(i<0) or np.any(j<0):
        print(f"w: {w}, h:{h}")
        raise IndexError("width or height are small")
    return FLOPs_array[i, j]


def Tiles2FLOPs(Tiles, FLOPs_array, shift, margin=0):
    """Return FLOPs for one or more tile dimensions from a tabular FLOPs as numpy array.
    Simple seperation of single tile and multiple tiles here
    Main function is Tile2FLOPs()
    """
    if Tiles.ndim == 1:
        w, h = Tiles[2], Tiles[3]
    else:
        w, h = Tiles[:,2], Tiles[:,3]
    
    FLOPs = Tile2FLOPs(FLOPs_array, shift, w, h, margin)
    
    return FLOPs

def Tiles2TotalFLOPs(Tiles, FLOPs_array, shift, margin=0):
    """Return total FLOPs for given tile configuration: simply sum over Tiles2FLOPs()"""
    FLOPs = Tiles2FLOPs(Tiles, FLOPs_array, shift, margin)
    FLOPs = np.sum(FLOPs)
    return FLOPs


def P2Tiles(P, mask = None, merge = False, merge_max_length = 1000):
    """
    Construct tile configuration for a given collocation points, 
    for (optionally) given mask,
    (optionally) merging the tile at the end
    """
    x, y = P
    x, y = np.asarray(x), np.asarray(y)
    w = x[1:] - x[:-1]
    h = y[1:] - y[:-1]

    Tiles = []
    for i,x_ in enumerate(x[:-1]):
        for j,y_ in enumerate(y[:-1]):
            Tiles.append([x_, y_, w[i], h[j]])
    Tiles = np.array(Tiles)

    if not mask is None:
        Tiles_filtered = []
        for tile in Tiles:
            in_pixels , _ = in_out_count(tile, mask)
            if in_pixels>0:
                Tiles_filtered.append(tile)
        Tiles = np.array(Tiles_filtered)
    
    if merge:
        Tiles = Merge(Tiles, merge_max_length)
    return Tiles

def Tiles2Prob(Tiles, projected_statistics):
    """
    Return (approximate) probability of presence of players in each tile for a given tile configuration
    """
    Probs = []
    for tile in Tiles:
        x, y, w, h = tile.astype(int)
        frame_tile = projected_statistics[x:x+w, y:y+h]
        Probs.append(np.sum(frame_tile))
    
    # return np.array(Probs)
    return np.clip(np.array(Probs), a_min=0, a_max=1)

def RejNetFLOPs(Tiles):
    """
    Return FLOPs of non-shared rejection head for a given tile configuration

    for simple MLP tested in the EAST-SPL paper
    # 128, 64, 32, 16: 26114
    # 64, 32, 16, 8: 7682
    # 64, 32, 16: 7426
    # 32, 16, 8: 2434
    # 32, 16: 2178
    # 16, 8: 834
    # 8, 4: 354
    """

    if len(Tiles)>0:
        total_flops = 26114 * np.ones_like(Tiles[:,1])
    else:
        total_flops = 0
        
    return total_flops

def in_out_count(tile,mask):
    """
    Return number of pixels inside and outside of the given mas to the given tile
    """
    x, y, w, h = tile.astype(int)
    x_end = min(x + w, mask.shape[0])
    y_end = min(y + h, mask.shape[1])
    tile_mask = mask[x : x_end, y : y_end]
    in_pixels = np.sum(tile_mask == 1)
    out_pixels = np.sum(tile_mask == 0)
    return in_pixels, out_pixels

class FLOPsCalculators():
    """
    Calculate statistical and non-statistical FLOPs for given tile configurations or Collocation points.

    The statistical objective combines rejection-head cost, shared-feature cost,
    primary-network cost, and projected tile probabilities.
    """
    def __init__(self, mask, RejectionHeadFLOPs, MainNetFLOPs, MainNetShift, SharedFLOPs, SharedShift, projectedStats, margin):
        self.mask = mask
        self.Rejectionhead = RejectionHeadFLOPs
        self.MainNet_FLOPs = MainNetFLOPs
        self.MainNet_Shift = MainNetShift
        self.Shared_FLOPs = SharedFLOPs
        self.Shared_Shift = SharedShift
        self.Stats = projectedStats
        self.margin = margin

    def StatFLOPsPoints(self, P):
        Tiles = P2Tiles(P, mask = self.mask)
        return self.StatFLOPsTiles(Tiles)
    
    def NonStatFLOPsPoints(self, P, merge = True, max_len = 1000):
        Tiles = P2Tiles(P, mask = self.mask)
        if merge:
            Tiles = Merge(Tiles, max_len)
        return self.NonStatFLOPsTiles(Tiles)

    def StatFLOPsTiles(self, Tiles):
        l1 = self.Rejectionhead(Tiles)
        l2 = Tiles2FLOPs(Tiles, self.MainNet_FLOPs, self.MainNet_Shift, self.margin)
        l3 = Tiles2FLOPs(Tiles, self.Shared_FLOPs, self.Shared_Shift, self.margin)
        weights = Tiles2Prob(Tiles, self.Stats)
        return np.sum(l1 + l3 + weights * (l2-l3))

    def NonStatFLOPsTiles(self, Tiles):
        l = Tiles2TotalFLOPs(Tiles, self.MainNet_FLOPs, self.MainNet_Shift, self.margin)
        return l
    
def Merge(tile_boxes, merge_max_length: int):
    """
    From DTSPL-BEV
    Return a tile config after vertically and horizontally until convergence of inputted tile config 
    """
    # DTSPL-BEV vertical and horizontal merging
    change_flag = True
    
    while change_flag:
        change_flag = False
        merged = []
        used = set()
        for i, tile_1 in enumerate(tile_boxes):
            if i in used:
                continue
            x1, y1, w1, h1 = tile_1
            for j, tile_2 in enumerate(tile_boxes):
                if (j==i) or (j in used):
                    continue
                x2, y2, w2, h2 = tile_2
                if (x1 == x2) and (w1 == w2) and (y2 == y1+h1) and (h1 + h2 <= merge_max_length):
                    merged.append([x1, y1, w1, h1+h2])
                    used.update([i, j])
                    change_flag = True
                    break
                if (y1 == y2) and (h1 == h2) and (x2 == x1+w1) and (w1 + w2 <= merge_max_length):
                    merged.append([x1, y1, w1+w2, h1])
                    used.update([i,j])
                    change_flag = True
                    break
            
        for i, tile in enumerate(tile_boxes):
            if not i in used:
                merged.append(tile)
        tile_boxes = np.array(merged)
    return tile_boxes

def PlotTilesProbOnImage(Tiles, img, projectedStats, title = None):
    probs = Tiles2Prob(Tiles, projectedStats)
    fig , ax = plt.subplots(figsize=(15, 12))
    ax.imshow(img)
    for i,tile in enumerate(Tiles):
        x, y, w, h = tile
        rect = patches.Rectangle(
                    (y, x), h, w,
                    linewidth=3,
                    edgecolor='red',
                    facecolor='white',
                    alpha = 0.4
                )
        ax.add_patch(rect)
        ax.text(y,x,f"{probs[i]*100:.1f}%",color='white',fontsize=12,fontweight='bold',va='top',ha='left')
    if title:
        plt.title(title)
    plt.show()

def PlotTilesOnImage(Tiles, img, title = None):
    fig , ax = plt.subplots(figsize=(15, 12))
    ax.imshow(img)
    for i,tile in enumerate(Tiles):
        x, y, w, h = tile
        rect = patches.Rectangle(
                    (y, x), h, w,
                    linewidth=3,
                    edgecolor='red',
                    facecolor='white',
                    alpha = 0.4
                )
        ax.add_patch(rect)
    if title:
        plt.title(title)
    plt.show()
    