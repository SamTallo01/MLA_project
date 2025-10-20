import os
import cv2
import numpy as np
import openslide
from PIL import Image
from tqdm import tqdm
from wsi_core.wsi_utils import savePatchIter_bag_hdf5, initialize_hdf5_bag, isBlackPatch, isWhitePatch
from wsi_core.util_classes import  isInContourV2, Contour_Checking_fn

Image.MAX_IMAGE_PIXELS = 933120000

class WholeSlideImage(object):
    def __init__(self, path):

        """
        Args:
            path (str): fullpath to WSI file
        """

#         self.name = ".".join(path.split("/")[-1].split('.')[:-1])
        self.name = os.path.splitext(os.path.basename(path))[0]
        self.wsi = openslide.open_slide(path)
        self.level_downsamples = self._assertLevelDownsamples()
        self.level_dim = self.wsi.level_dimensions
    
        self.contours_tissue = None
        self.contours_tumor = None
        self.hdf5_file = None

    def getOpenSlide(self):
        return self.wsi
            
    def segmentTissue(self, seg_level=0, sthresh=20, sthresh_up = 255, mthresh=7, close = 0, use_otsu=False, 
                            filter_params={'a_t':100}, ref_patch_size=512, exclude_ids=[], keep_ids=[]):
        """
            Segment the tissue via HSV -> Median thresholding -> Binary threshold
        """
        
        def _filter_contours(contours, hierarchy, filter_params):
            """
                Filter contours by: area.
            """
            filtered = []

            # find indices of foreground contours (parent == -1)
            hierarchy_1 = np.flatnonzero(hierarchy[:,1] == -1)
            all_holes = []
            
            # loop through foreground contour indices
            for cont_idx in hierarchy_1:
                # actual contour
                cont = contours[cont_idx]
                # indices of holes contained in this contour (children of parent contour)
                holes = np.flatnonzero(hierarchy[:, 1] == cont_idx)
                # take contour area (includes holes)
                a = cv2.contourArea(cont)
                # calculate the contour area of each hole
                hole_areas = [cv2.contourArea(contours[hole_idx]) for hole_idx in holes]
                # actual area of foreground contour region
                a = a - np.array(hole_areas).sum()
                if a == 0: continue
                if tuple((filter_params['a_t'],)) < tuple((a,)): 
                    filtered.append(cont_idx)
                    all_holes.append(holes)


            foreground_contours = [contours[cont_idx] for cont_idx in filtered]
            
            hole_contours = []

            for hole_ids in all_holes:
                unfiltered_holes = [contours[idx] for idx in hole_ids ]
                unfilered_holes = sorted(unfiltered_holes, key=cv2.contourArea, reverse=True)
                # take max_n_holes largest holes by area
                unfilered_holes = unfilered_holes[:filter_params['max_n_holes']]
                filtered_holes = []
                
                # filter these holes
                for hole in unfilered_holes:
                    if cv2.contourArea(hole) > filter_params['a_h']:
                        filtered_holes.append(hole)

                hole_contours.append(filtered_holes)

            return foreground_contours, hole_contours
        
        img = np.array(self.wsi.read_region((0,0), seg_level, self.level_dim[seg_level]))
        img_hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)  # Convert to HSV space
        img_med = cv2.medianBlur(img_hsv[:,:,1], mthresh)  # Apply median blurring
        
       
        # Thresholding
        if use_otsu:
            _, img_otsu = cv2.threshold(img_med, 0, sthresh_up, cv2.THRESH_OTSU+cv2.THRESH_BINARY)
        else:
            _, img_otsu = cv2.threshold(img_med, sthresh, sthresh_up, cv2.THRESH_BINARY)

        # Morphological closing
        if close > 0:
            kernel = np.ones((close, close), np.uint8)
            img_otsu = cv2.morphologyEx(img_otsu, cv2.MORPH_CLOSE, kernel)                 

        scale = self.level_downsamples[seg_level]
        scaled_ref_patch_area = int(ref_patch_size**2 / (scale[0] * scale[1]))
        filter_params = filter_params.copy()
        filter_params['a_t'] = filter_params['a_t'] * scaled_ref_patch_area
        filter_params['a_h'] = filter_params['a_h'] * scaled_ref_patch_area
        
        # Find and filter contours
        contours, hierarchy = cv2.findContours(img_otsu, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE) # Find contours 
        hierarchy = np.squeeze(hierarchy, axis=(0,))[:, 2:]
        if filter_params: foreground_contours, hole_contours = _filter_contours(contours, hierarchy, filter_params)  # Necessary for filtering out artifacts

        self.contours_tissue = self.scaleContourDim(foreground_contours, scale)
        self.holes_tissue = self.scaleHolesDim(hole_contours, scale)

        #exclude_ids = [0,7,9]
        if len(keep_ids) > 0:
            contour_ids = set(keep_ids) - set(exclude_ids)
        else:
            contour_ids = set(np.arange(len(self.contours_tissue))) - set(exclude_ids)

        self.contours_tissue = [self.contours_tissue[i] for i in contour_ids]
        self.holes_tissue = [self.holes_tissue[i] for i in contour_ids]


    def createPatches_bag_hdf5(self, save_path, patch_level=0, patch_size=256, step_size=256, save_coord=True, **kwargs):
        contours = self.contours_tissue
        contour_holes = self.holes_tissue

        print("Creating patches for: ", self.name, "...",)
        
        total_estimated_patches = 0
        # Stima totale di patch
        for idx, cont in enumerate(contours):
            x, y, w, h = cv2.boundingRect(cont)
            nx = max(1, (w + step_size - 1) // step_size)
            ny = max(1, (h + step_size - 1) // step_size)
            total_estimated_patches += nx * ny

        patch_bar = tqdm(total=total_estimated_patches, desc=f"Patches for {self.name}")

        for idx, cont in enumerate(contours):
            patch_gen = self._getPatchGenerator(cont, idx, patch_level, save_path, patch_size, step_size, **kwargs)
            
            if self.hdf5_file is None:
                try:
                    first_patch = next(patch_gen)
                except StopIteration:
                    continue

                file_path = initialize_hdf5_bag(first_patch, save_coord=save_coord)
                self.hdf5_file = file_path
                patch_bar.update(1)

            for patch in patch_gen:
                savePatchIter_bag_hdf5(patch)
                patch_bar.update(1)

        patch_bar.close()
        return self.hdf5_file


    def _getPatchGenerator(self, cont, cont_idx, patch_level, save_path, patch_size=256, step_size=256,
        white_black=True, contour_fn='four_pt', use_padding=True):
        start_x, start_y, w, h = cv2.boundingRect(cont) if cont is not None else (0, 0, self.level_dim[patch_level][0], self.level_dim[patch_level][1])
        print("Bounding Box:", start_x, start_y, w, h)
        print("Contour Area:", cv2.contourArea(cont))
        

        patch_downsample = (int(self.level_downsamples[patch_level][0]), int(self.level_downsamples[patch_level][1]))
        ref_patch_size = (patch_size*patch_downsample[0], patch_size*patch_downsample[1])
        
        step_size_x = step_size * patch_downsample[0]
        step_size_y = step_size * patch_downsample[1]
        
        if isinstance(contour_fn, str):
            if contour_fn == 'four_pt':
                cont_check_fn = isInContourV2(contour=cont, patch_size=ref_patch_size[0])
            else:
                raise NotImplementedError
        else:
            assert isinstance(contour_fn, Contour_Checking_fn)
            cont_check_fn = contour_fn

        img_w, img_h = self.level_dim[0]
        if use_padding:
            stop_y = start_y+h
            stop_x = start_x+w
        else:
            stop_y = min(start_y+h, img_h-ref_patch_size[1])
            stop_x = min(start_x+w, img_w-ref_patch_size[0])

        count = 0
        for y in range(start_y, stop_y, step_size_y):
            for x in range(start_x, stop_x, step_size_x):

                if not self.isInContours(cont_check_fn, (x,y), self.holes_tissue[cont_idx], ref_patch_size[0]): #point not inside contour and its associated holes
                    continue    
                
                count+=1
                patch_PIL = self.wsi.read_region((x,y), patch_level, (patch_size, patch_size)).convert('RGB')

                if white_black:
                    if isBlackPatch(np.array(patch_PIL)) or isWhitePatch(np.array(patch_PIL)): 
                        continue
                
                patch_info = {'x':x // (patch_downsample[0]), 'y':y // (patch_downsample[1]), 'cont_idx':cont_idx, 'patch_level':patch_level, 
                'downsample': self.level_downsamples[patch_level], 'downsampled_level_dim': tuple(np.array(self.level_dim[patch_level])), 'level_dim': self.level_dim[patch_level],
                'patch_PIL':patch_PIL, 'name':self.name, 'save_path':save_path}

                yield patch_info

        
        print("patches extracted: {}".format(count))

    @staticmethod
    def isInHoles(holes, pt, patch_size):
        for hole in holes:
            if cv2.pointPolygonTest(hole, (pt[0]+patch_size/2, pt[1]+patch_size/2), False) > 0:
                return 1
        
        return 0

    @staticmethod
    def isInContours(cont_check_fn, pt, holes=None, patch_size=256):
        if cont_check_fn(pt):
            if holes is not None:
                return not WholeSlideImage.isInHoles(holes, pt, patch_size)
            else:
                return 1
        return 0
    
    @staticmethod
    def scaleContourDim(contours, scale):
        return [np.array(cont * scale, dtype='int32') for cont in contours]

    @staticmethod
    def scaleHolesDim(contours, scale):
        return [[np.array(hole * scale, dtype = 'int32') for hole in holes] for holes in contours]

    def _assertLevelDownsamples(self):
        level_downsamples = []
        dim_0 = self.wsi.level_dimensions[0]
        
        for downsample, dim in zip(self.wsi.level_downsamples, self.wsi.level_dimensions):
            estimated_downsample = (dim_0[0]/float(dim[0]), dim_0[1]/float(dim[1]))
            level_downsamples.append(estimated_downsample) if estimated_downsample != (downsample, downsample) else level_downsamples.append((downsample, downsample))
        
        return level_downsamples

    @staticmethod
    def process_coord_candidate(coord, contour_holes, ref_patch_size, cont_check_fn):
        if WholeSlideImage.isInContours(cont_check_fn, coord, contour_holes, ref_patch_size):
            return coord
        else:
            return None
